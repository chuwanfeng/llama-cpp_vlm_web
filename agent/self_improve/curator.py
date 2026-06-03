"""
Skill Curator — 移植自 hermes-agent/agent/curator.py

Automatic lifecycle management for agent-created skills:
  - Transitions: active → stale → archived (based on usage inactivity)
  - LLM review: periodically reviews all agent-created skills, suggests
    consolidations (merge similar) and prunes (remove dead)
  - State persistence: data/curator_state.json
  - Reports: data/curator/YYYY-MM-DDTHHMMSS/

Key differences from hermes-agent:
  - No cron rewriting (no cron infrastructure)
  - No backup snapshots (no backup infra)
  - No hub/bundled distinction
  - Uses AgentLoop for LLM review (not AIAgent)
  - Simpler config (env-based, not hermes_cli.config)

Lifecycle:
    active   → default, in library
    stale    → no activity > stale_after_days (default 14)
    archived → no activity > archive_after_days (default 30); moved to .archive/
    pinned   → opt-out from auto transitions
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from config import PROJECT_ROOT

logger = logging.getLogger(__name__)

CURATOR_DIR = Path(PROJECT_ROOT) / "data" / "curator"
STATE_PATH = Path(PROJECT_ROOT) / "data" / "curator_state.json"

from services import skill_usage as _su


# ── Curator state ──────────────────────────────────────────────────────────


def _load_state() -> Dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_state(state: Dict[str, Any]) -> None:
    """Atomically write curator state to disk (mkstemp + replace)."""
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    import os as _os
    fd, tmp = tempfile.mkstemp(
        dir=str(STATE_PATH.parent),
        prefix=".curator_state_",
        suffix=".tmp",
    )
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=True, ensure_ascii=False)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp, str(STATE_PATH))
    except BaseException:
        try:
            _os.unlink(tmp)
        except OSError:
            pass
        raise




# ── Pause control ─────────────────────────────────────────────────────


def set_paused(paused: bool) -> None:
    """Pause or unpause the curator."""
    state = _load_state()
    state["paused"] = bool(paused)
    _save_state(state)


def is_paused() -> bool:
    """Return True if the curator is paused."""
    return bool(_load_state().get("paused"))

# ── Config helpers ─────────────────────────────────────────────────────────




def should_run_now(now: Optional[datetime] = None) -> bool:
    """Return True if the curator should run now.

    Gates:
      - Not paused
      - last_run_at is set AND older than interval_hours

    First-run behavior: seed last_run_at to now and defer.
    """
    if is_paused():
        return False

    state = _load_state()
    last = _parse_iso(state.get("last_run_at"))
    if last is None:
        # Seed on first observation
        if now is None:
            now = datetime.now(timezone.utc)
        try:
            state["last_run_at"] = now.isoformat()
            state["last_run_summary"] = (
                "deferred first run - will run after one interval"
            )
            _save_state(state)
        except Exception as e:
            logger.debug("Failed to seed last_run_at: %s", e)
        return False

    if now is None:
        now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    interval = timedelta(hours=_get_config_value("interval_hours", 168))  # default 7 days
    return (now - last) >= interval


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string to datetime."""
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except (TypeError, ValueError):
        return None


def _get_config_value(key: str, default: Any = None) -> Any:
    """Read config from env: CURATOR_<KEY> (uppercase, underscores)."""
    env_key = f"CURATOR_{key.upper()}"
    val = os.environ.get(env_key)
    if val is not None:
        if isinstance(default, int):
            try:
                return int(val)
            except ValueError:
                pass
        if isinstance(default, float):
            try:
                return float(val)
            except ValueError:
                pass
        if isinstance(default, bool):
            return val.lower() in ("1", "true", "yes", "on")
        return val
    return default


# ── LLM review prompt ──────────────────────────────────────────────────────


_CURATOR_SYSTEM_PROMPT = """
You are a skill librarian. Your job is to review a catalog of agent-created
skills and recommend which ones to keep, consolidate, or remove.

For each skill you review, classify it as:
- KEEP: useful, well-defined, actively relevant
- CONSOLIDATE: overlaps with another skill; merge into the better-defined one
- REMOVE: obsolete, superseded, too narrow, or never useful

Additional rules:
- Be conservative. When in doubt, KEEP.
- Pinned skills are off-limits — never CONSOLIDATE or REMOVE them.
- Stale skills are candidates for REMOVE, but only if truly dead.
- When consolidating, the best-defined skill should absorb the weaker one.
- Provide a brief reason for each recommendation.

Output a JSON object with this schema:
{
  "recommendations": [
    {
      "skill_name": "...",
      "action": "KEEP" | "CONSOLIDATE" | "REMOVE",
      "reason": "...",
      "target": "skill to absorb into (CONSOLIDATE only)"
    }
  ]
}
"""

_CURATOR_USER_PROMPT_TEMPLATE = """Review the following agent-created skills:

{skills_inventory}

Identify which skills to KEEP, CONSOLIDATE (merge similar), or REMOVE (dead).
Respond with a JSON object containing the recommendations.
Be concise: no more than 1-2 sentences per reason."""


# ── State machine transitions ──────────────────────────────────────────────



def _strip_aux_credential(value: Any) -> Optional[str]:
    """Normalize an auxiliary credential value (None → None, empty → None)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_review_runtime(
    backend_type: str = "vendor",
    vendor_id: str = "deepseek",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve provider/model and optional credentials for the curator review.

    Precedence (highest → lowest):
      1. Explicit arguments (backend_type, vendor_id, model)
      2. Environment overrides (CURATOR_PROVIDER, CURATOR_MODEL, etc.)
      3. config.py settings (AUXILIARY_PROVIDER, AUXILIARY_MODEL, etc.)
      4. Main chat model as fallback

    Returns dict with keys: provider, model, api_key (optional), base_url (optional)
    """
    result = {
        "backend_type": backend_type,
        "vendor_id": vendor_id,
        "model": model,
        "api_key": None,
        "base_url": None,
    }

    # 1. Check env overrides
    env_provider = _get_config_value("provider", None)
    env_model = _get_config_value("model", None)
    env_api_key = _strip_aux_credential(_get_config_value("api_key", None))
    env_base_url = _strip_aux_credential(_get_config_value("base_url", None))

    if env_provider:
        # Override backend_type/vendor_id based on provider
        if env_provider in ("openai", "deepseek", "anthropic", "gemini", "qwen", "zhipu"):
            result["backend_type"] = "vendor"
            result["vendor_id"] = env_provider
        else:
            result["backend_type"] = env_provider
        if env_model:
            result["model"] = env_model
        result["api_key"] = env_api_key
        result["base_url"] = env_base_url
        return result

    # 2. Use arguments (already set in result dict)
    if result["model"]:
        return result

    # 3. Fallback: use main chat model from config
    try:
        from config import LLM_PROVIDER, LLM_MODEL
        if result["backend_type"] == "vendor":
            result["vendor_id"] = LLM_PROVIDER
        result["model"] = LLM_MODEL
    except ImportError:
        pass

    return result

def apply_automatic_transitions() -> Dict[str, Any]:
    """Step through all agent-created skills and update lifecycle states.

    Returns a summary dict: {stale_count, archive_count, stale_names, archived_names}
    """
    stale_days = _get_config_value("stale_after_days", 14)
    archive_days = _get_config_value("archive_after_days", 30)
    now = datetime.now(timezone.utc)

    stale_names: List[str] = []
    archived_names: List[str] = []

    for name in _su.list_agent_created_skill_names():
        record = _su.get_record(name)
        if record.get("pinned"):
            continue

        if record.get("state") == _su.STATE_ARCHIVED:
            continue

        last_act = _su.latest_activity_at(record)
        # If no activity ever, use created_at as fallback
        anchor = last_act or record.get("created_at")
        day = _su._parse_iso_timestamp(anchor)
        if day is None:
            continue

        days_since = (now - day).days

        if record.get("state") == _su.STATE_ACTIVE and days_since >= stale_days:
            _su.set_state(name, _su.STATE_STALE)
            stale_names.append(name)

        if days_since >= archive_days and record.get("state") != _su.STATE_ARCHIVED:
            ok, msg = _su.archive_skill(name)
            if ok:
                archived_names.append(name)
                logger.info("Curator auto-archived: %s (%s)", name, msg)
            else:
                logger.warning("Curator auto-archive failed: %s (%s)", name, msg)

    return {
        "stale_count": len(stale_names),
        "archive_count": len(archived_names),
        "stale_names": stale_names,
        "archived_names": archived_names,
    }


# ── LLM review pass ────────────────────────────────────────────────────────


def _build_skills_inventory(skill_names: List[str]) -> str:
    """Build a text inventory of skills for the LLM review prompt."""
    lines = []
    for name in sorted(skill_names):
        rec = _su.get_record(name)
        state = rec.get("state", _su.STATE_ACTIVE)
        pinned = " [PINNED]" if rec.get("pinned") else ""
        last_act = _su.latest_activity_at(rec) or rec.get("created_at") or "unknown"
        act_count = _su.activity_count(rec)
        lines.append(
            f"- {name} | state={state}{pinned} | activity_count={act_count} | "
            f"last_activity={last_act[:19] if last_act != 'unknown' else last_act}"
        )
    return "\n".join(lines)


@dataclass
class CuratorResult:
    """Curator review 执行结果。"""
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    applied_keep: int = 0
    applied_consolidate: int = 0
    applied_remove: int = 0
    errors_applying: int = 0
    auto_stale: int = 0
    auto_archived: int = 0
    dry_run: bool = False
    report_path: Optional[str] = None
    llm_raw: Optional[str] = None


def run_curator_review(
    backend_type: str = "vendor",
    vendor_id: str = "deepseek",
    model: str = None,
    dry_run: bool = False,
    min_skills: int = 3,
) -> CuratorResult:
    """Run a full curator review pass.

    1. Apply automatic transitions (stale/archived)
    2. Gather all agent-created skills
    3. Run LLM review pass for consolidation/pruning recommendations
    4. Apply recommendations (or just report in dry-run mode)
    5. Generate report

    When backend_type/vendor_id/model are defaults ("vendor", "deepseek", None),
    _resolve_review_runtime() is called to resolve from CURATOR_* env vars.

    Args:
        backend_type: LLM backend for review
        vendor_id: vendor ID
        model: model name (None = resolve from env)
        dry_run: if True, only report recommendations without applying
        min_skills: minimum agent-created skills to trigger review

    Returns:
        CuratorResult with summary
    """
    # Resolve runtime from config/env when using defaults
    if model is None:
        runtime = _resolve_review_runtime(backend_type, vendor_id, model)
        backend_type = runtime["backend_type"]
        vendor_id = runtime["vendor_id"]
        model = runtime["model"]

    result = CuratorResult(dry_run=dry_run)

    # Step 1: Auto-transitions
    transitions = apply_automatic_transitions()
    result.auto_stale = transitions["stale_count"]
    result.auto_archived = transitions["archive_count"]

    # Step 2: Gather skills
    skill_names = _su.list_agent_created_skill_names()
    active_names = [
        n for n in skill_names
        if _su.get_record(n).get("state") != _su.STATE_ARCHIVED
    ]

    if len(active_names) < min_skills:
        logger.debug(
            "Curator skip: %d agent-created skills (<%d minimum)",
            len(active_names), min_skills,
        )
        return result

    # Step 3: LLM review pass
    try:
        llm_recommendations = _llm_review_pass(
            skill_names=active_names,
            backend_type=backend_type,
            vendor_id=vendor_id,
            model=model,
        )
    except Exception as e:
        logger.warning("Curator LLM review failed: %s", e)
        result.errors_applying = 1
        llm_recommendations = []

    result.recommendations = llm_recommendations
    result.llm_raw = json.dumps(llm_recommendations, indent=2, ensure_ascii=False)

    # Step 4: Apply recommendations
    if not dry_run:
        for rec in llm_recommendations:
            action = rec.get("action", "").upper()
            name = rec.get("skill_name", "")

            if not name:
                result.errors_applying += 1
                continue

            record = _su.get_record(name)

            if action == "CONSOLIDATE":
                target = rec.get("target", "")
                if target:
                    ok, msg = _apply_consolidation(name, target)
                    if ok:
                        result.applied_consolidate += 1
                        logger.info("Curator consolidated: %s → %s", name, target)
                    else:
                        result.errors_applying += 1
                        logger.warning("Curator consolidate failed: %s", msg)
                else:
                    result.errors_applying += 1

            elif action == "REMOVE":
                if record.get("pinned"):
                    logger.debug("Curator skip remove: %s is pinned", name)
                    continue
                ok, msg = _su.archive_skill(name)
                if ok:
                    result.applied_remove += 1
                    logger.info("Curator removed: %s", name)
                else:
                    result.errors_applying += 1
                    logger.warning("Curator remove failed: %s", msg)

            elif action == "KEEP":
                result.applied_keep += 1
    else:
        # Dry run: count recommendations
        for rec in llm_recommendations:
            action = rec.get("action", "").upper()
            if action == "KEEP":
                result.applied_keep += 1
            elif action == "CONSOLIDATE":
                result.applied_consolidate += 1
            elif action == "REMOVE":
                result.applied_remove += 1

    # Step 5: Generate report
    result.report_path = _write_report(result)
    _update_state_after_run(result)

    return result


def _llm_review_pass(
    skill_names: List[str],
    backend_type: str,
    vendor_id: str,
    model: str = None,
) -> List[Dict[str, Any]]:
    """Run the LLM review pass to get consolidation/pruning recommendations.

    Uses AgentLoop with limited tools (none — just a classification pass).
    """
    inventory = _build_skills_inventory(skill_names)
    prompt = _CURATOR_USER_PROMPT_TEMPLATE.format(skills_inventory=inventory)

    import asyncio
    from agent.loop import AgentLoop

    loop = AgentLoop(
        backend_type=backend_type,
        vendor_id=vendor_id,
        model=model,
        tool_schemas=[],  # No tools for this classification pass
        max_turns=1,
        temperature=0.3,
    )

    messages = [
        {"role": "system", "content": _CURATOR_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    async def _run():
        return await loop.run(messages)

    result = asyncio.run(_run())

    if not result.messages:
        return []

    # Extract the last assistant message
    assistant_msgs = [
        m for m in result.messages if m.get("role") == "assistant"
    ]
    if not assistant_msgs:
        return []

    last_msg = assistant_msgs[-1].get("content", "")
    return _parse_llm_response(last_msg)


def _parse_llm_response(text: str) -> List[Dict[str, Any]]:
    """Parse JSON from LLM response text (may contain markdown fences)."""
    # Try to extract JSON from markdown code fence
    import re

    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        text = json_match.group(1)
    else:
        # Try to find bare JSON object
        json_match = re.search(r'\{[\s\S]*"recommendations"[\s\S]*\}', text)
        if json_match:
            text = json_match.group(0)

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            recs = data.get("recommendations", [])
            if isinstance(recs, list):
                return recs
    except json.JSONDecodeError:
        logger.debug("Curator: failed to parse LLM response as JSON")

    return []


# ── Consolidation logic ────────────────────────────────────────────────────


def _apply_consolidation(from_name: str, into_name: str) -> Tuple[bool, str]:
    """Consolidate one skill into another.

    Strategy: merge the from-skill's SKILL.md content into into-skill's,
    then archive the from-skill.
    """
    from_dir = _su._find_skill_dir(from_name)
    into_dir = _su._find_skill_dir(into_name)

    if from_dir is None:
        return False, f"source skill '{from_name}' not found"
    if into_dir is None:
        return False, f"target skill '{into_name}' not found"

    # Read from skill content
    from_skill_md = from_dir / "SKILL.md"
    if not from_skill_md.exists():
        return False, f"source SKILL.md missing for '{from_name}'"

    from_content = from_skill_md.read_text(encoding="utf-8", errors="replace")

    # Merge into target: append a consolidated section
    into_skill_md = into_dir / "SKILL.md"
    into_content = into_skill_md.read_text(encoding="utf-8", errors="replace")

    merge_section = (
        f"\n\n---\n\n"
        f"## Consolidated from `{from_name}` ({datetime.now(timezone.utc).isoformat()[:19]})\n\n"
        f"{from_content}\n"
    )
    into_skill_md.write_text(into_content + merge_section, encoding="utf-8")

    # Archive the source
    ok, msg = _su.archive_skill(from_name)
    if not ok:
        return False, f"consolidated content but failed to archive source: {msg}"

    return True, f"consolidated '{from_name}' → '{into_name}'"


# ── Report generation ──────────────────────────────────────────────────────


def _write_report(result: CuratorResult) -> Optional[str]:
    """Write a curator run report to data/curator/<timestamp>/."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%S")
    run_dir = CURATOR_DIR / ts
    try:
        run_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None

    # Write run.json (detailed)
    run_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": result.dry_run,
        "applied_keep": result.applied_keep,
        "applied_consolidate": result.applied_consolidate,
        "applied_remove": result.applied_remove,
        "errors_applying": result.errors_applying,
        "auto_stale": result.auto_stale,
        "auto_archived": result.auto_archived,
        "recommendations": result.recommendations,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Write REPORT.md (human-readable)
    lines = [
        f"# Curator Run — {ts}",
        f"",
        f"- **Dry run**: {result.dry_run}",
        f"- **Auto transitions**: {result.auto_stale} marked stale, {result.auto_archived} auto-archived",
        f"",
        f"## LLM Recommendations",
        f"",
    ]
    for rec in result.recommendations:
        lines.append(
            f"- **{rec.get('action', '?')}** `{rec.get('skill_name', '?')}`: "
            f"{rec.get('reason', 'no reason')}"
        )

    if not result.recommendations:
        lines.append("_No recommendations returned by the LLM review pass._")

    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return str(run_dir)


def _update_state_after_run(result: CuratorResult) -> None:
    """Update curator state to record this run."""
    state = _load_state()
    if "runs" not in state:
        state["runs"] = []
    state["runs"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "applied_consolidate": result.applied_consolidate,
        "applied_remove": result.applied_remove,
        "auto_archived": result.auto_archived,
        "dry_run": result.dry_run,
    })
    # Keep last 50 runs
    state["runs"] = state["runs"][-50:]
    state["last_run"] = datetime.now(timezone.utc).isoformat()
    _save_state(state)


# ── maybe_run_curator — trigger-gated entry ────────────────────────────────


def maybe_run_curator(
    backend_type: str = "vendor",
    vendor_id: str = "deepseek",
    model: str = None,
    dry_run: bool = False,
    force: bool = False,
) -> Optional[CuratorResult]:
    """Entry point: run curator if conditions are met.

    Conditions (from config / env):
    - min_skills: minimum agent-created skills (default 3)
    - min_interval_hours: minimum hours between runs (default 6)

    Set force=True to bypass all conditions.
    """
    # Check interval using should_run_now()
    if not force and not should_run_now():
        logger.debug("Curator skip: not due yet (should_run_now returned False)")
        return None

    # Check skill count
    min_skills = _get_config_value("min_skills", 3)
    skill_names = _su.list_agent_created_skill_names()
    active = [
        n for n in skill_names
        if _su.get_record(n).get("state") != _su.STATE_ARCHIVED
    ]

    if not force and len(active) < min_skills:
        logger.debug(
            "Curator skip: %d agent-created skills (<%d minimum)",
            len(active), min_skills,
        )
        return None

    return run_curator_review(
        backend_type=backend_type,
        vendor_id=vendor_id,
        model=model,
        dry_run=dry_run,
        min_skills=0,  # Already checked above
    )