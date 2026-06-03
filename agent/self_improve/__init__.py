"""
自我进化模块（Self-Improvement）— 移植自 hermes-agent

核心机制：
- review.py: 后台 review fork，扫描对话发现模式/偏好/错误
- provenance.py: write origin 追踪（ContextVar 区分 foreground/background_review）
- curator.py: 自动管理 agent_created skills（合并/剪枝/归档）
"""

from agent.self_improve.provenance import (
    set_current_write_origin,
    reset_current_write_origin,
    get_current_write_origin,
    is_background_review,
    BACKGROUND_REVIEW,
)
from agent.self_improve.review import (
    spawn_background_review,
    ReviewResult,
)
from agent.self_improve.curator import (
    CuratorResult,
    run_curator_review,
    maybe_run_curator,
    apply_automatic_transitions,
)

__all__ = [
    # provenance
    "set_current_write_origin",
    "reset_current_write_origin",
    "get_current_write_origin",
    "is_background_review",
    "BACKGROUND_REVIEW",
    # review
    "spawn_background_review",
    "ReviewResult",
    # curator
    "CuratorResult",
    "run_curator_review",
    "maybe_run_curator",
    "apply_automatic_transitions",
]