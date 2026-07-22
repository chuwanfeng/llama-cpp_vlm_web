"""
图片/视频生成配置管理

配置文件: settings.json 中的 generation 键
可通过 /api/generation/config 读写

支持:
  - ComfyUI 本地/云端 URL 配置
  - 自定义工作流注册 (名称 → workflow JSON 文件路径或直接 JSON)
  - 工作流目录扫描
  - 热重载 (无需重启)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "settings.json")

# ── 默认生成配置 ──
DEFAULT_GEN_CONFIG: Dict[str, Any] = {
    "version": 1,
    "image": {
        "local_comfyui_url": "http://127.0.0.1:8188",
        "cloud_comfyui_url": "",
    },
    "video": {
        "cloud_comfyui_url": "",
        "workflow_dir": "",          # 自定义工作流 JSON 目录
    },
    "workflows": {
        # "z-image-turo": {
        #     "type": "image",        # image / video
        #     "source": "comfyui",    # comfyui / file
        #     "display_name": "Z-Image Turbo",
        #     "description": "快速图生图工作流",
        #     "workflow": { ... }     # 或 "workflow_file": "path/to/workflow.json"
        # },
    },
}

# 内存缓存
_config_cache: Optional[Dict[str, Any]] = None


def read_config() -> Dict[str, Any]:
    """读取设置文件中的 generation 配置"""
    global _config_cache
    if _config_cache is not None:
        return _config_cache
    try:
        raw = json.loads(Path(SETTINGS_PATH).read_text(encoding="utf-8"))
        cfg = raw.get("generation", dict(DEFAULT_GEN_CONFIG))
    except Exception:
        cfg = dict(DEFAULT_GEN_CONFIG)
    _config_cache = cfg
    return cfg


def write_config(partial: Dict[str, Any]) -> Dict[str, Any]:
    """部分更新 generation 配置, 返回完整配置"""
    global _config_cache
    try:
        raw = json.loads(Path(SETTINGS_PATH).read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    current = raw.get("generation", dict(DEFAULT_GEN_CONFIG))
    _deep_update(current, partial)
    current["version"] = current.get("version", 1)
    raw["generation"] = current
    Path(SETTINGS_PATH).write_text(
        json.dumps(raw, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _config_cache = current
    return current


def get_effective_image_urls() -> List[Tuple[str, str]]:
    """
    返回有效的图片 ComfyUI URL 列表 (provider_name, url)
    本地 + 云端 (如果配置了)
    """
    cfg = read_config()
    img = cfg.get("image", {})
    urls: List[Tuple[str, str]] = []
    local = (img.get("local_comfyui_url") or "").strip()
    if local:
        urls.append(("comfyui", local))
    cloud = (img.get("cloud_comfyui_url") or "").strip()
    if cloud:
        urls.append(("comfyui_cloud", cloud))
    return urls


def get_effective_video_url() -> Optional[str]:
    """返回视频 ComfyUI URL"""
    cfg = read_config()
    video = cfg.get("video", {})
    cloud = (video.get("cloud_comfyui_url") or "").strip()
    return cloud or None


def get_workflow_dir() -> Optional[str]:
    """返回工作流 JSON 目录"""
    cfg = read_config()
    d = (cfg.get("video", {}).get("workflow_dir") or "").strip()
    return d if d else None


def list_registered_workflows() -> Dict[str, Dict[str, Any]]:
    """列出所有已注册的自定义工作流"""
    cfg = read_config()
    workflows: Dict[str, Dict[str, Any]] = {}
    # 1. 从配置中的 workflows 键加载
    wf_map = cfg.get("workflows", {})
    for name, wf in wf_map.items():
        if isinstance(wf, dict):
            workflows[name] = dict(wf)
            if "name" not in wf:
                wf["name"] = name
    # 2. 从 workflow_dir 扫描 JSON 文件
    wf_dir = get_workflow_dir()
    if wf_dir:
        import glob
        for json_file in Path(wf_dir).glob("*.json"):
            wf_name = json_file.stem
            if wf_name not in workflows:
                try:
                    raw = json_file.read_text(encoding="utf-8")
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and "workflow" in parsed:
                        # meta+workflow 格式
                        workflows[wf_name] = {**parsed, "name": wf_name, "source": "file"}
                    else:
                        # 纯 workflow JSON
                        workflows[wf_name] = {
                            "name": wf_name,
                            "type": "image" if "KSampler" in raw else "video",
                            "display_name": wf_name,
                            "source": "file",
                            "workflow": parsed,
                        }
                except Exception as e:
                    log.warning("Failed to load workflow %s: %s", json_file, e)
    return workflows


def register_workflow(name: str, wf_def: Dict[str, Any]) -> Dict[str, Any]:
    """注册或更新一个自定义工作流"""
    cfg = read_config()
    workflows = cfg.get("workflows", {})
    wf_def["name"] = name  # inject name
    workflows[name] = wf_def
    write_config({"workflows": workflows})
    return wf_def


def remove_workflow(name: str) -> bool:
    """移除一个工作流"""
    cfg = read_config()
    workflows = dict(cfg.get("workflows", {}))  # shallow copy
    if name in workflows:
        del workflows[name]
        # 直接写入整个 settings 文件, 不用 _deep_update (它不删键)
        try:
            raw = json.loads(Path(SETTINGS_PATH).read_text(encoding="utf-8"))
        except Exception:
            raw = {}
        gen = raw.get("generation", dict(DEFAULT_GEN_CONFIG))
        gen["workflows"] = workflows
        raw["generation"] = gen
        Path(SETTINGS_PATH).write_text(
            json.dumps(raw, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        global _config_cache
        _config_cache = None  # invalidate cache
        return True
    return False


def _deep_update(target: Dict, source: Dict) -> None:
    """递归合并字典, source 覆盖 target.
    特殊规则: 如果 source 值为空 dict, 清空 target 中对应子字典
    """
    for key, value in source.items():
        if isinstance(value, dict):
            if not value and isinstance(target.get(key), dict):
                target[key] = {}  # 空 dict → 清空
            elif isinstance(target.get(key), dict):
                _deep_update(target[key], value)
            else:
                target[key] = value
        else:
            target[key] = value
