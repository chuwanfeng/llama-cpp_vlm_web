"""
视频生成工具 — 统一入口,分发到已注册后端
参照 hermes-agent tools/video_generation_tool.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from tools.registry import get_registry, tool_error, tool_result

registry = get_registry()

log = logging.getLogger(__name__)

# ── Schema ──
VIDEO_GENERATE_SCHEMA: Dict[str, Any] = {
    "name": "video_generate",
    "description": (
        "使用配置的视频生成后端生成 AI 视频。"
        "支持 text-to-video(纯文本生成视频)和 image-to-video(图片转视频)。"
        "生成需要较长时间(通常 30s-3min),请耐心等待。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "视频描述提示词,描述期望的画面内容、运动方式、风格等。英文效果更好。",
            },
            "image_url": {
                "type": "string",
                "description": "可选。提供一张图片 URL,后端将基于该图片生成视频(image-to-video)。不传则为 text-to-video。",
            },
            "model": {
                "type": "string",
                "description": "模型名或已注册的自定义工作流名称。空则用默认模型。",
            },
            "duration": {
                "type": "integer",
                "minimum": 3,
                "maximum": 10,
                "default": 5,
                "description": "期望的视频时长(秒),实际时长由后端模型决定",
            },
            "resolution": {
                "type": "string",
                "enum": ["720p", "1080p"],
                "default": "720p",
                "description": "视频分辨率",
            },
        },
        "required": ["prompt"],
    },
}


def check_video_gen_requirements() -> bool:
    """检查是否有可用后端"""
    try:
        from agent.video_gen_registry import get_active_provider
        return get_active_provider() is not None
    except Exception:
        return False


def _handle_video_generate(prompt: str = "", model: str = None, image_url: str = None,
                             duration: int = 5, resolution: str = "720p",
                             **_kw: Any) -> str:
    """处理 video_generate 工具调用"""
    prompt = prompt.strip()
    if not prompt:
        return tool_error("prompt is required for video generation")

    from agent.video_gen_registry import get_active_provider, list_providers

    provider = get_active_provider()
    if provider is None:
        providers = list_providers()
        names = [p.display_name for p in providers] if providers else ["(none configured)"]
        return tool_error(
            f"No video generation backend available. "
            f"Available: {', '.join(names)}. "
            f"Set FAL_KEY env var to enable FAL.ai video generation."
        )

    duration = max(3, min(10, int(duration)))

    try:
        result = provider.generate(
            prompt=prompt,
            model=model or None,
            image_url=image_url,
            duration=duration,
            resolution=resolution,
        )

        if result.get("error"):
            return tool_error(result["error"])

        video_url = result.get("video_url", "")
        if not video_url:
            return tool_error("Video generation produced no video URL")

        model = result.get("model", "unknown")
        return tool_result(
            data=(
                f"🎬 **视频已生成**\n\n"
                f"[点击查看/下载视频]({video_url})\n\n"
                f"模型: {model} | 时长: {duration}s | 后端: {result.get('provider', provider.name)}"
            )
        )

    except Exception as exc:
        log.error("Video generation failed: %s", exc)
        return tool_error(f"Video generation failed: {exc}")


# ── 注册 ──
registry.register(
    name="video_generate",
    toolset="video_gen",
    schema=VIDEO_GENERATE_SCHEMA,
    handler=_handle_video_generate,
    check_fn=check_video_gen_requirements,
)
