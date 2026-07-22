"""
图片生成工具 — 统一入口,分发到已注册后端
参照 hermes-agent tools/image_generation_tool.py
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from tools.registry import get_registry, tool_error, tool_result

registry = get_registry()

log = logging.getLogger(__name__)

# ── 尺寸预设 ──
SIZE_MAP: Dict[str, tuple] = {
    "square": (1024, 1024),
    "landscape": (1344, 768),
    "portrait": (768, 1344),
    "wide": (1536, 640),
}

# ── Schema ──
IMAGE_GENERATE_SCHEMA: Dict[str, Any] = {
    "name": "image_generate",
    "description": (
        "使用配置的图片生成后端(ComfyUI/Ollama Flux/FAL.ai)生成 AI 图片。"
        "支持 text-to-image 和 image-to-image(需后端支持)。"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "图片描述提示词,英文效果更好。越详细越好,可包含风格/光照/构图/色彩等描述。",
            },
            "model": {
                "type": "string",
                "description": "模型名或已注册的自定义工作流名称(如 z-image-turo, image_z_image)。空则用默认模型。",
            },
            "negative_prompt": {
                "type": "string",
                "description": "负面提示词,描述不希望出现的内容(如 blurry, low quality, distorted)",
            },
            "size": {
                "type": "string",
                "enum": list(SIZE_MAP.keys()),
                "default": "square",
                "description": f"尺寸预设: {', '.join(f'{k}({v[0]}x{v[1]})' for k, v in SIZE_MAP.items())}",
            },
            "num_images": {
                "type": "integer",
                "minimum": 1,
                "maximum": 4,
                "default": 1,
                "description": "生成图片数量",
            },
            "seed": {
                "type": "integer",
                "description": "随机种子(相同 seed+prompt 产生相同图片,不传则随机)",
            },
            "steps": {
                "type": "integer",
                "minimum": 1,
                "maximum": 50,
                "description": "采样步数。不传则使用工作流原生值(推荐,通常 20)。更多步数=更精细但更慢",
            },
            "cfg_scale": {
                "type": "number",
                "minimum": 1.0,
                "maximum": 20.0,
                "description": "CFG 引导强度。不传则使用工作流原生值(推荐)。越高越贴近提示词但可能过饱和",
            },
        },
        "required": ["prompt"],
    },
}


def check_image_gen_requirements() -> bool:
    """检查是否有可用后端"""
    try:
        from agent.image_gen_registry import get_active_provider
        return get_active_provider() is not None
    except Exception:
        return False


def _handle_image_generate(prompt: str = "", model: str = None, negative_prompt: str = None,
                             size: str = "square", num_images: int = 1, seed: int = None,
                             steps: int = None, cfg_scale: float = None,
                             **_kw: Any) -> str:
    """处理 image_generate 工具调用"""
    prompt = prompt.strip()
    if not prompt:
        return tool_error("prompt is required for image generation")

    from agent.image_gen_registry import get_active_provider, list_providers

    provider = get_active_provider()
    if provider is None:
        providers = list_providers()
        names = [p.display_name for p in providers] if providers else ["(none configured)"]
        return tool_error(
            f"No image generation backend available. "
            f"Available: {', '.join(names)}. "
            f"Install ComfyUI or set OLLAMA_FLUX_MODEL / FAL_KEY env vars."
        )

    # 解析参数
    size_key = size
    width, height = SIZE_MAP.get(size_key, (1024, 1024))
    num_images = max(1, min(4, int(num_images)))

    try:
        result = provider.generate(
            prompt=prompt,
            model=model or None,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_images=num_images,
            seed=seed,
            steps=steps or None,
            guidance_scale=cfg_scale,
        )

        if result.get("error"):
            return tool_error(result["error"])

        images = result.get("images", [])
        if not images:
            return tool_error("Image generation produced no images")

        # 构建响应（含图片 Markdown，LLM 可在回复中引用显示）
        # 前端工具结果显示时会剥离图片，只显示元数据链接
        lines = []
        for i, img in enumerate(images):
            url = img.get("url") or img.get("b64_json", "")
            if url:
                lines.append(f"![Generated Image {i+1}]({url})")

        meta_parts = [
            f"模型: {result.get('model', 'unknown')}",
            f"后端: {result.get('provider', provider.name)}",
            f"尺寸: {width}x{height}",
        ]
        if seed is not None:
            meta_parts.append(f"seed: {seed}")
        lines.append("\n" + " | ".join(meta_parts))

        return tool_result(data="\n".join(lines))

    except Exception as exc:
        log.error("Image generation failed: %s", exc)
        return tool_error(f"Image generation failed: {exc}")


# ── 注册 ──
registry.register(
    name="image_generate",
    toolset="image_gen",
    schema=IMAGE_GENERATE_SCHEMA,
    handler=_handle_image_generate,
    check_fn=check_image_gen_requirements,
)
