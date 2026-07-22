"""
插件自动加载器 — 启动时扫描并注册所有图片/视频生成后端

数据源:
  前端 settings.json → services/generation_config.py → 此模块

支持:
  - 本地 ComfyUI 图片
  - 云端 ComfyUI 图片
  - 云端 ComfyUI 视频
  - Ollama Flux 图片
  - FAL.ai 视频
  - 用户自定义工作流 (按名称注册)

热重载: reload_all() 重新读取配置并重建后端
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


def init_image_providers() -> int:
    """初始化图片生成后端, 返回成功注册数量"""
    from services.generation_config import get_effective_image_urls

    count = 0
    urls = get_effective_image_urls()

    for provider_name, url in urls:
        try:
            from plugins.image_gen.comfyui import ComfyUIImageGenProvider
            from agent.image_gen_registry import register_provider

            is_cloud = provider_name == "comfyui_cloud"
            display = f"ComfyUI ({'云端: ' + url if is_cloud else '本地'})"

            p = ComfyUIImageGenProvider(
                base_url=url,
                provider_name=provider_name,
                display_name=display,
            )
            register_provider(p)
            if p.is_available():
                log.info(f"✅ 图片: {display}")
            else:
                log.info(f"⏳ 图片: {display} 不可达")
            count += 1
        except Exception as e:
            log.debug("ComfyUI %s init failed: %s", provider_name, e)

    # Ollama Flux (环境变量, 无前端配置)
    if os.environ.get("OLLAMA_FLUX_MODEL") or os.environ.get("OLLAMA_BASE_URL"):
        try:
            from plugins.image_gen.ollama_flux import OllamaFluxImageGenProvider
            from agent.image_gen_registry import register_provider
            p = OllamaFluxImageGenProvider()
            register_provider(p)
            if p.is_available():
                log.info("✅ 图片: Ollama Flux")
            else:
                log.info("⏳ 图片: Ollama Flux 不可用")
            count += 1
        except Exception as e:
            log.debug("Ollama Flux init failed: %s", e)

    return count


def init_video_providers() -> int:
    """初始化视频生成后端, 返回成功注册数量"""
    from services.generation_config import get_effective_video_url

    count = 0
    video_url = get_effective_video_url()

    # ComfyUI 视频 (仅当前端配置了云端 URL 时加载)
    if video_url:
        try:
            from plugins.video_gen.comfyui import ComfyUIVideoGenProvider
            from agent.video_gen_registry import register_provider

            p = ComfyUIVideoGenProvider(base_url=video_url)
            register_provider(p)
            if p.is_available():
                log.info(f"✅ 视频: ComfyUI ({video_url})")
            else:
                log.info(f"⏳ 视频: ComfyUI 不可达 ({video_url})")
            count += 1
        except Exception as e:
            log.debug("ComfyUI video init failed: %s", e)

    # FAL.ai
    if os.environ.get("FAL_KEY"):
        try:
            from plugins.video_gen.fal import FalVideoGenProvider
            from agent.video_gen_registry import register_provider
            p = FalVideoGenProvider()
            register_provider(p)
            log.info("✅ 视频: FAL.ai")
            count += 1
        except Exception as e:
            log.debug("FAL.ai init failed: %s", e)

    return count


def init_all_providers() -> tuple:
    img = init_image_providers()
    vid = init_video_providers()
    if img == 0 and vid == 0:
        log.info("ℹ️  无图片/视频后端可用 (前端未配置云端 ComfyUI)")
    return img, vid


def reload_all() -> tuple:
    """
    热重载所有生成后端。
    先清除现有注册表, 再根据最新配置重新注册。
    """
    from agent.image_gen_registry import _reset_for_tests as reset_img
    from agent.video_gen_registry import _reset_for_tests as reset_vid
    from services.generation_config import _config_cache

    # 清除配置缓存 (强制重新读取 settings.json)
    import services.generation_config as gcfg
    gcfg._config_cache = None

    # 清除工作流缓存
    try:
        from plugins.video_gen.comfyui import _clear_workflow_cache
        _clear_workflow_cache()
    except Exception:
        pass

    # 清除注册表
    reset_img()
    reset_vid()

    # 重新初始化
    return init_all_providers()
