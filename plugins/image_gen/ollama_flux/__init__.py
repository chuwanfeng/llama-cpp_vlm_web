"""
Ollama Flux 图片生成后端
通过 Ollama API 调用 flux 模型生成图片(实验性)

配置:
    OLLAMA_FLUX_MODEL = "flux.1-dev"  # 环境变量可覆盖
    OLLAMA_BASE_URL = "http://127.0.0.1:11434"
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import ImageGenProvider

log = logging.getLogger(__name__)

OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_FLUX_MODEL = os.environ.get("OLLAMA_FLUX_MODEL", "flux.1-dev")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "120"))


class OllamaFluxImageGenProvider(ImageGenProvider):
    """Ollama Flux 后端 — 本地自托管图片生成"""

    name = "ollama_flux"
    display_name = "Ollama Flux (本地)"

    def is_available(self) -> bool:
        """检测 Ollama 是否有 flux 模型"""
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            r.raise_for_status()
            models = r.json().get("models", [])
            for m in models:
                name = m.get("name", "")
                if "flux" in name.lower() or "stable-diffusion" in name.lower():
                    return True
            return False
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """列出 Ollama 中可用的图像生成模型"""
        try:
            r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
            r.raise_for_status()
            models = []
            for m in r.json().get("models", []):
                name = m.get("name", "")
                if any(k in name.lower() for k in ("flux", "stable-diffusion", "sd", "dalle")):
                    models.append({
                        "id": name,
                        "display_name": name.split(":")[0] if ":" in name else name,
                        "max_resolution": "1024x1024",
                    })
            return models
        except Exception:
            return []

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        steps: int = 28,
        guidance_scale: float = 3.5,
        reference_images: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """通过 Ollama generate API 调用 flux 模型"""
        selected_model = model or OLLAMA_FLUX_MODEL

        # 构建增强提示词(包含尺寸和负面提示词)
        full_prompt = prompt
        if negative_prompt:
            full_prompt += f"\n\nNegative prompt: {negative_prompt}"
        full_prompt += f"\n\nSize: {width}x{height}"

        payload = {
            "model": selected_model,
            "prompt": full_prompt,
            "stream": False,
        }

        if seed is not None:
            payload["options"] = {"seed": seed}

        try:
            log.info("Calling Ollama generate: %s (%dx%d)", selected_model, width, height)
            r = requests.post(
                f"{OLLAMA_BASE}/api/generate",
                json=payload,
                timeout=OLLAMA_TIMEOUT,
            )
            r.raise_for_status()
            result = r.json()

            # Ollama 返回的 response 可能包含 base64 图片数据
            response_text = result.get("response", "")
            images = []

            # 尝试从 response 中提取 base64 图片
            if "data:image" in response_text:
                # 提取 data URL
                import re
                data_urls = re.findall(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', response_text)
                for data_url in data_urls:
                    saved = self.save_b64_image(data_url, prefix="flux")
                    images.append(saved)

            # 检查是否有直接的 images 字段
            raw_images = result.get("images", [])
            for img in raw_images:
                if isinstance(img, str):
                    saved = self.save_b64_image(img, prefix="flux")
                    images.append(saved)

            if not images:
                # 尝试将整个 response 当作图片处理
                if len(response_text) > 100 and response_text.startswith(("iVBOR", "/9j/", "R0lG")):
                    saved = self.save_b64_image(response_text, prefix="flux")
                    images.append(saved)

            if not images:
                return self.error_response(
                    f"Ollama {selected_model} returned no image data. "
                    f"Response preview: {response_text[:200]}"
                )

            return self.success_response(images=images, model=selected_model, seed=seed)

        except requests.exceptions.Timeout:
            return self.error_response(f"Ollama {selected_model} timed out after {OLLAMA_TIMEOUT}s")
        except Exception as exc:
            log.error("Ollama flux generation failed: %s", exc)
            return self.error_response(str(exc))
