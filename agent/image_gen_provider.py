"""
图片生成后端抽象基类
参照 hermes-agent agent/image_gen_provider.py
"""
from __future__ import annotations

import abc
import base64
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


class ImageGenProvider(abc.ABC):
    """图片生成后端抽象基类

    每个后端 (ComfyUI, Ollama Flux, FAL.ai 等) 实现此接口
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """唯一标识: 'comfyui' / 'ollama_flux' / 'fal'"""
        ...

    @property
    def display_name(self) -> str:
        """前端展示名称"""
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        """后端是否可用"""
        ...

    @abc.abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """可用模型列表
        Returns:
            [{"id": "flux-dev", "display_name": "FLUX.1 Dev", "max_resolution": "1024x1024"}, ...]
        """
        ...

    def default_model(self) -> Optional[str]:
        models = self.list_models()
        return models[0]["id"] if models else None

    def capabilities(self) -> Dict[str, Any]:
        """后端能力描述"""
        return {
            "text_to_image": True,
            "image_to_image": False,
            "inpainting": False,
            "max_images_per_request": 4,
        }

    @abc.abstractmethod
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
        """
        执行图片生成

        Returns:
            {
                "images": [{"url": "http://..."}, ...],
                "model": "flux-dev",
                "seed": 42,
                "provider": "comfyui"
            }
        """
        ...

    # ── 辅助方法 ──

    @staticmethod
    def _images_dir() -> Path:
        """生成图片保存目录"""
        d = Path("static/generated/images")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_b64_image(self, b64_data: str, prefix: str = "gen") -> Dict[str, str]:
        """保存 base64 图片到静态目录, 返回 {"url": "..."}"""
        data = b64_data
        if "," in data:
            data = data.split(",", 1)[1]
        fname = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        path = self._images_dir() / fname
        path.write_bytes(base64.b64decode(data))
        return {"url": f"/static/generated/images/{fname}"}

    def save_url_image(self, url: str, prefix: str = "gen") -> Dict[str, str]:
        """下载并保存远程图片"""
        import requests
        fname = f"{prefix}_{uuid.uuid4().hex[:10]}.png"
        path = self._images_dir() / fname
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return {"url": f"/static/generated/images/{fname}"}

    def success_response(
        self,
        images: List[Dict[str, str]],
        model: str,
        seed: Optional[int] = None,
        **extra,
    ) -> Dict[str, Any]:
        return {
            "images": images,
            "model": model,
            "seed": seed,
            "provider": self.name,
            **extra,
        }

    def error_response(self, message: str) -> Dict[str, Any]:
        return {
            "images": [],
            "error": message,
            "provider": self.name,
        }
