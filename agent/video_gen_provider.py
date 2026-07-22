"""
视频生成后端抽象基类
参照 hermes-agent agent/video_gen_provider.py
"""
from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class VideoGenProvider(abc.ABC):
    """视频生成后端抽象基类"""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """唯一标识: 'fal' / 'xai'"""
        ...

    @property
    def display_name(self) -> str:
        return self.name

    @abc.abstractmethod
    def is_available(self) -> bool:
        ...

    @abc.abstractmethod
    def list_models(self) -> List[Dict[str, Any]]:
        """可用模型
        Returns:
            [{
                "id": "veo3.1",
                "modes": ["text-to-video", "image-to-video"],
                "max_duration": 8,
                "resolutions": ["720p", "1080p"],
            }]
        """
        ...

    def default_model(self) -> Optional[str]:
        models = self.list_models()
        return models[0]["id"] if models else None

    def capabilities(self) -> Dict[str, Any]:
        return {
            "text_to_video": True,
            "image_to_video": False,
            "video_edit": False,
        }

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        duration: int = 5,
        resolution: str = "720p",
        **kwargs,
    ) -> Dict[str, Any]:
        """生成视频
        Returns: {"video_url": "...", "model": "veo3.1", "duration": 5}
        """
        ...

    def success_response(self, video_url: str, model: str, **extra) -> Dict[str, Any]:
        return {"video_url": video_url, "model": model, "provider": self.name, **extra}

    def error_response(self, message: str) -> Dict[str, Any]:
        return {"video_url": "", "error": message, "provider": self.name}
