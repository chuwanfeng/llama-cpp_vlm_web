"""
FAL.ai 视频生成后端
通过 FAL.ai API 调用 Veo/Kling 等视频生成模型

配置:
    FAL_KEY = "..."  # 环境变量,从 https://fal.ai/dashboard 获取
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from agent.video_gen_provider import VideoGenProvider

log = logging.getLogger(__name__)

FAL_KEY = os.environ.get("FAL_KEY", "")
FAL_TIMEOUT = int(os.environ.get("FAL_VIDEO_TIMEOUT", "300"))  # 最长 5 分钟

# 可用模型目录
MODELS_CATALOG: Dict[str, Dict[str, Any]] = {
    "fal-ai/veo3.1": {
        "display_name": "Google Veo 3.1",
        "modes": ["text-to-video", "image-to-video"],
        "max_duration": 8,
        "resolutions": ["720p", "1080p"],
        "default_duration": 5,
    },
    "fal-ai/kling-v2": {
        "display_name": "Kling v2",
        "modes": ["text-to-video", "image-to-video"],
        "max_duration": 10,
        "resolutions": ["720p", "1080p"],
        "default_duration": 5,
    },
    "fal-ai/ltx-video": {
        "display_name": "Lightricks LTX Video",
        "modes": ["text-to-video"],
        "max_duration": 6,
        "resolutions": ["720p"],
        "default_duration": 4,
    },
}


class FalVideoGenProvider(VideoGenProvider):
    """FAL.ai 视频生成后端"""

    name = "fal"
    display_name = "FAL.ai (云端)"

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or FAL_KEY

    def is_available(self) -> bool:
        return bool(self._api_key)

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": mid, **{k: v for k, v in info.items() if k != "display_name"},
             "display_name": info["display_name"]}
            for mid, info in MODELS_CATALOG.items()
        ]

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
        """通过 FAL REST API 提交视频生成任务并轮询"""
        if not self._api_key:
            return self.error_response("FAL_KEY not configured. Set FAL_KEY environment variable.")

        selected_model = model or "fal-ai/veo3.1"
        model_info = MODELS_CATALOG.get(selected_model, {})

        # 校验模式
        if image_url and "image-to-video" not in model_info.get("modes", []):
            return self.error_response(f"{selected_model} does not support image-to-video")

        # 构建 payload
        payload: Dict[str, Any] = {"prompt": prompt}
        if image_url:
            payload["image_url"] = image_url
        if video_url:
            payload["video_url"] = video_url
        payload["duration"] = min(duration, model_info.get("max_duration", 8))

        try:
            # 1. 提交任务
            log.info("Submitting FAL video: %s (duration=%ds)", selected_model, payload["duration"])
            submit_r = self._fal_request("POST", f"fal-queue/{selected_model}", payload)
            request_id = submit_r.get("request_id")
            if not request_id:
                return self.error_response(f"FAL submit failed: {submit_r}")

            # 2. 轮询状态
            video_url_result = self._poll_fal(request_id, selected_model)

            if not video_url_result:
                return self.error_response(f"FAL generation timed out after {FAL_TIMEOUT}s")

            return self.success_response(
                video_url=video_url_result,
                model=selected_model,
                duration=payload["duration"],
            )

        except Exception as exc:
            log.error("FAL video generation failed: %s", exc)
            return self.error_response(str(exc))

    def _fal_request(self, method: str, path: str, body: Optional[Dict] = None) -> Dict:
        """发送 FAL API 请求"""
        import requests as _requests
        url = f"https://queue.fal.run/{path}"
        headers = {
            "Authorization": f"Key {self._api_key}",
            "Content-Type": "application/json",
        }
        if method == "POST":
            r = _requests.post(url, json=body, headers=headers, timeout=30)
        else:
            r = _requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def _poll_fal(self, request_id: str, model: str) -> Optional[str]:
        """轮询 FAL 任务状态直到完成"""
        deadline = time.monotonic() + FAL_TIMEOUT
        poll_interval = 2.0  # 视频生成较慢,降低轮询频率

        while time.monotonic() < deadline:
            time.sleep(poll_interval)

            try:
                result = self._fal_request("GET", f"fal-queue/{model}/requests/{request_id}/status")
            except Exception as exc:
                log.warning("FAL poll error: %s", exc)
                poll_interval = min(poll_interval * 1.5, 10)
                continue

            status = result.get("status", "")
            if status == "COMPLETED":
                output = result.get("output", {}) or result.get("result", {})
                # 尝试多种可能的 URL 字段
                video_url = (
                    output.get("video", {}).get("url")
                    or output.get("video_url")
                    or output.get("url")
                )
                log.info("FAL video complete: %s", video_url[:80] if video_url else "None")
                return video_url
            elif status == "FAILED":
                log.error("FAL video failed: %s", result.get("error", result))
                return None
            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                continue

            poll_interval = min(poll_interval * 1.2, 8)

        return None
