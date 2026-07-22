"""
ComfyUI 视频生成后端
通过 ComfyUI HTTP API 调用视频生成工作流

支持两种模式:
  - text-to-video: AnimateDiff 工作流 (文本→视频)
  - image-to-video: SVD (Stable Video Diffusion) 工作流 (图片→视频)

配置:
    COMFYUI_VIDEO_URL = "http://your-cloud-comfyui:8188"    # 环境变量
    不设置则回退到 COMFYUI_BASE_URL,再回退到 http://127.0.0.1:8188

所需模型 (ComfyUI 端需预先安装):
    - AnimateDiff: mm_sd15_v2 motion module + sd15 checkpoint + vae
    - SVD: svd_xt checkpoint
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from agent.video_gen_provider import VideoGenProvider

log = logging.getLogger(__name__)

# 云端 ComfyUI 地址 (优先级: 视频专用 > 通用 > 本地默认)
COMFYUI_VIDEO_URL = os.environ.get(
    "COMFYUI_VIDEO_URL",
    os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188"),
).rstrip("/")
COMFYUI_VIDEO_TIMEOUT = int(os.environ.get("COMFYUI_VIDEO_TIMEOUT", "600"))
COMFYUI_POLL_INTERVAL = float(os.environ.get("COMFYUI_POLL_INTERVAL", "3.0"))

# 内置视频工作流 → 可通过 COMFYUI_VIDEO_WORKFLOW_PATH 覆盖
DEFAULT_WORKFLOWS: Dict[str, Dict[str, Any]] = {
    # ── AnimateDiff text-to-video (标准 16 帧) ──
    "animatediff_t2v": {
        "display_name": "AnimateDiff (文本生成视频)",
        "mode": "text-to-video",
        "max_duration": 4,
        "default_duration": 2,
        "resolutions": ["512x512"],
        "frame_count": 16,
        "fps": 8,
        "description": "基于 AnimateDiff motion module 的文本驱动视频生成,需要 mm_sd15_v2 + sd15 模型",
    },
    # ── SVD image-to-video ──
    "svd_i2v": {
        "display_name": "Stable Video Diffusion (图片转视频)",
        "mode": "image-to-video",
        "max_duration": 4,
        "default_duration": 2,
        "resolutions": ["576x1024", "768x768"],
        "frame_count": 14,
        "fps": 7,
        "description": "基于 Stability AI SVD 的图片驱动视频生成,需要 svd_xt.safetensors 模型",
    },
}

# 工作流 JSON 文件路径 (可选, 用于用户自定义工作流)
# 设置此环境变量指向包含工作流 JSON 的目录,每个 .json 文件对应一个工作流
_WORKFLOW_CACHE: Optional[Dict[str, Dict[str, Any]]] = None


class ComfyUIVideoGenProvider(VideoGenProvider):
    """ComfyUI 视频生成后端 — 本地或云端"""

    name = "comfyui"
    display_name = "ComfyUI (视频)"
    _base_url: str

    def __init__(self, base_url: Optional[str] = None):
        self._base_url = (base_url or COMFYUI_VIDEO_URL).rstrip("/")
        self._available: Optional[bool] = None
        self._available_check_time = 0.0

    # ── VideoGenProvider 接口 ──

    def is_available(self) -> bool:
        now = time.monotonic()
        if self._available is not None and (now - self._available_check_time) < 30:
            return self._available

        try:
            r = requests.get(f"{self._base_url}/system_stats", timeout=5)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        self._available_check_time = now
        return self._available

    def list_models(self) -> List[Dict[str, Any]]:
        workflows = self._load_workflows()
        return [
            {
                "id": wid,
                "display_name": wf.get("display_name", wid),
                "mode": wf.get("mode", "text-to-video"),
                "max_duration": wf.get("max_duration", 4),
                "resolutions": wf.get("resolutions", ["512x512"]),
            }
            for wid, wf in workflows.items()
        ]

    def capabilities(self) -> Dict[str, Any]:
        workflows = self._load_workflows()
        has_t2v = any(w.get("mode") == "text-to-video" for w in workflows.values())
        has_i2v = any(w.get("mode") == "image-to-video" for w in workflows.values())
        has_custom = bool(os.environ.get("COMFYUI_VIDEO_WORKFLOW_PATH"))
        return {
            "text_to_video": has_t2v,
            "image_to_video": has_i2v,
            "video_edit": False,
            "custom_workflows": has_custom,
            "base_url": self._base_url,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        image_url: Optional[str] = None,
        video_url: Optional[str] = None,
        duration: int = 2,
        resolution: str = "512x512",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        提交 ComfyUI 视频工作流 → 轮询 → 下载视频
        """
        workflows = self._load_workflows()

        # 自动选择工作流
        if model and model in workflows:
            workflow_id = model
        elif image_url:
            # 有图片 → 选 image-to-video 工作流
            workflow_id = next(
                (wid for wid, wf in workflows.items()
                 if wf.get("mode") == "image-to-video"),
                None,
            )
        else:
            # 纯文本 → 选 text-to-video 工作流
            workflow_id = next(
                (wid for wid, wf in workflows.items()
                 if wf.get("mode") == "text-to-video"),
                None,
            )

        if not workflow_id:
            return self.error_response(
                "No suitable video workflow found. "
                f"Available: {list(workflows.keys())}. "
                "Install AnimateDiff or SVD models in ComfyUI."
            )

        workflow_def = workflows.get(workflow_id, {})
        workflow_json = workflow_def.get("workflow", {})
        if not workflow_json:
            return self.error_response(
                f"Workflow '{workflow_id}' has no workflow JSON."
                "Set COMFYUI_VIDEO_WORKFLOW_PATH to directory with .json workflow files."
            )

        # 解析参数
        frame_count = workflow_def.get("frame_count", 16)
        fps = workflow_def.get("fps", 8)
        width, height = self._parse_resolution(
            resolution,
            workflow_def.get("resolutions", ["512x512"])[0],
        )

        try:
            # 1. 填充工作流参数
            filled_workflow = self._fill_video_workflow(
                workflow=workflow_json,
                workflow_id=workflow_id,
                prompt=prompt,
                image_url=image_url,
                duration=duration,
                frame_count=frame_count,
                fps=fps,
                width=width,
                height=height,
            )

            # 2. 提交
            log.info(
                "Submitting ComfyUI video: %s (%dx%d, %d frames, %d fps, to %s)",
                workflow_id, width, height, frame_count, fps, self._base_url,
            )
            submit_r = requests.post(
                f"{self._base_url}/prompt",
                json={"prompt": filled_workflow},
                timeout=30,
            )
            submit_r.raise_for_status()
            prompt_id = submit_r.json().get("prompt_id")
            if not prompt_id:
                return self.error_response("ComfyUI returned no prompt_id")

            # 3. 轮询等待 (视频生成较慢)
            output_files = self._poll_video_output(prompt_id, fps)

            if not output_files:
                return self.error_response(
                    f"ComfyUI video generation timeout after {COMFYUI_VIDEO_TIMEOUT}s"
                )

            # 4. 下载视频到本地 static 目录
            video_info = self._download_video_to_local(output_files)
            if not video_info:
                return self.error_response("Failed to download generated video")

            return self.success_response(
                video_url=video_info["local_url"],
                model=workflow_id,
                duration=duration,
                resolution=f"{width}x{height}",
                remote_url=video_info.get("remote_url"),
            )

        except requests.exceptions.ConnectionError:
            return self.error_response(
                f"Cannot connect to ComfyUI at {self._base_url}. "
                "Ensure ComfyUI is running and COMFYUI_VIDEO_URL is correct."
            )
        except Exception as exc:
            log.error("ComfyUI video generation failed: %s", exc, exc_info=True)
            return self.error_response(str(exc))

    # ── 工作流加载 ──

    def _load_workflows(self) -> Dict[str, Dict[str, Any]]:
        """
        加载所有可用视频工作流.
        优先级: 前端注册 > 自定义 JSON 文件 > 内置工作流
        """
        global _WORKFLOW_CACHE
        if _WORKFLOW_CACHE is not None:
            return _WORKFLOW_CACHE

        workflows: Dict[str, Dict[str, Any]] = {}

        # 1. 从前端 generation_config 加载已注册的视频工作流
        try:
            from services.generation_config import list_registered_workflows
            reg_wfs = list_registered_workflows()
            for wf_name, wf_def in reg_wfs.items():
                if wf_def.get("type") == "video" and wf_def.get("workflow"):
                    workflows[wf_name] = {
                        "display_name": wf_def.get("display_name", wf_name),
                        "mode": wf_def.get("mode", "text-to-video"),
                        "max_duration": wf_def.get("max_duration", 6),
                        "default_duration": wf_def.get("default_duration", 2),
                        "resolutions": wf_def.get("resolutions", ["512x512"]),
                        "frame_count": wf_def.get("frame_count", 16),
                        "fps": wf_def.get("fps", 8),
                        "description": wf_def.get("description", ""),
                        "workflow": wf_def["workflow"],
                    }
                    log.info("Video workflow loaded from config: %s", wf_name)
        except Exception as e:
            log.debug("Failed to load video workflows from config: %s", e)

        # 2. 加载自定义工作流 JSON 文件 (COMFYUI_VIDEO_WORKFLOW_PATH)
        custom_path = os.environ.get("COMFYUI_VIDEO_WORKFLOW_PATH", "")
        if custom_path:
            wp = Path(custom_path)
            if wp.is_dir():
                for json_file in wp.glob("*.json"):
                    try:
                        raw = json_file.read_text(encoding="utf-8")
                        parsed = json.loads(raw)
                        wid = json_file.stem
                        workflows[wid] = self._extract_workflow_meta(parsed, wid)
                    except Exception as exc:
                        log.warning("Failed to load workflow %s: %s", json_file, exc)
            elif wp.is_file() and wp.suffix == ".json":
                try:
                    raw = wp.read_text(encoding="utf-8")
                    parsed = json.loads(raw)
                    wid = wp.stem
                    workflows[wid] = self._extract_workflow_meta(parsed, wid)
                except Exception as exc:
                    log.warning("Failed to load workflow %s: %s", wp, exc)

        # 3. 回退到内置工作流
        if not workflows:
            log.info("No custom video workflows found, using built-in defaults")
            for wid, wf_def in DEFAULT_WORKFLOWS.items():
                workflow_json = self._build_builtin_workflow(wid)
                if workflow_json:
                    workflows[wid] = {**wf_def, "workflow": workflow_json}

        _WORKFLOW_CACHE = workflows
        return workflows

    def _extract_workflow_meta(self, raw: Any, name: str) -> Dict[str, Any]:
        """从加载的 JSON 中提取工作流元信息"""
        if isinstance(raw, dict) and "workflow" in raw:
            return {
                **raw,
                "display_name": raw.get("display_name", name),
            }
        # 纯 workflow JSON → 自动推断
        return {
            "workflow": raw,
            "display_name": name,
            "mode": "text-to-video",  # 默认, 用户可编辑
            "max_duration": 6,
            "default_duration": 2,
            "resolutions": ["512x512"],
            "frame_count": 16,
            "fps": 8,
        }

    def _build_builtin_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """
        构建内置 ComfyUI 视频工作流 JSON.

        节点编号约定:
          1 - CheckpointLoaderSimple
          2 - CLIPTextEncode (positive prompt)
          3 - CLIPTextEncode (negative prompt)
          4 - AnimateDiff Loader / SVD Loader
          5 - KSampler
          6 - VAE Decode
          7 - VHS Video Combine (将帧合并为视频)
          8 - SaveVideo
        """
        if workflow_id == "animatediff_t2v":
            return self._build_animatediff_workflow()
        elif workflow_id == "svd_i2v":
            return self._build_svd_workflow()
        return None

    def _build_animatediff_workflow(self) -> Dict[str, Any]:
        """构建 AnimateDiff text-to-video 工作流"""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd15.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "__PROMPT_PLACEHOLDER__",
                    "clip": ["1", 1],
                },
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "__NEGATIVE_PLACEHOLDER__",
                    "clip": ["1", 1],
                },
            },
            "4": {
                "class_type": "AnimateDiffLoaderV1",
                "inputs": {
                    "model": ["1", 0],
                    "motion_module": "mm_sd15_v2.safetensors",
                    "frame_count": 16,  # __FRAME_COUNT__
                    "fps": 8,           # __FPS__
                    "latent_format": "sd15",
                },
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": 512,   # __WIDTH__
                    "height": 512,  # __HEIGHT__
                    "batch_size": 16,  # same as frame_count
                },
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": -1,          # random
                    "steps": 25,
                    "cfg": 7.5,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["5", 0],
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["6", 0],
                    "vae": ["1", 2],
                },
            },
            "8": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["7", 0],
                    "frame_rate": 8,  # __FPS__
                    "format": "video/mp4",
                    "loop_count": 0,
                    "crf": 23,
                },
            },
            "9": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["7", 0]},
            },
        }

    def _build_svd_workflow(self) -> Dict[str, Any]:
        """构建 SVD (Stable Video Diffusion) image-to-video 工作流"""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "svd_xt.safetensors"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "__PROMPT_PLACEHOLDER__",
                    "clip": ["1", 2],  # SVD 的 clip_vision
                },
            },
            "3": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": "__IMAGE_PLACEHOLDER__",
                },
            },
            "4": {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels": ["3", 0],
                    "vae": ["1", 1],
                    "mask": ["3", 0],  # TODO: 实际需要 mask
                },
            },
            "5": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": -1,
                    "steps": 25,
                    "cfg": 2.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["6", 0],
                    "latent_image": ["4", 0],
                },
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "__NEGATIVE_PLACEHOLDER__",
                    "clip": ["1", 2],
                },
            },
            "7": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 1],
                },
            },
            "8": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["7", 0],
                    "frame_rate": 7,  # __FPS__
                    "format": "video/mp4",
                    "loop_count": 0,
                    "crf": 23,
                },
            },
            "9": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["7", 0]},
            },
        }

    def _fill_video_workflow(
        self,
        workflow: Dict[str, Any],
        workflow_id: str,
        prompt: str,
        image_url: Optional[str],
        duration: int,
        frame_count: int,
        fps: int,
        width: int,
        height: int,
    ) -> Dict[str, Any]:
        """
        遍历工作流 JSON, 填充占位符参数.
        替换模式:
          - __PROMPT_PLACEHOLDER__ → 用户提示词
          - __NEGATIVE_PLACEHOLDER__ → 负面提示词
          - __WIDTH__ / __HEIGHT__ → 尺寸
          - __FPS__ → 帧率
          - __FRAME_COUNT__ → 帧数 (duration * fps)
          - __IMAGE_PLACEHOLDER__ → 从 URL 下载的本地文件名
          - 数字字段: 按字段名自动更新 (frame_count, fps, width, height 等)
        """
        # 计算实际帧数
        actual_frames = min(int(duration * fps), 250)  # 安全上限

        # 如果 image_url 提供, 下载到 ComfyUI input 目录
        local_image: Optional[str] = None
        if image_url:
            local_image = self._download_to_comfyui_input(image_url)
            if not local_image:
                log.warning("Failed to download image to ComfyUI input: %s", image_url)

        import copy
        filled = copy.deepcopy(workflow)

        for node_id, node_data in filled.items():
            if not isinstance(node_data, dict):
                continue
            inputs = node_data.get("inputs", {})
            if not isinstance(inputs, dict):
                continue

            for key, value in inputs.items():
                if isinstance(value, str):
                    # 占位符替换
                    value = value.replace("__PROMPT_PLACEHOLDER__", prompt)
                    value = value.replace(
                        "__NEGATIVE_PLACEHOLDER__",
                        "blurry, low quality, distorted, jittery, bad anatomy, watermark, text",
                    )
                    if local_image:
                        value = value.replace("__IMAGE_PLACEHOLDER__", local_image)
                    inputs[key] = value

                # 按 key 名自动填充数值参数
                if key == "frame_count" and isinstance(value, (int, float)):
                    inputs[key] = actual_frames
                if key == "batch_size" and isinstance(value, (int, float)):
                    inputs[key] = actual_frames
                if key == "fps" and isinstance(value, (int, float)):
                    inputs[key] = fps
                if key == "width" and isinstance(value, (int, float)):
                    inputs[key] = width
                if key == "height" and isinstance(value, (int, float)):
                    inputs[key] = height
                if key == "frame_rate" and isinstance(value, (int, float)):
                    inputs[key] = fps

        return filled

    # ── ComfyUI 交互 ──

    def _poll_video_output(self, prompt_id: str, fps: int) -> List[Dict[str, str]]:
        """轮询 ComfyUI 历史接口获取视频输出文件"""
        deadline = time.monotonic() + COMFYUI_VIDEO_TIMEOUT

        while time.monotonic() < deadline:
            time.sleep(COMFYUI_POLL_INTERVAL)

            try:
                r = requests.get(f"{self._base_url}/history/{prompt_id}", timeout=10)
                r.raise_for_status()
                history = r.json()
            except Exception as exc:
                log.warning("ComfyUI video poll error: %s", exc)
                continue

            if prompt_id not in history:
                continue

            outputs = history[prompt_id].get("outputs", {})
            files: List[Dict[str, str]] = []

            for _node_id, node_output in outputs.items():
                # 检查视频输出
                for media in node_output.get("gifs", []):
                    files.append({
                        "filename": media["filename"],
                        "subfolder": media.get("subfolder", ""),
                        "type": media.get("type", "output"),
                        "format": "gif",
                    })
                # 检查图片序列 (VHS 组合前)
                for img_info in node_output.get("images", []):
                    files.append({
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                        "format": "image",
                    })

            if files:
                elapsed = time.monotonic() - (deadline - COMFYUI_VIDEO_TIMEOUT)
                log.info("ComfyUI video: %d output files in %.1fs", len(files), elapsed)
                return files

        log.error("ComfyUI video timeout after %ds for prompt %s", COMFYUI_VIDEO_TIMEOUT, prompt_id)
        return []

    def _download_to_comfyui_input(self, url: str) -> Optional[str]:
        """下载远程图片到 ComfyUI input 目录, 返回文件名"""
        # 如果是本地 URL, 直接提取文件名
        if url.startswith(f"{self._base_url}/"):
            return url.split("/")[-1]

        try:
            import requests as _r
            resp = _r.get(url, timeout=30, stream=True)
            resp.raise_for_status()

            # 通过 ComfyUI upload API 上传
            files = {"image": ("input_image.png", resp.content, "image/png")}
            upload_r = requests.post(
                f"{self._base_url}/upload/image",
                files=files,
                timeout=30,
            )
            if upload_r.status_code == 200:
                return upload_r.json().get("name", "input_image.png")
            log.warning("ComfyUI upload failed: %s", upload_r.text[:200])
            return None
        except Exception as exc:
            log.warning("Download to ComfyUI input failed: %s", exc)
            return None

    def _download_video_to_local(self, output_files: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
        """从 ComfyUI 下载生成的视频到本地 static 目录"""
        # 优先取 gif/mp4 文件
        video_files = [f for f in output_files if f.get("format") in ("gif", "mp4")]
        if not video_files:
            video_files = output_files[:1]  # 回退到第一个文件

        local_dir = Path("static/generated/videos")
        local_dir.mkdir(parents=True, exist_ok=True)

        for vf in video_files:
            filename = vf["filename"]
            params = f"filename={filename}&type={vf.get('type', 'output')}"
            subfolder = vf.get("subfolder", "")
            if subfolder:
                params += f"&subfolder={subfolder}"
            remote_url = f"{self._base_url}/view?{params}"

            try:
                r = requests.get(remote_url, timeout=60, stream=True)
                r.raise_for_status()

                # 保存到本地
                ext = Path(filename).suffix or (".gif" if vf.get("format") == "gif" else ".mp4")
                local_name = f"comfyui_{uuid.uuid4().hex[:10]}{ext}"
                local_path = local_dir / local_name
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)

                return {
                    "local_url": f"/static/generated/videos/{local_name}",
                    "remote_url": remote_url,
                    "filename": local_name,
                }
            except Exception as exc:
                log.warning("Failed to download video file %s: %s", filename, exc)
                continue

        return None

    # ── 工具方法 ──

    @staticmethod
    def _parse_resolution(res_str: str, default: str) -> tuple:
        """解析分辨率字符串 'WIDTHxHEIGHT' → (width, height)"""
        try:
            parts = res_str.lower().replace("p", "").split("x")
            if len(parts) == 2:
                return int(parts[0]), int(parts[1])
            # "720p" / "1080p" 快捷方式
            presets = {"720": (1280, 720), "1080": (1920, 1080)}
            if parts[0] in presets:
                return presets[parts[0]]
        except (ValueError, IndexError):
            pass
        try:
            w, h = default.split("x")
            return int(w), int(h)
        except (ValueError, IndexError):
            return 512, 512


def _clear_workflow_cache() -> None:
    """仅测试用"""
    global _WORKFLOW_CACHE
    _WORKFLOW_CACHE = None
