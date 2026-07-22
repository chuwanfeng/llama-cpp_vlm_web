"""
ComfyUI 图片生成后端
通过 ComfyUI HTTP API 调用本地/远程工作流生成图片

支持多实例: 本地 + 云端可同时注册

配置:
    COMFYUI_BASE_URL = "http://127.0.0.1:8188"       # 默认(本地)
    COMFYUI_CLOUD_URL = "http://your-server:8188"     # 云端(可选, 图片+视频)
    也可通过构造函数 explicit 传 URL
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from agent.image_gen_provider import ImageGenProvider

log = logging.getLogger(__name__)

COMFYUI_BASE = os.environ.get("COMFYUI_BASE_URL", "http://127.0.0.1:8188")
COMFYUI_TIMEOUT = int(os.environ.get("COMFYUI_TIMEOUT", "120"))
COMFYUI_POLL_INTERVAL = float(os.environ.get("COMFYUI_POLL_INTERVAL", "1.0"))


class ComfyUIImageGenProvider(ImageGenProvider):
    """ComfyUI 后端 — 支持本地/云端 Stable Diffusion / FLUX / SDXL"""

    name = "comfyui"
    display_name = "ComfyUI (本地)"

    def __init__(
        self,
        base_url: Optional[str] = None,
        provider_name: str = "comfyui",
        display_name: str = "ComfyUI (本地)",
    ):
        self._base_url = (base_url or COMFYUI_BASE).rstrip("/")
        self.name = provider_name
        self.display_name = display_name
        self._available: Optional[bool] = None
        self._available_check_time = 0.0
        self._models_cache: Optional[List[Dict[str, Any]]] = None
        self._models_cache_time = 0.0

    def is_available(self) -> bool:
        """检测 ComfyUI 服务是否可达 (含 30s 缓存)"""
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
        """从 ComfyUI CheckpointLoaderSimple 探测可用模型 (含 5min 缓存)"""
        now = time.monotonic()
        if self._models_cache is not None and (now - self._models_cache_time) < 300:
            return self._models_cache
        try:
            r = requests.get(
                f"{self._base_url}/object_info/CheckpointLoaderSimple",
                timeout=10,
            )
            info = r.json()
            ckpt_list = (
                info.get("CheckpointLoaderSimple", {})
                .get("input", {})
                .get("required", {})
                .get("ckpt_name", [])
            )
            if isinstance(ckpt_list, list) and len(ckpt_list) > 1:
                names = ckpt_list[1:] if isinstance(ckpt_list[0], str) and ckpt_list[0] == "COMBO" else ckpt_list
            else:
                names = []
            self._models_cache = [
                {"id": name, "display_name": name, "max_resolution": "1024x1024"}
                for name in names
            ]
            self._models_cache_time = now
            return self._models_cache
        except Exception as exc:
            log.warning("Failed to list ComfyUI models [%s]: %s", self._base_url, exc)
            return self._models_cache or []

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
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        reference_images: Optional[List[str]] = None,
        workflow_name: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """提交 ComfyUI workflow → 轮询 → 获取图片

        模式:
          1. workflow_name 指定自定义工作流 → 从 generation_config 查找
          2. model 匹配自定义工作流名 → 自动切换
          3. 默认 → 内置 KSampler txt2img
        """
        # 匹配自定义工作流: model 参数可能是 key 或 display_name
        custom_wfs = self._list_custom_workflows()
        wf_name = workflow_name
        if not wf_name and model:
            if model in custom_wfs:
                wf_name = model
            else:
                # 按 display_name 模糊匹配
                from services.generation_config import list_registered_workflows
                wf_reg = list_registered_workflows()
                for k, v in wf_reg.items():
                    dn = v.get("display_name", "")
                    if dn and dn.lower() == model.lower():
                        wf_name = k
                        break
                    # 还支持 "Z Image Turbo" → "image_z_image_turbo" 这种去空格+下划线映射
                    normalized_model = model.lower().replace(" ", "_")
                    if normalized_model == k.lower() or normalized_model in k.lower():
                        wf_name = k
                        break
        if wf_name:
            return self._run_custom_workflow(
                wf_name=wf_name, prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, num_images=num_images,
                seed=seed, steps=steps, guidance_scale=guidance_scale,
                reference_images=reference_images,
            )

        # 1. 选择模型 / 修复 seed (ComfyUI 要求 seed >= 0)
        models = self.list_models()
        checkpoint = model or (models[0]["id"] if models else "sd_xl_base_1.0.safetensors")
        actual_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**31 - 1)
        # 内置路径的默认值（用户未指定时使用 ComfyUI 通用默认）
        _steps = steps if steps is not None else 20
        _cfg = guidance_scale if guidance_scale is not None else 7.0

        # 2. 构建 workflow JSON
        workflow = self._build_txt2img_workflow(
            prompt=prompt,
            negative=negative_prompt or "blurry, low quality, distorted, ugly, bad anatomy",
            checkpoint=checkpoint,
            width=width,
            height=height,
            steps=_steps,
            cfg=_cfg,
            seed=actual_seed,
            batch_size=min(num_images, 4),
        )

        # 3. 提交 prompt
        log.info("ComfyUI [%s]: %s (%dx%d, %d steps)",
                 self._base_url, checkpoint, width, height, _steps)
        r = requests.post(
            f"{self._base_url}/prompt",
            json={"prompt": workflow},
            timeout=30,
        )
        r.raise_for_status()
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            return self.error_response("ComfyUI returned no prompt_id")

        # 4. 轮询等待完成
        images = self._poll_for_result(prompt_id)

        if not images:
            return self.error_response("ComfyUI generation produced no images")

        # 5. 返回结果
        return self.success_response(
            images=images,
            model=checkpoint,
            seed=actual_seed,
            steps=_steps,
            guidance_scale=_cfg,
        )

    def _build_txt2img_workflow(
        self,
        prompt: str,
        negative: str,
        checkpoint: str,
        width: int,
        height: int,
        steps: int,
        cfg: float,
        seed: int,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """构建标准 txt2img workflow (KSampler)"""
        return {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed, "steps": steps, "cfg": cfg,
                    "sampler_name": "euler", "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["4", 0], "positive": ["6", 0],
                    "negative": ["7", 0], "latent_image": ["5", 0],
                },
            },
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": checkpoint},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": batch_size},
            },
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["4", 1]},
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative, "clip": ["4", 1]},
            },
            "8": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
            },
            "9": {
                "class_type": "PreviewImage",
                "inputs": {"images": ["8", 0]},
            },
        }

    def _poll_for_result(self, prompt_id: str) -> List[Dict[str, str]]:
        """轮询 ComfyUI 历史接口获取生成结果"""
        deadline = time.monotonic() + COMFYUI_TIMEOUT

        while time.monotonic() < deadline:
            time.sleep(COMFYUI_POLL_INTERVAL)

            try:
                r = requests.get(f"{self._base_url}/history/{prompt_id}", timeout=10)
                r.raise_for_status()
                history = r.json()
            except Exception as exc:
                log.warning("ComfyUI poll error [%s]: %s", self._base_url, exc)
                continue

            if prompt_id not in history:
                continue

            outputs = history[prompt_id].get("outputs", {})
            images = []
            for _node_id, node_output in outputs.items():
                for img_info in node_output.get("images", []):
                    filename = img_info["filename"]
                    subfolder = img_info.get("subfolder", "")
                    img_type = img_info.get("type", "output")
                    params = f"filename={filename}&type={img_type}"
                    if subfolder:
                        params += f"&subfolder={subfolder}"
                    images.append({"url": f"{self._base_url}/view?{params}"})

            if images:
                elapsed = time.monotonic() - (deadline - COMFYUI_TIMEOUT)
                log.info("ComfyUI [%s]: %d images in %.1fs", self._base_url, len(images), elapsed)
                return images

        log.error("ComfyUI [%s] timeout after %ds for prompt %s",
                  self._base_url, COMFYUI_TIMEOUT, prompt_id)
        return []

    # ── 自定义工作流支持 ────────────────────────────────────────────────

    @staticmethod
    def _list_custom_workflows() -> set:
        """返回已注册的图片类自定义工作流名称集合"""
        try:
            from services.generation_config import list_registered_workflows
            wfs = list_registered_workflows()
            return {n for n, w in wfs.items() if w.get("type") == "image"}
        except Exception:
            return set()

    def _run_custom_workflow(
        self,
        wf_name: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        reference_images: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """加载自定义工作流 JSON → 替换占位符 → 展开子图 → 提交 ComfyUI"""
        from services.generation_config import list_registered_workflows

        wfs = list_registered_workflows()
        wf_def = wfs.get(wf_name)
        if not wf_def:
            return self.error_response(f"Custom workflow '{wf_name}' not found")

        wf = wf_def.get("workflow")
        if not wf:
            return self.error_response(f"Workflow '{wf_name}' has no 'workflow' key")

        # deep copy
        import copy
        wf = json.loads(json.dumps(wf))

        # 1) 替换占位符
        placeholder_values = {
            "__PROMPT__": prompt,
            "__NEGATIVE__": negative_prompt or "blurry, low quality",
            "__WIDTH__": str(width),
            "__HEIGHT__": str(height),
            "__SEED__": str(seed if (seed is not None and seed >= 0) else random.randint(0, 2**31 - 1)),
            "__STEPS__": str(steps) if steps is not None else "20",
            "__CFG__": str(guidance_scale) if guidance_scale is not None else "1.0",
            "__BATCH__": str(min(num_images, 4)),
        }
        self._replace_placeholders(wf, placeholder_values)

        # 2) 生成实际用的 seed（随机化，解决每次相同图片的问题）
        actual_seed = seed if (seed is not None and seed >= 0) else random.randint(0, 2**31 - 1)
        # 注入用户参数（包括 seed）到子图 KSampler 节点
        self._inject_user_params(wf, prompt, negative_prompt, width, height,
                                 num_images, actual_seed, steps, guidance_scale)

        log.info("ComfyUI [%s] custom workflow '%s': %s (%dx%d)",
                 self._base_url, wf_name, prompt, width, height)

        # 3) 将工作流转换为 /prompt API 格式
        prompt_wf = self._workflow_to_prompt(wf)
        extra_data = {"extra_pnginfo": {"workflow": wf}}

        r = requests.post(
            f"{self._base_url}/prompt",
            json={"prompt": prompt_wf, "extra_data": extra_data},
            timeout=30,
        )
        r.raise_for_status()
        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            return self.error_response("ComfyUI returned no prompt_id")

        images = self._poll_for_result(prompt_id)
        if not images:
            return self.error_response("Custom workflow produced no images")

        return self.success_response(images=images, model=wf_name)

    # KSampler 的 widget 名称→widgets_values 索引映射
    # randomize 对应 idx=1，无对应 input，导致后续 widget input 全部偏移
    _KSampler_WIDGET_MAP = {
        "seed": 0,
        "randomize": 1,
        "steps": 2,
        "cfg": 3,
        "sampler_name": 4,
        "scheduler": 5,
        "denoise": 6,
    }

    def _inject_user_params(
        self,
        wf: Dict[str, Any],
        prompt: str,
        negative_prompt: Optional[str],
        width: int,
        height: int,
        num_images: int,
        seed: int,
        steps: Optional[int],
        guidance_scale: Optional[float],
    ) -> None:
        """将用户参数注入子图内部节点

        通过 -10 虚节点的链接映射找到每个输入端口对应的内部节点和 widget 位置，
        直接修改 widgets_values。
        """
        subgraphs = wf.get("definitions", {}).get("subgraphs", [])
        if not subgraphs:
            return

        sg = subgraphs[0]
        sg_nodes = sg.get("nodes", [])
        sg_links = sg.get("links", [])
        sg_inputs = sg.get("inputs", [])

        # 构建链接索引
        sg_link_map = {}
        for link in sg_links:
            if isinstance(link, dict):
                sg_link_map[link["id"]] = (str(link["origin_id"]), link["origin_slot"],
                                           str(link["target_id"]), link.get("target_slot", 0))

        # 构建节点索引
        node_map = {str(n["id"]): n for n in sg_nodes}

        # -10 输入端口映射: 端口名 → 内部 (node_id, input_slot)
        port_map = {}
        for inp in sg_inputs:
            port_name = inp["name"]
            link_ids = inp.get("linkIds", [])
            if link_ids and link_ids[0] in sg_link_map:
                _, _, target_id, target_slot = sg_link_map[link_ids[0]]
                port_map[port_name] = (target_id, target_slot)

        # 注入参数: 找到目标节点→input→widget_values 的映射
        # 只注入非 None 的参数，保留工作流原生设定
        for port_name, value in [
            ("text", prompt),
            ("width", width),
            ("height", height),
            ("steps", steps),
            ("cfg", guidance_scale),
            ("batch_size", min(num_images, 4)),
        ]:
            if value is None or port_name not in port_map:
                continue
            target_id, target_slot = port_map[port_name]
            node = node_map.get(target_id)
            if not node:
                continue

            inputs = node.get("inputs", [])
            wv = node.get("widgets_values", [])

            if target_slot >= len(inputs):
                continue

            inp = inputs[target_slot]
            iname = inp.get("name", "")
            wname = inp.get("widget", {}).get("name", "")

            # 使用节点类型专属 widget 映射确定 widgets_values 索引
            # （KSampler 的 randomize 等无 input 对应部件会导致 widget_idx 计数偏移）
            widget_map = getattr(self, f"_{node.get('type', '')}_WIDGET_MAP", None)
            if widget_map:
                widget_idx = widget_map.get(wname, -1)
                if 0 <= widget_idx < len(wv):
                    wv[widget_idx] = value
                    log.debug("Injected %s=%s → node %s (%s) widgets[%d]",
                             port_name, value, target_id, node.get("type", ""), widget_idx)
            else:
                # 回退：按顺序计数
                widget_idx = 0
                for i in inputs:
                    if i.get("widget") is not None:
                        if i.get("name") == iname:
                            if widget_idx < len(wv):
                                wv[widget_idx] = value
                                log.debug("Injected %s=%s → node %s (%s) widgets[%d]",
                                         port_name, value, target_id, node.get("type", ""), widget_idx)
                            break
                        widget_idx += 1

        # 注入 seed（KSampler 的特殊处理）
        if seed is not None and seed >= 0:
            for node in sg_nodes:
                if node.get("type") == "KSampler":
                    wv = node.get("widgets_values", [])
                    if wv:
                        wv[0] = seed
                        log.debug("Injected seed=%d into KSampler", seed)
                    break

        # 注入 negative prompt（CLIPTextEncode 的负向判断）
        for node in sg_nodes:
            if node.get("type") == "CLIPTextEncode":
                wv = node.get("widgets_values", [])
                if wv and isinstance(wv[0], str):
                    txt = wv[0].lower()
                    is_negative = any(kw in txt for kw in
                                      ["worst quality", "low quality", "nsfw", "blurry", "deformed"])
                    if is_negative:
                        wv[0] = negative_prompt or "blurry, low quality, deformed"

        log.info("Injected user params into %d subgraph nodes", len(port_map) + 1)

    def _replace_placeholders(self, obj: Any, values: Dict[str, str]) -> None:
        """递归替换 workflow JSON 中的占位符

        支持:
        1. 文本占位符 (values dict 中 key→value)
        2. ComfyUI %date:format% 动态日期模板 (解析为当前时间)
           format 使用 ComfyUI 方言: yyyy MM dd hh mm ss
        """
        # 匹配 %date:格式化字符串% 例如 %date:yyyyMMdd_hhmmss%
        _DATE_RE = re.compile(r"%date:([^%]+)%")
        # ComfyUI 日期标记 → Python strftime 格式转换
        _DATE_MAP = {"yyyy": "%Y", "yy": "%y", "MM": "%m", "dd": "%d",
                     "hh": "%H", "mm": "%M", "ss": "%S"}

        def _resolve_date(m: re.Match) -> str:
            cfmt = m.group(1)
            for cf_key, py_key in _DATE_MAP.items():
                cfmt = cfmt.replace(cf_key, py_key)
            return datetime.now().strftime(cfmt)

        if isinstance(obj, dict):
            for k, v in list(obj.items()):
                if isinstance(v, str):
                    # 先精确匹配占位符 key
                    v2 = values.get(v, v)
                    obj[k] = _DATE_RE.sub(_resolve_date, v2)
                else:
                    self._replace_placeholders(v, values)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, str):
                    obj[i] = _DATE_RE.sub(_resolve_date, item)
                else:
                    self._replace_placeholders(item, values)

    # ── 工作流格式转换 ──────────────────────────────────────

    def _workflow_to_prompt(self, wf: Dict[str, Any]) -> Dict[str, Any]:
        """将 ComfyUI 完整工作流转换为 /prompt API 格式

        支持两种工作流格式:
        1. 标准格式: nodes 数组 + links 数组 → 转换为 prompt dict
        2. 子图格式: 含 definitions.subgraphs 的组节点 → 展开内部节点

        自动解开 workflow 包装层（ComfyUI 导出的 JSON 常含此层）

        /prompt API 格式:
            {"<node_id>": {"class_type": "...", "inputs": {"name": value}}}
        """
        # 解开 workflow 包装层
        if "workflow" in wf and isinstance(wf.get("workflow"), dict):
            wf = wf["workflow"]
        # 如果有子图定义，先展开
        subgraphs = wf.get("definitions", {}).get("subgraphs", [])
        if subgraphs:
            return self._expand_subgraphs(wf, subgraphs)

        return self._nodes_to_prompt(wf.get("nodes", []), wf.get("links", []))

    def _nodes_to_prompt(self, nodes: List[Dict], links_list: List[Any]) -> Dict[str, Any]:
        """将 nodes + links 列表转换为 prompt 格式"""
        link_map: Dict[int, List[Any]] = {}
        for link in links_list:
            if isinstance(link, dict):
                lid = link["id"]
                link_map[lid] = [str(link["origin_id"]), link.get("origin_slot", 0)]
            elif isinstance(link, (list, tuple)) and len(link) >= 5:
                lid = link[0]
                link_map[lid] = [str(link[1]), link[2]]

        prompt = {}
        for node in nodes:
            node_id = str(node["id"])
            node_type = node.get("type", node.get("class_type", ""))
            inputs_raw = node.get("inputs", [])
            widgets_values = node.get("widgets_values", [])

            inputs_out = {}
            # 获取该节点类型的 widget 名称→索引映射
            widget_map = getattr(self, f"_{node_type}_WIDGET_MAP", None)
            read_widget_idx = 0

            for inp in inputs_raw:
                name = inp.get("name", "")
                link_id = inp.get("link")

                if link_id is not None:
                    if link_id in link_map:
                        inputs_out[name] = link_map[link_id]
                    else:
                        log.warning("Link %s not found for input '%s' of node %s",
                                   link_id, name, node_id)
                        inputs_out[name] = None
                elif inp.get("widget") is not None:
                    wname = inp.get("widget", {}).get("name", "")
                    if widget_map:
                        pos = widget_map.get(wname, -1)
                        inputs_out[name] = widgets_values[pos] if 0 <= pos < len(widgets_values) else ""
                    else:
                        if read_widget_idx < len(widgets_values):
                            inputs_out[name] = widgets_values[read_widget_idx]
                        else:
                            inputs_out[name] = ""
                        read_widget_idx += 1
                else:
                    inputs_out[name] = None

            prompt[node_id] = {
                "class_type": node_type,
                "inputs": inputs_out,
            }

        return prompt

    def _expand_subgraphs(
        self, wf: Dict[str, Any], subgraphs: List[Dict]
    ) -> Dict[str, Any]:
        """展开子图为平铺的 prompt 格式

        -10 虚节点（子图输入）→ 使用子图节点自身 widgets_values（已注入）
        -20 虚节点（子图输出）→ 重定向到主图中 SaveImage 等外部节点的链接目标
        外部无链接节点（easy positive, CR Text 等）→ 跳过
        """
        main_nodes = wf.get("nodes", [])
        main_links = wf.get("links", [])
        sg = subgraphs[0]
        sg_id = sg.get("id", "")
        sg_nodes = sg.get("nodes", [])
        sg_links = sg.get("links", [])

        # 找到组节点
        group_node_id = None
        for n in main_nodes:
            if n.get("type") == sg_id:
                group_node_id = n["id"]
                break

        if group_node_id is None:
            return self._nodes_to_prompt(main_nodes, main_links)

        # 构建子图链接索引
        sg_link_map: Dict[int, List[Any]] = {}
        for link in sg_links:
            if isinstance(link, dict):
                sg_link_map[link["id"]] = [str(link["origin_id"]), link.get("origin_slot", 0)]

        # 找到 -20 的输出 link → (origin_node, origin_slot)
        output_ref = None
        for link in sg_links:
            if isinstance(link, dict):
                if str(link.get("target_id")) == "-20":
                    lid = link["id"]
                    if lid in sg_link_map:
                        output_ref = sg_link_map[lid]
                        break

        # 构建主图链接: link_id → [origin_id, origin_slot]
        main_link_map: Dict[int, List[Any]] = {}
        for link in main_links:
            if isinstance(link, dict):
                main_link_map[link["id"]] = [str(link["origin_id"]), link.get("origin_slot", 0)]
            elif isinstance(link, (list, tuple)) and len(link) >= 5:
                main_link_map[link[0]] = [str(link[1]), link[2]]

        prompt = {}

        # ── 1) 内部节点 ──
        for node in sg_nodes:
            nid = str(node["id"])
            ntype = node.get("type", "")
            inputs_raw = node.get("inputs", [])
            wv = node.get("widgets_values", [])

            inputs_out = {}
            widget_map_sg = getattr(self, f"_{ntype}_WIDGET_MAP", None)
            widget_idx_read = 0

            for inp in inputs_raw:
                name = inp.get("name", "")
                link_id = inp.get("link")
                has_widget = inp.get("widget") is not None

                resolved = None
                if link_id is not None and link_id in sg_link_map:
                    ref = sg_link_map[link_id]
                    if ref[0] != "-10" and ref[0] != "-20":
                        # 正常的内部链接
                        resolved = ref
                    # -10 / -20 → resolved stays None → falls through to widget

                if resolved is not None:
                    inputs_out[name] = resolved
                elif has_widget:
                    if widget_map_sg:
                        wname = inp.get("widget", {}).get("name", "")
                        pos = widget_map_sg.get(wname, -1)
                        inputs_out[name] = wv[pos] if 0 <= pos < len(wv) else ""
                    else:
                        if widget_idx_read < len(wv):
                            inputs_out[name] = wv[widget_idx_read]
                        else:
                            inputs_out[name] = ""
                        widget_idx_read += 1
                else:
                    inputs_out[name] = None

            prompt[nid] = {"class_type": ntype, "inputs": inputs_out}

        # ── 2) 外部节点（有连接的）──
        for node in main_nodes:
            nid = node["id"]
            if nid == group_node_id:
                continue

            nid_str = str(nid)
            ntype = node.get("type", "")
            inputs_raw = node.get("inputs", [])
            wv = node.get("widgets_values", [])

            inputs_out = {}
            widget_map_ext = getattr(self, f"_{ntype}_WIDGET_MAP", None)
            widget_idx_ext = 0
            has_valid_link = False

            for inp in inputs_raw:
                name = inp.get("name", "")
                link_id = inp.get("link")
                has_widget = inp.get("widget") is not None

                resolved = None
                if link_id is not None and link_id in main_link_map:
                    ref = main_link_map[link_id]
                    if ref[0] == str(group_node_id):
                        # 指向组节点 → 重定向到子图输出
                        if output_ref:
                            resolved = output_ref
                        has_valid_link = True
                    else:
                        resolved = ref
                        has_valid_link = True

                if resolved is not None:
                    inputs_out[name] = resolved
                elif has_widget:
                    if widget_map_ext:
                        wname = inp.get("widget", {}).get("name", "")
                        pos = widget_map_ext.get(wname, -1)
                        inputs_out[name] = wv[pos] if 0 <= pos < len(wv) else ""
                    else:
                        if widget_idx_ext < len(wv):
                            inputs_out[name] = wv[widget_idx_ext]
                        else:
                            inputs_out[name] = ""
                        widget_idx_ext += 1
                else:
                    inputs_out[name] = None

            # 只保留有有效链接的外部节点（如 SaveImage），跳过孤立的 prompt 模板节点
            if has_valid_link or ntype in ("SaveImage", "PreviewImage", "VHS_VideoCombine"):
                prompt[nid_str] = {"class_type": ntype, "inputs": inputs_out}

        log.info("Expanded subgraph: %d internal + %d external = %d prompt nodes",
                 len(sg_nodes), len(prompt) - len(sg_nodes), len(prompt))
        return prompt
