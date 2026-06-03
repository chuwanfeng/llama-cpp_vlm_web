"""GLM 4.7 工具调用解析器。

基于 GLM 4.5 扩展，更新正则表达式模式。
tool_call 标签包裹方式略有不同，arg 解析处理键值对之间的换行。

基于 VLLM 的 Glm47MoeModelToolParser（继承 Glm4MoeModelToolParser）。
"""

import re

from services.tool_call_parsers import register_parser
from services.tool_call_parsers.glm45_parser import Glm45ToolCallParser


@register_parser("glm47")
class Glm47ToolCallParser(Glm45ToolCallParser):
    """GLM 4.7 工具调用解析器。
    继承 GLM 4.5，使用更新的正则表达式模式。
    """

    def __init__(self):
        super().__init__()
        # GLM 4.7 的 FUNC_DETAIL_REGEX 包含 <tool_call> 包裹和可选 arg_key 内容
        self.FUNC_DETAIL_REGEX = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        # GLM 4.7 处理 arg_key/arg_value 之间的换行
        self.FUNC_ARG_REGEX = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )
