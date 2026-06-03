"""Agent tools - supports web search, file ops, terminal, and more.

Tool architecture:
    Each tool module calls registry.register() at module level.
    registry.py manages discovery, schemas, and handler dispatch.
    tool_parser.py handles XML/JSON extraction for llama-cpp backends.
"""

from tools.registry import ToolRegistry, get_registry, discover_tools

__all__ = ["ToolRegistry", "get_registry", "discover_tools"]
