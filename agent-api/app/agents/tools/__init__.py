"""
SmartDoc Tools Module
Herramientas y utilidades para agentes de investigación
"""

from .base_tool import (
    BaseTool,
    ToolResult,
    ToolInput,
    ToolCategory,
    ToolRegistry,
    global_tool_registry,
    register_tool,
    get_available_tools,
    get_tool_by_name,
    tool_registration,
    EchoTool
)

__all__ = [
    "BaseTool",
    "ToolResult", 
    "ToolInput",
    "ToolCategory",
    "ToolRegistry",
    "global_tool_registry",
    "register_tool",
    "get_available_tools",
    "get_tool_by_name",
    "tool_registration",
    "EchoTool"
]
