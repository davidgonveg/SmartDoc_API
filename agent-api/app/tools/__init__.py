# ===============================================
# agent-api/app/agents/tools/__init__.py
# ===============================================
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

# Auto-registrar herramientas disponibles
def _auto_register_tools():
    """Auto-registrar herramientas en el módulo"""
    # Por ahora solo EchoTool para testing
    echo_tool = EchoTool()
    register_tool(echo_tool)
    
# Ejecutar auto-registro
_auto_register_tools()

