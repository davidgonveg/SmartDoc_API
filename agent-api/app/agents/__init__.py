"""
SmartDoc Agents Module
Módulo principal para agentes de investigación inteligentes
"""

from .core.smart_agent import SmartDocAgent, create_smartdoc_agent, ResearchSession
from .prompts.react_templates import ReactTemplates, SMARTDOC_SYSTEM_PROMPT
from .tools.base_tool import (
    BaseTool, 
    ToolResult, 
    ToolInput, 
    ToolCategory,
    ToolRegistry,
    global_tool_registry,
    register_tool,
    get_available_tools,
    get_tool_by_name,
    tool_registration
)

__all__ = [
    "SmartDocAgent",
    "create_smartdoc_agent", 
    "ResearchSession",
    "ReactTemplates",
    "SMARTDOC_SYSTEM_PROMPT",
    "BaseTool",
    "ToolResult",
    "ToolInput", 
    "ToolCategory",
    "ToolRegistry",
    "global_tool_registry",
    "register_tool",
    "get_available_tools",
    "get_tool_by_name",  
    "tool_registration"
]
