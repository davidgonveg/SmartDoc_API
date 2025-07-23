# ===============================================
# agent-api/app/agents/__init__.py
# ===============================================
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

# Exponer las clases principales del módulo
__all__ = [
    # Core Agent
    "SmartDocAgent",
    "create_smartdoc_agent", 
    "ResearchSession",
    
    # Templates
    "ReactTemplates",
    "SMARTDOC_SYSTEM_PROMPT",
    
    # Tools Infrastructure
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

# Metadata del módulo
__version__ = "0.1.0"
__author__ = "SmartDoc Team"
__description__ = "Intelligent research agents with ReAct pattern and tool orchestration"

# Configuración de logging para el módulo
import logging
logger = logging.getLogger(__name__)
logger.info(f"SmartDoc Agents Module v{__version__} loaded")

# ===============================================
# agent-api/app/agents/core/__init__.py  
# ===============================================
"""
SmartDoc Core Agents Module
Implementaciones principales de agentes
"""

from .smart_agent import SmartDocAgent, create_smartdoc_agent, ResearchSession

__all__ = [
    "SmartDocAgent",
    "create_smartdoc_agent",
    "ResearchSession"
]

# ===============================================
# agent-api/app/agents/prompts/__init__.py
# ===============================================
"""
SmartDoc Prompts Module  
Templates y prompts para agentes ReAct
"""

from .react_templates import (
    ReactTemplates,
    SMARTDOC_SYSTEM_PROMPT,
    REACT_MAIN_TEMPLATE,
    ACADEMIC_RESEARCH_TEMPLATE,
    BUSINESS_RESEARCH_TEMPLATE,
    TECHNICAL_RESEARCH_TEMPLATE,
    SYNTHESIS_TEMPLATE,
    DEFAULT_PROMPT_CONFIGS,
    EXAMPLE_INPUTS
)

__all__ = [
    "ReactTemplates",
    "SMARTDOC_SYSTEM_PROMPT", 
    "REACT_MAIN_TEMPLATE",
    "ACADEMIC_RESEARCH_TEMPLATE",
    "BUSINESS_RESEARCH_TEMPLATE", 
    "TECHNICAL_RESEARCH_TEMPLATE",
    "SYNTHESIS_TEMPLATE",
    "DEFAULT_PROMPT_CONFIGS",
    "EXAMPLE_INPUTS"
]

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

# ===============================================
# agent-api/app/services/__init__.py (actualizado)
# ===============================================
"""
Services module - External service integrations
"""

from .ollama_client import OllamaClient, get_ollama_client

__all__ = ["OllamaClient", "get_ollama_client"]

# ===============================================
# agent-api/app/api/__init__.py (actualizado)
# ===============================================
"""
SmartDoc API Module
FastAPI routes and endpoints
"""

# Re-export common API components
from .routes import research, health, upload

__all__ = ["research", "health", "upload"]

# ===============================================
# agent-api/app/api/routes/__init__.py (actualizado)
# ===============================================
"""
API Routes Module
FastAPI route definitions
"""

from . import health, research, upload

__all__ = ["health", "research", "upload"]