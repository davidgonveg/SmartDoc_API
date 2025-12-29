"""
Web Search Tools Module - Safe Version
No instancia automáticamente para evitar errores de Pydantic
"""

# Solo importar las clases, sin instanciar
try:
    from .web_search_tool import WebSearchTool, create_web_search_tool, web_search
    WEB_SEARCH_AVAILABLE = True
except ImportError as e:
    # Fallback si hay problemas de import
    WEB_SEARCH_AVAILABLE = False
    
    class WebSearchTool:
        """Placeholder WebSearchTool"""
        def __init__(self):
            raise ImportError(f"WebSearchTool not available: {e}")
    
    def create_web_search_tool():
        raise ImportError("WebSearchTool not available")
    
    async def web_search(query: str, **kwargs):
        raise ImportError("WebSearchTool not available")

# Export only what's available
__all__ = [
    "WebSearchTool",
    "create_web_search_tool", 
    "web_search",
    "WEB_SEARCH_AVAILABLE"
]

# Module metadata
__version__ = "2.1.0"
__author__ = "SmartDoc Team"
__description__ = "Standalone web search tool for SmartDoc Agent"

# NO instanciar automáticamente - dejar que el usuario lo haga cuando necesite
# web_search_tool = WebSearchTool()  # ← REMOVIDO para evitar errores Pydantic

# Convenience function que instancia solo cuando se usa
async def search_web(query: str, **kwargs) -> str:
    """Quick web search function - creates tool on demand"""
    tool = create_web_search_tool()
    return await tool._arun(query, **kwargs)