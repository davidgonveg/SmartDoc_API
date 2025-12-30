"""
Verification Script for SmartDoc Project Status
Checks if critical features claimed to be 'In Development' are actually implemented.
"""

import pytest
import inspect
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../agent-api')))

from app.agents.core.smart_agent import SmartDocAgent
from app.agents.tools.web.web_search_tool import WebSearchTool

class TestProjectStatus:
    """Verify actual project status against README claims"""

    def test_langchain_implementation(self):
        """Verify SmartDocAgent implementation uses real LangChain primitives"""
        agent = SmartDocAgent()
        
        # Check source code for LangChain imports and usage
        source_code = inspect.getsource(SmartDocAgent)
        
        has_agent_executor = "AgentExecutor" in source_code
        has_create_react_agent = "create_react_agent" in source_code
        has_chat_ollama = "ChatOllama" in source_code or "OllamaLLM" in source_code
        
        print(f"\n[Feature Check] LangChain Agent Implementation:")
        print(f"  - Uses AgentExecutor: {'✅' if has_agent_executor else '❌'}")
        print(f"  - Uses create_react_agent: {'✅' if has_create_react_agent else '❌'}")
        
        assert has_agent_executor, "SmartDocAgent should use LangChain AgentExecutor"
        assert has_create_react_agent, "SmartDocAgent should use create_react_agent pattern"

    def test_web_search_tool_implementation(self):
        """Verify WebSearchTool is fully implemented, not a placeholder"""
        tool = WebSearchTool()
        
        # Check if it has real implementation features
        has_arun = hasattr(tool, '_arun')
        
        # Check for multi-engine support in source
        source_code = inspect.getsource(WebSearchTool)
        has_duckduckgo = "duckduckgo" in source_code.lower()
        has_bs4 = "BeautifulSoup" in source_code
        
        print(f"\n[Feature Check] Web Search Tool:")
        print(f"  - Implements _arun: {'✅' if has_arun else '❌'}")
        print(f"  - Supports DuckDuckGo: {'✅' if has_duckduckgo else '❌'}")
        print(f"  - Uses BeautifulSoup: {'✅' if has_bs4 else '❌'}")
        
        assert has_arun, "WebSearchTool must have async run method"
        assert has_duckduckgo, "WebSearchTool should support search engines"

    def test_ollama_integration(self):
        """Verify Ollama integration is set up for production"""
        source_code = inspect.getsource(SmartDocAgent._setup_llm_real)
        
        has_chato_llama = "ChatOllama" in source_code
        has_streaming = "streaming=" in source_code
        
        print(f"\n[Feature Check] Ollama Integration:")
        print(f"  - Uses ChatOllama (LangChain): {'✅' if has_chato_llama else '❌'}")
        print(f"  - Supports Streaming: {'✅' if has_streaming else '❌'}")
        
        assert has_chato_llama, "Should use ChatOllama for LangChain integration"

    def test_api_routes_existence(self):
        """Verify existence of critical API routes"""
        from app.api.routes import research
        
        routes = [route.path for route in research.router.routes]
        
        print(f"\n[Feature Check] API Routes (Research):")
        expected_routes = ["/session", "/chat/{session_id}", "/session/{session_id}/optimize"]
        
        for route in expected_routes:
            exists = any(route in r for r in routes)
            print(f"  - {route}: {'✅' if exists else '❌'}")
            
        # Optimize endpoint indicates v0.2+ feature
        assert any("/session/{session_id}/optimize" in r for r in routes), "Optimization endpoints missing"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
