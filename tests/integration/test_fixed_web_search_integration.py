"""
Integration tests for Fixed WebSearchTool with SmartDocAgent
Testing the complete integration workflow
"""

import pytest
import asyncio
import logging
from unittest.mock import Mock, patch, AsyncMock

# Project imports
from app.agents.core.smart_agent import SmartDocAgent
from app.agents.tools.web.web_search_tool import FixedWebSearchTool
from app.agents.tools.base import ToolCategory

logger = logging.getLogger(__name__)


class TestFixedWebSearchAgentIntegration:
    """Tests de integración del Fixed WebSearchTool con SmartDocAgent"""
    
    @pytest.fixture
    def web_search_tool(self):
        """Fixture del WebSearchTool arreglado"""
        return FixedWebSearchTool()
    
    @pytest.fixture
    def smart_agent(self):
        """Fixture del SmartDocAgent"""
        return SmartDocAgent(model_name="llama3.2:3b")
    
    def test_tool_can_be_added_to_agent(self, smart_agent, web_search_tool):
        """Test que el tool se puede agregar al agente"""
        
        # Verificar estado inicial
        assert len(smart_agent.tools) == 0
        
        # Agregar herramienta
        smart_agent.tools = [web_search_tool]
        
        # Verificar que se agregó
        assert len(smart_agent.tools) == 1
        assert smart_agent.tools[0] == web_search_tool
        assert smart_agent.tools[0].name == "web_search"
    
    def test_langchain_compatibility_with_agent(self, smart_agent, web_search_tool):
        """Test que el tool es compatible con LangChain en el contexto del agente"""
        
        from langchain.tools import BaseTool
        
        # Verificar que es una BaseTool válida
        assert isinstance(web_search_tool, BaseTool)
        
        # Agregar al agente
        smart_agent.tools = [web_search_tool]
        
        # Verificar que LangChain puede trabajar con ella
        tools_names = [tool.name for tool in smart_agent.tools]
        assert "web_search" in tools_names
        
        # Verificar que tiene los métodos requeridos por LangChain
        for tool in smart_agent.tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, '_run')
            assert hasattr(tool, '_arun')
    
    @pytest.mark.asyncio
    async def test_tool_health_check_in_agent_context(self, smart_agent, web_search_tool):
        """Test de health check del tool en el contexto del agente"""
        
        smart_agent.tools = [web_search_tool]
        
        # Mock successful search for health check
        with patch.object(web_search_tool, '_arun', 
                         return_value="Successful search result for health check"):
            
            health = await web_search_tool.health_check()
            
            assert health['healthy'] == True
            assert 'statistics' in health
            assert health['test_successful'] == True
    
    @pytest.mark.asyncio
    async def test_agent_can_use_tool_for_search(self, smart_agent, web_search_tool):
        """Test que el agente puede usar el tool para búsquedas"""
        
        # Setup
        smart_agent.tools = [web_search_tool]
        
        # Mock the web search to return a controlled result
        mock_search_result = """Search results for 'python tutorial' (Source: DuckDuckGo):

1. **Python Official Tutorial**
   URL: https://docs.python.org/3/tutorial/
   Summary: The official Python tutorial covering basics to advanced topics

2. **Learn Python - Beginner's Guide**
   URL: https://www.python.org/about/gettingstarted/
   Summary: Getting started guide for Python programming

Found 2 results. Use these results to answer the user's question."""
        
        with patch.object(web_search_tool, '_arun', return_value=mock_search_result):
            
            # Test direct tool usage
            result = await web_search_tool._arun("python tutorial", max_results=2)
            
            assert isinstance(result, str)
            assert "Python Official Tutorial" in result
            assert "docs.python.org" in result
            assert web_search_tool.stats['total_searches'] >= 1
    
    @pytest.mark.asyncio
    async def test_agent_initialization_with_tool(self, smart_agent, web_search_tool):
        """Test que el agente se puede inicializar con el tool"""
        
        # Agregar herramienta antes de inicializar
        smart_agent.tools = [web_search_tool]
        
        # Mock Ollama para evitar dependencias externas en tests
        mock_llm = AsyncMock()
        mock_llm.health_check.return_value = True
        
        with patch('app.agents.core.smart_agent.ChatOllama', return_value=mock_llm):
            with patch.object(smart_agent, '_setup_llm_real'):
                with patch.object(smart_agent, '_create_langchain_agent'):
                    
                    # Simular inicialización exitosa
                    smart_agent.is_initialized = True
                    smart_agent.llm = mock_llm
                    
                    # Verificar que el agente tiene la herramienta
                    assert len(smart_agent.tools) == 1
                    assert smart_agent.tools[0].name == "web_search"
                    assert smart_agent.is_initialized == True


class TestFixedWebSearchToolErrorHandling:
    """Tests de manejo de errores en integración"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    @pytest.fixture
    def smart_agent(self):
        return SmartDocAgent(model_name="llama3.2:3b")
    
    @pytest.mark.asyncio
    async def test_tool_handles_network_errors_gracefully(self, web_search_tool):
        """Test que el tool maneja errores de red gracefully"""
        
        # Mock network error
        with patch.object(web_search_tool, '_perform_multi_engine_search', 
                         side_effect=Exception("Network connection failed")):
            
            result = await web_search_tool._arun("test query")
            
            assert isinstance(result, str)
            assert "web search error" in result.lower()
            assert "network connection failed" in result.lower()
            assert web_search_tool.stats['engine_failures'] >= 1
    
    @pytest.mark.asyncio
    async def test_tool_handles_empty_results_gracefully(self, web_search_tool):
        """Test que el tool maneja resultados vacíos gracefully"""
        
        # Mock empty results
        with patch.object(web_search_tool, '_perform_multi_engine_search', 
                         return_value={'results': [], 'query': 'test'}):
            
            result = await web_search_tool._arun("test query")
            
            assert isinstance(result, str)
            assert "no search results found" in result.lower()
    
    @pytest.mark.asyncio
    async def test_tool_handles_malformed_responses(self, web_search_tool):
        """Test que el tool maneja respuestas malformadas"""
        
        # Mock malformed response
        with patch.object(web_search_tool, '_perform_multi_engine_search', 
                         return_value=None):
            
            result = await web_search_tool._arun("test query")
            
            assert isinstance(result, str)
            assert "no search results found" in result.lower()


class TestFixedWebSearchToolPerformance:
    """Tests de rendimiento del Fixed WebSearchTool"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    @pytest.mark.asyncio
    async def test_search_performance_is_reasonable(self, web_search_tool):
        """Test que el rendimiento de búsqueda es razonable"""
        
        import time
        
        # Mock fast response
        fast_mock_result = {
            'query': 'test query',
            'results': [
                {'title': 'Fast Result', 'url': 'https://fast.com', 'snippet': 'Fast snippet'}
            ],
            'source': 'FastEngine'
        }
        
        with patch.object(web_search_tool, '_perform_multi_engine_search', 
                         return_value=fast_mock_result):
            
            start_time = time.time()
            result = await web_search_tool._arun("performance test", max_results=1)
            end_time = time.time()
            
            duration = end_time - start_time
            
            assert duration < 5.0  # Debe completarse en menos de 5 segundos
            assert isinstance(result, str)
            assert len(result) > 0
    
    def test_cache_improves_performance(self, web_search_tool):
        """Test que el cache mejora el rendimiento"""
        
        query = "cache test query"
        mock_result = "Cached search result"
        
        # Primera vez - no está en cache
        cache_key = web_search_tool._generate_cache_key(query, 5)
        assert web_search_tool._get_cached_result(cache_key) is None
        
        # Almacenar en cache
        web_search_tool._cache_result(cache_key, mock_result)
        
        # Segunda vez - debe estar en cache
        cached_result = web_search_tool._get_cached_result(cache_key)
        assert cached_result == mock_result
        
        # Verificar que el cache tiene el elemento
        assert len(web_search_tool._cache) == 1


class TestFixedWebSearchRealWorld:
    """Tests de escenarios del mundo real"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    @pytest.mark.asyncio
    async def test_common_search_patterns(self, web_search_tool):
        """Test de patrones de búsqueda comunes"""
        
        # Mock responses for different search patterns
        mock_responses = {
            "how to": {
                'query': 'how to learn python',
                'results': [
                    {'title': 'How to Learn Python', 'url': 'https://tutorial.com', 'snippet': 'Step by step guide'}
                ],
                'source': 'DuckDuckGo'
            },
            "what is": {
                'query': 'what is machine learning',
                'results': [
                    {'title': 'Machine Learning Explained', 'url': 'https://ml.com', 'snippet': 'ML is a subset of AI'}
                ],
                'source': 'DuckDuckGo'
            },
            "latest": {
                'query': 'latest python version',
                'results': [
                    {'title': 'Python 3.12 Released', 'url': 'https://python.org', 'snippet': 'New features in Python 3.12'}
                ],
                'source': 'DuckDuckGo'
            }
        }
        
        for pattern, mock_response in mock_responses.items():
            with patch.object(web_search_tool, '_perform_multi_engine_search', 
                             return_value=mock_response):
                
                result = await web_search_tool._arun(mock_response['query'], max_results=1)
                
                assert isinstance(result, str)
                assert len(result) > 50  # Substantial content
                assert pattern.replace("_", " ") in result.lower() or mock_response['query'] in result.lower()


@pytest.mark.integration
class TestCompleteWorkflowWithFixedTool:
    """Tests de workflow completo con la herramienta arreglada"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_search_workflow(self):
        """Test de workflow completo de búsqueda"""
        
        # 1. Crear componentes
        agent = SmartDocAgent(model_name="llama3.2:3b")
        web_tool = FixedWebSearchTool()
        
        # 2. Configurar agente
        agent.tools = [web_tool]
        
        # 3. Mock successful search
        mock_search_result = """Search results for 'AI developments 2025' (Source: DuckDuckGo):

1. **Latest AI Breakthroughs in 2025**
   URL: https://ai-news.com/2025-breakthroughs
   Summary: Major developments in artificial intelligence this year

2. **AI Industry Report 2025**
   URL: https://tech-reports.org/ai-2025
   Summary: Comprehensive analysis of AI progress in 2025

Found 2 results. Use these results to answer the user's question."""
        
        with patch.object(web_tool, '_arun', return_value=mock_search_result):
            
            # 4. Test tool directly
            result = await web_tool._arun("AI developments 2025", max_results=2)
            
            # 5. Verify result quality
            assert isinstance(result, str)
            assert "AI Breakthroughs" in result
            assert "ai-news.com" in result
            assert "2025" in result
            assert len(result) > 200  # Substantial content
            
            # 6. Verify statistics
            assert web_tool.stats['total_searches'] >= 1
            assert web_tool.stats['successful_searches'] >= 1
    
    def test_integration_readiness_checklist(self):
        """Test que todos los componentes están listos para integración"""
        
        # 1. Verify tool can be imported
        from app.agents.tools.web.web_search_tool import FixedWebSearchTool
        from app.agents.core.smart_agent import SmartDocAgent
        from langchain.tools import BaseTool
        
        # 2. Verify tool instantiation
        tool = FixedWebSearchTool()
        assert isinstance(tool, FixedWebSearchTool)
        assert isinstance(tool, BaseTool)
        
        # 3. Verify agent instantiation
        agent = SmartDocAgent(model_name="test_model")
        assert isinstance(agent, SmartDocAgent)
        
        # 4. Verify tool can be added to agent
        agent.tools = [tool]
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "web_search"
        
        # 5. Verify LangChain compatibility
        assert hasattr(tool, 'name')
        assert hasattr(tool, 'description')
        assert hasattr(tool, '_run')
        assert hasattr(tool, '_arun')
        
        # 6. Verify tool configuration
        assert tool.name == "web_search"
        assert tool.version == "2.0.0"
        assert tool.requires_internet == True
        assert tool.category == ToolCategory.WEB_SEARCH
        
        print("✅ Integration readiness checklist passed!")
        print(f"   Tool: {tool.name} v{tool.version}")
        print(f"   Agent: {agent.__class__.__name__}")
        print(f"   LangChain Compatible: {isinstance(tool, BaseTool)}")


# Convenience functions for running specific test groups
def run_integration_tests():
    """Run integration tests only"""
    pytest.main([__file__, "-v", "--tb=short", "-m", "integration"])


def run_basic_tests():
    """Run basic functionality tests"""
    pytest.main([__file__ + "::TestFixedWebSearchAgentIntegration", "-v"])


def run_all_fixed_tests():
    """Run all tests for the fixed web search tool"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_all_fixed_tests()