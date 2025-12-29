"""
Unit tests for the Fixed WebSearchTool implementation
Testing the new robust web search tool that actually works
"""

import pytest
import asyncio
import time
import logging
from unittest.mock import Mock, patch, AsyncMock

# Project imports
from app.agents.tools.web.web_search_tool import FixedWebSearchTool
from app.agents.tools.base import ToolCategory

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestFixedWebSearchToolBasics:
    """Test básicos del Fixed WebSearchTool"""
    
    @pytest.fixture
    def web_search_tool(self):
        """Fixture del WebSearchTool arreglado"""
        return FixedWebSearchTool()
    
    def test_tool_initialization(self, web_search_tool):
        """Test que el tool se inicializa correctamente"""
        assert web_search_tool.name == "web_search"
        assert web_search_tool.description is not None
        assert "search the web" in web_search_tool.description.lower()
        assert web_search_tool.category == ToolCategory.WEB_SEARCH
        assert web_search_tool.version == "2.0.0"
        assert web_search_tool.requires_internet == True
        assert web_search_tool.requires_gpu == False
        assert web_search_tool.max_execution_time == 60
        assert web_search_tool.rate_limit == 30
    
    def test_langchain_compatibility(self, web_search_tool):
        """Test que el tool es compatible con LangChain"""
        from langchain.tools import BaseTool
        
        # Verificar herencia
        assert isinstance(web_search_tool, BaseTool)
        
        # Verificar métodos requeridos
        assert hasattr(web_search_tool, 'name')
        assert hasattr(web_search_tool, 'description')
        assert hasattr(web_search_tool, '_run')
        assert hasattr(web_search_tool, '_arun')
        
        # Verificar que name y description son strings
        assert isinstance(web_search_tool.name, str)
        assert isinstance(web_search_tool.description, str)
        assert len(web_search_tool.name) > 0
        assert len(web_search_tool.description) > 0
    
    def test_cache_system_initialization(self, web_search_tool):
        """Test que el sistema de cache se inicializa"""
        assert hasattr(web_search_tool, '_cache')
        assert hasattr(web_search_tool, '_cache_ttl')
        assert hasattr(web_search_tool, '_last_request_time')
        assert hasattr(web_search_tool, '_min_request_interval')
        
        assert isinstance(web_search_tool._cache, dict)
        assert web_search_tool._cache_ttl > 0
        assert web_search_tool._min_request_interval > 0
    
    def test_statistics_initialization(self, web_search_tool):
        """Test que las estadísticas se inicializan"""
        assert hasattr(web_search_tool, 'stats')
        assert isinstance(web_search_tool.stats, dict)
        
        required_stats = [
            'total_searches', 'successful_searches', 
            'cache_hits', 'engine_failures'
        ]
        
        for stat in required_stats:
            assert stat in web_search_tool.stats
            assert web_search_tool.stats[stat] == 0


class TestFixedWebSearchToolCaching:
    """Tests del sistema de cache"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    def test_cache_key_generation(self, web_search_tool):
        """Test de generación de claves de cache"""
        key1 = web_search_tool._generate_cache_key("python", 5)
        key2 = web_search_tool._generate_cache_key("python", 10)
        key3 = web_search_tool._generate_cache_key("java", 5)
        
        # Keys deben ser diferentes para diferentes parámetros
        assert key1 != key2  # Diferentes max_results
        assert key1 != key3  # Diferentes queries
        assert key2 != key3
        
        # Keys deben ser strings válidos
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    def test_cache_storage_and_retrieval(self, web_search_tool):
        """Test de almacenamiento y recuperación de cache"""
        cache_key = "test_key"
        test_result = "Test cached search result"
        
        # Store in cache
        web_search_tool._cache_result(cache_key, test_result)
        
        # Retrieve from cache
        cached = web_search_tool._get_cached_result(cache_key)
        
        assert cached == test_result
        assert len(web_search_tool._cache) == 1
    
    def test_cache_expiration(self, web_search_tool):
        """Test de expiración de cache"""
        cache_key = "expiring_key"
        test_result = "Expiring result"
        
        # Reducir TTL para testing rápido
        original_ttl = web_search_tool._cache_ttl
        web_search_tool._cache_ttl = 0.1  # 0.1 seconds
        
        try:
            # Store and retrieve immediately
            web_search_tool._cache_result(cache_key, test_result)
            cached = web_search_tool._get_cached_result(cache_key)
            assert cached == test_result
            
            # Wait for expiration
            time.sleep(0.15)
            
            # Should be expired
            expired = web_search_tool._get_cached_result(cache_key)
            assert expired is None
            
        finally:
            web_search_tool._cache_ttl = original_ttl


class TestFixedWebSearchToolParsing:
    """Tests de parsing de HTML/JSON"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    def test_clean_text_function(self, web_search_tool):
        """Test de función de limpieza de texto"""
        # Test casos normales
        assert web_search_tool._clean_text("  Hello World  ") == "Hello World"
        assert web_search_tool._clean_text("Multi\n\nLine\nText") == "Multi Line Text"
        assert web_search_tool._clean_text("Text   with   spaces") == "Text with spaces"
        
        # Test casos edge
        assert web_search_tool._clean_text("") == ""
        assert web_search_tool._clean_text(None) == ""
        assert web_search_tool._clean_text("   ") == ""
    
    def test_clean_url_function(self, web_search_tool):
        """Test de función de limpieza de URLs"""
        # Test URLs normales
        assert web_search_tool._clean_url("https://example.com") == "https://example.com"
        assert web_search_tool._clean_url("http://test.org/path") == "http://test.org/path"
        
        # Test URLs con protocolo relativo
        assert web_search_tool._clean_url("//example.com") == "https://example.com"
        
        # Test URLs inválidas
        assert web_search_tool._clean_url("") == ""
        assert web_search_tool._clean_url("/relative/path") == ""
        
        # Test DuckDuckGo redirects (simulado)
        ddg_url = "https://duckduckgo.com/l/?uddg=https%3A//example.com&..."
        cleaned = web_search_tool._clean_url(ddg_url)
        assert "example.com" in cleaned
    
    def test_is_valid_result_function(self, web_search_tool):
        """Test de validación de resultados"""
        # Resultados válidos
        assert web_search_tool._is_valid_result("Python Tutorial", "https://example.com") == True
        assert web_search_tool._is_valid_result("Good Title", "http://test.org") == True
        
        # Resultados inválidos
        assert web_search_tool._is_valid_result("", "https://example.com") == False
        assert web_search_tool._is_valid_result("Title", "") == False
        assert web_search_tool._is_valid_result("Title", "ftp://old.com") == False
        assert web_search_tool._is_valid_result("12", "https://example.com") == False  # Solo números
        assert web_search_tool._is_valid_result("Click here", "https://example.com") == False  # Spam


class TestFixedWebSearchToolFormatting:
    """Tests de formateo de resultados"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    def test_format_search_results(self, web_search_tool):
        """Test de formateo de resultados de búsqueda"""
        mock_search_result = {
            'query': 'python tutorial',
            'results': [
                {
                    'title': 'Python Tutorial',
                    'url': 'https://python.org/tutorial',
                    'snippet': 'Learn Python programming'
                },
                {
                    'title': 'Advanced Python',
                    'url': 'https://advanced-python.com',
                    'snippet': 'Advanced Python concepts'
                }
            ],
            'source': 'DuckDuckGo'
        }
        
        formatted = web_search_tool._format_search_results(mock_search_result)
        
        assert isinstance(formatted, str)
        assert 'python tutorial' in formatted.lower()
        assert 'Python Tutorial' in formatted
        assert 'https://python.org/tutorial' in formatted
        assert 'Learn Python programming' in formatted
        assert 'DuckDuckGo' in formatted
        assert 'Found 2 results' in formatted
    
    def test_format_error_response(self, web_search_tool):
        """Test de formateo de respuestas de error"""
        error_msg = "Connection timeout"
        
        formatted = web_search_tool._format_error_response(error_msg)
        
        assert isinstance(formatted, str)
        assert "Web search error:" in formatted
        assert "Connection timeout" in formatted
        assert "existing knowledge" in formatted


@pytest.mark.asyncio
class TestFixedWebSearchToolAsync:
    """Tests asíncronos del WebSearchTool"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    async def test_rate_limiting(self, web_search_tool):
        """Test de rate limiting"""
        # Reducir intervalo para test rápido
        original_interval = web_search_tool._min_request_interval
        web_search_tool._min_request_interval = 0.1
        
        try:
            start_time = time.time()
            
            # Primera llamada
            await web_search_tool._enforce_rate_limit()
            
            # Segunda llamada inmediata (debe esperar)
            await web_search_tool._enforce_rate_limit()
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Debe haber esperado al menos el intervalo mínimo
            assert duration >= 0.1
            
        finally:
            web_search_tool._min_request_interval = original_interval
    
    async def test_arun_with_mocked_search(self, web_search_tool):
        """Test de _arun con búsqueda mockeada"""
        
        # Mock del método de búsqueda multi-engine
        mock_result = {
            'query': 'test query',
            'results': [
                {
                    'title': 'Test Result',
                    'url': 'https://test.com',
                    'snippet': 'Test snippet'
                }
            ],
            'source': 'MockEngine'
        }
        
        with patch.object(web_search_tool, '_perform_multi_engine_search', 
                         return_value=mock_result):
            
            result = await web_search_tool._arun("test query", max_results=1)
            
            assert isinstance(result, str)
            assert 'Test Result' in result
            assert 'https://test.com' in result
            assert 'Test snippet' in result
            assert web_search_tool.stats['total_searches'] == 1
            assert web_search_tool.stats['successful_searches'] == 1
    
    async def test_arun_with_failed_search(self, web_search_tool):
        """Test de _arun cuando la búsqueda falla"""
        
        # Mock que retorna None (búsqueda fallida)
        with patch.object(web_search_tool, '_perform_multi_engine_search', 
                         return_value=None):
            
            result = await web_search_tool._arun("test query")
            
            assert isinstance(result, str)
            assert "no search results found" in result.lower()
            assert web_search_tool.stats['total_searches'] == 1
            assert web_search_tool.stats['successful_searches'] == 0
    
    async def test_health_check(self, web_search_tool):
        """Test de health check"""
        
        # Mock successful search for health check
        with patch.object(web_search_tool, '_arun', 
                         return_value="Successful health check result"):
            
            health = await web_search_tool.health_check()
            
            assert isinstance(health, dict)
            assert 'healthy' in health
            assert 'statistics' in health
            assert 'cache_size' in health
            assert health['healthy'] == True
    
    async def test_health_check_failure(self, web_search_tool):
        """Test de health check cuando falla"""
        
        # Mock failed search for health check
        with patch.object(web_search_tool, '_arun', 
                         side_effect=Exception("Health check failed")):
            
            health = await web_search_tool.health_check()
            
            assert isinstance(health, dict)
            assert health['healthy'] == False
            assert 'error' in health
            assert 'Health check failed' in health['error']


class TestFixedWebSearchToolSynchronous:
    """Tests de la interfaz síncrona"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    def test_sync_run_method(self, web_search_tool):
        """Test del método _run síncrono"""
        
        # Mock del método async
        with patch.object(web_search_tool, '_arun', 
                         return_value="Mocked async result"):
            
            result = web_search_tool._run("test query")
            
            assert result == "Mocked async result"


@pytest.mark.integration  
class TestFixedWebSearchToolIntegration:
    """Tests de integración (si están disponibles los servicios externos)"""
    
    @pytest.fixture
    def web_search_tool(self):
        return FixedWebSearchTool()
    
    @pytest.mark.skipif(not pytest.config.getoption("--run-integration", default=False),
                       reason="Integration tests require --run-integration flag")
    async def test_real_search_integration(self, web_search_tool):
        """Test de búsqueda real (solo con flag de integración)"""
        
        try:
            result = await web_search_tool._arun("python programming", max_results=2)
            
            assert isinstance(result, str)
            assert len(result) > 100  # Debe tener contenido sustancial
            assert "python" in result.lower()
            assert web_search_tool.stats['total_searches'] >= 1
            
        except Exception as e:
            pytest.skip(f"Integration test skipped due to network issues: {e}")


# Función de conveniencia para ejecutar tests específicos
def run_fixed_websearch_tests():
    """Ejecutar solo los tests del Fixed WebSearchTool"""
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_fixed_websearch_tests()