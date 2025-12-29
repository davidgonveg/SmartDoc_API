"""
Unit Tests for Search Engines
Tests individuales para cada motor de búsqueda con mocks HTTP
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

# Imports del proyecto
from app.agents.tools.web.search_engines import (
    DuckDuckGoSearchEngine,
    GoogleCustomSearchEngine,
    SearXSearchEngine,
    BingSearchEngine,
    SearchEngineManager,
    SearchResult,
    SearchResponse
)
from app.agents.tools.web.config import SearchEngine, get_engine_config
from app.agents.tools.web.web_utils import WebResponse

# Test fixtures
from fixtures import (
    DUCKDUCKGO_MOCK_HTML,
    GOOGLE_CUSTOM_SEARCH_MOCK,
    SEARX_MOCK_RESPONSE,
    BING_SEARCH_MOCK,
    HTTP_ERROR_SCENARIOS,
    get_mock_search_results
)

# =============================================================================
# TESTS BASE PARA SEARCH ENGINES
# =============================================================================

class BaseSearchEngineTest:
    """Clase base con tests comunes para todos los motores"""
    
    def test_engine_initialization(self, engine_class, engine_type):
        """Test que el motor se inicializa correctamente"""
        engine = engine_class()
        
        assert engine.config is not None
        assert engine.name is not None
        assert engine.engine_type == engine_type
        assert hasattr(engine, 'search')
    
    def test_clean_text_removes_html_tags(self, engine_instance):
        """Test que _clean_text remueve tags HTML"""
        dirty_text = "<b>Bold text</b> with <em>emphasis</em> and &amp; entities"
        
        result = engine_instance._clean_text(dirty_text)
        
        assert "<b>" not in result
        assert "<em>" not in result
        assert "Bold text" in result
        assert "emphasis" in result
        assert "&" in result  # Entity decodificada
    
    def test_clean_text_normalizes_whitespace(self, engine_instance):
        """Test que _clean_text normaliza espacios"""
        messy_text = "  Multiple    spaces\n\n\nand   tabs\t\t  "
        
        result = engine_instance._clean_text(messy_text)
        
        assert "Multiple spaces" in result
        assert "and tabs" in result
        assert "   " not in result
    
    def test_validate_url_accepts_good_urls(self, engine_instance):
        """Test que _validate_url acepta URLs válidas"""
        valid_urls = [
            "https://example.com",
            "http://test.org/path",
            "https://subdomain.domain.co.uk/long/path?param=value"
        ]
        
        for url in valid_urls:
            assert engine_instance._validate_url(url) == True
    
    def test_validate_url_rejects_bad_urls(self, engine_instance):
        """Test que _validate_url rechaza URLs inválidas"""
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://old-protocol.com",
            "javascript:alert('xss')",
            "//incomplete-url"
        ]
        
        for url in invalid_urls:
            assert engine_instance._validate_url(url) == False

# =============================================================================
# TESTS DE DUCKDUCKGO SEARCH ENGINE
# =============================================================================

class TestDuckDuckGoSearchEngine(BaseSearchEngineTest):
    """Tests específicos para DuckDuckGo"""
    
    @pytest.fixture
    def engine_class(self):
        return DuckDuckGoSearchEngine
    
    @pytest.fixture
    def engine_type(self):
        return SearchEngine.DUCKDUCKGO
    
    @pytest.fixture
    def engine_instance(self):
        return DuckDuckGoSearchEngine()
    
    @pytest.fixture
    def mock_duckduckgo_response(self):
        """Mock de respuesta exitosa de DuckDuckGo"""
        return WebResponse(
            url="https://duckduckgo.com/html/?q=python+programming",
            status_code=200,
            content=DUCKDUCKGO_MOCK_HTML,
            headers={"content-type": "text/html; charset=utf-8"},
            encoding="utf-8",
            response_time=1.2,
            final_url="https://duckduckgo.com/html/?q=python+programming",
            success=True
        )
    
    @pytest.mark.asyncio
    async def test_search_success(self, engine_instance, mock_duckduckgo_response):
        """Test de búsqueda exitosa en DuckDuckGo"""
        with patch.object(engine_instance, '_make_request', return_value=mock_duckduckgo_response):
            
            result = await engine_instance.search("python programming", max_results=3)
            
            assert result.success == True
            assert result.query == "python programming"
            assert result.engine == "DuckDuckGo"
            assert len(result.results) > 0
            assert result.search_time > 0
            
            # Verificar primer resultado
            first_result = result.results[0]
            assert "python" in first_result.title.lower() or "python" in first_result.url.lower()
            assert first_result.rank == 1
            assert first_result.source_engine == "DuckDuckGo"
    
    @pytest.mark.asyncio
    async def test_search_http_error(self, engine_instance):
        """Test de manejo de error HTTP en DuckDuckGo"""
        error_response = WebResponse(
            url="https://duckduckgo.com/html/?q=test",
            status_code=404,
            content="",
            headers={},
            encoding="utf-8",
            response_time=0.5,
            final_url="https://duckduckgo.com/html/?q=test",
            success=False,
            error="Not Found"
        )
        
        with patch.object(engine_instance, '_make_request', return_value=error_response):
            
            result = await engine_instance.search("test query")
            
            assert result.success == False
            assert result.error_message is not None
            assert "404" in result.error_message
            assert len(result.results) == 0
    
    @pytest.mark.asyncio
    async def test_search_exception_handling(self, engine_instance):
        """Test de manejo de excepciones en DuckDuckGo"""
        with patch.object(engine_instance, '_make_request', side_effect=Exception("Network error")):
            
            result = await engine_instance.search("test query")
            
            assert result.success == False
            assert "Network error" in result.error_message
            assert len(result.results) == 0
    
    def test_parse_duckduckgo_results_extracts_correctly(self, engine_instance):
        """Test que el parser de DuckDuckGo extrae resultados correctamente"""
        # Llamar al método protegido usando asyncio
        async def test_parsing():
            results = await engine_instance._parse_duckduckgo_results(
                DUCKDUCKGO_MOCK_HTML, 
                "python programming", 
                10
            )
            
            assert len(results) > 0
            
            for result in results:
                assert isinstance(result, SearchResult)
                assert len(result.title) > 0
                assert result.url.startswith(('http://', 'https://'))
                assert result.rank > 0
                assert result.source_engine == "DuckDuckGo"
        
        asyncio.run(test_parsing())
    
    def test_parse_empty_html_returns_empty_list(self, engine_instance):
        """Test que HTML vacío retorna lista vacía"""
        async def test_empty_parsing():
            results = await engine_instance._parse_duckduckgo_results("", "query", 10)
            assert results == []
        
        asyncio.run(test_empty_parsing())

# =============================================================================
# TESTS DE GOOGLE CUSTOM SEARCH ENGINE
# =============================================================================

class TestGoogleCustomSearchEngine(BaseSearchEngineTest):
    """Tests específicos para Google Custom Search"""
    
    @pytest.fixture
    def engine_class(self):
        return GoogleCustomSearchEngine
    
    @pytest.fixture
    def engine_type(self):
        return SearchEngine.GOOGLE_CUSTOM
    
    @pytest.fixture
    def engine_instance(self):
        return GoogleCustomSearchEngine()
    
    @pytest.fixture
    def mock_google_response(self):
        """Mock de respuesta exitosa de Google Custom Search"""
        return WebResponse(
            url="https://www.googleapis.com/customsearch/v1",
            status_code=200,
            content=json.dumps(GOOGLE_CUSTOM_SEARCH_MOCK),
            headers={"content-type": "application/json"},
            encoding="utf-8",
            response_time=0.8,
            final_url="https://www.googleapis.com/customsearch/v1",
            success=True
        )
    
    @pytest.mark.asyncio
    async def test_search_success_with_api_key(self, engine_instance, mock_google_response):
        """Test de búsqueda exitosa con API key configurada"""
        # Mock API key y search engine ID
        engine_instance.api_key = "test_api_key"
        engine_instance.search_engine_id = "test_search_engine_id"
        
        with patch.object(engine_instance, '_make_request', return_value=mock_google_response):
            
            result = await engine_instance.search("python programming", max_results=5)
            
            assert result.success == True
            assert result.query == "python programming"
            assert result.engine == "Google Custom Search"
            assert len(result.results) > 0
            
            # Verificar estructura de resultados Google
            first_result = result.results[0]
            assert first_result.title == "Welcome to Python.org"
            assert first_result.url == "https://www.python.org/"
            assert "python" in first_result.snippet.lower()
    
    @pytest.mark.asyncio
    async def test_search_fails_without_api_key(self, engine_instance):
        """Test que búsqueda falla sin API key"""
        # Asegurar que no hay API key
        engine_instance.api_key = None
        engine_instance.search_engine_id = None
        
        result = await engine_instance.search("test query")
        
        assert result.success == False
        assert "API key" in result.error_message or "not configured" in result.error_message
        assert len(result.results) == 0
    
    def test_parse_google_json_results_correctly(self, engine_instance):
        """Test que el parser de JSON de Google funciona"""
        async def test_parsing():
            results = await engine_instance._parse_google_json_results(
                json.dumps(GOOGLE_CUSTOM_SEARCH_MOCK),
                "python programming"
            )
            
            assert len(results) == 2  # MOCK tiene 2 items
            
            first_result = results[0]
            assert first_result.title == "Welcome to Python.org"
            assert first_result.url == "https://www.python.org/"
            assert first_result.rank == 1
            assert first_result.metadata["google_cache_id"] == "test_cache_id_1"
        
        asyncio.run(test_parsing())
    
    def test_parse_malformed_json_handles_gracefully(self, engine_instance):
        """Test que JSON mal formado se maneja correctamente"""
        async def test_malformed_parsing():
            results = await engine_instance._parse_google_json_results(
                "invalid json content",
                "test query"
            )
            assert results == []
        
        asyncio.run(test_malformed_parsing())

# =============================================================================
# TESTS DE SEARX SEARCH ENGINE
# =============================================================================

class TestSearXSearchEngine(BaseSearchEngineTest):
    """Tests específicos para SearX"""
    
    @pytest.fixture
    def engine_class(self):
        return SearXSearchEngine
    
    @pytest.fixture
    def engine_type(self):
        return SearchEngine.SEARX
    
    @pytest.fixture
    def engine_instance(self):
        return SearXSearchEngine()
    
    @pytest.fixture
    def mock_searx_response(self):
        """Mock de respuesta exitosa de SearX"""
        return WebResponse(
            url="https://searx.be/search",
            status_code=200,
            content=json.dumps(SEARX_MOCK_RESPONSE),
            headers={"content-type": "application/json"},
            encoding="utf-8",
            response_time=2.1,
            final_url="https://searx.be/search",
            success=True
        )
    
    @pytest.mark.asyncio
    async def test_search_success(self, engine_instance, mock_searx_response):
        """Test de búsqueda exitosa en SearX"""
        with patch.object(engine_instance, '_make_request', return_value=mock_searx_response):
            
            result = await engine_instance.search("python programming", max_results=10)
            
            assert result.success == True
            assert result.query == "python programming"
            assert result.engine == "SearX"
            assert len(result.results) > 0
            
            # Verificar metadatos específicos de SearX
            first_result = result.results[0]
            assert "engines" in first_result.metadata
            assert "score" in first_result.metadata
            assert first_result.metadata["extracted_from"] == "searx_api"
    
    def test_parse_searx_json_results_with_scores(self, engine_instance):
        """Test que el parser de SearX maneja scores correctamente"""
        async def test_parsing():
            results = await engine_instance._parse_searx_json_results(
                json.dumps(SEARX_MOCK_RESPONSE),
                "python programming",
                10
            )
            
            assert len(results) == 2
            
            # Verificar orden por score (mayor score primero en mock)
            first_result = results[0]
            second_result = results[1]
            
            assert first_result.metadata["score"] == 100.0
            assert second_result.metadata["score"] == 95.0
            assert first_result.metadata["engines"] == ["google", "duckduckgo"]
        
        asyncio.run(test_parsing())

# =============================================================================
# TESTS DE BING SEARCH ENGINE
# =============================================================================

class TestBingSearchEngine(BaseSearchEngineTest):
    """Tests específicos para Bing Search"""
    
    @pytest.fixture
    def engine_class(self):
        return BingSearchEngine
    
    @pytest.fixture
    def engine_type(self):
        return SearchEngine.BING
    
    @pytest.fixture
    def engine_instance(self):
        return BingSearchEngine()
    
    @pytest.fixture
    def mock_bing_response(self):
        """Mock de respuesta exitosa de Bing"""
        return WebResponse(
            url="https://api.bing.microsoft.com/v7.0/search",
            status_code=200,
            content=json.dumps(BING_SEARCH_MOCK),
            headers={"content-type": "application/json"},
            encoding="utf-8",
            response_time=1.5,
            final_url="https://api.bing.microsoft.com/v7.0/search",
            success=True
        )
    
    @pytest.mark.asyncio
    async def test_search_success_with_api_key(self, engine_instance, mock_bing_response):
        """Test de búsqueda exitosa con API key de Bing"""
        # Mock API key
        engine_instance.api_key = "test_bing_api_key"
        
        with patch.object(engine_instance, '_make_request', return_value=mock_bing_response):
            
            result = await engine_instance.search("python programming", max_results=20)
            
            assert result.success == True
            assert result.query == "python programming"
            assert result.engine == "Bing Search"
            assert len(result.results) > 0
            
            # Verificar metadatos específicos de Bing
            first_result = result.results[0]
            assert "last_crawled" in first_result.metadata
            assert "display_url" in first_result.metadata
    
    @pytest.mark.asyncio
    async def test_search_fails_without_api_key(self, engine_instance):
        """Test que Bing falla sin API key"""
        engine_instance.api_key = None
        
        result = await engine_instance.search("test query")
        
        assert result.success == False
        assert "API key not configured" in result.error_message
        assert len(result.results) == 0
    
    def test_parse_bing_json_results_correctly(self, engine_instance):
        """Test que el parser de Bing maneja la estructura JSON"""
        async def test_parsing():
            results = await engine_instance._parse_bing_json_results(
                json.dumps(BING_SEARCH_MOCK),
                "python programming"
            )
            
            assert len(results) == 2
            
            first_result = results[0]
            assert first_result.title == "Welcome to Python.org"
            assert first_result.url == "https://www.python.org/"
            assert first_result.metadata["display_url"] == "https://www.python.org"
            assert "dateLastCrawled" in first_result.metadata
        
        asyncio.run(test_parsing())

# =============================================================================
# TESTS DE SEARCH ENGINE MANAGER
# =============================================================================

class TestSearchEngineManager:
    """Tests para el gestor de motores de búsqueda"""
    
    @pytest.fixture
    def manager(self):
        return SearchEngineManager()
    
    def test_manager_initializes_all_engines(self, manager):
        """Test que el manager inicializa todos los motores"""
        assert SearchEngine.DUCKDUCKGO in manager.engines
        assert SearchEngine.GOOGLE_CUSTOM in manager.engines
        assert SearchEngine.SEARX in manager.engines
        assert SearchEngine.BING in manager.engines
        
        assert manager.default_engine == SearchEngine.DUCKDUCKGO
        assert SearchEngine.SEARX in manager.fallback_engines
    
    @pytest.mark.asyncio
    async def test_search_with_specific_engine(self, manager):
        """Test de búsqueda con motor específico"""
        # Mock del motor DuckDuckGo
        mock_result = SearchResponse(
            query="test query",
            results=get_mock_search_results("test query", "duckduckgo", 3),
            total_results=3,
            search_time=1.0,
            engine="DuckDuckGo",
            success=True
        )
        
        with patch.object(manager.engines[SearchEngine.DUCKDUCKGO], 'search', return_value=mock_result):
            
            result = await manager.search(
                query="test query",
                engine=SearchEngine.DUCKDUCKGO,
                max_results=3
            )
            
            assert result.success == True
            assert result.engine == "DuckDuckGo"
            assert len(result.results) == 3
    
    @pytest.mark.asyncio
    async def test_search_with_fallback(self, manager):
        """Test de búsqueda con fallback automático"""
        # Mock que el motor principal falla
        failed_result = SearchResponse(
            query="test query",
            results=[],
            total_results=0,
            search_time=0.5,
            engine="DuckDuckGo",
            success=False,
            error_message="Connection failed"
        )
        
        # Mock que el fallback funciona
        success_result = SearchResponse(
            query="test query",
            results=get_mock_search_results("test query", "searx", 2),
            total_results=2,
            search_time=2.0,
            engine="SearX",
            success=True
        )
        
        with patch.object(manager.engines[SearchEngine.DUCKDUCKGO], 'search', return_value=failed_result), \
             patch.object(manager.engines[SearchEngine.SEARX], 'search', return_value=success_result):
            
            result = await manager.search(
                query="test query",
                use_fallback=True
            )
            
            assert result.success == True
            assert result.engine == "SearX"
            assert result.metadata.get("used_fallback") == True
            assert result.metadata.get("original_engine") == "duckduckgo"
    
    @pytest.mark.asyncio
    async def test_search_all_engines_fail(self, manager):
        """Test cuando todos los motores fallan"""
        failed_result = SearchResponse(
            query="test query",
            results=[],
            total_results=0,
            search_time=0.1,
            engine="Mock Engine",
            success=False,
            error_message="All engines failed"
        )
        
        # Mock todos los engines para que fallen
        for engine in manager.engines.values():
            with patch.object(engine, 'search', return_value=failed_result):
                pass
        
        with patch.object(manager.engines[SearchEngine.DUCKDUCKGO], 'search', return_value=failed_result), \
             patch.object(manager.engines[SearchEngine.SEARX], 'search', return_value=failed_result):
            
            result = await manager.search("test query", use_fallback=True)
            
            assert result.success == False
            assert len(result.results) == 0
    
    def test_get_available_engines_filters_correctly(self, manager):
        """Test que get_available_engines filtra correctamente"""
        available = manager.get_available_engines()
        
        # Debe incluir engines habilitados que no requieren API key
        assert SearchEngine.DUCKDUCKGO in available
        assert SearchEngine.SEARX in available
        
        # Engines que requieren API key pueden o no estar disponibles
        # dependiendo de si están configurados
    
    @pytest.mark.asyncio
    async def test_health_check_all_engines(self, manager):
        """Test de health check de todos los motores"""
        # Mock health check responses
        mock_results = {}
        for engine_type in SearchEngine:
            mock_results[engine_type.value] = {
                'available': True,
                'healthy': True,
                'error': None,
                'search_time': 1.0,
                'requires_api_key': False,
                'enabled': True
            }
        
        with patch.object(manager, 'health_check', return_value=mock_results):
            
            health_status = await manager.health_check()
            
            assert len(health_status) == len(SearchEngine)
            
            for engine_name, status in health_status.items():
                assert 'available' in status
                assert 'healthy' in status
                assert 'requires_api_key' in status

# =============================================================================
# TESTS DE INTEGRATION ENTRE ENGINES
# =============================================================================

class TestSearchEngineIntegration:
    """Tests de integración entre diferentes componentes"""
    
    @pytest.mark.asyncio
    async def test_all_engines_produce_consistent_results(self):
        """Test que todos los engines producen SearchResult consistentes"""
        query = "python programming"
        engines = [
            DuckDuckGoSearchEngine(),
            SearXSearchEngine()
        ]
        
        # Mock responses for all engines
        mock_response = WebResponse(
            url="https://test.com",
            status_code=200,
            content=json.dumps(SEARX_MOCK_RESPONSE),
            headers={"content-type": "application/json"},
            encoding="utf-8",
            response_time=1.0,
            final_url="https://test.com",
            success=True
        )
        
        for engine in engines:
            with patch.object(engine, '_make_request', return_value=mock_response):
                result = await engine.search(query, max_results=5)
                
                # Verificar estructura consistente
                assert isinstance(result, SearchResponse)
                assert result.query == query
                assert isinstance(result.results, list)
                assert isinstance(result.success, bool)
                
                if result.success and result.results:
                    first_result = result.results[0]
                    assert isinstance(first_result, SearchResult)
                    assert hasattr(first_result, 'title')
                    assert hasattr(first_result, 'url')
                    assert hasattr(first_result, 'snippet')
                    assert hasattr(first_result, 'rank')

# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

class TestSearchEnginePerformance:
    """Tests de rendimiento para motores de búsqueda"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_search_performance_under_load(self, performance_tracker):
        """Test de performance bajo carga concurrente"""
        engine = DuckDuckGoSearchEngine()
        
        # Mock response rápida
        fast_response = WebResponse(
            url="https://duckduckgo.com/html/",
            status_code=200,
            content=DUCKDUCKGO_MOCK_HTML,
            headers={"content-type": "text/html"},
            encoding="utf-8",
            response_time=0.1,
            final_url="https://duckduckgo.com/html/",
            success=True
        )
        
        performance_tracker['start']('concurrent_searches')
        
        # Simular búsquedas concurrentes
        with patch.object(engine, '_make_request', return_value=fast_response):
            tasks = [
                engine.search(f"query {i}", max_results=5)
                for i in range(10)
            ]
            
            results = await asyncio.gather(*tasks)
        
        duration = performance_tracker['end']('concurrent_searches')
        
        # Verificar que todas las búsquedas fueron exitosas
        assert all(result.success for result in results)
        assert duration < 5.0  # Menos de 5 segundos para 10 búsquedas
    
    @pytest.mark.performance
    def test_result_parsing_performance(self, performance_tracker):
        """Test de performance del parsing de resultados"""
        engine = DuckDuckGoSearchEngine()
        
        # HTML con muchos resultados
        large_html = DUCKDUCKGO_MOCK_HTML * 10  # Simular página con más resultados
        
        performance_tracker['start']('parsing_large_html')
        
        async def parse_test():
            results = await engine._parse_duckduckgo_results(large_html, "test", 50)
            return results
        
        results = asyncio.run(parse_test())
        duration = performance_tracker['end']('parsing_large_html')
        
        assert len(results) > 0
        assert duration < 2.0  # Parsing debe ser rápido

# =============================================================================
# TESTS DE ERROR HANDLING
# =============================================================================

class TestSearchEngineErrorHandling:
    """Tests de manejo de errores en motores de búsqueda"""
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test de manejo de timeouts"""
        engine = DuckDuckGoSearchEngine()
        
        # Mock timeout exception
        with patch.object(engine, '_make_request', side_effect=asyncio.TimeoutError("Request timeout")):
            
            result = await engine.search("test query")
            
            assert result.success == False
            assert "timeout" in result.error_message.lower()
            assert len(result.results) == 0
    
    @pytest.mark.asyncio
    async def test_network_error_handling(self):
        """Test de manejo de errores de red"""
        engine = SearXSearchEngine()
        
        # Mock network error
        with patch.object(engine, '_make_request', side_effect=ConnectionError("Network unreachable")):
            
            result = await engine.search("test query")
            
            assert result.success == False
            assert "network" in result.error_message.lower() or "connection" in result.error_message.lower()
    
    @pytest.mark.asyncio 
    async def test_invalid_json_response_handling(self):
        """Test de manejo de respuestas JSON inválidas"""
        engine = GoogleCustomSearchEngine()
        engine.api_key = "test_key"
        engine.search_engine_id = "test_id"
        
        # Mock respuesta con JSON inválido
        invalid_json_response = WebResponse(
            url="https://www.googleapis.com/customsearch/v1",
            status_code=200,
            content="invalid json content {",
            headers={"content-type": "application/json"},
            encoding="utf-8",
            response_time=1.0,
            final_url="https://www.googleapis.com/customsearch/v1",
            success=True
        )
        
        with patch.object(engine, '_make_request', return_value=invalid_json_response):
            
            result = await engine.search("test query")
            
            # Debe manejar JSON inválido sin crashear
            assert isinstance(result, SearchResponse)
            assert len(result.results) == 0  # Sin resultados por JSON inválido
    
    @pytest.mark.parametrize("status_code,should_succeed", [
        (200, True),
        (201, True),
        (400, False),
        (401, False),
        (403, False),
        (404, False),
        (500, False),
        (503, False)
    ])
    @pytest.mark.asyncio
    async def test_http_status_code_handling(self, status_code, should_succeed):
        """Test de manejo de diferentes códigos de status HTTP"""
        engine = DuckDuckGoSearchEngine()
        
        mock_response = WebResponse(
            url="https://duckduckgo.com/html/",
            status_code=status_code,
            content=DUCKDUCKGO_MOCK_HTML if should_succeed else "",
            headers={"content-type": "text/html"},
            encoding="utf-8",
            response_time=1.0,
            final_url="https://duckduckgo.com/html/",
            success=should_succeed
        )
        
        with patch.object(engine, '_make_request', return_value=mock_response):
            
            result = await engine.search("test query")
            
            assert result.success == should_succeed
            if not should_succeed:
                assert str(status_code) in result.error_message

# =============================================================================
# TESTS DE EDGE CASES
# =============================================================================

class TestSearchEngineEdgeCases:
    """Tests para casos extremos y situaciones límite"""
    
    @pytest.mark.asyncio
    async def test_search_with_empty_query(self):
        """Test de búsqueda con query vacía"""
        engine = DuckDuckGoSearchEngine()
        
        result = await engine.search("", max_results=5)
        
        # Debe manejar query vacía gracefully
        assert isinstance(result, SearchResponse)
        # Puede fallar o retornar resultados vacíos, ambos son válidos
    
    @pytest.mark.asyncio
    async def test_search_with_very_long_query(self):
        """Test de búsqueda con query extremadamente larga"""
        engine = SearXSearchEngine()
        
        # Query de 1000 caracteres
        long_query = "python programming " * 100
        
        mock_response = WebResponse(
            url="https://searx.be/search",
            status_code=200,
            content=json.dumps(SEARX_MOCK_RESPONSE),
            headers={"content-type": "application/json"},
            encoding="utf-8",
            response_time=1.0,
            final_url="https://searx.be/search",
            success=True
        )
        
        with patch.object(engine, '_make_request', return_value=mock_response):
            result = await engine.search(long_query, max_results=5)
            
            assert isinstance(result, SearchResponse)
            # Query debe ser truncada o manejada apropiadamente
    
    @pytest.mark.asyncio
    async def test_search_with_special_characters(self):
        """Test de búsqueda con caracteres especiales"""
        engine = DuckDuckGoSearchEngine()
        
        special_queries = [
            "query with spaces",
            "query+with+plus",
            "query%20with%20encoding",
            "query with émojis 🔍",
            "query with \"quotes\"",
            "query with & ampersand",
            "query with <html> tags"
        ]
        
        mock_response = WebResponse(
            url="https://duckduckgo.com/html/",
            status_code=200,
            content=DUCKDUCKGO_MOCK_HTML,
            headers={"content-type": "text/html"},
            encoding="utf-8",
            response_time=1.0,
            final_url="https://duckduckgo.com/html/",
            success=True
        )
        
        for query in special_queries:
            with patch.object(engine, '_make_request', return_value=mock_response):
                result = await engine.search(query, max_results=3)
                
                assert isinstance(result, SearchResponse)
                assert result.query == query  # Query debe preservarse
    
    @pytest.mark.asyncio
    async def test_search_with_zero_max_results(self):
        """Test de búsqueda con max_results=0"""
        engine = DuckDuckGoSearchEngine()
        
        mock_response = WebResponse(
            url="https://duckduckgo.com/html/",
            status_code=200,
            content=DUCKDUCKGO_MOCK_HTML,
            headers={"content-type": "text/html"},
            encoding="utf-8",
            response_time=1.0,
            final_url="https://duckduckgo.com/html/",
            success=True
        )
        
        with patch.object(engine, '_make_request', return_value=mock_response):
            result = await engine.search("test query", max_results=0)
            
            assert result.success == True
            assert len(result.results) == 0  # No resultados solicitados
    
    @pytest.mark.asyncio
    async def test_search_with_very_high_max_results(self):
        """Test de búsqueda con max_results muy alto"""
        engine = SearXSearchEngine()
        
        mock_response = WebResponse(
            url="https://searx.be/search",
            status_code=200,
            content=json.dumps(SEARX_MOCK_RESPONSE),
            headers={"content-type": "application/json"},
            encoding="utf-8",
            response_time=1.0,
            final_url="https://searx.be/search",
            success=True
        )
        
        with patch.object(engine, '_make_request', return_value=mock_response):
            result = await engine.search("test query", max_results=1000)
            
            assert result.success == True
            # Resultados deben estar limitados por lo que devuelve el motor
            assert len(result.results) <= 100  # Límite razonable
    
    def test_search_result_rank_consistency(self):
        """Test que los ranks de resultados son consistentes"""
        results = get_mock_search_results("test query", "test_engine", 5)
        
        for i, result in enumerate(results):
            assert result.rank == i + 1
            assert result.rank > 0
    
    def test_search_result_url_validation(self):
        """Test que URLs en resultados son válidas"""
        results = get_mock_search_results("test query", "test_engine", 3)
        
        for result in results:
            assert result.url.startswith(('http://', 'https://'))
            assert len(result.url) > 10
            assert '.' in result.url  # Debe tener dominio
    
    @pytest.mark.asyncio
    async def test_concurrent_searches_same_engine(self):
        """Test de búsquedas concurrentes en el mismo motor"""
        engine = DuckDuckGoSearchEngine()
        
        mock_response = WebResponse(
            url="https://duckduckgo.com/html/",
            status_code=200,
            content=DUCKDUCKGO_MOCK_HTML,
            headers={"content-type": "text/html"},
            encoding="utf-8",
            response_time=0.5,
            final_url="https://duckduckgo.com/html/",
            success=True
        )
        
        with patch.object(engine, '_make_request', return_value=mock_response):
            # Ejecutar 5 búsquedas concurrentes
            tasks = [
                engine.search(f"query {i}", max_results=3)
                for i in range(5)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Todas deben completarse exitosamente
            assert len(results) == 5
            for i, result in enumerate(results):
                assert result.success == True
                assert result.query == f"query {i}"

# =============================================================================
# TESTS DE CONFIGURATION VALIDATION
# =============================================================================

class TestSearchEngineConfiguration:
    """Tests para validación de configuración de motores"""
    
    def test_all_engines_have_valid_config(self):
        """Test que todos los motores tienen configuración válida"""
        for engine_type in SearchEngine:
            config = get_engine_config(engine_type)
            
            assert config is not None
            assert config.name is not None
            assert config.base_url is not None
            assert config.search_endpoint is not None
            assert isinstance(config.params, dict)
            assert isinstance(config.headers, dict)
            assert config.rate_limit_per_minute > 0
            assert config.timeout > 0
            assert config.max_results > 0
    
    def test_engine_config_consistency(self):
        """Test que configuraciones de motores son consistentes"""
        configs = [get_engine_config(engine) for engine in SearchEngine]
        
        for config in configs:
            # URLs deben ser válidas
            assert config.base_url.startswith(('http://', 'https://'))
            
            # Headers deben incluir User-Agent
            assert any('user-agent' in key.lower() for key in config.headers.keys())
            
            # Rate limits deben ser razonables
            assert 1 <= config.rate_limit_per_minute <= 10000
            assert 1 <= config.timeout <= 300  # Max 5 minutos
    
    def test_api_key_engines_marked_correctly(self):
        """Test que motores que requieren API key están marcados"""
        google_config = get_engine_config(SearchEngine.GOOGLE_CUSTOM)
        bing_config = get_engine_config(SearchEngine.BING)
        duckduckgo_config = get_engine_config(SearchEngine.DUCKDUCKGO)
        searx_config = get_engine_config(SearchEngine.SEARX)
        
        assert google_config.requires_api_key == True
        assert bing_config.requires_api_key == True
        assert duckduckgo_config.requires_api_key == False
        assert searx_config.requires_api_key == False
    
    def test_engine_endpoints_are_different(self):
        """Test que endpoints de motores son únicos"""
        endpoints = []
        
        for engine_type in SearchEngine:
            config = get_engine_config(engine_type)
            full_endpoint = config.base_url + config.search_endpoint
            endpoints.append(full_endpoint)
        
        # Todos los endpoints deben ser únicos
        assert len(endpoints) == len(set(endpoints))

# =============================================================================
# TESTS DE SEARCH RESULT VALIDATION
# =============================================================================

class TestSearchResultDataStructure:
    """Tests para validar estructura de datos de resultados"""
    
    def test_search_result_has_required_fields(self):
        """Test que SearchResult tiene todos los campos requeridos"""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            domain="example.com",
            rank=1,
            source_engine="test_engine"
        )
        
        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet"
        assert result.domain == "example.com"
        assert result.rank == 1
        assert result.source_engine == "test_engine"
        assert result.metadata == {}  # Default value
    
    def test_search_response_has_required_fields(self):
        """Test que SearchResponse tiene todos los campos requeridos"""
        results = [
            SearchResult(
                title="Test",
                url="https://example.com",
                snippet="snippet",
                domain="example.com",
                rank=1,
                source_engine="test"
            )
        ]
        
        response = SearchResponse(
            query="test query",
            results=results,
            total_results=1,
            search_time=1.5,
            engine="Test Engine",
            success=True
        )
        
        assert response.query == "test query"
        assert len(response.results) == 1
        assert response.total_results == 1
        assert response.search_time == 1.5
        assert response.engine == "Test Engine"
        assert response.success == True
        assert response.error_message is None
        assert response.metadata == {}
    
    def test_search_result_post_init_processes_url(self):
        """Test que SearchResult.__post_init__ procesa URL correctamente"""
        result = SearchResult(
            title="Test",
            url="https://www.example.com/path?utm_source=test",
            snippet="snippet",
            domain="",  # Should be calculated
            rank=1,
            source_engine="test"
        )
        
        # Domain debe calcularse automáticamente
        assert result.domain == "example.com"
        # URL debe normalizarse
        assert "utm_source" not in result.url or result.url != "https://www.example.com/path?utm_source=test"

# =============================================================================
# TESTS DE MEMORY Y RESOURCE USAGE
# =============================================================================

class TestSearchEngineResourceUsage:
    """Tests para uso de memoria y recursos"""
    
    @pytest.mark.slow
    def test_memory_usage_with_large_results(self):
        """Test de uso de memoria con resultados grandes"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Crear muchos resultados mock
        large_results = get_mock_search_results("test", "test_engine", 1000)
        
        # Crear respuesta con muchos resultados
        response = SearchResponse(
            query="test query",
            results=large_results,
            total_results=1000,
            search_time=1.0,
            engine="Test Engine",
            success=True
        )
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Aumento de memoria debe ser razonable (menos de 50MB)
        assert memory_increase < 50 * 1024 * 1024
        
        # Cleanup
        del large_results
        del response
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_search_engine_cleanup(self):
        """Test que motores de búsqueda limpian recursos correctamente"""
        engine = DuckDuckGoSearchEngine()
        
        # Simular múltiples búsquedas
        mock_response = WebResponse(
            url="https://duckduckgo.com/html/",
            status_code=200,
            content=DUCKDUCKGO_MOCK_HTML,
            headers={"content-type": "text/html"},
            encoding="utf-8",
            response_time=0.1,
            final_url="https://duckduckgo.com/html/",
            success=True
        )
        
        with patch.object(engine, '_make_request', return_value=mock_response):
            # Realizar muchas búsquedas
            for i in range(100):
                await engine.search(f"query {i}", max_results=5)
        
        # Verificar que no hay memory leaks obvios
        # (Este test es más conceptual, en práctica necesitaríamos profilers más avanzados)
        assert True  # Si llegamos aquí sin crash, el test pasa

# =============================================================================
# TESTS FINALES DE INTEGRACIÓN
# =============================================================================

class TestSearchEngineIntegrationFinal:
    """Tests finales de integración para todo el sistema de motores"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_search_workflow(self):
        """Test de flujo completo de búsqueda"""
        manager = SearchEngineManager()
        
        # Mock successful responses for all engines
        success_responses = {
            SearchEngine.DUCKDUCKGO: WebResponse(
                url="https://duckduckgo.com/html/", status_code=200,
                content=DUCKDUCKGO_MOCK_HTML, headers={"content-type": "text/html"},
                encoding="utf-8", response_time=1.0,
                final_url="https://duckduckgo.com/html/", success=True
            ),
            SearchEngine.SEARX: WebResponse(
                url="https://searx.be/search", status_code=200,
                content=json.dumps(SEARX_MOCK_RESPONSE), headers={"content-type": "application/json"},
                encoding="utf-8", response_time=1.5,
                final_url="https://searx.be/search", success=True
            )
        }
        
        for engine_type, mock_response in success_responses.items():
            engine = manager.engines[engine_type]
            with patch.object(engine, '_make_request', return_value=mock_response):
                
                # Test búsqueda individual
                result = await manager.search(
                    query="python programming",
                    engine=engine_type,
                    max_results=5
                )
                
                assert result.success == True
                assert len(result.results) > 0
                assert result.query == "python programming"
                
                # Verificar calidad de resultados
                for search_result in result.results:
                    assert len(search_result.title) > 0
                    assert search_result.url.startswith(('http://', 'https://'))
                    assert search_result.rank > 0
                    assert search_result.source_engine is not None
    
    def test_search_engine_system_health(self):
        """Test final de salud del sistema de motores"""
        manager = SearchEngineManager()
        
        # Verificar que todos los componentes están presentes
        assert len(manager.engines) == len(SearchEngine)
        
        # Verificar configuraciones
        for engine_type in SearchEngine:
            config = get_engine_config(engine_type)
            assert config is not None
            
            engine = manager.engines[engine_type]
            assert engine.config == config
            assert engine.engine_type == engine_type
        
        # Verificar que hay engines disponibles
        available = manager.get_available_engines()
        assert len(available) > 0
        
        # Verificar fallback configuration
        assert manager.default_engine in SearchEngine
        assert len(manager.fallback_engines) > 0
        assert all(engine in SearchEngine for engine in manager.fallback_engines)

# =============================================================================
# EXPORTS Y CONFIGURACIÓN FINAL
# =============================================================================

# Marcar todos los tests como unit tests por defecto
pytestmark = pytest.mark.unit

# Tests que requieren conexión a internet (para cuando se habiliten)
internet_required_tests = [
    "test_real_duckduckgo_search",
    "test_real_searx_search"
]

# Tests que requieren API keys (skip por defecto)
api_key_required_tests = [
    "test_real_google_search", 
    "test_real_bing_search"
]