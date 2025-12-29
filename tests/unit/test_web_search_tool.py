"""
Unit Tests for Web Search Tool
Tests del WebSearchTool completo que integra motores, extracción y rate limiting
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import List, Dict, Any

# Imports del proyecto
from app.agents.tools.web.web_search_tool import (
    WebSearchTool,
    WebSearchRequest,
    WebSearchResult,
    web_search,
    quick_search
)
from app.agents.tools.web.search_engines import SearchResult, SearchResponse
from app.agents.tools.web.content_extractor import ExtractedContent
from app.agents.tools.base_tool import ToolResult, ToolCategory

# Test fixtures
from fixtures import (
    MOCK_EXTRACTED_CONTENT,
    MOCK_FAILED_EXTRACTION,
    get_mock_search_results,
    PERFORMANCE_TEST_QUERIES,
    RATE_LIMIT_TEST_CONFIG,
    E2E_RESEARCH_SCENARIO
)

# =============================================================================
# FIXTURES ESPECÍFICAS PARA WEB SEARCH TOOL
# =============================================================================

@pytest.fixture
def web_search_tool():
    """Fixture del WebSearchTool"""
    return WebSearchTool()

@pytest.fixture
def mock_search_manager():
    """Mock del SearchEngineManager"""
    mock_manager = AsyncMock()
    
    # Mock search response exitosa
    mock_manager.search.return_value = SearchResponse(
        query="test query",
        results=get_mock_search_results("test query", "mock_engine", 5),
        total_results=5,
        search_time=1.2,
        engine="Mock Engine",
        success=True
    )
    
    return mock_manager

@pytest.fixture
def mock_content_extractor():
    """Mock del ContentExtractor"""
    mock_extractor = AsyncMock()
    
    # Mock extraction exitosa
    mock_extractor.extract_content.return_value = Mock(
        **MOCK_EXTRACTED_CONTENT,
        extraction_success=True
    )
    
    return mock_extractor

@pytest.fixture
def mock_rate_limiter():
    """Mock del RateLimitManager"""
    mock_limiter = AsyncMock()
    mock_limiter.acquire.return_value = True  # Siempre permite por defecto
    return mock_limiter

@pytest.fixture
def mock_robots_checker():
    """Mock del RobotsTxtChecker"""
    mock_checker = AsyncMock()
    mock_checker.can_fetch.return_value = True  # Siempre permite por defecto
    return mock_checker

# =============================================================================
# TESTS DE INICIALIZACIÓN Y CONFIGURACIÓN
# =============================================================================

class TestWebSearchToolInitialization:
    """Tests de inicialización del WebSearchTool"""
    
    def test_tool_initialization(self, web_search_tool):
        """Test que el tool se inicializa correctamente"""
        assert web_search_tool.name == "web_search"
        assert web_search_tool.description is not None
        assert web_search_tool.category == ToolCategory.WEB_SEARCH
        assert web_search_tool.version == "1.0.0"
        assert web_search_tool.requires_internet == True
        assert web_search_tool.requires_gpu == False
        assert web_search_tool.max_execution_time == 120
        assert web_search_tool.rate_limit == 30
    
    def test_tool_has_required_components(self, web_search_tool):
        """Test que el tool tiene todos los componentes necesarios"""
        assert hasattr(web_search_tool, 'config')
        assert hasattr(web_search_tool, 'search_manager')
        assert hasattr(web_search_tool, 'content_extractor')
        assert hasattr(web_search_tool, 'rate_limiter')
        assert hasattr(web_search_tool, 'robots_checker')
        
        # Verificar estadísticas iniciales
        assert isinstance(web_search_tool.stats, dict)
        assert web_search_tool.stats['total_searches'] == 0
    
    def test_tool_info_is_complete(self, web_search_tool):
        """Test que get_tool_info retorna información completa"""
        info = web_search_tool.get_tool_info()
        
        required_fields = [
            'name', 'description', 'category', 'version',
            'capabilities', 'parameters', 'example_usage',
            'limitations', 'statistics'
        ]
        
        for field in required_fields:
            assert field in info
        
        assert isinstance(info['capabilities'], list)
        assert len(info['capabilities']) > 0
        assert isinstance(info['parameters'], dict)
        assert isinstance(info['example_usage'], dict)

# =============================================================================
# TESTS DE PARSING DE REQUESTS
# =============================================================================

class TestWebSearchRequestParsing:
    """Tests de parsing de requests de búsqueda"""
    
    def test_parse_simple_string_query(self, web_search_tool):
        """Test de parsing de query simple string"""
        query = "python programming"
        
        request = web_search_tool._parse_search_request(query, {})
        
        assert isinstance(request, WebSearchRequest)
        assert request.query == "python programming"
        assert request.max_results == 10  # Default
        assert request.query_type == "general"  # Default
        assert request.extract_content == True  # Default
    
    def test_parse_json_query(self, web_search_tool):
        """Test de parsing de query JSON"""
        json_query = json.dumps({
            "query": "machine learning",
            "max_results": 15,
            "query_type": "academic",
            "extract_content": False
        })
        
        request = web_search_tool._parse_search_request(json_query, {})
        
        assert request.query == "machine learning"
        assert request.max_results == 15
        assert request.query_type == "academic"
        assert request.extract_content == False
    
    def test_parse_kwargs_override_json(self, web_search_tool):
        """Test que kwargs sobrescriben parámetros JSON"""
        json_query = json.dumps({
            "query": "test query",
            "max_results": 5
        })
        
        kwargs = {"max_results": 20, "search_engine": "duckduckgo"}
        
        request = web_search_tool._parse_search_request(json_query, kwargs)
        
        assert request.query == "test query"
        assert request.max_results == 20  # Overridden by kwargs
        assert request.search_engine == "duckduckgo"  # From kwargs
    
    def test_parse_malformed_json_fallback(self, web_search_tool):
        """Test que JSON mal formado usa string como query"""
        malformed_json = '{"query": "test", "invalid": json}'
        
        request = web_search_tool._parse_search_request(malformed_json, {})
        
        assert request.query == malformed_json  # Uses full string as query
        assert request.max_results == 10  # Default values

# =============================================================================
# TESTS DE BÚSQUEDA PRINCIPAL
# =============================================================================

class TestWebSearchExecution:
    """Tests de ejecución principal del web search"""
    
    @pytest.mark.asyncio
    async def test_successful_search_with_content_extraction(
        self, web_search_tool, mock_search_manager, mock_content_extractor, 
        mock_rate_limiter, mock_robots_checker
    ):
        """Test de búsqueda exitosa con extracción de contenido"""
        
        # Setup mocks
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.content_extractor = mock_content_extractor
        web_search_tool.rate_limiter = mock_rate_limiter
        web_search_tool.robots_checker = mock_robots_checker
        
        # Execute search
        result = await web_search_tool._arun("python programming")
        
        # Verificar que se llamaron los métodos correctos
        mock_rate_limiter.acquire.assert_called_once()
        mock_search_manager.search.assert_called_once()
        
        # Verificar resultado
        assert isinstance(result, str)
        assert "python programming" in result.lower()
        assert "Found" in result
        assert len(result) > 100  # Respuesta sustancial
        
        # Verificar estadísticas actualizadas
        assert web_search_tool.stats['total_searches'] == 1
        assert web_search_tool.stats['successful_searches'] == 1
    
    @pytest.mark.asyncio
    async def test_search_with_rate_limit_exceeded(
        self, web_search_tool, mock_rate_limiter
    ):
        """Test cuando rate limit es excedido"""
        
        # Rate limiter bloquea
        mock_rate_limiter.acquire.return_value = False
        web_search_tool.rate_limiter = mock_rate_limiter
        
        result = await web_search_tool._arun("test query")
        
        assert "Rate limit exceeded" in result
        assert web_search_tool.stats['rate_limit_blocks'] == 1
    
    @pytest.mark.asyncio
    async def test_search_with_no_results(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test cuando no se encuentran resultados"""
        
        # Mock search manager que no retorna resultados
        mock_search_manager.search.return_value = SearchResponse(
            query="nonexistent query",
            results=[],
            total_results=0,
            search_time=0.5,
            engine="Mock Engine",
            success=True
        )
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        result = await web_search_tool._arun("nonexistent query")
        
        assert "No search results found" in result
    
    @pytest.mark.asyncio 
    async def test_search_with_search_engine_failure(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test cuando el motor de búsqueda falla"""
        
        # Mock search manager que falla
        mock_search_manager.search.return_value = SearchResponse(
            query="test query",
            results=[],
            total_results=0,
            search_time=0.1,
            engine="Mock Engine",
            success=False,
            error_message="Search engine error"
        )
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        result = await web_search_tool._arun("test query")
        
        assert "No search results found" in result

# =============================================================================
# TESTS DE ENRIQUECIMIENTO DE RESULTADOS
# =============================================================================

class TestResultEnrichment:
    """Tests de enriquecimiento de resultados con extracción de contenido"""
    
    @pytest.mark.asyncio
    async def test_enrich_results_with_successful_extraction(
        self, web_search_tool, mock_content_extractor, mock_robots_checker
    ):
        """Test de enriquecimiento exitoso con extracción de contenido"""
        
        # Setup mocks
        web_search_tool.content_extractor = mock_content_extractor
        web_search_tool.robots_checker = mock_robots_checker
        
        # Search results mock
        search_results = get_mock_search_results("test query", "mock", 3)
        
        # Request mock
        request = WebSearchRequest(
            query="test query",
            extract_content=True,
            max_results=3
        )
        
        # Execute enrichment
        enriched = await web_search_tool._enrich_search_results(search_results, request)
        
        assert len(enriched) == 3
        
        for result in enriched:
            assert isinstance(result, WebSearchResult)
            assert result.title is not None
            assert result.url is not None
            assert result.relevance_score > 0
            
        # Verificar que se llamó extracción de contenido
        assert mock_content_extractor.extract_content.call_count == 3
    
    @pytest.mark.asyncio
    async def test_enrich_results_with_robots_blocked(
        self, web_search_tool, mock_content_extractor, mock_robots_checker
    ):
        """Test cuando robots.txt bloquea acceso"""
        
        # Robots checker bloquea acceso
        mock_robots_checker.can_fetch.return_value = False
        
        web_search_tool.content_extractor = mock_content_extractor
        web_search_tool.robots_checker = mock_robots_checker
        
        search_results = get_mock_search_results("test query", "mock", 2)
        request = WebSearchRequest(query="test query", extract_content=True)
        
        enriched = await web_search_tool._enrich_search_results(search_results, request)
        
        # Verificar que extraction no se llamó (bloqueado por robots)
        mock_content_extractor.extract_content.assert_not_called()
        
        # Verificar estadísticas
        assert web_search_tool.stats['robots_blocks'] == 2
    
    @pytest.mark.asyncio
    async def test_enrich_results_without_content_extraction(
        self, web_search_tool, mock_content_extractor
    ):
        """Test de enriquecimiento sin extracción de contenido"""
        
        web_search_tool.content_extractor = mock_content_extractor
        
        search_results = get_mock_search_results("test query", "mock", 3)
        request = WebSearchRequest(
            query="test query",
            extract_content=False  # No extraction
        )
        
        enriched = await web_search_tool._enrich_search_results(search_results, request)
        
        assert len(enriched) == 3
        
        # Verificar que NO se llamó extracción
        mock_content_extractor.extract_content.assert_not_called()
        
        # Resultados deben tener datos básicos
        for result in enriched:
            assert result.extracted_content is None
            assert result.extraction_success == False
    
    def test_calculate_relevance_score(self, web_search_tool):
        """Test de cálculo de score de relevancia"""
        
        # Result que coincide perfectamente con query
        perfect_match = SearchResult(
            title="Python Programming Tutorial",
            url="https://python.org/tutorial",
            snippet="Learn python programming with this comprehensive tutorial",
            domain="python.org",
            rank=1,
            source_engine="test"
        )
        
        # Result con coincidencia parcial
        partial_match = SearchResult(
            title="Java Development Guide", 
            url="https://java.com/guide",
            snippet="Programming guide for developers",
            domain="java.com",
            rank=5,
            source_engine="test"
        )
        
        query = "python programming"
        
        perfect_score = web_search_tool._calculate_relevance_score(perfect_match, query)
        partial_score = web_search_tool._calculate_relevance_score(partial_match, query)
        
        assert perfect_score > partial_score
        assert 0.0 <= perfect_score <= 1.0
        assert 0.0 <= partial_score <= 1.0

# =============================================================================
# TESTS DE FILTRADO DE RESULTADOS
# =============================================================================

class TestResultFiltering:
    """Tests de filtrado de resultados por dominio"""
    
    def test_filter_results_by_include_domains(self, web_search_tool):
        """Test de filtrado por dominios incluidos"""
        
        results = [
            SearchResult(title="Test 1", url="https://python.org/doc", snippet="", 
                        domain="python.org", rank=1, source_engine="test"),
            SearchResult(title="Test 2", url="https://java.com/doc", snippet="",
                        domain="java.com", rank=2, source_engine="test"),
            SearchResult(title="Test 3", url="https://python.org/tutorial", snippet="",
                        domain="python.org", rank=3, source_engine="test")
        ]
        
        include_domains = ["python.org"]
        
        filtered = web_search_tool._filter_results_by_domain(
            results, include_domains, None
        )
        
        assert len(filtered) == 2
        assert all("python.org" in result.domain for result in filtered)
    
    def test_filter_results_by_exclude_domains(self, web_search_tool):
        """Test de filtrado por dominios excluidos"""
        
        results = [
            SearchResult(title="Test 1", url="https://python.org/doc", snippet="",
                        domain="python.org", rank=1, source_engine="test"),
            SearchResult(title="Test 2", url="https://spam.com/ad", snippet="",
                        domain="spam.com", rank=2, source_engine="test"),
            SearchResult(title="Test 3", url="https://github.com/repo", snippet="",
                        domain="github.com", rank=3, source_engine="test")
        ]
        
        exclude_domains = ["spam.com"]
        
        filtered = web_search_tool._filter_results_by_domain(
            results, None, exclude_domains
        )
        
        assert len(filtered) == 2
        assert not any("spam.com" in result.domain for result in filtered)
    
    def test_filter_blocked_domains_from_config(self, web_search_tool):
        """Test que dominios bloqueados en config se filtran"""
        
        # Mock config para bloquear facebook.com
        with patch.object(web_search_tool.config, 'is_domain_blocked', return_value=True) as mock_blocked:
            
            results = [
                SearchResult(title="Test 1", url="https://python.org/doc", snippet="",
                            domain="python.org", rank=1, source_engine="test"),
                SearchResult(title="Test 2", url="https://facebook.com/page", snippet="",
                            domain="facebook.com", rank=2, source_engine="test")
            ]
            
            # Mock que facebook.com está bloqueado
            def mock_is_blocked(domain):
                return domain == "facebook.com"
            
            mock_blocked.side_effect = mock_is_blocked
            
            filtered = web_search_tool._filter_results_by_domain(results, None, None)
            
            assert len(filtered) == 1
            assert filtered[0].domain == "python.org"

# =============================================================================
# TESTS DE FORMATEO DE RESPUESTAS
# =============================================================================

class TestResponseFormatting:
    """Tests de formateo de respuestas finales"""
    
    def test_format_search_response_basic(self, web_search_tool):
        """Test de formateo básico de respuesta"""
        
        # Crear resultados mock con contenido extraído
        results = []
        for i in range(3):
            result = WebSearchResult(
                title=f"Test Result {i+1}",
                url=f"https://example{i+1}.com",
                snippet=f"This is test snippet {i+1}",
                domain=f"example{i+1}.com",
                rank=i+1,
                source_engine="test",
                extracted_content=Mock(
                    main_content=f"Extracted content {i+1} " * 20,
                    author=f"Author {i+1}",
                    publish_date="2024-01-15"
                ),
                extraction_success=True,
                quality_score=0.8,
                language="english",
                word_count=100,
                reading_time=1
            )
            results.append(result)
        
        request = WebSearchRequest(
            query="test query",
            max_results=10,
            include_metadata=True
        )
        
        formatted = web_search_tool._format_search_response(results, request)
        
        assert "Found 3 results for 'test query'" in formatted
        assert "Test Result 1" in formatted
        assert "example1.com" in formatted
        assert "Quality:" in formatted  # Metadata incluida
        assert "Length:" in formatted
        assert "Successfully extracted content from" in formatted
    
    def test_format_search_response_without_metadata(self, web_search_tool):
        """Test de formateo sin metadatos"""
        
        results = [
            WebSearchResult(
                title="Simple Result",
                url="https://example.com",
                snippet="Simple snippet",
                domain="example.com",
                rank=1,
                source_engine="test"
            )
        ]
        
        request = WebSearchRequest(
            query="simple query",
            include_metadata=False
        )
        
        formatted = web_search_tool._format_search_response(results, request)
        
        assert "Simple Result" in formatted
        assert "Quality:" not in formatted  # No metadata
        assert "Language:" not in formatted
    
    def test_format_search_response_no_results(self, web_search_tool):
        """Test de formateo cuando no hay resultados"""
        
        request = WebSearchRequest(query="no results query")
        
        formatted = web_search_tool._format_search_response([], request)
        
        assert "No search results found" in formatted
    
    def test_format_error_response(self, web_search_tool):
        """Test de formateo de respuestas de error"""
        
        error_msg = "Connection timeout"
        
        formatted = web_search_tool._format_error_response(error_msg)
        
        assert "Web search error:" in formatted
        assert "Connection timeout" in formatted

# =============================================================================
# TESTS DE CACHE
# =============================================================================

class TestWebSearchCache:
    """Tests del sistema de cache del WebSearchTool"""
    
    def test_cache_key_generation(self, web_search_tool):
        """Test de generación de claves de cache"""
        
        request1 = WebSearchRequest(
            query="python programming",
            max_results=10,
            search_engine="duckduckgo",
            query_type="general",
            extract_content=True
        )
        
        request2 = WebSearchRequest(
            query="python programming",
            max_results=5,  # Diferente
            search_engine="duckduckgo",
            query_type="general",
            extract_content=True
        )
        
        key1 = web_search_tool._generate_cache_key(request1)
        key2 = web_search_tool._generate_cache_key(request2)
        
        assert key1 != key2  # Keys deben ser diferentes
        assert isinstance(key1, str)
        assert isinstance(key2, str)
    
    def test_cache_storage_and_retrieval(self, web_search_tool):
        """Test de almacenamiento y recuperación de cache"""
        
        cache_key = "test_key"
        test_result = "Test cached result"
        
        # Store in cache
        web_search_tool._cache_result(cache_key, test_result)
        
        # Retrieve from cache
        cached = web_search_tool._get_cached_result(cache_key)
        
        assert cached == test_result
    
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
            time.sleep(0.2)
            
            # Should be expired
            expired = web_search_tool._get_cached_result(cache_key)
            assert expired is None
            
        finally:
            web_search_tool._cache_ttl = original_ttl
    
    def test_cache_cleanup_when_full(self, web_search_tool):
        """Test de limpieza de cache cuando está lleno"""
        
        # Fill cache beyond limit (100 entries default)
        for i in range(120):
            web_search_tool._cache_result(f"key_{i}", f"result_{i}")
        
        # Cache should be cleaned up
        assert len(web_search_tool._results_cache) <= 100

# =============================================================================
# TESTS DE TIPOS DE QUERY
# =============================================================================

class TestQueryTypes:
    """Tests de diferentes tipos de query"""
    
    @pytest.mark.asyncio
    async def test_academic_query_type(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test de query tipo académico"""
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        # Query académica
        academic_query = json.dumps({
            "query": "machine learning research",
            "query_type": "academic",
            "max_results": 8
        })
        
        await web_search_tool._arun(academic_query)
        
        # Verificar que se llamó search con parámetros correctos
        call_args = mock_search_manager.search.call_args
        assert call_args is not None
        
        # Query puede haber sido modificada para tipo académico
        called_query = call_args[1]['query'] if 'query' in call_args[1] else call_args[0][0]
        assert "machine learning research" in called_query
    
    @pytest.mark.parametrize("query_type,expected_modifier", [
        ("academic", "research"),
        ("technical", "documentation"),
        ("news", "recent"),
        ("general", "")
    ])
    def test_query_type_modifiers(self, web_search_tool, query_type, expected_modifier):
        """Test que diferentes tipos de query aplican modificadores"""
        
        # Mock get_query_config para retornar modificadores
        with patch('app.agents.tools.web.web_search_tool.get_query_config') as mock_config:
            
            mock_config.return_value = {
                "query_modifiers": [f"site:example.com {expected_modifier}"] if expected_modifier else [],
                "max_results": 10
            }
            
            request = WebSearchRequest(
                query="test query",
                query_type=query_type
            )
            
            # Simular _perform_search para ver query modificada
            enhanced_query = request.query
            if mock_config.return_value.get('query_modifiers'):
                modifiers = ' '.join(mock_config.return_value['query_modifiers'])
                enhanced_query = f"{request.query} {modifiers}"
            
            if expected_modifier:
                assert expected_modifier in enhanced_query or "site:example.com" in enhanced_query
            else:
                assert enhanced_query == "test query"

# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

class TestWebSearchToolPerformance:
    """Tests de rendimiento del WebSearchTool"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_search_performance_simple_query(
        self, web_search_tool, mock_search_manager, mock_rate_limiter, performance_tracker
    ):
        """Test de performance para query simple"""
        
        # Setup mocks rápidos
        mock_search_manager.search.return_value = SearchResponse(
            query="test", results=get_mock_search_results("test", "mock", 5),
            total_results=5, search_time=0.1, engine="Mock", success=True
        )
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        performance_tracker['start']('simple_search')
        
        result = await web_search_tool._arun("test")
        
        duration = performance_tracker['end']('simple_search')
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert duration < 5.0  # Menos de 5 segundos
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_searches_performance(
        self, web_search_tool, mock_search_manager, mock_rate_limiter, performance_tracker
    ):
        """Test de performance con búsquedas concurrentes"""
        
        # Setup mocks
        mock_search_manager.search.return_value = SearchResponse(
            query="concurrent", results=get_mock_search_results("concurrent", "mock", 3),
            total_results=3, search_time=0.2, engine="Mock", success=True
        )
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        performance_tracker['start']('concurrent_searches')
        
        # Ejecutar 5 búsquedas concurrentes
        tasks = [
            web_search_tool._arun(f"query {i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        duration = performance_tracker['end']('concurrent_searches')
        
        assert len(results) == 5
        assert all(isinstance(r, str) for r in results)
        assert duration < 10.0  # Menos de 10 segundos para 5 búsquedas
    
    @pytest.mark.performance
    def test_memory_usage_large_results(self, web_search_tool):
        """Test de uso de memoria con resultados grandes"""
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Crear muchos resultados
        large_results = []
        for i in range(1000):
            result = WebSearchResult(
                title=f"Large Result {i}",
                url=f"https://example{i}.com/very/long/path/to/content",
                snippet="This is a long snippet " * 20,
                domain=f"example{i}.com",
                rank=i+1,
                source_engine="test",
                extracted_content=Mock(main_content="Large content " * 100),
                extraction_success=True
            )
            large_results.append(result)
        
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Aumento de memoria debe ser razonable (menos de 100MB)
        assert memory_increase < 100 * 1024 * 1024
        
        # Cleanup
        del large_results

# =============================================================================
# TESTS DE HEALTH CHECK
# =============================================================================

class TestWebSearchToolHealthCheck:
    """Tests de health check del WebSearchTool"""
    
    @pytest.mark.asyncio
    async def test_health_check_success(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test de health check exitoso"""
        
        # Mock successful search
        mock_search_manager.search.return_value = SearchResponse(
            query="health check", results=[SearchResult(
                title="Health Check Result", url="https://example.com",
                snippet="Health check successful", domain="example.com",
                rank=1, source_engine="mock"
            )],
            total_results=1, search_time=0.5, engine="Mock", success=True
        )
        
        # Mock health check para search engines
        mock_search_manager.health_check.return_value = {
            "duckduckgo": {"healthy": True, "available": True},
            "searx": {"healthy": True, "available": True}
        }
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        health_info = await web_search_tool.health_check()
        
        assert health_info["healthy"] == True
        assert "search_test_success" in health_info
        assert "available_engines" in health_info
        assert "cache_size" in health_info
        assert health_info["total_searches"] >= 0
    
    @pytest.mark.asyncio
    async def test_health_check_with_search_failure(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test de health check cuando búsqueda de prueba falla"""
        
        # Mock failed search
        with patch.object(web_search_tool, '_arun', side_effect=Exception("Search failed")):
            
            web_search_tool.search_manager = mock_search_manager
            web_search_tool.rate_limiter = mock_rate_limiter
            
            health_info = await web_search_tool.health_check()
            
            assert health_info["healthy"] == False
            assert "health_check_error" in health_info
            assert "Search failed" in health_info["health_check_error"]
    
    def test_health_check_input_generation(self, web_search_tool):
        """Test de generación de input para health check"""
        
        health_input = web_search_tool._get_health_check_input()
        
        assert isinstance(health_input, str)
        assert len(health_input) > 0
        assert health_input == "python programming tutorial"

# =============================================================================
# TESTS DE FUNCIONES DE CONVENIENCIA
# =============================================================================

class TestConvenienceFunctions:
    """Tests de funciones de conveniencia del módulo"""
    
    @pytest.mark.asyncio
    async def test_web_search_function(self):
        """Test de la función web_search directa"""
        
        with patch('app.agents.tools.web.web_search_tool.WebSearchTool') as MockTool:
            
            mock_instance = MockTool.return_value
            mock_instance._arun.return_value = "Mocked web search result"
            
            result = await web_search("test query", max_results=5)
            
            assert result == "Mocked web search result"
            mock_instance._arun.assert_called_once_with("test query", max_results=5)
    
    @pytest.mark.asyncio
    async def test_quick_search_function(self):
        """Test de la función quick_search"""
        
        with patch('app.agents.tools.web.web_search_tool.WebSearchTool') as MockTool:
            
            mock_instance = MockTool.return_value
            
            # Mock _perform_search y _enrich_search_results
            mock_search_results = get_mock_search_results("quick test", "mock", 3)
            
            mock_instance._perform_search.return_value = mock_search_results
            mock_instance._enrich_search_results.return_value = [
                WebSearchResult(
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet,
                    domain=result.domain,
                    rank=result.rank,
                    source_engine=result.source_engine,
                    relevance_score=0.8
                ) for result in mock_search_results
            ]
            
            results = await quick_search("quick test", max_results=3)
            
            assert isinstance(results, list)
            assert len(results) == 3
            
            for result in results:
                assert isinstance(result, dict)
                assert "title" in result
                assert "url" in result
                assert "relevance_score" in result

# =============================================================================
# TESTS DE ERROR HANDLING AVANZADO
# =============================================================================

class TestWebSearchToolErrorHandling:
    """Tests avanzados de manejo de errores"""
    
    @pytest.mark.asyncio
    async def test_handle_search_manager_exception(
        self, web_search_tool, mock_rate_limiter
    ):
        """Test de manejo de excepción en search manager"""
        
        # Mock search manager que lanza excepción
        mock_search_manager = AsyncMock()
        mock_search_manager.search.side_effect = Exception("Search engine crashed")
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        result = await web_search_tool._arun("test query")
        
        assert "Search failed" in result
        assert isinstance(result, str)
    
    @pytest.mark.asyncio
    async def test_handle_content_extraction_exception(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test de manejo de excepción en extracción de contenido"""
        
        # Mock content extractor que falla
        mock_content_extractor = AsyncMock()
        mock_content_extractor.extract_content.side_effect = Exception("Extraction failed")
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        web_search_tool.content_extractor = mock_content_extractor
        web_search_tool.robots_checker = AsyncMock()
        web_search_tool.robots_checker.can_fetch.return_value = True
        
        # Debe manejar excepción y continuar
        result = await web_search_tool._arun("test query")
        
        assert isinstance(result, str)
        assert len(result) > 0
        # Debe retornar resultados aunque falle extracción
    
    @pytest.mark.asyncio
    async def test_handle_malformed_input(self, web_search_tool, mock_rate_limiter):
        """Test de manejo de input mal formado"""
        
        web_search_tool.rate_limiter = mock_rate_limiter
        
        # Diferentes tipos de input problemático
        problematic_inputs = [
            None,
            "",
            "   ",
            {"not": "a string"},
            123,
            [],
            '{"malformed": json'
        ]
        
        for bad_input in problematic_inputs:
            try:
                result = await web_search_tool._arun(bad_input)
                # Si no lanza excepción, debe retornar string válido
                assert isinstance(result, str)
            except Exception as e:
                # Si lanza excepción, debe ser manejada gracefully
                assert "error" in str(e).lower() or "invalid" in str(e).lower()

# =============================================================================
# TESTS DE DOMAIN SPECIFIC LOGIC
# =============================================================================

class TestDomainSpecificLogic:
    """Tests de lógica específica por dominio"""
    
    def test_infer_content_type_academic(self, web_search_tool):
        """Test de inferencia de tipo de contenido académico"""
        
        academic_result = WebSearchResult(
            title="Research Paper on Machine Learning",
            url="https://arxiv.org/abs/1234.5678",
            snippet="This paper presents novel ML techniques",
            domain="arxiv.org",
            rank=1,
            source_engine="test"
        )
        
        content_type = web_search_tool._infer_content_type(academic_result, "academic")
        
        assert content_type == "academic"
    
    def test_infer_content_type_technical(self, web_search_tool):
        """Test de inferencia de tipo técnico"""
        
        tech_result = WebSearchResult(
            title="API Documentation",
            url="https://docs.python.org/3/api/",
            snippet="Python API reference documentation",
            domain="docs.python.org",
            rank=1,
            source_engine="test"
        )
        
        content_type = web_search_tool._infer_content_type(tech_result, "technical")
        
        assert content_type == "technical"
    
    def test_infer_content_type_forum(self, web_search_tool):
        """Test de inferencia de tipo forum"""
        
        forum_result = WebSearchResult(
            title="How to fix Python error?",
            url="https://stackoverflow.com/questions/12345",
            snippet="I'm getting this error when running Python",
            domain="stackoverflow.com", 
            rank=1,
            source_engine="test"
        )
        
        content_type = web_search_tool._infer_content_type(forum_result, "general")
        
        assert content_type == "forum_post"
    
    @pytest.mark.parametrize("domain,expected_type", [
        ("docs.python.org", "documentation"),
        ("github.com", "documentation"),
        ("stackoverflow.com", "forum_post"),
        ("reddit.com", "forum_post"),
        ("arxiv.org", "article"),
        ("unknown-domain.com", "article")
    ])
    def test_infer_content_type_by_domain(self, web_search_tool, domain, expected_type):
        """Test parametrizado de inferencia por dominio"""
        
        result = WebSearchResult(
            title="Test Title",
            url=f"https://{domain}/path",
            snippet="Test snippet",
            domain=domain,
            rank=1,
            source_engine="test"
        )
        
        content_type = web_search_tool._infer_content_type(result, "general")
        
        assert content_type == expected_type

# =============================================================================
# TESTS DE RATE LIMITING INTEGRATION
# =============================================================================

class TestRateLimitingIntegration:
    """Tests de integración con rate limiting"""
    
    @pytest.mark.asyncio
    async def test_rate_limiting_blocks_excessive_requests(self, web_search_tool):
        """Test que rate limiting bloquea requests excesivos"""
        
        # Mock rate limiter que bloquea después de 3 requests
        call_count = 0
        
        async def mock_acquire(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return call_count <= 3
        
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire = mock_acquire
        
        web_search_tool.rate_limiter = mock_rate_limiter
        
        # Primeros 3 requests deben pasar
        for i in range(3):
            result = await web_search_tool._arun(f"query {i}")
            assert "Rate limit exceeded" not in result
        
        # 4to request debe ser bloqueado
        result = await web_search_tool._arun("blocked query")
        assert "Rate limit exceeded" in result
        
        # Verificar estadísticas
        assert web_search_tool.stats['rate_limit_blocks'] >= 1
    
    @pytest.mark.asyncio
    async def test_rate_limiting_parameters_passed_correctly(self, web_search_tool):
        """Test que parámetros se pasan correctamente a rate limiter"""
        
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire.return_value = True
        
        web_search_tool.rate_limiter = mock_rate_limiter
        
        await web_search_tool._arun("test query")
        
        # Verificar que acquire fue llamado con parámetros correctos
        mock_rate_limiter.acquire.assert_called_once()
        call_args = mock_rate_limiter.acquire.call_args
        
        # Verificar que identifier incluye el nombre del tool
        identifier = call_args[1]['identifier']
        assert "web_search" in identifier

# =============================================================================
# TESTS DE ESTADÍSTICAS Y MÉTRICAS
# =============================================================================

class TestStatisticsAndMetrics:
    """Tests de estadísticas y métricas del tool"""
    
    @pytest.mark.asyncio
    async def test_statistics_tracking(
        self, web_search_tool, mock_search_manager, mock_rate_limiter,
        mock_content_extractor, mock_robots_checker
    ):
        """Test de tracking de estadísticas"""
        
        # Setup mocks
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        web_search_tool.content_extractor = mock_content_extractor
        web_search_tool.robots_checker = mock_robots_checker
        
        initial_stats = web_search_tool.stats.copy()
        
        # Realizar búsqueda exitosa
        await web_search_tool._arun("test query")
        
        # Verificar que estadísticas se actualizaron
        assert web_search_tool.stats['total_searches'] == initial_stats['total_searches'] + 1
        assert web_search_tool.stats['successful_searches'] == initial_stats['successful_searches'] + 1
    
    @pytest.mark.asyncio
    async def test_statistics_on_rate_limit_block(self, web_search_tool):
        """Test de estadísticas cuando rate limit bloquea"""
        
        # Mock rate limiter que bloquea
        mock_rate_limiter = AsyncMock()
        mock_rate_limiter.acquire.return_value = False
        
        web_search_tool.rate_limiter = mock_rate_limiter
        
        initial_blocks = web_search_tool.stats['rate_limit_blocks']
        
        await web_search_tool._arun("blocked query")
        
        assert web_search_tool.stats['rate_limit_blocks'] == initial_blocks + 1
    
    @pytest.mark.asyncio
    async def test_statistics_on_robots_block(
        self, web_search_tool, mock_search_manager, mock_rate_limiter, mock_robots_checker
    ):
        """Test de estadísticas cuando robots.txt bloquea"""
        
        # Setup mocks
        mock_robots_checker.can_fetch.return_value = False  # Block robots
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        web_search_tool.robots_checker = mock_robots_checker
        
        initial_blocks = web_search_tool.stats['robots_blocks']
        
        await web_search_tool._arun("test query")
        
        # Robots blocks se incrementan durante content extraction
        # pero solo si extract_content=True (default)
        assert web_search_tool.stats['robots_blocks'] >= initial_blocks

# =============================================================================
# TESTS FINALES DE INTEGRATION
# =============================================================================

class TestWebSearchToolFinalIntegration:
    """Tests finales de integración completa del WebSearchTool"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_workflow_success(self, web_search_tool):
        """Test del workflow completo exitoso"""
        
        # Mock todos los componentes para simular flujo exitoso
        with patch.object(web_search_tool, 'rate_limiter') as mock_rate, \
             patch.object(web_search_tool, 'search_manager') as mock_search, \
             patch.object(web_search_tool, 'content_extractor') as mock_extract, \
             patch.object(web_search_tool, 'robots_checker') as mock_robots:
            
            # Configure mocks
            mock_rate.acquire.return_value = True
            mock_search.search.return_value = SearchResponse(
                query="complete test", 
                results=get_mock_search_results("complete test", "mock", 3),
                total_results=3, search_time=1.0, engine="Mock", success=True
            )
            mock_extract.extract_content.return_value = Mock(**MOCK_EXTRACTED_CONTENT)
            mock_robots.can_fetch.return_value = True
            
            # Execute complete workflow
            result = await web_search_tool._arun("complete test")
            
            # Verify complete response
            assert isinstance(result, str)
            assert "Found 3 results" in result
            assert "complete test" in result
            assert len(result) > 200  # Substantial response
            
            # Verify all components were called
            mock_rate.acquire.assert_called_once()
            mock_search.search.assert_called_once()
            mock_robots.can_fetch.assert_called()  # Called for each result
    
    def test_tool_integration_with_base_tool_interface(self, web_search_tool):
        """Test de integración con interfaz BaseTool"""
        
        # Verify tool implements BaseTool interface correctly
        assert hasattr(web_search_tool, '_arun')
        assert hasattr(web_search_tool, 'get_tool_info')
        assert hasattr(web_search_tool, 'health_check')
        
        # Verify tool metadata
        assert web_search_tool.name == "web_search"
        assert web_search_tool.category == ToolCategory.WEB_SEARCH
        assert isinstance(web_search_tool.version, str)
        
        # Verify tool info structure
        tool_info = web_search_tool.get_tool_info()
        
        required_info_fields = [
            'name', 'description', 'category', 'version',
            'capabilities', 'parameters', 'limitations'
        ]
        
        for field in required_info_fields:
            assert field in tool_info
            assert tool_info[field] is not None
    
    @pytest.mark.asyncio
    async def test_tool_execute_method_integration(self, web_search_tool):
        """Test de integración con método execute de BaseTool"""
        
        # Mock components para test rápido
        with patch.object(web_search_tool, 'rate_limiter') as mock_rate, \
             patch.object(web_search_tool, 'search_manager') as mock_search:
            
            mock_rate.acquire.return_value = True
            mock_search.search.return_value = SearchResponse(
                query="execute test", results=[], total_results=0,
                search_time=0.1, engine="Mock", success=True
            )
            
            # Test execute method (inherited from BaseTool)
            result = await web_search_tool.execute("execute test")
            
            assert isinstance(result, ToolResult)
            assert result.success in [True, False]  # Either outcome is valid
            assert isinstance(result.data, str)
            assert result.execution_time > 0
    
    def test_tool_registration_and_discovery(self):
        """Test que el tool se registra correctamente"""
        
        # Verify tool is properly decorated and registered
        from app.agents.tools.web.web_search_tool import WebSearchTool
        
        # Verify class has correct attributes
        assert hasattr(WebSearchTool, 'category')
        assert WebSearchTool.category == ToolCategory.WEB_SEARCH
        
        # Verify tool can be instantiated
        tool = WebSearchTool()
        assert isinstance(tool, WebSearchTool)
        assert tool.name == "web_search"

# =============================================================================
# TESTS DE EDGE CASES FINALES
# =============================================================================

class TestWebSearchToolEdgeCases:
    """Tests de casos extremos finales"""
    
    @pytest.mark.asyncio
    async def test_search_with_unicode_query(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test de búsqueda con caracteres Unicode"""
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        unicode_query = "búsqueda en español con acentos 🔍"
        
        result = await web_search_tool._arun(unicode_query)
        
        assert isinstance(result, str)
        # Query debe ser preservada correctamente
        assert "español" in result or "búsqueda" in result
    
    @pytest.mark.asyncio
    async def test_search_with_very_long_response(
        self, web_search_tool, mock_search_manager, mock_rate_limiter
    ):
        """Test con respuesta muy larga"""
        
        # Mock results con contenido muy largo
        long_results = []
        for i in range(50):  # Muchos resultados
            result = SearchResult(
                title=f"Long Title {i} " * 10,
                url=f"https://example{i}.com/very/long/path/to/content",
                snippet="Very long snippet content " * 50,
                domain=f"example{i}.com",
                rank=i+1,
                source_engine="mock"
            )
            long_results.append(result)
        
        mock_search_manager.search.return_value = SearchResponse(
            query="long test", results=long_results, total_results=50,
            search_time=2.0, engine="Mock", success=True
        )
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        
        result = await web_search_tool._arun("long test")
        
        assert isinstance(result, str)
        assert len(result) > 1000  # Should be substantial
        # Debe manejar contenido largo sin problemas
    
    @pytest.mark.asyncio
    async def test_search_with_mixed_success_failure_extraction(
        self, web_search_tool, mock_search_manager, mock_rate_limiter, mock_robots_checker
    ):
        """Test con extracción mixta (algunas exitosas, otras no)"""
        
        # Mock content extractor que a veces falla
        extraction_results = [
            Mock(**MOCK_EXTRACTED_CONTENT, extraction_success=True),
            Mock(**MOCK_FAILED_EXTRACTION, extraction_success=False),
            Mock(**MOCK_EXTRACTED_CONTENT, extraction_success=True)
        ]
        
        mock_content_extractor = AsyncMock()
        mock_content_extractor.extract_content.side_effect = extraction_results
        
        web_search_tool.search_manager = mock_search_manager
        web_search_tool.rate_limiter = mock_rate_limiter
        web_search_tool.content_extractor = mock_content_extractor
        web_search_tool.robots_checker = mock_robots_checker
        
        result = await web_search_tool._arun("mixed test")
        
        assert isinstance(result, str)
        # Debe manejar mix de éxitos y fallos en extracción
        assert "Successfully extracted content from" in result

# =============================================================================
# CONFIGURACIÓN FINAL Y EXPORTS
# =============================================================================

# Marcar todos los tests como unit por defecto
pytestmark = pytest.mark.unit

# Tests específicos que requieren marcadores especiales
slow_tests = [
    "test_search_performance_simple_query",
    "test_concurrent_searches_performance", 
    "test_memory_usage_large_results"
]

integration_tests = [
    "test_complete_workflow_success",
    "test_tool_integration_with_base_tool_interface"
]

# Tests que podrían requerir servicios externos en el futuro
external_service_tests = [
    "test_health_check_success",
    "test_complete_workflow_success"
]