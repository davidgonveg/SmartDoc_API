"""
Unit Tests for Content Extraction
Tests para extracción de contenido HTML sin dependencias externas
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from bs4 import BeautifulSoup

# Imports del proyecto
from app.agents.tools.web.web_utils import ContentCleaner, URLNormalizer, get_content_hash
from app.agents.tools.web.content_extractor import ContentExtractor, ExtractedContent
from app.agents.tools.web.config.extraction_rules import (
    get_extraction_rule, 
    ContentType, 
    is_content_valid,
    calculate_content_quality_score,
    get_domain_from_url
)

# Test fixtures
from fixtures import (
    SIMPLE_HTML,
    COMPLEX_HTML,
    WIKIPEDIA_HTML,
    GITHUB_HTML,
    STACKOVERFLOW_HTML,
    MALFORMED_HTML,
    UNICODE_HTML,
    MOCK_EXTRACTED_CONTENT
)

# =============================================================================
# TESTS DE CONTENT CLEANER
# =============================================================================

class TestContentCleaner:
    """Tests para la clase ContentCleaner"""
    
    def test_clean_simple_html(self):
        """Test básico de limpieza de HTML"""
        result = ContentCleaner.clean_html(SIMPLE_HTML)
        
        assert "Main Article Title" in result
        assert "first paragraph" in result
        assert "second paragraph" in result
        assert "<html>" not in result
        assert "<p>" not in result
    
    def test_clean_complex_html_removes_unwanted_elements(self):
        """Test que elementos no deseados son removidos"""
        result = ContentCleaner.clean_html(COMPLEX_HTML)
        
        # Contenido principal debe estar presente
        assert "Main Article Title" in result
        assert "main content paragraph" in result
        assert "Final paragraph" in result
        
        # Elementos no deseados deben ser removidos
        assert "Navigation menu" not in result
        assert "Sidebar content" not in result
        assert "Footer content" not in result
        assert "console.log" not in result
        assert "body { margin: 0; }" not in result
    
    def test_clean_html_preserves_links_when_requested(self):
        """Test que links se preservan cuando se solicita"""
        html_with_link = '<p>Check out <a href="http://example.com">this link</a></p>'
        
        # Sin preservar links
        result_no_links = ContentCleaner.clean_html(html_with_link, preserve_links=False)
        assert "http://example.com" not in result_no_links
        assert "this link" in result_no_links
        
        # Preservando links
        result_with_links = ContentCleaner.clean_html(html_with_link, preserve_links=True)
        assert "this link [http://example.com]" in result_with_links
    
    def test_clean_text_normalizes_whitespace(self):
        """Test de normalización de espacios en blanco"""
        messy_text = "  Multiple    spaces\n\n\n\nand\t\ttabs  \n  "
        
        result = ContentCleaner.clean_text(messy_text)
        
        assert result == "Multiple spaces\n\nand tabs"
    
    def test_clean_text_handles_unicode(self):
        """Test de manejo de caracteres Unicode"""
        unicode_text = "Café with émojis 🐍 and áccénts"
        
        result = ContentCleaner.clean_text(unicode_text)
        
        assert "Café" in result
        assert "émojis" in result
        assert "áccénts" in result
        assert "🐍" in result
    
    def test_clean_malformed_html(self):
        """Test de HTML mal formado"""
        result = ContentCleaner.clean_html(MALFORMED_HTML)
        
        # Debe extraer algo de contenido aunque el HTML esté mal formado
        assert len(result) > 0
        assert "Missing closing tags" in result
        assert "alert('malicious script')" not in result  # Scripts removidos
    
    def test_extract_main_content_with_article_tag(self):
        """Test de extracción usando tag article"""
        soup = BeautifulSoup(COMPLEX_HTML, 'html.parser')
        
        result = ContentCleaner.extract_main_content(soup)
        
        assert "Main Article Title" in result
        assert "main content paragraph" in result
        # Sidebar no debe estar en contenido principal
        assert "Sidebar content" not in result
    
    def test_extract_main_content_fallback(self):
        """Test de fallback cuando no hay elementos principales claros"""
        simple_soup = BeautifulSoup(SIMPLE_HTML, 'html.parser')
        
        result = ContentCleaner.extract_main_content(simple_soup)
        
        assert "Main Article Title" in result
        assert "first paragraph" in result

# =============================================================================
# TESTS DE URL NORMALIZER
# =============================================================================

class TestURLNormalizer:
    """Tests para normalización de URLs"""
    
    def test_normalize_basic_url(self):
        """Test de normalización básica"""
        url = "http://example.com/path"
        
        result = URLNormalizer.normalize_url(url)
        
        assert result == "https://example.com/path"
    
    def test_normalize_removes_www(self):
        """Test que www se remueve"""
        url = "https://www.example.com/path"
        
        result = URLNormalizer.normalize_url(url)
        
        assert result == "https://example.com/path"
    
    def test_normalize_removes_trailing_slash(self):
        """Test que trailing slash se remueve (excepto root)"""
        url1 = "https://example.com/path/"
        url2 = "https://example.com/"
        
        result1 = URLNormalizer.normalize_url(url1)
        result2 = URLNormalizer.normalize_url(url2)
        
        assert result1 == "https://example.com/path"
        assert result2 == "https://example.com/"  # Root slash se mantiene
    
    def test_normalize_filters_tracking_params(self):
        """Test que parámetros de tracking se filtran"""
        url = "https://example.com/page?utm_source=google&utm_medium=cpc&useful_param=value"
        
        result = URLNormalizer.normalize_url(url)
        
        assert "utm_source" not in result
        assert "utm_medium" not in result
        assert "useful_param=value" in result
    
    def test_normalize_adds_protocol(self):
        """Test que protocolo se agrega si falta"""
        url = "example.com/path"
        
        result = URLNormalizer.normalize_url(url)
        
        assert result.startswith("https://")
        assert "example.com/path" in result
    
    def test_get_url_info_extracts_components(self):
        """Test de extracción de componentes de URL"""
        url = "https://subdomain.example.com/path/to/page?param=value#section"
        
        info = URLNormalizer.get_url_info(url)
        
        assert info.domain == "example.com"
        assert info.subdomain == "subdomain"
        assert info.path == "/path/to/page"
        assert info.query_params == {"param": "value"}
        assert info.is_secure == True
        assert info.is_valid == True
        assert len(info.url_hash) == 12
    
    def test_get_url_info_handles_invalid_url(self):
        """Test de manejo de URL inválida"""
        invalid_url = "not-a-url"
        
        info = URLNormalizer.get_url_info(invalid_url)
        
        assert info.is_valid == False
        assert info.domain == ""

# =============================================================================
# TESTS DE EXTRACTION RULES
# =============================================================================

class TestExtractionRules:
    """Tests para reglas de extracción"""
    
    def test_get_extraction_rule_returns_general_rule(self):
        """Test que devuelve regla general por defecto"""
        rule = get_extraction_rule(ContentType.TITLE)
        
        assert rule is not None
        assert "title" in rule.selectors
        assert "h1" in rule.selectors
        assert rule.required == True
    
    def test_get_extraction_rule_returns_domain_specific(self):
        """Test que devuelve regla específica del dominio"""
        rule = get_extraction_rule(ContentType.TITLE, "wikipedia.org")
        
        assert rule is not None
        assert "#firstHeading" in rule.selectors
    
    def test_get_domain_from_url(self):
        """Test de extracción de dominio de URL"""
        test_cases = [
            ("https://www.example.com/path", "www.example.com"),
            ("http://subdomain.test.org/page", "subdomain.test.org"),
            ("https://simple.com", "simple.com"),
            ("invalid-url", "")
        ]
        
        for url, expected_domain in test_cases:
            result = get_domain_from_url(url)
            assert result == expected_domain
    
    def test_is_content_valid_accepts_good_content(self):
        """Test que contenido válido es aceptado"""
        good_content = "This is a good piece of content with sufficient length and meaning."
        
        assert is_content_valid(good_content) == True
    
    def test_is_content_valid_rejects_bad_content(self):
        """Test que contenido inválido es rechazado"""
        test_cases = [
            "",  # Vacío
            "short",  # Muy corto
            "404 not found",  # Página de error
            "javascript required",  # Mensaje de JS requerido
            "   \n  \t  ",  # Solo espacios
        ]
        
        for bad_content in test_cases:
            assert is_content_valid(bad_content) == False
    
    def test_calculate_content_quality_score(self):
        """Test de cálculo de score de calidad"""
        # Contenido de alta calidad
        high_quality = """
        This is a well-written article about machine learning research. 
        The study shows significant improvements in accuracy. 
        According to the methodology section, the researchers used cross-validation.
        Therefore, these findings provide strong evidence for the proposed approach.
        However, further analysis is needed to confirm the results.
        """
        
        # Contenido de baja calidad
        low_quality = "Click here! Buy now! Amazing deal! Limited time!"
        
        high_score = calculate_content_quality_score(high_quality)
        low_score = calculate_content_quality_score(low_quality)
        
        assert high_score > low_score
        assert high_score > 0.5
        assert low_score < 0.5

# =============================================================================
# TESTS DE CONTENT EXTRACTOR
# =============================================================================

class TestContentExtractor:
    """Tests para el extractor de contenido principal"""
    
    @pytest.fixture
    def content_extractor(self):
        """Fixture del content extractor"""
        return ContentExtractor()
    
    @pytest.fixture
    def mock_web_response(self):
        """Mock de WebResponse"""
        mock_response = Mock()
        mock_response.success = True
        mock_response.content = SIMPLE_HTML
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html; charset=utf-8"}
        mock_response.final_url = "https://example.com/test"
        mock_response.response_time = 1.5
        return mock_response
    
    @pytest.mark.asyncio
    async def test_extract_content_from_html_simple(self, content_extractor):
        """Test de extracción de HTML simple"""
        url = "https://example.com/test"
        
        with patch('app.agents.tools.web.content_extractor.WebRequestManager') as mock_manager:
            # Setup mock
            mock_context = AsyncMock()
            mock_manager.return_value.__aenter__.return_value = mock_context
            mock_context.fetch_url.return_value = Mock(
                success=True,
                content=SIMPLE_HTML,
                status_code=200,
                headers={"content-type": "text/html"},
                final_url=url,
                response_time=1.0
            )
            
            result = await content_extractor.extract_content(url)
            
            assert result.extraction_success == True
            assert "Main Article Title" in result.title
            assert "simple test page" in result.description.lower()
            assert "Test Author" in result.author
            assert len(result.main_content) > 0
            assert result.language in ["english", "unknown"]
            assert result.word_count > 0
    
    @pytest.mark.asyncio
    async def test_extract_content_from_html_wikipedia(self, content_extractor):
        """Test de extracción específica de Wikipedia"""
        url = "https://en.wikipedia.org/wiki/Python"
        
        with patch('app.agents.tools.web.content_extractor.WebRequestManager') as mock_manager:
            mock_context = AsyncMock()
            mock_manager.return_value.__aenter__.return_value = mock_context
            mock_context.fetch_url.return_value = Mock(
                success=True,
                content=WIKIPEDIA_HTML,
                status_code=200,
                headers={"content-type": "text/html"},
                final_url=url,
                response_time=1.2
            )
            
            result = await content_extractor.extract_content(url, content_type_hint="article")
            
            assert result.extraction_success == True
            assert "Python (programming language)" in result.title
            assert "high-level" in result.main_content
            assert result.domain == "wikipedia.org"
    
    @pytest.mark.asyncio
    async def test_extract_content_handles_http_error(self, content_extractor):
        """Test de manejo de errores HTTP"""
        url = "https://example.com/notfound"
        
        with patch('app.agents.tools.web.content_extractor.WebRequestManager') as mock_manager:
            mock_context = AsyncMock()
            mock_manager.return_value.__aenter__.return_value = mock_context
            mock_context.fetch_url.return_value = Mock(
                success=False,
                content="",
                status_code=404,
                error="Not Found",
                final_url=url,
                response_time=0.5
            )
            
            result = await content_extractor.extract_content(url)
            
            assert result.extraction_success == False
            assert result.title == ""
            assert result.main_content == ""
            assert "404" in result.error or "Not Found" in result.error
    
    @pytest.mark.asyncio
    async def test_extract_content_handles_malformed_html(self, content_extractor):
        """Test de manejo de HTML mal formado"""
        url = "https://example.com/malformed"
        
        with patch('app.agents.tools.web.content_extractor.WebRequestManager') as mock_manager:
            mock_context = AsyncMock()
            mock_manager.return_value.__aenter__.return_value = mock_context
            mock_context.fetch_url.return_value = Mock(
                success=True,
                content=MALFORMED_HTML,
                status_code=200,
                headers={"content-type": "text/html"},
                final_url=url,
                response_time=1.0
            )
            
            result = await content_extractor.extract_content(url)
            
            # Debe manejar HTML mal formado sin crashear
            assert result.extraction_success == True
            assert len(result.main_content) > 0
            assert result.quality_score >= 0.0
    
    def test_extract_content_from_html_direct(self, content_extractor):
        """Test de extracción directa de HTML (método sincrónico)"""
        result = content_extractor.extract_content_from_html(
            html=COMPLEX_HTML,
            url="https://example.com/test",
            content_type_hint="article"
        )
        
        assert result.extraction_success == True
        assert "Main Article Title" in result.title
        assert "main content paragraph" in result.main_content
        assert result.domain == "example.com"
        assert result.quality_score > 0.0
    
    def test_extract_metadata_from_html(self, content_extractor):
        """Test de extracción de metadatos"""
        html_with_meta = """
        <html>
        <head>
            <title>Test Page</title>
            <meta name="description" content="Test description">
            <meta name="author" content="Test Author">
            <meta name="keywords" content="test, html, extraction">
            <meta property="og:title" content="OG Title">
            <meta property="article:published_time" content="2024-01-15T10:00:00Z">
        </head>
        <body>
            <h1>Main Content</h1>
            <p>Content paragraph.</p>
        </body>
        </html>
        """
        
        result = content_extractor.extract_content_from_html(
            html=html_with_meta,
            url="https://example.com/meta-test"
        )
        
        assert result.title == "Test Page"
        assert result.description == "Test description"
        assert result.author == "Test Author"
        assert "test" in result.keywords
        assert "html" in result.keywords
        assert result.publish_date == "2024-01-15T10:00:00Z"

# =============================================================================
# TESTS DE UTILIDADES
# =============================================================================

class TestContentUtilities:
    """Tests para funciones de utilidad"""
    
    def test_get_content_hash_consistent(self):
        """Test que hash de contenido es consistente"""
        content = "This is test content for hashing."
        
        hash1 = get_content_hash(content)
        hash2 = get_content_hash(content)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length
    
    def test_get_content_hash_different_for_different_content(self):
        """Test que contenido diferente produce hash diferente"""
        content1 = "First piece of content"
        content2 = "Second piece of content"
        
        hash1 = get_content_hash(content1)
        hash2 = get_content_hash(content2)
        
        assert hash1 != hash2
    
    def test_get_content_hash_handles_empty_content(self):
        """Test de manejo de contenido vacío"""
        empty_content = ""
        
        result = get_content_hash(empty_content)
        
        assert result == ""

# =============================================================================
# TESTS DE INTEGRATION CON BEAUTIFULSOUP
# =============================================================================

class TestBeautifulSoupIntegration:
    """Tests de integración con BeautifulSoup"""
    
    def test_handles_different_html_parsers(self):
        """Test que funciona con diferentes parsers de HTML"""
        html = SIMPLE_HTML
        
        # Test con diferentes parsers
        parsers = ['html.parser', 'lxml', 'html5lib']
        
        for parser in parsers:
            try:
                soup = BeautifulSoup(html, parser)
                content = ContentCleaner.extract_main_content(soup)
                assert len(content) > 0
            except Exception:
                # Si parser no está disponible, skip
                continue
    
    def test_handles_unicode_correctly(self):
        """Test de manejo correcto de Unicode con BeautifulSoup"""
        result = ContentCleaner.clean_html(UNICODE_HTML)
        
        assert "Título con Acentos" in result
        assert "áéíóú ñÑ" in result
        assert "🐍" in result
        assert "こんにちは" in result

# =============================================================================
# TESTS DE PERFORMANCE
# =============================================================================

class TestContentExtractionPerformance:
    """Tests de rendimiento para extracción de contenido"""
    
    @pytest.mark.performance
    def test_extraction_performance_simple_html(self, performance_tracker):
        """Test de performance para HTML simple"""
        performance_tracker['start']('simple_extraction')
        
        result = ContentCleaner.clean_html(SIMPLE_HTML)
        
        duration = performance_tracker['end']('simple_extraction')
        
        assert len(result) > 0
        assert duration < 1.0  # Menos de 1 segundo
    
    @pytest.mark.performance
    def test_extraction_performance_complex_html(self, performance_tracker):
        """Test de performance para HTML complejo"""
        performance_tracker['start']('complex_extraction')
        
        result = ContentCleaner.clean_html(COMPLEX_HTML)
        
        duration = performance_tracker['end']('complex_extraction')
        
        assert len(result) > 0
        assert duration < 2.0  # Menos de 2 segundos
    
    @pytest.mark.performance
    def test_url_normalization_performance(self, performance_tracker):
        """Test de performance para normalización de URLs"""
        urls = [
            "http://www.example.com/path/?utm_source=test&param=value",
            "https://subdomain.test.org/long/path/with/many/segments/",
            "ftp://old-protocol.com/file.txt",
            "//protocol-relative.com/path"
        ] * 100  # 400 URLs total
        
        performance_tracker['start']('url_normalization')
        
        results = [URLNormalizer.normalize_url(url) for url in urls]
        
        duration = performance_tracker['end']('url_normalization')
        
        assert len(results) == 400
        assert duration < 1.0  # Menos de 1 segundo para 400 URLs

# =============================================================================
# TESTS DE EDGE CASES
# =============================================================================

class TestContentExtractionEdgeCases:
    """Tests para casos extremos"""
    
    def test_extract_from_empty_html(self):
        """Test de extracción de HTML vacío"""
        result = ContentCleaner.clean_html("")
        assert result == ""
    
    def test_extract_from_none_html(self):
        """Test de extracción de HTML None"""
        result = ContentCleaner.clean_html(None)
        assert result == ""
    
    def test_extract_from_very_large_html(self):
        """Test de extracción de HTML muy grande"""
        large_html = f"<html><body>{'<p>Large content paragraph.</p>' * 1000}</body></html>"
        
        result = ContentCleaner.clean_html(large_html)
        
        assert len(result) > 0
        assert "Large content paragraph" in result
    
    def test_extract_from_html_with_only_scripts(self):
        """Test de HTML que solo contiene scripts"""
        script_only_html = """
        <html>
        <head>
            <script>console.log('test');</script>
        </head>
        <body>
            <script>alert('more scripts');</script>
        </body>
        </html>
        """
        
        result = ContentCleaner.clean_html(script_only_html)
        
        # Debe remover scripts y retornar cadena vacía o casi vacía
        assert len(result.strip()) == 0 or "console.log" not in result
    
    def test_normalize_already_normalized_url(self):
        """Test de normalización de URL ya normalizada"""
        clean_url = "https://example.com/path"
        
        result = URLNormalizer.normalize_url(clean_url)
        
        assert result == clean_url