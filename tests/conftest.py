"""
Pytest Configuration and Global Fixtures
Configuración global y fixtures compartidas para todos los tests
"""

import pytest
import asyncio
import httpx
import logging
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, patch
import os
import sys

# Agregar el directorio agent-api al path para imports
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), '..', 'agent-api')))

from test_config import (
    TEST_SERVICES,
    WEB_SEARCH_CONFIG,
    AGENT_CONFIG,
    MOCK_CONFIG,
    get_test_env_vars,
    setup_test_logging,
    is_service_available
)

# =============================================================================
# CONFIGURACIÓN GLOBAL DE PYTEST
# =============================================================================

def pytest_configure(config):
    """Configuración global de pytest"""
    # Setup logging
    setup_test_logging()
    
    # Configurar variables de entorno para testing
    test_env = get_test_env_vars()
    for key, value in test_env.items():
        os.environ[key] = value
    
    # Registrar markers personalizados
    config.addinivalue_line("markers", "unit: Unit tests - fast, no external dependencies")
    config.addinivalue_line("markers", "integration: Integration tests - may call external services")
    config.addinivalue_line("markers", "e2e: End-to-end tests - require full system running")
    config.addinivalue_line("markers", "slow: Tests that take more than 10 seconds")
    config.addinivalue_line("markers", "web: Tests that require internet connection")
    config.addinivalue_line("markers", "ollama: Tests that require Ollama running")
    config.addinivalue_line("markers", "agent: Tests specific to SmartDoc Agent")

def pytest_collection_modifyitems(config, items):
    """Modificar items de test collection"""
    for item in items:
        # Auto-marcar tests basado en nombre de archivo
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Auto-marcar tests que requieren servicios externos
        if "web_search" in str(item.fspath) or "web" in item.name:
            item.add_marker(pytest.mark.web)
        
        if "agent" in str(item.fspath) or "ollama" in item.name:
            item.add_marker(pytest.mark.ollama)

# =============================================================================
# FIXTURES DE CONFIGURACIÓN
# =============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Fixture con toda la configuración de testing"""
    return {
        'services': TEST_SERVICES,
        'web_search': WEB_SEARCH_CONFIG,
        'agent': AGENT_CONFIG,
        'mock': MOCK_CONFIG
    }

@pytest.fixture(scope="session")
def event_loop():
    """Fixture para event loop de asyncio"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# =============================================================================
# FIXTURES DE SERVICIOS EXTERNOS
# =============================================================================

@pytest.fixture(scope="session")
async def http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Cliente HTTP compartido para todos los tests"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        yield client

@pytest.fixture(scope="function")
def mock_http_client():
    """Mock del cliente HTTP para tests unitarios"""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    
    # Configurar respuestas por defecto
    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.text = MOCK_CONFIG.MOCK_HTML_CONTENT
    mock_response.json.return_value = {"status": "ok"}
    mock_response.headers = {"content-type": "text/html"}
    
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response
    
    return mock_client

@pytest.fixture(scope="session")
def check_services():
    """Verificar qué servicios están disponibles"""
    services_status = {
        'api': is_service_available(f"{TEST_SERVICES.api_base_url}/health"),
        'ollama': is_service_available(f"{TEST_SERVICES.ollama_base_url}/api/version"),
        'chromadb': is_service_available(f"http://{TEST_SERVICES.chromadb_host}:{TEST_SERVICES.chromadb_port}/api/v1/heartbeat"),
        'redis': False,  # Verificación de Redis requiere cliente específico
        'internet': is_service_available("https://httpbin.org/get", timeout=5)
    }
    
    # Log status de servicios
    logger = logging.getLogger(__name__)
    logger.info(f"Services availability: {services_status}")
    
    return services_status

# =============================================================================
# FIXTURES DE OLLAMA
# =============================================================================

@pytest.fixture(scope="session")
def ollama_available(check_services):
    """Verificar si Ollama está disponible"""
    return check_services['ollama']

@pytest.fixture(scope="function")
def mock_ollama_client():
    """Mock del cliente Ollama"""
    mock_client = AsyncMock()
    
    # Mock health check
    mock_client.health_check.return_value = True
    
    # Mock generate response
    mock_client.generate.return_value = {
        "success": True,
        "response": "This is a mocked response from Ollama.",
        "model": AGENT_CONFIG.TEST_MODEL,
        "total_duration": 1000000000
    }
    
    return mock_client

@pytest.fixture(scope="function")
async def ollama_client():
    """Cliente Ollama real (requiere servicio corriendo)"""
    pytest.importorskip("app.services.ollama_client")
    
    from app.services.ollama_client import get_ollama_client
    
    # Skip si Ollama no está disponible
    if not is_service_available(TEST_SERVICES.ollama_base_url):
        pytest.skip("Ollama service not available")
    
    client = await get_ollama_client()
    
    # Verificar que el modelo de test está disponible
    health_ok = await client.health_check()
    if not health_ok:
        pytest.skip("Ollama health check failed")
    
    yield client
    await client.close()

# =============================================================================
# FIXTURES DE WEB SEARCH
# =============================================================================

@pytest.fixture(scope="function")
def mock_search_engine():
    """Mock de un motor de búsqueda"""
    mock_engine = AsyncMock()
    
    # Mock search response
    mock_engine.search.return_value = Mock(
        success=True,
        results=MOCK_CONFIG.MOCK_SEARCH_RESULTS,
        total_results=len(MOCK_CONFIG.MOCK_SEARCH_RESULTS),
        search_time=0.5,
        engine="mock_engine",
        error_message=None
    )
    
    return mock_engine

@pytest.fixture(scope="function")
def mock_content_extractor():
    """Mock del extractor de contenido"""
    mock_extractor = AsyncMock()
    
    # Mock extracted content
    mock_extractor.extract_content.return_value = Mock(
        main_content="This is extracted content from a web page.",
        title="Test Page Title",
        author="Test Author",
        publish_date="2024-01-01",
        language="english",
        word_count=50,
        reading_time=1,
        quality_score=0.8,
        extraction_success=True
    )
    
    return mock_extractor

@pytest.fixture(scope="function")
def web_search_tool_mock():
    """Mock completo del WebSearchTool"""
    with patch('app.agents.tools.web.web_search_tool.WebSearchTool') as mock_tool:
        # Configurar el mock
        mock_instance = mock_tool.return_value
        mock_instance._arun.return_value = "Mocked web search results"
        mock_instance.get_tool_info.return_value = {
            "name": "web_search",
            "description": "Mocked web search tool",
            "version": "1.0.0"
        }
        
        yield mock_instance

# =============================================================================
# FIXTURES DE AGENT
# =============================================================================

@pytest.fixture(scope="function")
def mock_smartdoc_agent():
    """Mock del SmartDoc Agent"""
    mock_agent = AsyncMock()
    
    # Mock initialization
    mock_agent.initialize.return_value = True
    mock_agent.is_initialized = True
    
    # Mock session creation
    mock_agent.create_research_session.return_value = "test-session-123"
    
    # Mock query processing
    mock_agent.process_query.return_value = {
        "response": "This is a mocked agent response.",
        "sources": [
            {"type": "web", "url": "https://example.com", "relevance": 0.9}
        ],
        "reasoning": "Mocked reasoning steps",
        "confidence": 0.8,
        "session_id": "test-session-123"
    }
    
    # Mock session status
    mock_agent.get_session_status.return_value = {
        "session_id": "test-session-123",
        "topic": "test topic",
        "message_count": 1,
        "created_at": "2024-01-01T00:00:00"
    }
    
    return mock_agent

@pytest.fixture(scope="function")
async def real_smartdoc_agent(ollama_available):
    """SmartDoc Agent real (requiere Ollama)"""
    if not ollama_available:
        pytest.skip("Ollama not available for agent testing")
    
    pytest.importorskip("app.agents.core.smart_agent")
    
    from app.agents.core.smart_agent import create_smartdoc_agent
    
    try:
        agent = await create_smartdoc_agent(
            model_name=AGENT_CONFIG.TEST_MODEL,
            research_style="general",
            max_iterations=3  # Menos iteraciones para testing rápido
        )
        yield agent
        await agent.close()
    except Exception as e:
        pytest.skip(f"Could not create SmartDoc agent: {e}")

# =============================================================================
# FIXTURES DE DATOS DE TESTING
# =============================================================================

@pytest.fixture(scope="function")
def sample_html_content():
    """Contenido HTML de ejemplo para tests"""
    return MOCK_CONFIG.MOCK_HTML_CONTENT

@pytest.fixture(scope="function")
def sample_search_query():
    """Query de búsqueda de ejemplo"""
    return WEB_SEARCH_CONFIG.SAFE_TEST_QUERIES[0]

@pytest.fixture(scope="function")
def sample_search_results():
    """Resultados de búsqueda de ejemplo"""
    return MOCK_CONFIG.MOCK_SEARCH_RESULTS.copy()

@pytest.fixture(scope="function")
def test_urls():
    """URLs de testing"""
    return {
        'reliable': WEB_SEARCH_CONFIG.RELIABLE_TEST_URLS,
        'unreliable': WEB_SEARCH_CONFIG.UNRELIABLE_TEST_URLS
    }

# =============================================================================
# FIXTURES DE SESSION Y CLEANUP
# =============================================================================

@pytest.fixture(scope="function")
def research_session_id():
    """ID de sesión de investigación para testing"""
    return "test-session-" + str(hash("test"))[:8]

@pytest.fixture(scope="function")
def cleanup_test_files():
    """Limpiar archivos temporales después del test"""
    temp_files = []
    
    def add_temp_file(filepath):
        temp_files.append(filepath)
    
    yield add_temp_file
    
    # Cleanup
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            logging.warning(f"Could not clean up temp file {filepath}: {e}")

# =============================================================================
# FIXTURES DE ERROR TESTING
# =============================================================================

@pytest.fixture(scope="function")
def http_error_responses():
    """Respuestas HTTP de error para testing"""
    return {
        'not_found': Mock(status_code=404, text="Not Found"),
        'server_error': Mock(status_code=500, text="Internal Server Error"),
        'timeout': httpx.TimeoutException("Request timeout"),
        'connection_error': httpx.ConnectError("Connection failed")
    }

@pytest.fixture(scope="function")
def failing_search_scenarios():
    """Escenarios de fallo para web search testing"""
    return {
        'empty_query': "",
        'invalid_query': "askjdhlaksjdhlaksjdh",
        'blocked_domain': "https://blocked-domain.com",
        'timeout_url': "https://httpstat.us/408"
    }

# =============================================================================
# FIXTURES DE PERFORMANCE TESTING
# =============================================================================

@pytest.fixture(scope="function")
def performance_tracker():
    """Tracker para medir performance de tests"""
    import time
    
    times = {}
    
    def start_timer(name: str):
        times[name] = time.time()
    
    def end_timer(name: str) -> float:
        if name in times:
            duration = time.time() - times[name]
            del times[name]
            return duration
        return 0.0
    
    return {
        'start': start_timer,
        'end': end_timer,
        'times': times
    }

# =============================================================================
# SKIP CONDITIONS
# =============================================================================

skip_if_no_internet = pytest.mark.skipif(
    not is_service_available("https://httpbin.org/get", timeout=5),
    reason="No internet connection available"
)

skip_if_no_ollama = pytest.mark.skipif(
    not is_service_available(TEST_SERVICES.ollama_base_url),
    reason="Ollama service not available"
)

skip_if_no_api = pytest.mark.skipif(
    not is_service_available(f"{TEST_SERVICES.api_base_url}/health"),
    reason="SmartDoc API not available"
)

# =============================================================================
# PARAMETRIZED FIXTURES
# =============================================================================

@pytest.fixture(params=WEB_SEARCH_CONFIG.SAFE_TEST_QUERIES)
def test_query(request):
    """Parametrized fixture con diferentes queries de test"""
    return request.param

@pytest.fixture(params=["duckduckgo", "searx"])
def search_engine_name(request):
    """Parametrized fixture con diferentes motores de búsqueda"""
    return request.param

@pytest.fixture(params=["general", "academic", "technical"])
def query_type(request):
    """Parametrized fixture con diferentes tipos de query"""
    return request.param

# =============================================================================
# LOGGING HELPERS
# =============================================================================

@pytest.fixture(scope="function")
def test_logger():
    """Logger específico para testing"""
    logger = logging.getLogger("test")
    logger.setLevel(logging.DEBUG)
    return logger

# =============================================================================
# EXPORT DE FIXTURES IMPORTANTES
# =============================================================================

__all__ = [
    # Configuración
    "test_config",
    "check_services",
    
    # HTTP
    "http_client",
    "mock_http_client",
    
    # Ollama
    "ollama_available",
    "mock_ollama_client",
    "ollama_client",
    
    # Web Search
    "mock_search_engine",
    "mock_content_extractor", 
    "web_search_tool_mock",
    
    # Agent
    "mock_smartdoc_agent",
    "real_smartdoc_agent",
    
    # Datos
    "sample_html_content",
    "sample_search_query",
    "sample_search_results",
    "test_urls",
    
    # Session
    "research_session_id",
    "cleanup_test_files",
    
    # Error testing
    "http_error_responses",
    "failing_search_scenarios",
    
    # Performance
    "performance_tracker",
    
    # Skip conditions
    "skip_if_no_internet",
    "skip_if_no_ollama", 
    "skip_if_no_api",
    
    # Parametrized
    "test_query",
    "search_engine_name",
    "query_type",
    
    # Logging
    "test_logger"
]