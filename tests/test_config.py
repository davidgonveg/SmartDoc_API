"""
Test Configuration for SmartDoc Research Agent
Configuraciones centralizadas para todos los tests
"""

import os
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# PATHS Y DIRECTORIOS
# =============================================================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
AGENT_API_ROOT = PROJECT_ROOT / "agent-api"
TESTS_ROOT = PROJECT_ROOT / "tests"

# Directorios de datos de testing
TEST_DATA_DIR = TESTS_ROOT / "data"
FIXTURES_DIR = TESTS_ROOT / "fixtures"
MOCK_DATA_DIR = TESTS_ROOT / "mock_data"

# Crear directorios si no existen
TEST_DATA_DIR.mkdir(exist_ok=True)
FIXTURES_DIR.mkdir(exist_ok=True)
MOCK_DATA_DIR.mkdir(exist_ok=True)

# =============================================================================
# CONFIGURACIÓN DE SERVICIOS
# =============================================================================

@dataclass
class TestServiceConfig:
    """Configuración para servicios en testing"""
    
    # FastAPI
    api_base_url: str = "http://localhost:8001"
    api_timeout: int = 30
    
    # Streamlit  
    ui_base_url: str = "http://localhost:8501"
    ui_timeout: int = 10
    
    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_timeout: int = 60
    ollama_test_model: str = "llama3.2:3b"
    
    # ChromaDB
    chromadb_host: str = "localhost"
    chromadb_port: int = 8000
    chromadb_timeout: int = 15
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_timeout: int = 5

# Instancia global de configuración
TEST_SERVICES = TestServiceConfig()

# =============================================================================
# CONFIGURACIÓN DE WEB SEARCH TESTING
# =============================================================================

class WebSearchTestConfig:
    """Configuración específica para testing de web search"""
    
    # Queries de testing seguras
    SAFE_TEST_QUERIES = [
        "python programming",
        "machine learning basics",
        "weather forecast",
        "latest technology news",
        "cooking recipes pasta"
    ]
    
    # Queries que pueden fallar (para testing de errores)
    FAILING_TEST_QUERIES = [
        "askjdhlaksjdhlaksjdh",  # Query sin sentido
        "site:nonexistent-domain-12345.com",  # Sitio inexistente
        ""  # Query vacía
    ]
    
    # URLs de testing confiables
    RELIABLE_TEST_URLS = [
        "https://httpbin.org/html",
        "https://example.com",
        "https://www.python.org",
        "https://en.wikipedia.org/wiki/Python_(programming_language)"
    ]
    
    # URLs que pueden fallar
    UNRELIABLE_TEST_URLS = [
        "https://httpstat.us/404",  # 404 error
        "https://httpstat.us/500",  # 500 error
        "https://this-domain-does-not-exist-12345.com"  # Domain error
    ]
    
    # Timeouts para testing
    QUICK_TIMEOUT = 5  # Para tests rápidos
    NORMAL_TIMEOUT = 15  # Para tests normales
    SLOW_TIMEOUT = 30  # Para tests que pueden tardar
    
    # Rate limiting
    TEST_RATE_LIMIT = 10  # Requests por minuto en testing
    
    # Content extraction
    MIN_CONTENT_LENGTH = 10
    MAX_CONTENT_LENGTH = 1000  # Para testing, no necesitamos contenido largo

WEB_SEARCH_CONFIG = WebSearchTestConfig()

# =============================================================================
# CONFIGURACIÓN DE AGENT TESTING
# =============================================================================

class AgentTestConfig:
    """Configuración para testing del SmartDoc Agent"""
    
    # Configuración de modelo
    TEST_MODEL = "llama3.2:3b"
    FALLBACK_MODEL = "llama3.2:1b"  # Si el principal no está disponible
    
    # Configuración de generación
    MAX_TOKENS = 512  # Tokens limitados para testing rápido
    TEMPERATURE = 0.1  # Baja para resultados predecibles
    
    # Timeouts
    AGENT_INIT_TIMEOUT = 30
    AGENT_RESPONSE_TIMEOUT = 60
    
    # Queries de testing para el agent
    SIMPLE_QUERIES = [
        "¿Qué es Python?",
        "Explica machine learning",
        "¿Cómo funciona internet?"
    ]
    
    COMPLEX_QUERIES = [
        "Investiga los últimos avances en inteligencia artificial",
        "Compara diferentes frameworks de machine learning",
        "Analiza las tendencias tecnológicas para 2024"
    ]
    
    # Configuración de herramientas
    ENABLE_WEB_SEARCH = True
    ENABLE_PDF_READER = False  # Aún no implementado
    ENABLE_CALCULATOR = False  # Aún no implementado
    
    # Research session config
    DEFAULT_MAX_SOURCES = 5  # Menos fuentes para testing rápido
    DEFAULT_RESEARCH_DEPTH = "basic"

AGENT_CONFIG = AgentTestConfig()

# =============================================================================
# CONFIGURACIÓN DE MOCKING
# =============================================================================

class MockConfig:
    """Configuración para mocks y fixtures"""
    
    # Mock responses para diferentes escenarios
    MOCK_OLLAMA_RESPONSE = {
        "model": "llama3.2:3b",
        "response": "This is a test response from Ollama.",
        "done": True,
        "total_duration": 1000000000,
        "load_duration": 500000000,
        "prompt_eval_count": 10,
        "prompt_eval_duration": 200000000,
        "eval_count": 20,
        "eval_duration": 300000000
    }
    
    MOCK_SEARCH_RESULTS = [
        {
            "title": "Test Result 1",
            "url": "https://example.com/1",
            "snippet": "This is a test search result snippet.",
            "domain": "example.com",
            "rank": 1
        },
        {
            "title": "Test Result 2", 
            "url": "https://test.com/2",
            "snippet": "Another test search result for validation.",
            "domain": "test.com",
            "rank": 2
        }
    ]
    
    MOCK_HTML_CONTENT = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Test Page</title>
        <meta name="description" content="Test page description">
    </head>
    <body>
        <h1>Test Article Title</h1>
        <p>This is test content for extraction testing.</p>
        <p>Multiple paragraphs help test content cleaning.</p>
    </body>
    </html>
    """
    
    # HTTP Status codes para testing
    HTTP_SUCCESS_CODES = [200, 201, 202]
    HTTP_CLIENT_ERROR_CODES = [400, 401, 403, 404]
    HTTP_SERVER_ERROR_CODES = [500, 502, 503, 504]

MOCK_CONFIG = MockConfig()

# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================

class PytestConfig:
    """Configuración específica de pytest"""
    
    # Markers personalizados
    MARKERS = [
        "unit: Unit tests - fast, no external dependencies",
        "integration: Integration tests - may call external services", 
        "e2e: End-to-end tests - require full system running",
        "slow: Tests that take more than 10 seconds",
        "web: Tests that require internet connection",
        "ollama: Tests that require Ollama running",
        "agent: Tests specific to SmartDoc Agent"
    ]
    
    # Configuración de asyncio
    ASYNCIO_MODE = "auto"
    
    # Directorio de tests
    TESTPATHS = ["tests"]
    
    # Patrones de archivos de test
    PYTHON_FILES = ["test_*.py", "*_test.py"]
    PYTHON_CLASSES = ["Test*"]
    PYTHON_FUNCTIONS = ["test_*"]
    
    # Opciones por defecto
    ADDOPTS = [
        "-v",  # Verbose
        "--tb=short",  # Short traceback format
        "--strict-markers",  # Strict marker checking
        "--disable-warnings",  # Disable warnings (optional)
    ]

PYTEST_CONFIG = PytestConfig()

# =============================================================================
# ENVIRONMENT VARIABLES PARA TESTING
# =============================================================================

def get_test_env_vars() -> Dict[str, str]:
    """Obtener variables de entorno para testing"""
    return {
        # Configuración de testing
        "ENVIRONMENT": "testing",
        "LOG_LEVEL": "DEBUG",
        
        # Servicios
        "OLLAMA_HOST": TEST_SERVICES.ollama_base_url.replace("http://", "").split(":")[0],
        "OLLAMA_PORT": "11434",
        "CHROMADB_HOST": TEST_SERVICES.chromadb_host,
        "CHROMADB_PORT": str(TEST_SERVICES.chromadb_port),
        "REDIS_HOST": TEST_SERVICES.redis_host,
        "REDIS_PORT": str(TEST_SERVICES.redis_port),
        
        # Agent config
        "DEFAULT_MODEL": AGENT_CONFIG.TEST_MODEL,
        "MAX_TOKENS": str(AGENT_CONFIG.MAX_TOKENS),
        "TEMPERATURE": str(AGENT_CONFIG.TEMPERATURE),
        
        # Web search config
        "USE_MOCK_RESPONSES": "true",
        "TEST_MODE": "true",
        "RATE_LIMIT_TESTING": "true",
        
        # API Keys (vacías para testing)
        "GOOGLE_SEARCH_API_KEY": "",
        "BING_SEARCH_API_KEY": "",
    }

# =============================================================================
# UTILIDADES DE TESTING
# =============================================================================

def is_service_available(url: str, timeout: int = 5) -> bool:
    """Verificar si un servicio está disponible"""
    import httpx
    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code < 500
    except Exception:
        return False

def skip_if_service_unavailable(service_name: str, url: str):
    """Decorator para skip tests si servicio no disponible"""
    import pytest
    return pytest.mark.skipif(
        not is_service_available(url),
        reason=f"{service_name} service not available at {url}"
    )

def requires_internet():
    """Decorator para tests que requieren internet"""
    import pytest
    return pytest.mark.web

def requires_ollama():
    """Decorator para tests que requieren Ollama"""
    import pytest
    return pytest.mark.skipif(
        not is_service_available(TEST_SERVICES.ollama_base_url),
        reason="Ollama not available"
    )

# =============================================================================
# CONFIGURACIÓN DE LOGGING PARA TESTING
# =============================================================================

import logging

def setup_test_logging():
    """Configurar logging para testing"""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(TESTS_ROOT / "test.log")
        ]
    )
    
    # Silenciar logs muy verbosos durante testing
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

# =============================================================================
# VALIDACIÓN DE CONFIGURACIÓN
# =============================================================================

def validate_test_config():
    """Validar que la configuración de testing es correcta"""
    errors = []
    
    # Verificar directorios
    if not PROJECT_ROOT.exists():
        errors.append(f"Project root not found: {PROJECT_ROOT}")
    
    if not AGENT_API_ROOT.exists():
        errors.append(f"Agent API root not found: {AGENT_API_ROOT}")
    
    # Verificar configuración básica
    if not TEST_SERVICES.api_base_url:
        errors.append("API base URL not configured")
    
    if not AGENT_CONFIG.TEST_MODEL:
        errors.append("Test model not configured")
    
    if errors:
        raise ValueError(f"Test configuration errors: {errors}")
    
    return True

# Validar configuración al importar
validate_test_config()

# =============================================================================
# EXPORTS PARA FÁCIL IMPORTACIÓN
# =============================================================================

__all__ = [
    # Paths
    "PROJECT_ROOT",
    "AGENT_API_ROOT", 
    "TESTS_ROOT",
    "TEST_DATA_DIR",
    "FIXTURES_DIR",
    "MOCK_DATA_DIR",
    
    # Configs
    "TEST_SERVICES",
    "WEB_SEARCH_CONFIG",
    "AGENT_CONFIG",
    "MOCK_CONFIG",
    "PYTEST_CONFIG",
    
    # Utilities
    "get_test_env_vars",
    "is_service_available",
    "skip_if_service_unavailable", 
    "requires_internet",
    "requires_ollama",
    "setup_test_logging",
    "validate_test_config",
]