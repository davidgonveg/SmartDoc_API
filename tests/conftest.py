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
from pathlib import Path

# Agregar directorios al path para imports
project_root = Path(__file__).parent.parent
agent_api_root = project_root / "agent-api"
tests_root = Path(__file__).parent

# Añadir paths necesarios
for path in [str(project_root), str(agent_api_root), str(tests_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Importar configuración de tests con fallback
try:
    from test_config import (
        TEST_SERVICES,
        WEB_SEARCH_CONFIG,
        AGENT_CONFIG,
        MOCK_CONFIG,
        get_test_env_vars,
        setup_test_logging,
        is_service_available
    )
except ImportError:
    # Fallback configuration si test_config no se puede importar
    from dataclasses import dataclass
    
    @dataclass
    class TestServiceConfig:
        api_base_url: str = "http://localhost:8001"
        api_timeout: int = 30
        ollama_base_url: str = "http://localhost:11434"
        ollama_timeout: int = 60
    
    @dataclass
    class WebSearchConfig:
        SAFE_TEST_QUERIES: list = None
        RELIABLE_TEST_URLS: list = None
        UNRELIABLE_TEST_URLS: list = None
        
        def __post_init__(self):
            self.SAFE_TEST_QUERIES = ["python programming", "test query"]
            self.RELIABLE_TEST_URLS = ["https://example.com"]
            self.UNRELIABLE_TEST_URLS = ["https://nonexistent.com"]
    
    @dataclass
    class AgentConfig:
        TEST_MODEL: str = "llama3.2:3b"
        DEFAULT_TIMEOUT: int = 30
    
    @dataclass 
    class MockConfig:
        MOCK_SEARCH_RESULTS: list = None
        
        def __post_init__(self):
            self.MOCK_SEARCH_RESULTS = [
                {"title": "Test Result", "url": "https://test.com", "snippet": "Test snippet"}
            ]
    
    # Crear instancias fallback
    TEST_SERVICES = TestServiceConfig()
    WEB_SEARCH_CONFIG = WebSearchConfig()
    AGENT_CONFIG = AgentConfig()
    MOCK_CONFIG = MockConfig()
    
    def get_test_env_vars():
        return {"TEST_ENV": "true", "LOG_LEVEL": "INFO"}
    
    def setup_test_logging():
        logging.basicConfig(level=logging.INFO)
    
    def is_service_available(url):
        return False

# =============================================================================
# CONFIGURACIÓN GLOBAL DE PYTEST
# =============================================================================

def pytest_configure(config):
    """Configuración global de pytest"""
    setup_test_logging()
    
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
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        if "web_search" in str(item.fspath) or "web" in item.name:
            item.add_marker(pytest.mark.web)
        
        if "agent" in str(item.fspath) or "ollama" in item.name:
            item.add_marker(pytest.mark.ollama)

# =============================================================================
# FIXTURES BÁSICAS
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

@pytest.fixture(scope="function")
def mock_http_client():
    """Mock del cliente HTTP para tests unitarios"""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>Test content</body></html>"
    mock_response.json.return_value = {"status": "ok", "data": "test"}
    
    mock_client.get.return_value = mock_response
    mock_client.post.return_value = mock_response
    
    return mock_client

@pytest.fixture(scope="function")
def mock_smart_agent():
    """Mock del SmartDocAgent"""
    mock_agent = AsyncMock()
    
    mock_agent.is_initialized = True
    mock_agent.active_sessions = {}
    mock_agent.tools = []
    
    mock_agent.create_research_session.return_value = {
        "session_id": "test-session-123",
        "topic": "test topic",
        "status": "created",
        "created_at": "2024-01-01T00:00:00Z"
    }
    
    mock_agent.chat.return_value = {
        "response": "This is a test response from the SmartDoc agent.",
        "sources": [{"type": "web", "url": "https://example.com", "title": "Test Source"}],
        "reasoning": "Test reasoning steps",
        "confidence": 0.85
    }
    
    return mock_agent

@pytest.fixture(scope="function")
def research_session_id():
    """ID de sesión de investigación para testing"""
    return "test-session-" + str(hash("test"))[:8]

# =============================================================================
# MISSING FIXTURES - Added by final fix
# =============================================================================

@pytest.fixture(scope="function")
def performance_tracker():
    """Simple performance tracking fixture"""
    tracking = {}
    
    def start_timing(key):
        import time
        tracking[key] = time.time()
    
    def end_timing(key):
        import time
        if key in tracking:
            return time.time() - tracking[key]
        return 0.0
    
    return {
        'start': start_timing,
        'end': end_timing
    }

@pytest.fixture(scope="function")
async def fully_mocked_agent():
    """Fully mocked agent with all dependencies mocked"""
    from unittest.mock import AsyncMock, Mock
    
    # Mock Ollama client
    mock_ollama = AsyncMock()
    mock_ollama.health_check.return_value = True
    mock_ollama.generate.return_value = {
        "success": True,
        "response": "Mock response from fully mocked agent",
        "model": "test-model"
    }
    
    # Mock web search tool
    mock_web_tool = AsyncMock()
    mock_web_tool.name = "web_search"
    mock_web_tool._arun.return_value = "Mock web search results"
    
    # Mock agent
    mock_agent = AsyncMock()
    mock_agent.initialized = True
    mock_agent.tools = [mock_web_tool]
    mock_agent.llm = mock_ollama
    
    # Mock methods
    mock_agent.initialize.return_value = None
    mock_agent.create_research_session.return_value = "test-session-123"
    mock_agent.process_query.return_value = {
        "success": True,
        "response": "Mock query response",
        "sources": []
    }
    mock_agent.get_session_status.return_value = {
        "message_count": 2,
        "status": "active"
    }
    mock_agent.close.return_value = None
    
    return mock_agent, mock_ollama, mock_web_tool

