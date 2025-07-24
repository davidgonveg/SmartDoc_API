"""
SmartDoc Research Agent - Unit Tests Package

Este paquete contiene tests unitarios para el agente de investigación SmartDoc.
Los tests unitarios se enfocan en probar componentes individuales de forma aislada.

Estructura de tests unitarios:
    test_smart_agent.py         # Tests del SmartDocAgent core
    test_web_search_tool.py     # Tests del WebSearchTool
    test_content_extraction.py  # Tests de extracción de contenido
    test_search_engines.py      # Tests de motores de búsqueda
    test_rate_limiter.py        # Tests del rate limiter
    test_user_agents.py         # Tests de user agents
    test_web_utils.py           # Tests de utilidades web

Convenciones para tests unitarios:
- Prefijo test_ para funciones de test
- Uso extensivo de mocks para dependencias externas
- Tests rápidos (< 1 segundo cada uno)
- Sin dependencias de servicios externos
- Cobertura completa de casos edge

Ejecución:
    pytest tests/unit/                    # Todos los tests unitarios
    pytest tests/unit/test_smart_agent.py # Test específico
    pytest -m unit                        # Solo tests marcados como unit
"""

import os
import sys
import pytest
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Agregar paths para imports
project_root = Path(__file__).parent.parent.parent
agent_api_root = project_root / "agent-api"
tests_root = project_root / "tests"

for path in [str(project_root), str(agent_api_root), str(tests_root)]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Configuración específica para tests unitarios
UNIT_TEST_CONFIG = {
    "timeout": 5.0,                    # Tests unitarios deben ser rápidos
    "max_iterations": 2,               # Límite para evitar loops infinitos
    "mock_external_services": True,    # Siempre mockear servicios externos
    "enable_logging": False,           # Desactivar logs por defecto
    "use_real_network": False          # Nunca usar red real en unit tests
}

# Mock data para tests unitarios
UNIT_TEST_MOCK_DATA = {
    "web_search_results": [
        {
            "title": "Unit Test Result 1",
            "url": "https://example.com/unit-test-1",
            "snippet": "This is a mock search result for unit testing.",
            "score": 0.95
        },
        {
            "title": "Unit Test Result 2", 
            "url": "https://example.com/unit-test-2",
            "snippet": "Another mock search result for comprehensive testing.",
            "score": 0.87
        }
    ],
    "agent_responses": {
        "simple": "This is a simple mock response from the agent.",
        "complex": "This is a more complex mock response that includes multiple sentences and covers various aspects of the query.",
        "with_sources": "Based on the available sources, here is a comprehensive response with references."
    },
    "session_data": {
        "session_id": "unit-test-session-123",
        "topic": "Unit Test Research Topic",
        "status": "active",
        "created_at": "2024-01-01T00:00:00Z"
    }
}

# =============================================================================
# HELPER FUNCTIONS PARA UNIT TESTS
# =============================================================================

def get_mock_agent():
    """Crear un mock del SmartDocAgent para tests unitarios"""
    mock_agent = AsyncMock()
    mock_agent.is_initialized = True
    mock_agent.model_name = "test-model"
    mock_agent.max_iterations = UNIT_TEST_CONFIG["max_iterations"]
    mock_agent.tools = []
    mock_agent.active_sessions = {}
    
    # Mock métodos principales
    mock_agent.initialize.return_value = None
    mock_agent.create_research_session.return_value = UNIT_TEST_MOCK_DATA["session_data"]["session_id"]
    mock_agent.process_query.return_value = {
        "success": True,
        "response": UNIT_TEST_MOCK_DATA["agent_responses"]["simple"],
        "sources": [],
        "reasoning": "Unit test reasoning",
        "confidence": 0.85
    }
    
    return mock_agent

def get_mock_web_search_tool():
    """Crear un mock del WebSearchTool para tests unitarios"""
    mock_tool = AsyncMock()
    mock_tool.name = "web_search"
    mock_tool.description = "Mock web search tool for unit testing"
    mock_tool.version = "1.0.0"
    mock_tool.enabled = True
    
    # Mock del método principal
    mock_tool._arun.return_value = "Mock web search results: " + str(UNIT_TEST_MOCK_DATA["web_search_results"])
    mock_tool.search.return_value = UNIT_TEST_MOCK_DATA["web_search_results"]
    
    return mock_tool

def get_mock_ollama_client():
    """Crear un mock del cliente Ollama para tests unitarios"""
    mock_client = AsyncMock()
    mock_client.health_check.return_value = True
    mock_client.generate.return_value = {
        "success": True,
        "response": UNIT_TEST_MOCK_DATA["agent_responses"]["simple"],
        "model": "test-model",
        "total_duration": 1000000000  # 1 segundo en nanosegundos
    }
    
    return mock_client

def get_mock_http_response(status_code=200, content="Mock content"):
    """Crear un mock de respuesta HTTP"""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.text = content
    mock_response.content = content.encode()
    mock_response.json.return_value = {"status": "ok", "data": content}
    mock_response.headers = {"content-type": "text/html"}
    
    return mock_response

def setup_unit_test_environment():
    """Configurar el entorno para tests unitarios"""
    # Configurar logging para tests
    if not UNIT_TEST_CONFIG["enable_logging"]:
        logging.disable(logging.CRITICAL)
    
    # Configurar variables de entorno para tests
    os.environ["TEST_ENV"] = "unit"
    os.environ["LOG_LEVEL"] = "ERROR"
    os.environ["USE_REAL_SERVICES"] = "false"

def teardown_unit_test_environment():
    """Limpiar después de tests unitarios"""
    # Re-habilitar logging
    logging.disable(logging.NOTSET)
    
    # Limpiar variables de entorno
    test_env_vars = ["TEST_ENV", "LOG_LEVEL", "USE_REAL_SERVICES"]
    for var in test_env_vars:
        os.environ.pop(var, None)

# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_valid_agent_response(response):
    """Verificar que una respuesta del agent es válida"""
    assert isinstance(response, dict), "Agent response must be a dict"
    assert "success" in response, "Response must include success field"
    assert "response" in response, "Response must include response field"
    assert isinstance(response["response"], str), "Response text must be a string"
    assert len(response["response"]) > 0, "Response text cannot be empty"

def assert_valid_search_results(results):
    """Verificar que los resultados de búsqueda son válidos"""
    assert isinstance(results, list), "Search results must be a list"
    for result in results:
        assert isinstance(result, dict), "Each result must be a dict"
        assert "title" in result, "Result must have title"
        assert "url" in result, "Result must have URL"
        assert "snippet" in result, "Result must have snippet"

def assert_valid_session_data(session_data):
    """Verificar que los datos de sesión son válidos"""
    assert isinstance(session_data, dict), "Session data must be a dict"
    assert "session_id" in session_data, "Session must have ID"
    assert "status" in session_data, "Session must have status"
    assert isinstance(session_data["session_id"], str), "Session ID must be string"

def assert_mock_called_with_partial(mock, **partial_kwargs):
    """Verificar que un mock fue llamado con argumentos parciales"""
    assert mock.called, "Mock was not called"
    
    call_args = mock.call_args
    if call_args is None:
        pytest.fail("Mock was called but no call args recorded")
    
    args, kwargs = call_args
    for key, expected_value in partial_kwargs.items():
        if key in kwargs:
            assert kwargs[key] == expected_value, f"Expected {key}={expected_value}, got {kwargs[key]}"
        else:
            pytest.fail(f"Expected argument {key} not found in call")

def assert_response_time_acceptable(duration, max_time=UNIT_TEST_CONFIG["timeout"]):
    """Verificar que el tiempo de respuesta es aceptable para unit tests"""
    assert duration < max_time, f"Response took {duration:.2f}s, max allowed for unit tests: {max_time}s"

# =============================================================================
# TEST DATA PATHS
# =============================================================================

def get_test_data_path(filename):
    """Obtener la ruta a un archivo de datos de test"""
    test_data_dir = Path(__file__).parent / "data"
    test_data_dir.mkdir(exist_ok=True)
    return test_data_dir / filename

def get_mock_data_path(filename):
    """Obtener la ruta a un archivo de datos mock"""
    mock_data_dir = Path(__file__).parent / "mock_data"
    mock_data_dir.mkdir(exist_ok=True)
    return mock_data_dir / filename

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "UNIT_TEST_CONFIG",
    "UNIT_TEST_MOCK_DATA",
    
    # Mock helpers
    "get_mock_agent",
    "get_mock_web_search_tool", 
    "get_mock_ollama_client",
    "get_mock_http_response",
    
    # Environment setup
    "setup_unit_test_environment",
    "teardown_unit_test_environment",
    
    # Assertion helpers
    "assert_valid_agent_response",
    "assert_valid_search_results",
    "assert_valid_session_data",
    "assert_mock_called_with_partial",
    "assert_response_time_acceptable",
    
    # Path helpers
    "get_test_data_path",
    "get_mock_data_path"
]

# Metadata del paquete
__version__ = "1.0.0"
__author__ = "SmartDoc Team"
__description__ = "Unit tests package for SmartDoc Research Agent"

# Auto-setup cuando se importa el paquete
setup_unit_test_environment()
