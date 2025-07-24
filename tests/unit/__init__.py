"""
SmartDoc Research Agent - Unit Tests Package

Este paquete contiene todos los tests unitarios para los componentes individuales
del agente de investigación SmartDoc.

Los tests unitarios se enfocan en probar:
- Componentes aislados sin dependencias externas
- Lógica de negocio específica de cada clase/función
- Casos edge y manejo de errores
- Validación de interfaces y contratos

Módulos de tests unitarios:
    test_smart_agent.py          # Tests del agente principal
    test_web_search_tool.py      # Tests de la herramienta de búsqueda web
    test_content_extractor.py    # Tests del extractor de contenido
    test_search_engines.py       # Tests de los motores de búsqueda
    test_base_tool.py           # Tests de la clase base de herramientas
    test_session_manager.py     # Tests del gestor de sesiones
    test_prompt_templates.py    # Tests de las plantillas de prompts

Convenciones:
- Cada archivo test_*.py corresponde a un módulo específico
- Se usan mocks extensivamente para aislar componentes
- Tests rápidos (< 1 segundo por test individual)
- Cobertura alta de casos edge y errores
- Nomenclatura descriptiva: test_method_name_when_condition_should_result

Ejecución:
    pytest tests/unit/                    # Todos los tests unitarios
    pytest tests/unit/test_smart_agent.py # Test específico
    pytest -m unit                        # Solo tests marcados como unit
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any, Optional
import asyncio

# Imports del proyecto para validación
try:
    from app.agents.core.smart_agent import SmartDocAgent
    from app.agents.tools.base_tool import BaseTool
    from app.agents.tools.web.web_search_tool import WebSearchTool
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Configuración específica para tests unitarios
UNIT_TEST_CONFIG = {
    "default_timeout": 5.0,        # Tests unitarios deben ser rápidos
    "max_test_duration": 10.0,     # Límite superior para tests individuales
    "mock_by_default": True,       # Mockear dependencias externas por defecto
    "allow_real_network": False,   # No permitir llamadas de red reales
    "allow_real_llm": False,       # No permitir llamadas a LLM reales
    "allow_real_db": False,        # No permitir acceso a DB real
}

# Fixtures comunes para tests unitarios
@pytest.fixture
def mock_ollama_llm():
    """Mock del LLM Ollama para tests unitarios"""
    mock_llm = AsyncMock()
    mock_llm.health_check.return_value = True
    mock_llm.generate.return_value = {
        "success": True,
        "response": "Mocked LLM response for unit testing"
    }
    mock_llm.model_name = "test_model"
    return mock_llm

@pytest.fixture
def mock_web_search_engine():
    """Mock del motor de búsqueda para tests unitarios"""
    mock_engine = Mock()
    mock_engine.search.return_value = [
        {
            "title": "Unit Test Result 1",
            "url": "https://example.com/unit-test-1",
            "snippet": "This is a mocked search result for unit testing purposes."
        },
        {
            "title": "Unit Test Result 2", 
            "url": "https://example.com/unit-test-2",
            "snippet": "Another mocked search result to ensure comprehensive testing."
        }
    ]
    mock_engine.name = "mock_search_engine"
    return mock_engine

@pytest.fixture
def mock_content_extractor():
    """Mock del extractor de contenido para tests unitarios"""
    mock_extractor = Mock()
    mock_extractor.extract_content.return_value = {
        "title": "Mocked Page Title",
        "content": "This is mocked page content extracted for unit testing. " * 10,
        "metadata": {
            "word_count": 100,
            "reading_time": 2,
            "language": "en"
        }
    }
    return mock_extractor

@pytest.fixture
def sample_agent_config():
    """Configuración de ejemplo para el agente en tests unitarios"""
    return {
        "model_name": "test_model",
        "temperature": 0.7,
        "max_tokens": 1000,
        "timeout": 30.0,
        "retry_attempts": 3,
        "tools": ["web_search"],
        "session_config": {
            "max_messages": 100,
            "context_window": 4000
        }
    }

@pytest.fixture
def sample_research_session():
    """Sesión de investigación de ejemplo para tests"""
    return {
        "session_id": "test_session_12345",
        "topic": "Unit Testing Best Practices",
        "objectives": [
            "Learn about test isolation principles",
            "Understand mocking strategies",
            "Find code coverage best practices"
        ],
        "research_depth": "standard",
        "max_sources": 10,
        "created_at": "2024-01-25T10:00:00Z",
        "message_count": 0,
        "status": "active"
    }

@pytest.fixture
def mock_session_manager():
    """Mock del gestor de sesiones para tests unitarios"""
    mock_manager = Mock()
    mock_manager.create_session.return_value = "test_session_12345"
    mock_manager.get_session.return_value = {
        "session_id": "test_session_12345",
        "topic": "Test Topic",
        "objectives": ["Test Objective"],
        "message_count": 0
    }
    mock_manager.update_session.return_value = True
    mock_manager.delete_session.return_value = True
    mock_manager.list_sessions.return_value = []
    return mock_manager

# Utilidades para tests unitarios
class UnitTestHelpers:
    """Utilidades compartidas para tests unitarios"""
    
    @staticmethod
    def create_mock_agent(with_tools: bool = True) -> Mock:
        """Crear un mock del SmartDocAgent para tests"""
        mock_agent = AsyncMock(spec=SmartDocAgent)
        mock_agent.initialized = True
        mock_agent.model_name = "test_model"
        
        if with_tools:
            mock_tool = Mock(spec=BaseTool)
            mock_tool.name = "mock_tool"
            mock_tool._arun = AsyncMock(return_value="Mock tool result")
            mock_agent.tools = [mock_tool]
        else:
            mock_agent.tools = []
        
        # Métodos principales
        mock_agent.initialize.return_value = None
        mock_agent.create_research_session.return_value = "test_session_12345"
        mock_agent.process_query.return_value = {
            "success": True,
            "response": "Mock agent response",
            "session_id": "test_session_12345"
        }
        mock_agent.get_session_status.return_value = {
            "session_id": "test_session_12345",
            "topic": "Test Topic",
            "message_count": 1
        }
        mock_agent.close.return_value = None
        
        return mock_agent
    
    @staticmethod
    def create_mock_web_tool() -> Mock:
        """Crear un mock de WebSearchTool para tests"""
        mock_tool = AsyncMock(spec=WebSearchTool)
        mock_tool.name = "web_search"
        mock_tool.description = "Mock web search tool for unit testing"
        
        mock_tool._arun.return_value = "Mock web search results: Found 5 relevant results about the topic."
        mock_tool.get_tool_info.return_value = {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {"query": "string"}
        }
        
        return mock_tool
    
    @staticmethod
    def assert_mock_called_with_partial(mock