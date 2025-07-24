"""
SmartDoc Research Agent - Integration Tests Package

Este paquete contiene todos los tests de integración para verificar que los
componentes del agente de investigación SmartDoc funcionan correctamente juntos.

Los tests de integración se enfocan en:
- Interacciones entre componentes múltiples
- Flujos de trabajo completos end-to-end
- Integración con servicios externos (mocked cuando sea necesario)
- Comportamiento del sistema como un todo
- Manejo de datos reales o realistas

Módulos de tests de integración:
    test_agent_web_integration.py     # Integración agente + web search
    test_full_workflow.py            # Flujos completos E2E
    test_research_session.py         # Sistema de sesiones completo
    test_multi_tool_coordination.py  # Coordinación entre herramientas
    test_memory_persistence.py      # Persistencia y memoria del agente
    test_error_recovery.py          # Recuperación de errores en flujos

Características:
- Tests más lentos que unitarios (pueden tomar varios segundos)
- Uso de mocks para servicios externos pero componentes internos reales
- Validación de interfaces entre componentes
- Verificación de flujos de datos completos
- Testing de escenarios realistas de uso

Convenciones:
- Prefijo test_integration_ para tests puramente de integración
- Uso de fixtures compartidas para setup complejo
- Validación de estado del sistema después de operaciones
- Cleanup apropiado de recursos después de tests

Ejecución:
    pytest tests/integration/                    # Todos los tests de integración
    pytest tests/integration/test_full_workflow.py # Test específico
    pytest -m integration                       # Solo tests marcados como integration
    pytest -m "integration and not slow"        # Integración excluyendo tests lentos
"""

import pytest
import asyncio
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import time

# Imports del proyecto
try:
    from app.agents.core.smart_agent import SmartDocAgent
    from app.agents.tools.web.web_search_tool import WebSearchTool
    from app.agents.tools.base_tool import BaseTool
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Configuración específica para tests de integración
INTEGRATION_TEST_CONFIG = {
    "default_timeout": 30.0,          # Tests de integración pueden tomar más tiempo
    "slow_test_timeout": 120.0,       # Tests E2E completos
    "max_concurrent_sessions": 5,     # Límite para tests concurrentes
    "allow_real_llm": False,          # Usar mocks para LLM por defecto
    "allow_real_web_search": False,   # Usar mocks para búsquedas web
    "allow_real_db": False,           # Usar mocks para base de datos
    "enable_performance_tracking": True,  # Tracking de performance en integración
    "cleanup_after_tests": True,      # Limpiar recursos después de tests
}

# Logger específico para tests de integración
integration_logger = logging.getLogger('smartdoc.integration_tests')
integration_logger.setLevel(logging.INFO)

# Fixtures compartidas para tests de integración
@pytest.fixture(scope="session")
async def integration_test_session():
    """Session-level fixture para configuración global de tests de integración"""
    
    # Setup global
    session_data = {
        "start_time": time.time(),
        "test_sessions_created": [],
        "performance_metrics": {},
        "cleanup_tasks": []
    }
    
    integration_logger.info("Starting integration test session")
    
    yield session_data
    
    # Cleanup global
    cleanup_time = time.time()
    session_duration = cleanup_time - session_data["start_time"]
    
    integration_logger.info(f"Integration test session completed in {session_duration:.2f}s")
    integration_logger.info(f"Sessions created: {len(session_data['test_sessions_created'])}")
    
    # Ejecutar tareas de cleanup
    for cleanup_task in session_data["cleanup_tasks"]:
        try:
            if asyncio.iscoroutinefunction(cleanup_task):
                await cleanup_task()
            else:
                cleanup_task()
        except Exception as e:
            integration_logger.warning(f"Cleanup task failed: {e}")

@pytest.fixture
async def integration_agent():
    """Agent configurado para tests de integración con mocks apropiados"""
    
    # Mock del LLM con respuestas realistas
    mock_ollama = AsyncMock()
    mock_ollama.health_check.return_value = True
    
    def realistic_llm_response(prompt: str, **kwargs):
        """Generar respuestas realistas basadas en el contexto del prompt"""
        
        if "search" in prompt.lower() or "web_search" in prompt.lower():
            return {
                "success": True,
                "response": "Based on my search, I found relevant information about the topic. Let me provide you with a comprehensive summary of the key findings."
            }
        elif "error" in prompt.lower() or "failed" in prompt.lower():
            return {
                "success": True, 
                "response": "I encountered an issue but I'll try an alternative approach to help you with your request."
            }
        else:
            return {
                "success": True,
                "response": "I understand your request and I'll help you research this topic thoroughly."
            }
    
    mock_ollama.generate.side_effect = realistic_llm_response
    
    # Crear agente con LLM mocked
    with patch('app.agents.core.smart_agent.OllamaLLM', return_value=mock_ollama):
        agent = SmartDocAgent(model_name="integration_test_model")
        
        # Crear mock de web search tool más realista
        mock_web_tool = AsyncMock(spec=WebSearchTool)
        mock_web_tool.name = "web_search"
        
        def realistic_web_search(query: str):
            """Simular búsquedas web realistas"""
            return f"Found 8 relevant results for '{query}': " + \
                   "Technical documentation, research papers, tutorials, and practical examples " + \
                   "covering various aspects of the topic with detailed information."
        
        mock_web_tool._arun.side_effect = realistic_web_search
        mock_web_tool.get_tool_info.return_value = {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {"query": "string"}
        }
        
        # Asignar herramientas al agente
        agent.tools = [mock_web_tool]
        
        # Inicializar agente
        await agent.initialize()
    
    yield agent, mock_ollama, mock_web_tool
    
    # Cleanup del agente
    try:
        await agent.close()
    except Exception as e:
        integration_logger.warning(f"Agent cleanup failed: {e}")

@pytest.fixture
def performance_tracker():
    """Fixture para tracking de performance en tests de integración"""
    
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
            self.start_times = {}
        
        def start_timing(self, operation_name: str):
            """Iniciar medición de tiempo para una operación"""
            self.start_times[operation_name] = time.time()
        
        def end_timing(self, operation_name: str) -> float:
            """Finalizar medición y retornar duración"""
            if operation_name in self.start_times:
                duration = time.time() - self.start_times[operation_name]
                self.metrics[operation_name] = duration
                del self.start_times[operation_name]
                return duration
            return 0.0
        
        def get_metrics(self) -> Dict[str, float]:
            """Obtener todas las métricas registradas"""
            return self.metrics.copy()
        
        def assert_performance(self, operation_name: str, max_duration: float):
            """Assert que una operación no exceda el tiempo máximo"""
            actual_duration = self.metrics.get(operation_name, float('inf'))
            assert actual_duration <= max_duration, \
                f"Operation '{operation_name}' took {actual_duration:.2f}s, max allowed: {max_duration}s"
    
    return PerformanceTracker()

@pytest.fixture
async def multi_session_environment():
    """Fixture para tests que requieren múltiples sesiones"""
    
    sessions = {}
    
    class MultiSessionManager:
        def __init__(self):
            self.sessions = sessions
        
        async def create_test_session(self, agent, topic: str, session_id: str = None) -> str:
            """Crear una sesión de test"""
            if session_id is None:
                session_id = f"test_session_{len(self.sessions) + 1}"
            
            actual_session_id = await agent.create_research_session(
                topic=topic,
                objectives=[f"Test objective for {topic}"]
            )
            
            self.sessions[session_id] = {
                "actual_id": actual_session_id,
                "topic": topic,
                "created_at": time.time()
            }
            
            return actual_session_id
        
        def get_session_count(self) -> int:
            """Obtener número de sesiones activas"""
            return len(self.sessions)
        
        def get_session_info(self, session_id: str) -> Dict:
            """Obtener información de una sesión"""
            return self.sessions.get(session_id, {})
        
        async def cleanup_all_sessions(self, agent):
            """Limpiar todas las sesiones de test"""
            for session_data in self.sessions.values():
                try:
                    # Si el agente tiene método de cleanup de sesión
                    if hasattr(agent, 'close_session'):
                        await agent.close_session(session_data["actual_id"])
                except Exception as e:
                    integration_logger.warning(f"Session cleanup failed: {e}")
            
            self.sessions.clear()
    
    manager = MultiSessionManager()
    
    yield manager
    
    # Cleanup automático al final
    # (Las sesiones se limpiarán cuando el agente se cierre)

# Utilidades para tests de integración
class IntegrationTestHelpers:
    """Utilidades específicas para tests de integración"""
    
    @staticmethod
    async def wait_for_condition(condition_func, timeout: float = 10.0, interval: float = 0.1) -> bool:
        """Esperar hasta que una condición se cumpla o timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
                    return True
            except Exception:
                pass
            
            await asyncio.sleep(interval)
        
        return False
    
    @staticmethod
    async def simulate_user_interaction_sequence(agent, session_id: str, queries: List[str]) -> List[Dict]:
        """Simular una secuencia de interacciones de usuario"""
        results = []
        
        for i, query in enumerate(queries):
            integration_logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            result = await agent.process_query(session_id, query)
            results.append(result)
            
            # Pequeña pausa para simular tiempo de lectura del usuario
            await asyncio.sleep(0.1)
        
        return results
    
    @staticmethod
    def validate_workflow_continuity(results: List[Dict]) -> bool:
        """Validar que los resultados de un workflow muestran continuidad"""
        if len(results) < 2:
            return True
        
        # Verificar que todos los resultados son exitosos
        for result in results:
            if not result.get("success", False):
                return False
        
        # Verificar que las sesiones son consistentes
        session_ids = [result.get("session_id") for result in results]
        if not all(sid == session_ids[0] for sid in session_ids):
            return False
        
        # Verificar que las respuestas no son idénticas (indicating real processing)
        responses = [result.get("response", "") for result in results]
        if len(set(responses)) < len(responses) * 0.8:  # Al menos 80% únicos
            return False
        
        return True
    
    @staticmethod
    async def test_concurrent_operations(operations: List, max_concurrency: int = 5) -> List:
        """Ejecutar operaciones concurrentemente con límite de concurrencia"""
        semaphore = asyncio.Semaphore(max_concurrency)
        
        async def limited_operation(op):
            async with semaphore:
                if asyncio.iscoroutinefunction(op):
                    return await op()
                else:
                    return op()
        
        tasks = [limited_operation(op) for op in operations]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Decoradores para tests de integración
def integration_test(func):
    """Decorator para marcar tests como de integración"""
    return pytest.mark.integration(func)

def slow_integration_test(func):
    """Decorator para tests de integración lentos"""
    return pytest.mark.integration(pytest.mark.slow(func))

def requires_real_services(services: List[str]):
    """Decorator para tests que requieren servicios reales"""
    def decorator(func):
        skip_reason = f"Requires real services: {', '.join(services)}"
        return pytest.mark.skipif(
            not INTEGRATION_TEST_CONFIG.get("allow_real_services", False),
            reason=skip_reason
        )(func)
    return decorator

def concurrent_test(max_workers: int = 3):
    """Decorator para tests que prueban concurrencia"""
    def decorator(func):
        func._max_workers = max_workers
        return pytest.mark.concurrent(func)
    return decorator

# Assertions específicas para integración
class IntegrationAssertions:
    """Assertions personalizadas para tests de integración"""
    
    @staticmethod
    def assert_workflow_completed_successfully(results: List[Dict]):
        """Verificar que un workflow completo fue exitoso"""
        assert len(results) > 0, "No results from workflow"
        
        for i, result in enumerate(results):
            assert result.get("success") != False, f"Step {i+1} failed: {result}"
            assert isinstance(result.get("response"), str), f"Step {i+1} invalid response format"
            assert len(result.get("response", "")) > 10, f"Step {i+1} response too short"
    
    @staticmethod
    def assert_performance_within_limits(metrics: Dict[str, float], limits: Dict[str, float]):
        """Verificar que las métricas de performance están dentro de límites"""
        for operation, actual_time in metrics.items():
            if operation in limits:
                max_time = limits[operation]
                assert actual_time <= max_time, \
                    f"Operation '{operation}' took {actual_time:.2f}s, max allowed: {max_time}s"
    
    @staticmethod
    def assert_session_state_consistency(agent, session_id: str, expected_state: Dict):
        """Verificar consistencia del estado de sesión"""
        actual_state = agent.get_session_status(session_id)
        
        for key, expected_value in expected_state.items():
            assert key in actual_state, f"Missing state key: {key}"
            assert actual_state[key] == expected_value, \
                f"State mismatch for {key}: expected {expected_value}, got {actual_state[key]}"

# Configuración de logging para integración
def setup_integration_logging():
    """Configurar logging específico para tests de integración"""
    
    # Handler para archivo específico de integración
    from tests import get_test_config
    logs_dir = get_test_config("logs_dir")
    
    log_file = logs_dir / "integration_tests.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Formato detallado para integración
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    if not integration_logger.handlers:
        integration_logger.addHandler(file_handler)

# Setup inicial
setup_integration_logging()

# Datos específicos para tests de integración
INTEGRATION_TEST_DATA = {
    "realistic_queries": [
        "What are the latest developments in artificial intelligence?",
        "How do I implement a REST API with authentication?",
        "Compare different machine learning frameworks",
        "What are the best practices for microservices architecture?"
    ],
    "complex_topics": [
        "Advanced Machine Learning Techniques",
        "Modern Web Development Architecture",
        "Cloud Computing Security Best Practices",
        "DevOps Pipeline Optimization"
    ],
    "multi_step_scenarios": [
        {
            "topic": "Python Web Development",
            "steps": [
                "What is Flask and how does it work?",
                "How to add authentication to a Flask app?",
                "Best practices for Flask project structure?",
                "How to deploy Flask apps to production?"
            ]
        },
        {
            "topic": "Machine Learning Project",
            "steps": [
                "How to choose the right ML algorithm?",
                "Data preprocessing best practices?",
                "Model evaluation and validation techniques?",
                "MLOps and model deployment strategies?"
            ]
        }
    ],
    "performance_benchmarks": {
        "session_creation": 2.0,
        "simple_query": 10.0,
        "complex_query": 30.0,
        "multi_tool_query": 45.0,
        "concurrent_queries": 60.0
    }
}

# Exports
__all__ = [
    "INTEGRATION_TEST_CONFIG",
    "IntegrationTestHelpers",
    "IntegrationAssertions", 
    "INTEGRATION_TEST_DATA",
    "integration_test",
    "slow_integration_test",
    "requires_real_services",
    "concurrent_test",
    "integration_logger"
]

# Información del paquete
__version__ = "1.0.0"
__description__ = "Integration tests package for SmartDoc Research Agent"