"""
Tests específicos para el sistema de sesiones de investigación de SmartDoc

Este módulo se enfoca en probar el manejo de sesiones de investigación:
- Creación y gestión de sesiones
- Persistencia de contexto entre queries
- Manejo de memoria y estado
- Funcionalidades específicas de sesiones
- Límites y validaciones de sesiones
"""

import pytest
import asyncio
import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional

from app.agents.core.smart_agent import SmartDocAgent
from app.agents.tools.web.web_search_tool import WebSearchTool
from tests.fixtures import (
    MOCK_RESEARCH_SESSIONS,
    SAMPLE_RESEARCH_TOPICS,
    PERFORMANCE_BENCHMARKS,
    CONCURRENCY_TEST_CONFIG
)


# =============================================================================
# TESTS DE CREACIÓN Y GESTIÓN DE SESIONES
# =============================================================================

class TestResearchSessionCreation:
    """Tests para creación y configuración inicial de sesiones"""
    
    @pytest.fixture
    async def mocked_agent(self):
        """Agent con mocks básicos para test de sesiones"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Session created successfully. Ready to help with research."
        }
        
        with patch('app.agents.core.smart_agent.OllamaLLM', return_value=mock_ollama):
            agent = SmartDocAgent(model_name="test_model")
            agent.initialized = True
            
        return agent
    
    @pytest.mark.asyncio
    async def test_create_basic_research_session(self, mocked_agent):
        """Test de creación básica de sesión de investigación"""
        
        agent = mocked_agent
        
        # Crear sesión con parámetros mínimos
        session_id = await agent.create_research_session(
            topic="Python programming",
            objectives=["Learn basic concepts", "Find tutorials"]
        )
        
        # Verificaciones básicas
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        # Verificar que la sesión existe en el estado del agente
        status = agent.get_session_status(session_id)
        assert status is not None
        assert status["topic"] == "Python programming"
        assert len(status["objectives"]) == 2
        assert "Learn basic concepts" in status["objectives"]
        assert "Find tutorials" in status["objectives"]
    
    @pytest.mark.asyncio
    async def test_create_session_with_all_parameters(self, mocked_agent):
        """Test de creación de sesión con todos los parámetros"""
        
        agent = mocked_agent
        
        session_id = await agent.create_research_session(
            topic="Advanced Machine Learning",
            objectives=[
                "Understand deep learning architectures",
                "Research latest papers in transformer models",
                "Find practical implementation examples",
                "Compile comprehensive resource list"
            ],
            max_sources=15,
            research_depth="deep",
            research_style="academic"
        )
        
        assert session_id is not None
        
        status = agent.get_session_status(session_id)
        assert status["topic"] == "Advanced Machine Learning"
        assert len(status["objectives"]) == 4
        assert status.get("max_sources", 10) == 15
        assert status.get("research_depth", "standard") == "deep"
        assert status.get("research_style", "casual") == "academic"
    
    @pytest.mark.asyncio
    async def test_session_id_uniqueness(self, mocked_agent):
        """Test que los IDs de sesión son únicos"""
        
        agent = mocked_agent
        
        # Crear múltiples sesiones
        session_ids = []
        for i in range(5):
            session_id = await agent.create_research_session(
                topic=f"Topic {i+1}",
                objectives=[f"Objective {i+1}"]
            )
            session_ids.append(session_id)
        
        # Verificar unicidad
        assert len(session_ids) == len(set(session_ids))
        
        # Verificar que todas las sesiones existen
        for session_id in session_ids:
            status = agent.get_session_status(session_id)
            assert status is not None
    
    @pytest.mark.asyncio
    async def test_invalid_session_parameters(self, mocked_agent):
        """Test de validación de parámetros inválidos"""
        
        agent = mocked_agent
        
        # Topic vacío o None
        with pytest.raises((ValueError, TypeError)):
            await agent.create_research_session(
                topic="",
                objectives=["Test objective"]
            )
        
        # Objetivos vacíos
        with pytest.raises((ValueError, TypeError)):
            await agent.create_research_session(
                topic="Valid topic",
                objectives=[]
            )
        
        # Parámetros numéricos inválidos
        with pytest.raises((ValueError, TypeError)):
            await agent.create_research_session(
                topic="Valid topic",
                objectives=["Valid objective"],
                max_sources=-5  # Número negativo
            )


# =============================================================================
# TESTS DE CONTEXTO Y MEMORIA DE SESIÓN
# =============================================================================

class TestSessionContextAndMemory:
    """Tests para contexto y memoria entre queries en una sesión"""
    
    @pytest.fixture
    async def session_with_history(self):
        """Sesión con historial de conversación para tests"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        
        # Responses que muestran continuidad
        def contextual_response(prompt: str, **kwargs):
            if "first" in prompt.lower() or "initial" in prompt.lower():
                return {
                    "success": True,
                    "response": "This is my first response about Python. I'll remember this context."
                }
            elif "second" in prompt.lower() or "follow" in prompt.lower():
                return {
                    "success": True,
                    "response": "Following up on my previous Python response, here's additional information."
                }
            elif "summary" in prompt.lower():
                return {
                    "success": True,
                    "response": "Based on our conversation about Python, here's a comprehensive summary."
                }
            else:
                return {
                    "success": True,
                    "response": "I understand your question in the context of our ongoing research."
                }
        
        mock_ollama.generate.side_effect = contextual_response
        
        with patch('app.agents.core.smart_agent.OllamaLLM', return_value=mock_ollama):
            agent = SmartDocAgent(model_name="test_model")
            agent.initialized = True
            
        # Crear sesión con historial
        session_id = await agent.create_research_session(
            topic="Python Programming Deep Dive",
            objectives=["Learn advanced concepts", "Build practical projects"]
        )
        
        return agent, session_id
    
    @pytest.mark.asyncio
    async def test_context_continuity_between_queries(self, session_with_history):
        """Test de continuidad de contexto entre múltiples queries"""
        
        agent, session_id = session_with_history
        
        # Primera query
        result1 = await agent.process_query(
            session_id, 
            "Tell me about Python data structures (first query)"
        )
        
        assert result1["success"] != False
        assert "first response" in result1["response"]
        
        # Segunda query que referencia la primera
        result2 = await agent.process_query(
            session_id,
            "Can you expand on what you just told me? (second query)"
        )
        
        assert result2["success"] != False
        assert "previous" in result2["response"] or "follow" in result2["response"]
        
        # Verificar que la sesión mantiene el conteo de mensajes
        status = agent.get_session_status(session_id)
        assert status["message_count"] >= 4  # 2 queries + 2 responses
    
    @pytest.mark.asyncio
    async def test_session_memory_persistence(self, session_with_history):
        """Test de persistencia de memoria en la sesión"""
        
        agent, session_id = session_with_history
        
        # Secuencia de queries que construyen conocimiento
        queries = [
            "What are Python decorators?",
            "How do they relate to what we just discussed?",
            "Can you give me a practical example based on our conversation?",
            "Summarize everything we've covered about Python"
        ]
        
        responses = []
        for i, query in enumerate(queries):
            result = await agent.process_query(session_id, query)
            assert result["success"] != False
            responses.append(result)
            
            # Verificar que el conteo de mensajes aumenta
            status = agent.get_session_status(session_id)
            expected_count = (i + 1) * 2  # queries + responses
            assert status["message_count"] >= expected_count
        
        # La última respuesta (summary) debería mostrar conocimiento acumulado
        final_response = responses[-1]["response"]
        assert "summary" in final_response or "conversation" in final_response
    
    @pytest.mark.asyncio
    async def test_session_context_isolation(self, mocked_agent):
        """Test que las sesiones mantienen contextos aislados"""
        
        agent = mocked_agent
        
        # Crear dos sesiones diferentes
        session1_id = await agent.create_research_session(
            topic="Python Programming",
            objectives=["Learn basics"]
        )
        
        session2_id = await agent.create_research_session(
            topic="JavaScript Development", 
            objectives=["Learn frameworks"]
        )
        
        # Queries en sesión 1
        result1 = await agent.process_query(
            session1_id,
            "Tell me about Python variables"
        )
        
        # Query en sesión 2
        result2 = await agent.process_query(
            session2_id,
            "What are JavaScript variables?"
        )
        
        # Verificar que las sesiones son independientes
        status1 = agent.get_session_status(session1_id)
        status2 = agent.get_session_status(session2_id)
        
        assert status1["topic"] != status2["topic"]
        assert status1["message_count"] >= 2
        assert status2["message_count"] >= 2
        
        # Las sesiones no deben interferir entre sí
        assert session1_id != session2_id


# =============================================================================
# TESTS DE LÍMITES Y VALIDACIONES DE SESIÓN
# =============================================================================

class TestSessionLimitsAndValidation:
    """Tests para límites y validaciones del sistema de sesiones"""
    
    @pytest.mark.asyncio
    async def test_session_message_limits(self, mocked_agent):
        """Test de límites de mensajes por sesión"""
        
        agent = mocked_agent
        
        session_id = await agent.create_research_session(
            topic="Stress test topic",
            objectives=["Test limits"]
        )
        
        # Simular muchos mensajes
        message_limit = 50  # Ajustar según configuración real
        
        for i in range(message_limit + 5):  # Exceder el límite
            try:
                result = await agent.process_query(
                    session_id,
                    f"Test message #{i+1}"
                )
                
                if i < message_limit:
                    assert result["success"] != False
                else:
                    # Después del límite, debería manejar graciosamente
                    # (puede truncar historial o dar warning)
                    pass
                    
            except Exception as e:
                # Si hay límite estricto, debería fallar graciosamente
                assert "limit" in str(e).lower() or "exceeded" in str(e).lower()
                break
    
    @pytest.mark.asyncio
    async def test_invalid_session_operations(self, mocked_agent):
        """Test de operaciones con sesiones inválidas"""
        
        agent = mocked_agent
        
        # Session ID inexistente
        fake_session_id = str(uuid.uuid4())
        
        with pytest.raises((ValueError, KeyError)):
            await agent.process_query(fake_session_id, "Test query")
        
        # Session ID malformado
        with pytest.raises((ValueError, TypeError)):
            await agent.process_query("invalid-session", "Test query")
        
        # Query vacía
        valid_session_id = await agent.create_research_session(
            topic="Valid topic",
            objectives=["Valid objective"]
        )
        
        with pytest.raises((ValueError, TypeError)):
            await agent.process_query(valid_session_id, "")
    
    @pytest.mark.asyncio
    async def test_session_timeout_handling(self, mocked_agent):
        """Test de manejo de timeouts de sesión"""
        
        agent = mocked_agent
        
        session_id = await agent.create_research_session(
            topic="Timeout test",
            objectives=["Test timeouts"]
        )
        
        # Simular sesión activa
        result = await agent.process_query(session_id, "Initial query")
        assert result["success"] != False
        
        # Verificar timestamp de última actividad
        status = agent.get_session_status(session_id)
        assert "last_activity" in status or "created_at" in status
        
        # Una sesión recién creada no debería estar expirada
        # (Los tests de expiración real requerirían manipulación de tiempo)


# =============================================================================
# TESTS DE CONCURRENCIA DE SESIONES
# =============================================================================

class TestSessionConcurrency:
    """Tests para manejo concurrente de múltiples sesiones"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_session_creation(self, mocked_agent):
        """Test de creación concurrente de sesiones"""
        
        agent = mocked_agent
        
        # Crear múltiples sesiones concurrentemente
        async def create_session(topic_num: int):
            return await agent.create_research_session(
                topic=f"Concurrent Topic {topic_num}",
                objectives=[f"Objective {topic_num}"]
            )
        
        # Crear 10 sesiones concurrentemente
        tasks = [create_session(i) for i in range(10)]
        session_ids = await asyncio.gather(*tasks)
        
        # Verificar que todas se crearon exitosamente
        assert len(session_ids) == 10
        assert len(set(session_ids)) == 10  # Todos únicos
        
        # Verificar que todas las sesiones son válidas
        for session_id in session_ids:
            status = agent.get_session_status(session_id)
            assert status is not None
            assert "Concurrent Topic" in status["topic"]
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_query_processing(self, mocked_agent):
        """Test de procesamiento concurrente de queries en diferentes sesiones"""
        
        agent = mocked_agent
        
        # Crear varias sesiones
        sessions = []
        for i in range(3):
            session_id = await agent.create_research_session(
                topic=f"Concurrent Session {i+1}",
                objectives=[f"Process queries concurrently {i+1}"]
            )
            sessions.append(session_id)
        
        # Función para procesar query en una sesión específica
        async def process_in_session(session_id: str, query_num: int):
            return await agent.process_query(
                session_id,
                f"Concurrent query #{query_num} in session {session_id[:8]}"
            )
        
        # Procesar queries concurrentemente en diferentes sesiones
        tasks = []
        for i, session_id in enumerate(sessions):
            for j in range(2):  # 2 queries por sesión
                task = process_in_session(session_id, j+1)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verificar resultados
        successful_results = 0
        for result in results:
            if isinstance(result, dict) and result.get("success") != False:
                successful_results += 1
        
        expected_results = len(sessions) * 2  # 3 sesiones * 2 queries
        success_rate = successful_results / expected_results
        
        assert success_rate >= 0.8, f"Success rate too low: {success_rate:.2f}"
        
        # Verificar que cada sesión mantuvo su estado independiente
        for session_id in sessions:
            status = agent.get_session_status(session_id)
            assert status["message_count"] >= 4  # 2 queries + 2 responses


# =============================================================================
# TESTS DE FUNCIONALIDADES AVANZADAS DE SESIÓN
# =============================================================================

class TestAdvancedSessionFeatures:
    """Tests para funcionalidades avanzadas del sistema de sesiones"""
    
    @pytest.mark.asyncio
    async def test_session_objective_tracking(self, mocked_agent):
        """Test de seguimiento de objetivos de investigación"""
        
        agent = mocked_agent
        
        objectives = [
            "Learn Python basics",
            "Find practical examples", 
            "Build a simple project",
            "Understand best practices"
        ]
        
        session_id = await agent.create_research_session(
            topic="Python Learning Journey",
            objectives=objectives
        )
        
        # Procesar queries relacionadas con objetivos
        objective_queries = [
            "What are Python variables and data types?",  # Objetivo 1
            "Show me examples of Python functions",        # Objetivo 2
            "Help me design a simple calculator project",  # Objetivo 3
            "What are Python coding best practices?"       # Objetivo 4
        ]
        
        for query in objective_queries:
            result = await agent.process_query(session_id, query)
            assert result["success"] != False
        
        # Verificar que el estado de la sesión refleja el progreso
        status = agent.get_session_status(session_id)
        assert len(status["objectives"]) == len(objectives)
        
        # Podría haber tracking de progreso por objetivo (si está implementado)
        if "objective_progress" in status:
            assert isinstance(status["objective_progress"], dict)
    
    @pytest.mark.asyncio
    async def test_session_export_functionality(self, mocked_agent):
        """Test de funcionalidad de exportación de sesión"""
        
        agent = mocked_agent
        
        session_id = await agent.create_research_session(
            topic="Export Test Session",
            objectives=["Test export functionality"]
        )
        
        # Añadir contenido a la sesión
        await agent.process_query(session_id, "First query for export test")
        await agent.process_query(session_id, "Second query for export test")
        
        # Intentar exportar sesión (si está implementado)
        if hasattr(agent, 'export_session'):
            export_data = agent.export_session(session_id)
            
            assert export_data is not None
            assert export_data["session_id"] == session_id
            assert "topic" in export_data
            assert "objectives" in export_data
            assert "messages" in export_data or "conversation_history" in export_data
        else:
            # Si no está implementado, verificar que los datos están disponibles
            status = agent.get_session_status(session_id)
            assert status["message_count"] >= 4
    
    @pytest.mark.asyncio
    async def test_session_research_depth_modes(self, mocked_agent):
        """Test de diferentes modos de profundidad de investigación"""
        
        agent = mocked_agent
        
        depth_modes = ["quick", "standard", "deep"]
        
        sessions = {}
        
        for depth in depth_modes:
            session_id = await agent.create_research_session(
                topic=f"Research depth test - {depth}",
                objectives=[f"Test {depth} research"],
                research_depth=depth
            )
            sessions[depth] = session_id
        
        # Procesar la misma query en diferentes profundidades
        test_query = "Explain machine learning algorithms"
        
        for depth, session_id in sessions.items():
            result = await agent.process_query(session_id, test_query)
            assert result["success"] != False
            
            # Verificar que la profundidad se refleja en el resultado
            status = agent.get_session_status(session_id)
            if "research_depth" in status:
                assert status["research_depth"] == depth


# =============================================================================
# TESTS DE PERFORMANCE DE SESIONES
# =============================================================================

class TestSessionPerformance:
    """Tests de performance específicos para el sistema de sesiones"""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_session_creation_performance(self, mocked_agent):
        """Test de performance en creación de sesiones"""
        
        agent = mocked_agent
        
        import time
        
        # Medir tiempo de creación de sesiones
        start_time = time.time()
        
        session_count = 20
        for i in range(session_count):
            session_id = await agent.create_research_session(
                topic=f"Performance test session {i+1}",
                objectives=[f"Performance objective {i+1}"]
            )
            assert session_id is not None
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_session = total_time / session_count
        
        # Verificar que la creación es razonablemente rápida
        assert avg_time_per_session < 0.1, f"Session creation too slow: {avg_time_per_session:.3f}s per session"
        
        print(f"✅ Created {session_count} sessions in {total_time:.2f}s (avg: {avg_time_per_session:.3f}s/session)")
    
    @pytest.mark.asyncio
    async def test_session_memory_usage(self, mocked_agent):
        """Test de uso de memoria con múltiples sesiones"""
        
        agent = mocked_agent
        
        # Crear muchas sesiones para verificar uso de memoria
        session_ids = []
        
        for i in range(50):
            session_id = await agent.create_research_session(
                topic=f"Memory test session {i+1}",
                objectives=[f"Memory test objective {i+1}"]
            )
            session_ids.append(session_id)
            
            # Añadir algo de contenido a cada sesión
            await agent.process_query(session_id, f"Test query for session {i+1}")
        
        # Verificar que todas las sesiones siguen siendo accesibles
        accessible_sessions = 0
        for session_id in session_ids:
            try:
                status = agent.get_session_status(session_id)
                if status is not None:
                    accessible_sessions += 1
            except:
                pass
        
        # Al menos el 90% de las sesiones deberían ser accesibles
        accessibility_rate = accessible_sessions / len(session_ids)
        assert accessibility_rate >= 0.9, f"Too many sessions lost: {accessibility_rate:.2f}"
        
        print(f"✅ {accessible_sessions}/{len(session_ids)} sessions remain accessible")


# =============================================================================
# FIXTURES Y UTILIDADES ESPECÍFICAS
# =============================================================================

@pytest.fixture
def sample_session_configs():
    """Configuraciones de ejemplo para tests de sesiones"""
    return [
        {
            "topic": "Python Web Development",
            "objectives": ["Learn Flask", "Build REST API", "Deploy to cloud"],
            "research_depth": "standard",
            "max_sources": 10
        },
        {
            "topic": "Machine Learning Fundamentals", 
            "objectives": ["Understand algorithms", "Practice with datasets"],
            "research_depth": "deep",
            "max_sources": 20
        },
        {
            "topic": "Quick JavaScript Reference",
            "objectives": ["Find syntax examples"],
            "research_depth": "quick",
            "max_sources": 5
        }
    ]


if __name__ == "__main__":
    # Ejecutar tests específicos si se ejecuta directamente
    pytest.main([__file__, "-v", "--tb=short", "--asyncio-mode=auto"])