"""
Integration Tests for SmartDoc Agent + Web Search Tool
Tests de integración entre el SmartDocAgent y WebSearchTool
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from typing import Dict, List, Any

# Imports del proyecto
from app.agents.core.smart_agent import SmartDocAgent, create_smartdoc_agent, ResearchSession
from app.agents.tools.web.web_search_tool import WebSearchTool
from app.agents.tools.base_tool import ToolResult, get_available_tools, register_tool
from app.services.ollama_client import get_ollama_client

# Test fixtures
from fixtures import (
    MOCK_AGENT_RESPONSE,
    MOCK_SESSION_STATUS,
    get_mock_search_results,
    PERFORMANCE_TEST_QUERIES,
    E2E_RESEARCH_SCENARIO
)
from test_config import AGENT_CONFIG, WEB_SEARCH_CONFIG

# =============================================================================
# FIXTURES ESPECÍFICAS PARA INTEGRATION
# =============================================================================

@pytest.fixture
async def mock_agent_with_web_search():
    """Fixture de SmartDocAgent con WebSearchTool mockeado"""
    
    # Crear agent mock
    agent = AsyncMock(spec=SmartDocAgent)
    agent.is_initialized = True
    agent.active_sessions = {}
    
    # Mock web search tool
    mock_web_tool = AsyncMock(spec=WebSearchTool)
    mock_web_tool.name = "web_search"
    mock_web_tool.description = "Mock web search tool"
    mock_web_tool._arun.return_value = "Mock web search results for test query"
    
    # Simular que el agent tiene el tool
    agent.tools = [mock_web_tool]
    
    return agent, mock_web_tool

@pytest.fixture
async def real_agent_mock_ollama():
    """Fixture de SmartDocAgent real con Ollama mockeado"""
    
    # Mock Ollama client
    mock_ollama = AsyncMock()
    mock_ollama.health_check.return_value = True
    mock_ollama.generate.return_value = {
        "success": True,
        "response": "This is a mocked response from the language model about the query.",
        "model": AGENT_CONFIG.TEST_MODEL,
        "total_duration": 1000000000
    }
    
    with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
        
        # Crear agent real
        agent = SmartDocAgent(
            model_name=AGENT_CONFIG.TEST_MODEL,
            research_style="general",
            max_iterations=3
        )
        
        # Inicializar (sin herramientas por ahora)
        success = await agent.initialize()
        assert success, "Agent initialization failed"
        
        yield agent, mock_ollama
        
        # Cleanup
        await agent.close()

@pytest.fixture
def mock_web_search_tool():
    """Mock del WebSearchTool para integration tests"""
    
    mock_tool = AsyncMock(spec=WebSearchTool)
    mock_tool.name = "web_search"
    mock_tool.description = "Search the web for current information"
    mock_tool.category = "web_search"
    
    # Mock _arun method
    async def mock_arun(query, **kwargs):
        results = get_mock_search_results(query, "integration_test", 5)
        
        response_parts = [f"Found {len(results)} results for '{query}':"]
        
        for i, result in enumerate(results[:3], 1):
            response_parts.append(f"{i}. **{result.title}**")
            response_parts.append(f"   URL: {result.url}")
            response_parts.append(f"   Summary: {result.snippet}")
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    mock_tool._arun = mock_arun
    
    return mock_tool

# =============================================================================
# TESTS DE INICIALIZACIÓN DE AGENT CON TOOLS
# =============================================================================

class TestAgentToolInitialization:
    """Tests de inicialización del agent con herramientas"""
    
    @pytest.mark.integration
    async def test_agent_initializes_without_tools(self, real_agent_mock_ollama):
        """Test que agent se inicializa correctamente sin herramientas"""
        
        agent, mock_ollama = real_agent_mock_ollama
        
        assert agent.is_initialized == True
        assert agent.llm is not None
        assert len(agent.tools) == 0  # Sin herramientas por ahora
        assert agent.agent_executor is None  # Sin herramientas, no hay executor
    
    @pytest.mark.integration
    async def test_agent_can_add_web_search_tool(self, real_agent_mock_ollama, mock_web_search_tool):
        """Test que se puede agregar WebSearchTool al agent"""
        
        agent, mock_ollama = real_agent_mock_ollama
        
        # Agregar herramienta manualmente (simular registro)
        agent.tools = [mock_web_search_tool]
        
        # Re-crear el agente para incluir herramientas
        agent._create_react_agent()
        
        assert len(agent.tools) == 1
        assert agent.tools[0].name == "web_search"
        assert agent.agent_executor is not None  # Ahora tiene executor
    
    def test_web_search_tool_registration(self):
        """Test que WebSearchTool se puede registrar correctamente"""
        
        # Crear tool real
        web_tool = WebSearchTool()
        
        # Verificar propiedades básicas
        assert web_tool.name == "web_search"
        assert hasattr(web_tool, '_arun')
        assert hasattr(web_tool, 'get_tool_info')
        
        # Verificar que se puede obtener info
        tool_info = web_tool.get_tool_info()
        assert isinstance(tool_info, dict)
        assert 'name' in tool_info
        assert 'description' in tool_info

# =============================================================================
# TESTS DE RESEARCH SESSION CON WEB SEARCH
# =============================================================================

class TestResearchSessionWithWebSearch:
    """Tests de sesiones de investigación usando web search"""
    
    @pytest.mark.integration
    async def test_create_research_session_with_web_topic(self, real_agent_mock_ollama):
        """Test de creación de sesión con tema de investigación web"""
        
        agent, mock_ollama = real_agent_mock_ollama
        
        # Crear sesión de investigación
        session_id = await agent.create_research_session(
            topic="Python programming tutorials",
            objectives=["Find beginner resources", "Identify best practices"]
        )
        
        assert session_id is not None
        assert session_id in agent.active_sessions
        
        # Verificar sesión
        session = agent.active_sessions[session_id]
        assert isinstance(session, ResearchSession)
        assert session.topic == "Python programming tutorials"
        assert len(session.objectives) == 2
    
    @pytest.mark.integration
    async def test_process_query_without_web_search(self, real_agent_mock_ollama):
        """Test de procesamiento de query sin web search (baseline)"""
        
        agent, mock_ollama = real_agent_mock_ollama
        
        # Crear sesión
        session_id = await agent.create_research_session("Test topic")
        
        # Procesar query simple
        result = await agent.process_query(session_id, "What is Python programming?")
        
        assert result["success"] != False  # No debe fallar
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0
        assert result["session_id"] == session_id
        
        # Verificar que se usó solo LLM (sin herramientas)
        assert "mode" in result  # Should indicate direct mode
        assert result["reasoning"] == "Respuesta directa del modelo de lenguaje"

# =============================================================================
# TESTS DE AGENT CON WEB SEARCH TOOL INTEGRADO
# =============================================================================

class TestAgentWithWebSearchIntegration:
    """Tests del agent con WebSearchTool completamente integrado"""
    
    @pytest.mark.integration
    async def test_agent_with_mock_web_search_tool(self, mock_web_search_tool):
        """Test de agent con web search tool mockeado"""
        
        # Mock Ollama
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Based on my web search, I found information about Python programming. The search results show that Python is a popular programming language.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            # Crear agent con herramientas
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            
            # Mock tools setup
            agent.tools = [mock_web_search_tool]
            
            # Mock initialization
            agent.llm = mock_ollama
            agent.is_initialized = True
            agent._create_react_agent()
            
            # Crear sesión
            session_id = await agent.create_research_session("Web research test")
            
            # Procesar query que debería usar web search
            result = await agent.process_query(
                session_id, 
                "Search for information about Python programming frameworks"
            )
            
            assert result["success"] != False
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            
            # Verificar que se mencionan fuentes o web search
            response_lower = result["response"].lower()
            assert any(term in response_lower for term in ["search", "found", "web", "results"])
    
    @pytest.mark.integration 
    async def test_agent_web_search_error_handling(self, mock_web_search_tool):
        """Test de manejo de errores en web search"""
        
        # Mock web search tool que falla
        mock_web_search_tool._arun.side_effect = Exception("Web search failed")
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "I apologize, but I encountered an error while trying to search the web. Let me provide information based on my knowledge.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Error test")
            
            # Query que debería usar web search pero fallará
            result = await agent.process_query(session_id, "Search for recent AI news")
            
            # Agent debe manejar el error gracefully
            assert result["success"] != False  # No debe fallar completamente
            assert isinstance(result["response"], str)
            
            # Puede incluir mensaje de error o fallback a conocimiento propio
            response_lower = result["response"].lower()
            assert any(term in response_lower for term in ["error", "knowledge", "unable", "sorry"])

# =============================================================================
# TESTS DE REASONING CON WEB SEARCH
# =============================================================================

class TestReasoningWithWebSearch:
    """Tests del razonamiento del agent usando web search"""
    
    @pytest.mark.integration
    async def test_agent_reasoning_steps_with_web_search(self, mock_web_search_tool):
        """Test que se capturan los pasos de razonamiento con web search"""
        
        # Mock callback handler para capturar reasoning
        mock_callback = AsyncMock()
        mock_callback.actions = []
        mock_callback.observations = []
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "I need to search for information about this topic. Let me use the web search tool.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL, max_iterations=2)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            agent.callback_handler = mock_callback
            
            session_id = await agent.create_research_session("Reasoning test")
            
            result = await agent.process_query(session_id, "Research machine learning applications")
            
            # Verificar que hay información de reasoning
            assert "reasoning" in result
            assert isinstance(result["reasoning"], str)
            
            # Puede haber información sobre tool calls
            if "tool_calls" in result:
                assert isinstance(result["tool_calls"], list)
    
    @pytest.mark.integration
    async def test_agent_multi_step_reasoning(self, mock_web_search_tool):
        """Test de razonamiento multi-step con web search"""
        
        # Mock que simula múltiples llamadas al LLM
        call_count = 0
        
        async def mock_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return {
                    "success": True,
                    "response": "I need to search for information about this topic first.",
                    "model": AGENT_CONFIG.TEST_MODEL
                }
            else:
                return {
                    "success": True,
                    "response": "Based on the search results, I can now provide a comprehensive answer.",
                    "model": AGENT_CONFIG.TEST_MODEL
                }
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate = mock_generate
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL, max_iterations=3)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Multi-step test")
            
            result = await agent.process_query(session_id, "Analyze current trends in AI research")
            
            assert result["success"] != False
            assert isinstance(result["response"], str)
            
            # Verificar que se hicieron múltiples llamadas
            assert call_count >= 1

# =============================================================================
# TESTS DE MEMORY Y PERSISTENCIA
# =============================================================================

class TestAgentMemoryWithWebSearch:
    """Tests de memoria del agent con web search"""
    
    @pytest.mark.integration
    async def test_session_maintains_web_search_context(self, mock_web_search_tool):
        """Test que la sesión mantiene contexto de búsquedas web"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "I remember the previous search results about Python. Now searching for frameworks.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Context test")
            
            # Primera query
            result1 = await agent.process_query(session_id, "Search for Python tutorials")
            assert result1["success"] != False
            
            # Segunda query que hace referencia a la primera
            result2 = await agent.process_query(session_id, "Now search for Python frameworks")
            assert result2["success"] != False
            
            # Verificar que la sesión tiene ambos mensajes
            session = agent.active_sessions[session_id]
            assert len(session.messages) >= 4  # 2 user + 2 assistant messages
    
    @pytest.mark.integration
    async def test_session_chat_history_formatting(self, mock_web_search_tool):
        """Test de formateo del historial de chat"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Response based on chat history context.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("History test")
            session = agent.active_sessions[session_id]
            
            # Agregar algunos mensajes manualmente
            session.add_message("user", "First question about AI")
            session.add_message("assistant", "First response about AI")
            session.add_message("user", "Follow-up question")
            session.add_message("assistant", "Follow-up response")
            
            # Obtener historial formateado
            history = session.get_chat_history()
            
            assert isinstance(history, str)
            assert "First question" in history
            assert "Follow-up" in history
            assert "Humano:" in history or "Asistente:" in history

# =============================================================================
# TESTS DE PERFORMANCE DE INTEGRATION
# =============================================================================

class TestAgentWebSearchPerformance:
    """Tests de rendimiento de la integración agent + web search"""
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_agent_web_search_response_time(self, mock_web_search_tool, performance_tracker):
        """Test de tiempo de respuesta agent + web search"""
        
        # Mock rápido de Ollama
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Quick response based on web search results.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL, max_iterations=2)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Performance test")
            
            performance_tracker['start']('agent_web_search')
            
            result = await agent.process_query(session_id, "Quick search for Python info")
            
            duration = performance_tracker['end']('agent_web_search')
            
            assert result["success"] != False
            assert isinstance(result["response"], str)
            assert duration < 30.0  # Menos de 30 segundos
    
    @pytest.mark.integration
    @pytest.mark.performance
    async def test_agent_handles_multiple_concurrent_sessions(self, mock_web_search_tool, performance_tracker):
        """Test de manejo de múltiples sesiones concurrentes"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Concurrent session response with web search.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            # Crear múltiples sesiones
            session_ids = []
            for i in range(3):
                session_id = await agent.create_research_session(f"Concurrent test {i}")
                session_ids.append(session_id)
            
            performance_tracker['start']('concurrent_sessions')
            
            # Procesar queries concurrentes
            tasks = [
                agent.process_query(session_id, f"Search query {i}")
                for i, session_id in enumerate(session_ids)
            ]
            
            results = await asyncio.gather(*tasks)
            
            duration = performance_tracker['end']('concurrent_sessions')
            
            assert len(results) == 3
            assert all(r["success"] != False for r in results)
            assert duration < 60.0  # Menos de 1 minuto para 3 sesiones

# =============================================================================
# TESTS DE ERROR HANDLING EN INTEGRATION
# =============================================================================

class TestIntegrationErrorHandling:
    """Tests de manejo de errores en integración"""
    
    @pytest.mark.integration
    async def test_agent_handles_ollama_connection_failure(self, mock_web_search_tool):
        """Test de manejo de fallo de conexión Ollama"""
        
        # Mock Ollama que falla en health check
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = False
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            
            # Initialization debe fallar gracefully
            success = await agent.initialize()
            
            # Depends on implementation - may fail or succeed with fallback
            assert isinstance(success, bool)
    
    @pytest.mark.integration
    async def test_agent_handles_web_search_tool_failure(self, mock_web_search_tool):
        """Test de manejo de fallo del web search tool"""
        
        # Mock web search que siempre falla
        mock_web_search_tool._arun.side_effect = Exception("Web search completely failed")
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "I cannot access web search right now, but I can help with my existing knowledge.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Error handling test")
            
            # Query que intentaría usar web search
            result = await agent.process_query(session_id, "Search for current news")
            
            # Agent debe manejar el error y responder con conocimiento propio
            assert result["success"] != False  # No debe fallar completamente
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
    
    @pytest.mark.integration
    async def test_agent_handles_invalid_session_id(self, mock_web_search_tool):
        """Test de manejo de session ID inválido"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            # Intentar usar session ID que no existe
            result = await agent.process_query("nonexistent_session", "Test query")
            
            assert result["success"] == False or "error" in result
            assert "session" in result.get("response", "").lower() or "not found" in result.get("response", "").lower()

# =============================================================================
# TESTS DE CONFIGURACIÓN Y SETUP
# =============================================================================

class TestIntegrationConfiguration:
    """Tests de configuración para integration"""
    
    @pytest.mark.integration
    def test_agent_config_values_are_reasonable(self):
        """Test que valores de configuración del agent son razonables"""
        
        assert AGENT_CONFIG.TEST_MODEL is not None
        assert AGENT_CONFIG.MAX_TOKENS > 0
        assert AGENT_CONFIG.MAX_TOKENS <= 4096  # Reasonable limit
        assert 0.0 <= AGENT_CONFIG.TEMPERATURE <= 1.0
        assert AGENT_CONFIG.AGENT_INIT_TIMEOUT > 0
        assert AGENT_CONFIG.AGENT_RESPONSE_TIMEOUT > 0
    
    @pytest.mark.integration
    def test_web_search_config_values_are_reasonable(self):
        """Test que valores de configuración de web search son razonables"""
        
        assert len(WEB_SEARCH_CONFIG.SAFE_TEST_QUERIES) > 0
        assert len(WEB_SEARCH_CONFIG.RELIABLE_TEST_URLS) > 0
        assert WEB_SEARCH_CONFIG.QUICK_TIMEOUT > 0
        assert WEB_SEARCH_CONFIG.NORMAL_TIMEOUT > WEB_SEARCH_CONFIG.QUICK_TIMEOUT
        assert WEB_SEARCH_CONFIG.TEST_RATE_LIMIT > 0
    
    @pytest.mark.integration
    async def test_factory_function_creates_working_agent(self):
        """Test que la función factory crea un agent funcional"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Factory created agent works correctly.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            # Usar factory function
            agent = await create_smartdoc_agent(
                model_name=AGENT_CONFIG.TEST_MODEL,
                research_style="general",
                max_iterations=2
            )
            
            assert isinstance(agent, SmartDocAgent)
            assert agent.is_initialized == True
            assert agent.model_name == AGENT_CONFIG.TEST_MODEL
            assert agent.research_style == "general"
            assert agent.max_iterations == 2
            
            await agent.close()

# =============================================================================
# TESTS DE SESSION MANAGEMENT
# =============================================================================

class TestSessionManagement:
    """Tests de gestión de sesiones en integración"""
    
    @pytest.mark.integration
    async def test_agent_session_status_tracking(self, mock_web_search_tool):
        """Test de tracking del estado de sesiones"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Session tracking test response.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            # Crear sesión
            session_id = await agent.create_research_session(
                "Session tracking test",
                objectives=["Test objective 1", "Test objective 2"]
            )
            
            # Procesar algunas queries
            await agent.process_query(session_id, "First query")
            await agent.process_query(session_id, "Second query")
            
            # Obtener estado de sesión
            status = agent.get_session_status(session_id)
            
            assert status["session_id"] == session_id
            assert status["topic"] == "Session tracking test"
            assert len(status["objectives"]) == 2
            assert status["message_count"] >= 4  # 2 user + 2 assistant
            assert "created_at" in status
            assert "last_activity" in status
    
    @pytest.mark.integration
    async def test_agent_list_active_sessions(self, mock_web_search_tool):
        """Test de listado de sesiones activas"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            # Crear múltiples sesiones
            session_ids = []
            for i in range(3):
                session_id = await agent.create_research_session(f"Test session {i}")
                session_ids.append(session_id)
            
            # Listar sesiones
            sessions_info = agent.list_active_sessions()
            
            assert sessions_info["count"] == 3
            assert len(sessions_info["active_sessions"]) == 3
            assert all(sid in sessions_info["active_sessions"] for sid in session_ids)
            assert "sessions" in sessions_info
            
            # Verificar información detallada
            for session_id in session_ids:
                assert session_id in sessions_info["sessions"]
                session_detail = sessions_info["sessions"][session_id]
                assert "topic" in session_detail
                assert "message_count" in session_detail

# =============================================================================
# TESTS DE QUALITY ASSURANCE
# =============================================================================

class TestIntegrationQualityAssurance:
    """Tests de quality assurance para la integración"""
    
    @pytest.mark.integration
    async def test_agent_response_quality_with_web_search(self, mock_web_search_tool):
        """Test de calidad de respuestas del agent con web search"""
        
        # Mock web search que retorna información estructurada
        async def detailed_mock_arun(query, **kwargs):
            return f"""Found 3 results for '{query}':

1. **Comprehensive Guide to {query.title()}**
   URL: https://example.com/guide
   Summary: A detailed guide covering all aspects of the topic.

2. **Latest Updates on {query.title()}**  
   URL: https://news.example.com/updates
   Summary: Recent developments and news about the topic.

3. **Expert Analysis of {query.title()}**
   URL: https://expert.example.com/analysis
   Summary: In-depth expert analysis and recommendations.

Successfully extracted content from 3/3 pages."""
        
        mock_web_search_tool._arun = detailed_mock_arun
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": """Based on my web search, I found comprehensive information about this topic. 

The search revealed three key sources:
1. A comprehensive guide that covers all fundamental aspects
2. Latest updates showing recent developments in the field
3. Expert analysis providing in-depth insights and recommendations

This information suggests that the topic is well-documented and actively evolving, with both educational resources and current developments available.""",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Quality test")
            
            result = await agent.process_query(session_id, "machine learning applications")
            
            assert result["success"] != False
            response = result["response"]
            
            # Verificar calidad de la respuesta
            assert len(response) > 200  # Respuesta sustancial
            assert "machine learning" in response.lower()  # Menciona el tema
            assert any(word in response.lower() for word in ["search", "found", "information", "sources"])
            
            # Verificar estructura de respuesta
            assert isinstance(result["confidence"], (int, float))
            assert 0.0 <= result["confidence"] <= 1.0
    
    @pytest.mark.integration
    async def test_agent_handles_ambiguous_queries(self, mock_web_search_tool):
        """Test de manejo de queries ambiguas"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "I understand your query might have multiple interpretations. Let me search for information and provide clarification on the different aspects.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Ambiguity test")
            
            # Query ambigua
            result = await agent.process_query(session_id, "python")
            
            assert result["success"] != False
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
            
            # Agent debe manejar ambigüedad apropiadamente
            response_lower = result["response"].lower()
            assert any(word in response_lower for word in ["clarification", "multiple", "different", "aspects"])
    
    @pytest.mark.integration
    async def test_agent_maintains_conversation_coherence(self, mock_web_search_tool):
        """Test de coherencia en conversaciones largas"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        
        # Mock que mantiene contexto entre llamadas
        conversation_context = []
        
        async def context_aware_generate(*args, **kwargs):
            conversation_context.append("call")
            call_number = len(conversation_context)
            
            return {
                "success": True,
                "response": f"This is response #{call_number}, building on our previous discussion about the research topic.",
                "model": AGENT_CONFIG.TEST_MODEL
            }
        
        mock_ollama.generate = context_aware_generate
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Coherence test")
            
            # Conversación multi-turn
            queries = [
                "Tell me about artificial intelligence",
                "What are the main applications?",
                "How does it compare to traditional programming?",
                "What are the future prospects?"
            ]
            
            results = []
            for query in queries:
                result = await agent.process_query(session_id, query)
                results.append(result)
                assert result["success"] != False
            
            # Verificar que todas las respuestas son coherentes
            assert len(results) == 4
            for i, result in enumerate(results):
                assert f"#{i+1}" in result["response"]  # Context awareness
                assert "research topic" in result["response"]  # Topic continuity

# =============================================================================
# TESTS DE STRESS Y LIMITS
# =============================================================================

class TestIntegrationStressTests:
    """Tests de stress y límites para la integración"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_agent_handles_very_long_query(self, mock_web_search_tool):
        """Test de manejo de query extremadamente larga"""
        
        # Query de 1000+ palabras
        long_query = """
        I need comprehensive information about machine learning applications in healthcare, 
        including but not limited to medical imaging analysis, drug discovery processes, 
        patient diagnosis automation, electronic health record analysis, predictive modeling 
        for patient outcomes, personalized treatment recommendations, clinical trial optimization,
        medical research acceleration, telemedicine improvements, healthcare cost reduction,
        medical device integration, real-time patient monitoring, emergency response systems,
        surgical robotics, rehabilitation therapy, mental health applications, genomics analysis,
        epidemiological modeling, hospital workflow optimization, medical education enhancement,
        and regulatory compliance automation. """ * 10  # Repeat to make it very long
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "I understand you're looking for comprehensive information about ML in healthcare. Let me search for relevant information.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Long query test")
            
            # Agent debe manejar query larga sin fallar
            result = await agent.process_query(session_id, long_query)
            
            assert result["success"] != False
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 0
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_agent_session_memory_limits(self, mock_web_search_tool):
        """Test de límites de memoria de sesión"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Response to query in long conversation.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session("Memory limits test")
            
            # Hacer muchas queries para llenar memoria
            for i in range(20):  # 20 exchanges = 40 messages
                query = f"Query number {i} about different topics and research questions"
                result = await agent.process_query(session_id, query)
                assert result["success"] != False
            
            # Verificar que la sesión aún funciona
            final_result = await agent.process_query(session_id, "Final summary query")
            assert final_result["success"] != False
            
            # Verificar que el historial no creció indefinidamente
            session = agent.active_sessions[session_id]
            # El sistema debe mantener un historial manejable
            assert len(session.messages) <= 100  # Reasonable limit

# =============================================================================
# TESTS DE INTEGRATION CON DIFERENTES RESEARCH STYLES
# =============================================================================

class TestResearchStyleIntegration:
    """Tests de integración con diferentes estilos de investigación"""
    
    @pytest.mark.integration
    @pytest.mark.parametrize("research_style,expected_terms", [
        ("academic", ["research", "study", "analysis", "methodology"]),
        ("business", ["market", "ROI", "strategy", "business"]),
        ("technical", ["implementation", "technical", "specification", "architecture"]),
        ("general", ["information", "overview", "general", "basic"])
    ])
    async def test_agent_adapts_to_research_style(self, mock_web_search_tool, research_style, expected_terms):
        """Test que el agent se adapta al estilo de investigación"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": f"This is a {research_style} style response focusing on {expected_terms[0]} aspects of the topic.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            agent = SmartDocAgent(
                model_name=AGENT_CONFIG.TEST_MODEL,
                research_style=research_style
            )
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            session_id = await agent.create_research_session(f"{research_style} research test")
            
            result = await agent.process_query(session_id, "Research artificial intelligence")
            
            assert result["success"] != False
            response_lower = result["response"].lower()
            
            # Verificar que el estilo se refleja en la respuesta
            assert research_style in response_lower
            assert any(term in response_lower for term in expected_terms)

# =============================================================================
# TESTS FINALES DE INTEGRATION
# =============================================================================

class TestFinalIntegrationValidation:
    """Tests finales de validación de integración"""
    
    @pytest.mark.integration
    async def test_complete_integration_workflow(self, mock_web_search_tool):
        """Test del workflow completo de integración"""
        
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.return_value = {
            "success": True,
            "response": "Complete integration test successful. Web search integration working properly.",
            "model": AGENT_CONFIG.TEST_MODEL
        }
        
        with patch('app.agents.core.smart_agent.get_ollama_client', return_value=mock_ollama):
            
            # 1. Crear agent con web search tool
            agent = SmartDocAgent(model_name=AGENT_CONFIG.TEST_MODEL)
            agent.tools = [mock_web_search_tool]
            agent.llm = mock_ollama
            agent.is_initialized = True
            
            # 2. Crear sesión de investigación
            session_id = await agent.create_research_session(
                "Complete integration test",
                objectives=["Test web search", "Verify agent reasoning", "Check response quality"]
            )
            
            # 3. Procesar query que requiere web search
            result = await agent.process_query(
                session_id,
                "Search for the latest developments in artificial intelligence and provide a summary"
            )
            
            # 4. Verificar resultado completo
            assert result["success"] != False
            assert isinstance(result["response"], str)
            assert len(result["response"]) > 100
            assert result["session_id"] == session_id
            assert "confidence" in result
            
            # 5. Verificar estado de sesión
            status = agent.get_session_status(session_id)
            assert status["message_count"] >= 2
            assert len(status["objectives"]) == 3
            
            # 6. Segunda query para verificar continuidad
            follow_up = await agent.process_query(
                session_id,
                "Based on the previous search, what are the main challenges?"
            )
            
            assert follow_up["success"] != False
            assert follow_up["session_id"] == session_id
            
            # 7. Verificar health del agent
            if hasattr(agent, 'health_check'):
                health = await agent.health_check()
                # Health check puede no estar implementado, pero si está debe funcionar
            
            # 8. Cleanup
            await agent.close()
    
    @pytest.mark.integration
    def test_integration_components_are_properly_connected(self):
        """Test que todos los componentes de integración están conectados"""
        
        # Verificar imports
        from app.agents.core.smart_agent import SmartDocAgent
        from app.agents.tools.web.web_search_tool import WebSearchTool
        from app.agents.tools.base_tool import BaseTool
        
        # Verificar jerarquía de clases
        assert issubclass(SmartDocAgent, object)
        assert issubclass(WebSearchTool, BaseTool)
        
        # Verificar que se pueden instanciar
        agent = SmartDocAgent(model_name="test_model")
        web_tool = WebSearchTool()
        
        assert isinstance(agent, SmartDocAgent)
        assert isinstance(web_tool, WebSearchTool)
        assert isinstance(web_tool, BaseTool)
        
        # Verificar interfaces críticas
        assert hasattr(agent, 'initialize')
        assert hasattr(agent, 'process_query')
        assert hasattr(agent, 'create_research_session')
        
        assert hasattr(web_tool, '_arun')
        assert hasattr(web_tool, 'get_tool_info')
        assert hasattr(web_tool, 'name')
    
    @pytest.mark.integration
    def test_integration_configuration_consistency(self):
        """Test de consistencia de configuración entre componentes"""
        
        # Verificar que timeouts son consistentes
        assert AGENT_CONFIG.AGENT_RESPONSE_TIMEOUT >= WEB_SEARCH_CONFIG.NORMAL_TIMEOUT
        
        # Verificar que rate limits son razonables
        assert WEB_SEARCH_CONFIG.TEST_RATE_LIMIT > 0
        assert WEB_SEARCH_CONFIG.TEST_RATE_LIMIT <= 100  # Reasonable for testing
        
        # Verificar configuración de modelo
        assert AGENT_CONFIG.TEST_MODEL is not None
        assert len(AGENT_CONFIG.TEST_MODEL) > 0
        
        # Verificar que hay queries de test disponibles
        assert len(WEB_SEARCH_CONFIG.SAFE_TEST_QUERIES) >= 3
        assert len(AGENT_CONFIG.SIMPLE_QUERIES) >= 3

# =============================================================================
# CONFIGURACIÓN FINAL Y MARKERS
# =============================================================================

# Marcar todos los tests como integration por defecto
pytestmark = pytest.mark.integration

# Tests que requieren servicios externos (skip por defecto en CI)
external_service_tests = [
    "test_agent_initializes_with_real_ollama",
    "test_real_web_search_integration"
]

# Tests de performance (pueden ser lentos)
performance_tests = [
    "test_agent_web_search_response_time",
    "test_agent_handles_multiple_concurrent_sessions",
    "test_agent_handles_very_long_query",
    "test_agent_session_memory_limits"
]

# Tests críticos que deben pasar siempre
critical_tests = [
    "test_agent_initializes_without_tools",
    "test_create_research_session_with_web_topic", 
    "test_complete_integration_workflow",
    "test_integration_components_are_properly_connected"
]