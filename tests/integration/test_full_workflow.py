"""
Tests de flujo completo (End-to-End) para SmartDoc Research Agent

Este módulo contiene tests que verifican el funcionamiento completo del agente
desde la inicialización hasta la generación de reportes, incluyendo:
- Creación de sesiones de investigación
- Procesamiento de queries complejas
- Uso de múltiples herramientas
- Mantenimiento de contexto
- Generación de respuestas coherentes
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, List, Any

from app.agents.core.smart_agent import SmartDocAgent
from app.agents.tools.web.web_search_tool import WebSearchTool
from tests.fixtures import (
    E2E_RESEARCH_SCENARIO,
    MOCK_SEARCH_RESULTS,
    MOCK_WEB_CONTENT,
    PERFORMANCE_BENCHMARKS
)


# =============================================================================
# TESTS DE FLUJO COMPLETO - ESCENARIOS REALISTAS
# =============================================================================

class TestCompleteWorkflow:
    """Tests de flujo completo del agente de investigación"""
    
    @pytest.fixture
    async def fully_mocked_agent(self):
        """Agent completamente mockeado para tests E2E"""
        
        # Mock del LLM
        mock_ollama = AsyncMock()
        mock_ollama.health_check.return_value = True
        mock_ollama.generate.side_effect = self._generate_realistic_responses
        
        # Mock del WebSearchTool
        mock_web_tool = AsyncMock(spec=WebSearchTool)
        mock_web_tool.name = "web_search"
        mock_web_tool._arun.side_effect = self._mock_web_search_responses
        mock_web_tool.get_tool_info.return_value = {
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {"query": "string"}
        }
        
        # Crear agente con mocks
        with patch('app.agents.core.smart_agent.OllamaLLM', return_value=mock_ollama):
            agent = SmartDocAgent(model_name="test_model")
            
            # Inject mocked tool
            agent.tools = [mock_web_tool]
            agent.initialized = True
            
        return agent, mock_ollama, mock_web_tool
    
    def _generate_realistic_responses(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generar respuestas realistas del LLM basadas en el prompt"""
        
        if "THOUGHT:" in prompt or "planning" in prompt.lower():
            return {
                "success": True,
                "response": """THOUGHT: I need to search for information about this topic to provide a comprehensive answer.

ACTION: web_search
ACTION_INPUT: {"query": "artificial intelligence latest developments 2024"}

OBSERVATION: [This will be filled by the web search tool]

THOUGHT: Based on the search results, I can now provide a detailed response about AI developments."""
            }
        
        elif "search results" in prompt.lower() or "web_search" in prompt:
            return {
                "success": True,
                "response": """Based on my search, here are the latest developments in artificial intelligence:

1. **Large Language Models**: Significant advances in model efficiency and capabilities
2. **Multimodal AI**: Integration of text, image, and audio processing
3. **AI Safety**: Increased focus on alignment and safety research
4. **Edge Computing**: More AI applications running on local devices
5. **Industry Applications**: Healthcare, finance, and education seeing major AI adoption

The field is rapidly evolving with new breakthroughs emerging regularly."""
            }
        
        elif "follow up" in prompt.lower() or "previous" in prompt.lower():
            return {
                "success": True,
                "response": """Based on my previous research on AI developments, the main challenges include:

1. **Technical Challenges**:
   - Computational requirements and energy consumption
   - Model interpretability and explainability
   - Handling bias and fairness in AI systems

2. **Ethical Challenges**:
   - Privacy and data protection
   - Job displacement concerns
   - AI safety and alignment

3. **Regulatory Challenges**:
   - Need for updated legislation
   - International coordination on AI governance
   - Balancing innovation with safety

These challenges require collaborative efforts from researchers, policymakers, and industry leaders."""
            }
        
        else:
            return {
                "success": True,
                "response": "I understand your request. Let me help you with that information."
            }
    
    def _mock_web_search_responses(self, query: str) -> str:
        """Generar respuestas de web search basadas en la query"""
        
        if "artificial intelligence" in query.lower():
            return """Found 8 relevant results about artificial intelligence developments:

1. "AI Breakthrough: New Language Models Show Enhanced Reasoning" - TechNews
2. "2024 AI Industry Report: Key Trends and Predictions" - ResearchFirm  
3. "Multimodal AI Systems: The Next Frontier" - ScienceJournal
4. "AI Safety Research: Recent Advances" - SafetyInstitute
5. "Edge AI: Bringing Intelligence to Local Devices" - TechReview
6. "Healthcare AI Applications: Current State" - MedicalJournal
7. "AI in Finance: Transforming the Industry" - FinanceNews
8. "Educational AI: Personalized Learning Systems" - EduTech

These sources provide comprehensive coverage of current AI developments."""
        
        elif "challenges" in query.lower():
            return """Found 6 relevant results about AI challenges:

1. "Major Challenges Facing AI Development" - AIResearch
2. "Ethical Considerations in AI Systems" - EthicsJournal  
3. "Technical Hurdles in Large Scale AI" - TechAnalysis
4. "AI Regulation: Global Perspectives" - PolicyReview
5. "Bias and Fairness in Machine Learning" - MLJournal
6. "Energy Consumption of AI Systems" - GreenTech

These sources detail the key challenges in AI development and deployment."""
        
        else:
            return f"Found 5 relevant results for '{query}' with detailed information from various sources."

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_complete_research_workflow(self, fully_mocked_agent):
        """Test del flujo completo de investigación"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        # 1. Inicializar agente
        await agent.initialize()
        assert agent.initialized
        
        # 2. Crear sesión de investigación
        session_id = await agent.create_research_session(
            topic=E2E_RESEARCH_SCENARIO["topic"],
            objectives=E2E_RESEARCH_SCENARIO["objectives"]
        )
        
        assert session_id is not None
        assert len(session_id) > 0
        
        # 3. Procesar queries secuenciales
        responses = []
        
        for i, query in enumerate(E2E_RESEARCH_SCENARIO["queries"]):
            print(f"\n--- Processing Query #{i+1}: {query} ---")
            
            result = await agent.process_query(session_id, query)
            
            # Verificaciones básicas
            assert result["success"] != False
            assert isinstance(result["response"], str)
            assert len(result["response"]) >= 50
            assert result["session_id"] == session_id
            
            responses.append(result)
            
            # Pequeña pausa para simular uso real
            await asyncio.sleep(0.1)
        
        # 4. Verificar continuidad entre respuestas
        assert len(responses) == len(E2E_RESEARCH_SCENARIO["queries"])
        
        # Verificar que web search fue llamado
        assert mock_web_tool._arun.call_count >= len(responses)
        
        # 5. Verificar estado final de sesión
        session_status = agent.get_session_status(session_id)
        assert session_status["message_count"] >= len(responses) * 2  # Queries + responses
        assert session_status["topic"] == E2E_RESEARCH_SCENARIO["topic"]
        
        # 6. Cleanup
        await agent.close()
        
        print(f"\n✅ Complete workflow test passed with {len(responses)} queries processed")

    @pytest.mark.integration
    async def test_multi_tool_coordination(self, fully_mocked_agent):
        """Test de coordinación entre múltiples herramientas"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        # Mock de herramienta adicional (calculator)
        mock_calc_tool = AsyncMock()
        mock_calc_tool.name = "calculator"
        mock_calc_tool._arun.return_value = "42"
        mock_calc_tool.get_tool_info.return_value = {
            "name": "calculator", 
            "description": "Perform calculations"
        }
        
        # Añadir herramienta adicional
        agent.tools.append(mock_calc_tool)
        
        await agent.initialize()
        
        # Crear sesión
        session_id = await agent.create_research_session(
            "Data analysis project",
            objectives=["Research methods", "Calculate statistics", "Generate insights"]
        )
        
        # Query que requiere múltiples herramientas
        complex_query = """
        I need to research machine learning algorithms and then calculate 
        the accuracy improvement: if model A has 85% accuracy and model B 
        has 92% accuracy, what's the percentage improvement?
        """
        
        result = await agent.process_query(session_id, complex_query)
        
        # Verificaciones
        assert result["success"] != False
        assert len(result["response"]) > 100
        
        # Al menos una herramienta debería haberse usado
        total_tool_calls = mock_web_tool._arun.call_count + mock_calc_tool._arun.call_count
        assert total_tool_calls >= 1
        
        await agent.close()

    @pytest.mark.integration
    async def test_error_recovery_workflow(self, fully_mocked_agent):
        """Test de recuperación de errores durante el flujo"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        # Configurar web tool para fallar la primera vez
        call_count = 0
        
        def failing_web_search(query: str):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise Exception("Simulated network error")
            else:
                return "Successfully recovered and found relevant information about the topic."
        
        mock_web_tool._arun.side_effect = failing_web_search
        
        await agent.initialize()
        
        session_id = await agent.create_research_session(
            "Error recovery test",
            objectives=["Test error handling"]
        )
        
        # Esta query debería fallar primero, luego recuperarse
        result = await agent.process_query(
            session_id, 
            "Search for information about error handling in AI systems"
        )
        
        # El agente debería manejar el error graciosamente
        assert result["success"] != False
        assert isinstance(result["response"], str)
        
        # Verificar que se intentó la recuperación
        assert call_count >= 1
        
        await agent.close()

# =============================================================================
# TESTS DE PERFORMANCE E2E
# =============================================================================

class TestWorkflowPerformance:
    """Tests de performance del flujo completo"""
    
    @pytest.mark.integration
    @pytest.mark.slow
    async def test_workflow_response_time(self, fully_mocked_agent):
        """Test de tiempo de respuesta del flujo completo"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        await agent.initialize()
        
        session_id = await agent.create_research_session(
            "Performance test",
            objectives=["Test response time"]
        )
        
        import time
        
        start_time = time.time()
        
        result = await agent.process_query(
            session_id,
            "Quick search for Python programming best practices"
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Verificar que la respuesta fue exitosa
        assert result["success"] != False
        
        # Verificar tiempo de respuesta (con mocks debería ser rápido)
        max_time = PERFORMANCE_BENCHMARKS["agent_response"]["simple_query"]["max_time"]
        assert response_time < max_time, f"Response took {response_time:.2f}s, max allowed: {max_time}s"
        
        await agent.close()
        
        print(f"✅ Workflow completed in {response_time:.2f} seconds")

    @pytest.mark.integration 
    @pytest.mark.slow
    async def test_concurrent_sessions_workflow(self, fully_mocked_agent):
        """Test de múltiples sesiones concurrentes"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        await agent.initialize()
        
        # Crear múltiples sesiones
        sessions = []
        for i in range(3):
            session_id = await agent.create_research_session(
                f"Concurrent test session {i+1}",
                objectives=[f"Test concurrency {i+1}"]
            )
            sessions.append(session_id)
        
        # Procesar queries concurrentemente
        async def process_session_query(session_id: str, query_num: int):
            return await agent.process_query(
                session_id,
                f"Concurrent query #{query_num} for session {session_id[:8]}"
            )
        
        # Ejecutar queries concurrentes
        tasks = []
        for i, session_id in enumerate(sessions):
            task = process_session_query(session_id, i+1)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verificar que todas las sesiones funcionaron
        successful_results = 0
        for result in results:
            if isinstance(result, dict) and result.get("success") != False:
                successful_results += 1
        
        assert successful_results >= 2, f"Only {successful_results}/3 concurrent sessions succeeded"
        
        await agent.close()
        
        print(f"✅ {successful_results}/3 concurrent sessions completed successfully")

# =============================================================================
# TESTS DE INTEGRACIÓN CON CONTEXTO REALISTA
# =============================================================================

class TestRealisticScenarios:
    """Tests con escenarios realistas de uso"""
    
    @pytest.mark.integration
    async def test_academic_research_scenario(self, fully_mocked_agent):
        """Test de escenario de investigación académica"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        await agent.initialize()
        
        # Escenario: Estudiante investigando para tesis
        session_id = await agent.create_research_session(
            "Machine Learning in Healthcare - Literature Review",
            objectives=[
                "Find recent research papers on ML in healthcare",
                "Identify key applications and success stories", 
                "Understand current challenges and limitations",
                "Compile comprehensive bibliography"
            ]
        )
        
        # Secuencia de queries típica de investigación académica
        academic_queries = [
            "What are the most recent applications of machine learning in healthcare?",
            "Can you find specific case studies of successful ML implementations in hospitals?",
            "What are the main challenges and ethical considerations?",
            "Based on this research, what would be good future research directions?"
        ]
        
        responses = []
        for query in academic_queries:
            result = await agent.process_query(session_id, query)
            assert result["success"] != False
            assert len(result["response"]) >= 100  # Respuestas académicas detalladas
            responses.append(result)
        
        # Verificar que las respuestas muestran continuidad académica
        final_response = responses[-1]["response"]
        assert "research" in final_response.lower()
        assert len(responses) == 4
        
        await agent.close()

    @pytest.mark.integration
    async def test_business_analysis_scenario(self, fully_mocked_agent):
        """Test de escenario de análisis de negocio"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        await agent.initialize()
        
        # Escenario: Analista investigando competencia
        session_id = await agent.create_research_session(
            "Competitive Analysis - AI Startups 2024",
            objectives=[
                "Identify key players in AI startup ecosystem",
                "Analyze funding trends and valuations",
                "Understand market opportunities",
                "Generate strategic recommendations"
            ]
        )
        
        business_queries = [
            "Who are the leading AI startups that received funding in 2024?",
            "What are the main market trends in AI investments?",
            "Which sectors are seeing the most AI startup activity?",
            "What strategic opportunities exist for new AI companies?"
        ]
        
        responses = []
        for query in business_queries:
            result = await agent.process_query(session_id, query)
            assert result["success"] != False
            responses.append(result)
        
        # Verificar enfoque de negocio en las respuestas
        business_keywords = ["market", "investment", "competition", "strategy", "opportunity"]
        
        combined_responses = " ".join([r["response"] for r in responses])
        keyword_matches = sum(1 for keyword in business_keywords 
                            if keyword in combined_responses.lower())
        
        assert keyword_matches >= 3, "Responses should contain business-focused content"
        
        await agent.close()

# =============================================================================
# TESTS DE VALIDACIÓN FINAL
# =============================================================================

class TestWorkflowValidation:
    """Tests de validación final del sistema completo"""
    
    @pytest.mark.integration
    async def test_end_to_end_system_health(self, fully_mocked_agent):
        """Test de salud completa del sistema E2E"""
        
        agent, mock_ollama, mock_web_tool = fully_mocked_agent
        
        # 1. Verificar inicialización
        await agent.initialize()
        assert agent.initialized
        
        # 2. Verificar creación de sesión
        session_id = await agent.create_research_session("Health check", ["Test system"])
        assert session_id is not None
        
        # 3. Verificar procesamiento básico
        result = await agent.process_query(session_id, "Simple health check query")
        assert result["success"] != False
        
        # 4. Verificar estado de sesión
        status = agent.get_session_status(session_id)
        assert status["message_count"] >= 2
        
        # 5. Verificar herramientas
        assert len(agent.tools) >= 1
        assert any(tool.name == "web_search" for tool in agent.tools)
        
        # 6. Verificar cleanup
        await agent.close()
        
        print("✅ End-to-end system health check passed")

    @pytest.mark.integration
    def test_workflow_configuration_validation(self):
        """Test de validación de configuración del workflow"""
        
        # Verificar imports críticos
        from app.agents.core.smart_agent import SmartDocAgent
        from app.agents.tools.web.web_search_tool import WebSearchTool
        
        # Verificar que las clases se pueden instanciar
        agent = SmartDocAgent(model_name="test")
        web_tool = WebSearchTool()
        
        assert isinstance(agent, SmartDocAgent)
        assert isinstance(web_tool, WebSearchTool)
        
        # Verificar interfaces críticas
        required_agent_methods = [
            'initialize', 'create_research_session', 
            'process_query', 'get_session_status', 'close'
        ]
        
        for method in required_agent_methods:
            assert hasattr(agent, method), f"Agent missing required method: {method}"
        
        # Verificar configuración de herramientas
        assert hasattr(web_tool, 'name')
        assert hasattr(web_tool, '_arun')
        assert web_tool.name == "web_search"
        
        print("✅ Workflow configuration validation passed")


if __name__ == "__main__":
    # Ejecutar tests específicos si se ejecuta directamente
    pytest.main([__file__, "-v", "--tb=short"])