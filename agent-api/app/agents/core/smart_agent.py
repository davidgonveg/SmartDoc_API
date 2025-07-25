"""
SmartDoc Agent - Clase principal del agente de investigación
Implementación del agente usando LangChain + Ollama con patrón ReAct
"""


import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# IMPORTS LANGCHAIN REALES - CRÍTICOS
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_ollama import ChatOllama  # ← CAMBIADO de OllamaLLM
from langchain.prompts import PromptTemplate

# Imports del proyecto
from app.agents.prompts.react_templates import ReactTemplates, SMARTDOC_SYSTEM_PROMPT, REACT_MAIN_TEMPLATE
from app.agents.tools.web.web_search_tool import WebSearchTool  # ← IMPORT DEL TOOL REAL
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class SmartDocCallbackHandler(AsyncCallbackHandler):
    """Callback handler para capturar el pensamiento del agente"""
    
    def __init__(self):
        self.thoughts = []
        self.actions = []
        self.observations = []
        self.current_step = 0
        
    async def on_agent_action(self, action, **kwargs):
        """Captura acciones del agente"""
        self.actions.append({
            "step": self.current_step,
            "tool": action.tool,
            "input": action.tool_input,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"Agente ejecutando: {action.tool} con input: {action.tool_input}")
        
    async def on_agent_finish(self, finish, **kwargs):
        """Captura finalización del agente"""
        logger.info(f"Agente completado con output: {finish.return_values}")
        
    async def on_tool_end(self, output, **kwargs):
        """Captura resultados de herramientas"""
        self.observations.append({
            "step": self.current_step,
            "output": str(output)[:500],  # Limitar longitud
            "timestamp": datetime.now().isoformat()
        })
        self.current_step += 1

class ResearchSession:
    """Clase para manejar sesiones de investigación"""
    
    def __init__(self, session_id: str, topic: str, objectives: List[str] = None):
        self.session_id = session_id
        self.topic = topic
        self.objectives = objectives or []
        self.created_at = datetime.now()
        self.messages = []
        self.sources_found = []
        self.confidence_scores = []
        self.reasoning_steps = []
        
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Agregar mensaje a la sesión"""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
        
    def add_source(self, source_type: str, content: str, relevance: float, url: str = None):
        """Agregar fuente encontrada"""
        self.sources_found.append({
            "type": source_type,
            "content": content[:200],  # Resumen
            "relevance": relevance,
            "url": url,
            "timestamp": datetime.now().isoformat()
        })
        
    def get_chat_history(self) -> str:
        """Obtener historial formateado para el prompt"""
        if not self.messages:
            return "No hay mensajes previos."
            
        history = []
        for msg in self.messages[-6:]:  # Solo últimos 6 mensajes
            role = "Humano" if msg["role"] == "user" else "Asistente"
            history.append(f"{role}: {msg['content']}")
            
        return "\n".join(history)

class SmartDocAgent:
    """Agente principal de investigación SmartDoc"""
    
    def __init__(self, 
                 model_name: str = None,
                 ollama_host: str = None,
                 research_style: str = "general",
                 max_iterations: int = 5):
        
        self.settings = get_settings()
        self.model_name = model_name or self.settings.default_model
        self.ollama_host = ollama_host or self.settings.ollama_host
        self.research_style = research_style
        self.max_iterations = max_iterations
        
        # Componentes principales
        self.llm = None
        self.agent_executor = None
        self.tools = []
        self.memory = None
        self.callback_handler = None
        
        # Estado del agente
        self.is_initialized = False
        self.active_sessions: Dict[str, ResearchSession] = {}
        
        logger.info(f"SmartDocAgent creado - Modelo: {self.model_name}, Estilo: {research_style}")
    
    async def initialize(self) -> bool:
        """Inicializar el agente LangChain REAL"""
        try:
            logger.info("🚀 Inicializando SmartDocAgent con LangChain...")
            
            # 1. CONFIGURAR LLM REAL - ChatOllama
            await self._setup_llm_real()
            
            # 2. CONFIGURAR HERRAMIENTAS REALES
            await self._setup_tools_real()
            
            # 3. CREAR AGENT LANGCHAIN REAL
            await self._create_langchain_agent()
            
            # 4. Configurar memoria (básica por ahora)
            self._setup_memory()
            
            # 5. Configurar callback handler
            self.callback_handler = SmartDocCallbackHandler()
            
            self.is_initialized = True
            logger.info("✅ SmartDocAgent inicializado correctamente con LangChain")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando SmartDocAgent: {e}")
            return False

    async def _setup_llm_real(self):
        """Configurar LLM real usando ChatOllama"""
        try:
            # Crear instancia ChatOllama real
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=f"http://{self.ollama_host}:11434",
                temperature=0.1,
                top_p=0.9,
                num_predict=2048,
                verbose=True
            )
            
            # Test de conexión REAL
            logger.info(f"🔗 Conectando con Ollama en {self.ollama_host}:11434...")
            test_response = await self.llm.ainvoke("Hello, are you working?")
            logger.info(f"✅ LLM conectado correctamente: {test_response.content[:100]}...")
            
        except Exception as e:
            logger.error(f"❌ Error configurando LLM: {e}")
            raise

    async def _setup_tools_real(self):
        """Configurar herramientas reales"""
        try:
            # Crear WebSearchTool real
            web_search_tool = WebSearchTool()
            self.tools = [web_search_tool]
            
            logger.info(f"✅ Configuradas {len(self.tools)} herramientas:")
            for tool in self.tools:
                logger.info(f"  - {tool.name}: {tool.description[:80]}...")
                
        except Exception as e:
            logger.error(f"❌ Error configurando herramientas: {e}")
            raise

    async def _create_langchain_agent(self):
        """Crear AgentExecutor real con LangChain"""
        try:
            # Crear prompt template para ReAct
            react_prompt = PromptTemplate.from_template(REACT_MAIN_TEMPLATE)
            
            # Crear agent ReAct real
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
            )
            
            # Crear AgentExecutor real
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=self.max_iterations,
                callbacks=[self.callback_handler] if self.callback_handler else None
            )
            
            logger.info("✅ AgentExecutor LangChain creado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error creando AgentExecutor: {e}")
            raise
    
    async def _setup_llm(self):
        """Configurar el modelo de lenguaje Ollama"""
        try:
            base_url = f"http://{self.ollama_host}:{self.settings.ollama_port}"
            
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=base_url,
                temperature=self.settings.temperature,
                num_ctx=4096,  # Context length
                num_predict=self.settings.max_tokens,
                top_p=0.9,
                repeat_penalty=1.1
            )
            
            # Test de conexión
            test_response = await self.llm.ainvoke("Hello, respond with 'OK'")
            logger.info(f"LLM conectado - Test response: {test_response[:50]}")
            
        except Exception as e:
            logger.error(f"Error configurando LLM: {e}")
            raise
    
    async def _setup_tools(self):
        """Configurar herramientas disponibles para el agente"""
        # Por ahora, lista vacía - se agregaran tools en próximos pasos
        self.tools = []
        logger.info(f"Tools configuradas: {len(self.tools)} herramientas disponibles")
    
    def _setup_memory(self):
        """Configurar sistema de memoria del agente"""
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Mantener últimos 10 intercambios
            return_messages=True,
            memory_key="chat_history",
            input_key="input"
        )
        logger.info("Sistema de memoria configurado")
    
    def _create_react_agent(self):
        """Crear el agente ReAct con LangChain"""
        try:
            # Obtener template según estilo de investigación
            react_template = ReactTemplates.get_research_template(self.research_style)
            
            if self.tools:
                # Con herramientas - usar create_react_agent
                self.agent = create_react_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=react_template
                )
                
                self.agent_executor = AgentExecutor(
                    agent=self.agent,
                    tools=self.tools,
                    memory=self.memory,
                    verbose=True,
                    max_iterations=self.max_iterations,
                    handle_parsing_errors=True,
                    callbacks=[self.callback_handler] if self.callback_handler else []
                )
                logger.info("Agente ReAct creado con herramientas")
            else:
                # Sin herramientas - modo directo
                self.agent_executor = None
                logger.info("Agente en modo directo (sin herramientas)")
                
        except Exception as e:
            logger.error(f"Error creando agente ReAct: {e}")
            raise
    
    async def _test_agent(self) -> bool:
        """Probar funcionamiento básico del agente"""
        try:
            test_query = "¿Qué eres y cómo puedes ayudarme?"
            
            if self.agent_executor:
                result = await self.agent_executor.ainvoke({
                    "input": test_query,
                    "topic": "test",
                    "objectives": [],
                    "tools": "",  # Lista vacía formateada
                    "chat_history": ""
                })
                response = result.get("output", "")
            else:
                # Modo directo
                prompt = f"{SMARTDOC_SYSTEM_PROMPT}\n\nPregunta: {test_query}\nRespuesta:"
                response = await self.llm.ainvoke(prompt)
            
            if response and len(response) > 10:
                logger.info(f"Test exitoso - Respuesta: {response[:100]}...")
                return True
            else:
                logger.error("Test falló - Respuesta vacía o muy corta")
                return False
                
        except Exception as e:
            logger.error(f"Test del agente falló: {e}")
            return False
    
    async def create_research_session(self, topic: str, objectives: List[str] = None) -> str:
        """Crear una nueva sesión de investigación"""
        session_id = str(uuid.uuid4())
        
        session = ResearchSession(
            session_id=session_id,
            topic=topic,
            objectives=objectives or []
        )
        
        self.active_sessions[session_id] = session
        
        logger.info(f"Sesión creada: {session_id} - Tema: {topic}")
        return session_id
    
    async def process_query(self, session_id: str, query: str) -> Dict[str, Any]:
        """Procesar query usando AgentExecutor REAL de LangChain"""
        
        try:
            # Verificar que el agente está inicializado
            if not self.is_initialized or not self.agent_executor:
                return self._error_response("Agent not initialized", session_id)
            
            # Obtener sesión
            session = self.active_sessions.get(session_id)
            if not session:
                return self._error_response("Session not found", session_id)
            
            logger.info(f"🤖 Procesando query con LangChain: {query[:100]}...")
            
            # Agregar mensaje del usuario
            session.add_message("user", query)
            
            # EJECUTAR CON LANGCHAIN REAL
            result = await self._execute_with_langchain(session, query)
            
            # Agregar respuesta del agente
            session.add_message("assistant", result["response"])
            
            logger.info(f"✅ Query procesada exitosamente")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error procesando query: {e}")
            return self._error_response(f"Error: {str(e)}", session_id)

    async def _execute_with_langchain(self, session: ResearchSession, query: str) -> Dict[str, Any]:
        """Ejecutar query con AgentExecutor real"""
        
        try:
            # Preparar input para LangChain
            agent_input = {
                "input": query,
                "topic": session.topic,
                "objectives": ", ".join(session.objectives) if session.objectives else "General research",
                "chat_history": session.get_chat_history()
            }
            
            logger.info("🔄 Ejecutando con AgentExecutor...")
            
            # LLAMADA REAL A LANGCHAIN
            result = await self.agent_executor.ainvoke(agent_input)
            
            # Extraer respuesta
            response = result.get("output", "No response generated")
            
            # Preparar resultado estructurado
            return {
                "success": True,
                "response": response,
                "sources": self._extract_sources_from_callback(),
                "reasoning": self._extract_reasoning_from_callback(),
                "confidence": self._calculate_confidence_from_response(response),
                "session_id": session.session_id,
                "model_used": self.model_name,
                "mode": "langchain_agent"
            }
            
        except Exception as e:
            logger.error(f"❌ Error en ejecución LangChain: {e}")
            raise

    def _extract_sources_from_callback(self) -> List[Dict[str, Any]]:
        """Extraer fuentes de las acciones del callback"""
        sources = []
        
        if self.callback_handler and hasattr(self.callback_handler, 'actions'):
            for action in self.callback_handler.actions:
                if action['tool'] == 'web_search':
                    sources.append({
                        "type": "web_search",
                        "tool": action['tool'],
                        "input": action['input'],
                        "timestamp": action['timestamp']
                    })
        
        return sources

    def _extract_reasoning_from_callback(self) -> List[str]:
        """Extraer pasos de razonamiento del callback"""
        reasoning = []
        
        if self.callback_handler and hasattr(self.callback_handler, 'actions'):
            for i, action in enumerate(self.callback_handler.actions):
                reasoning.append(f"Step {i+1}: Used {action['tool']} with input: {action['input']}")
        
        return reasoning

    def _calculate_confidence_from_response(self, response: str) -> float:
        """Calcular confianza basada en la respuesta"""
        # Heurística simple basada en longitud y contenido
        if len(response) > 200:
            confidence = 0.8
        elif len(response) > 100:
            confidence = 0.6
        else:
            confidence = 0.4
        
        # Aumentar si menciona fuentes
        if any(word in response.lower() for word in ["source", "found", "search", "according"]):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _process_with_agent_executor(self, session: ResearchSession, query: str) -> Dict[str, Any]:
        """Procesar usando AgentExecutor con herramientas"""
        
        agent_input = {
            "input": query,
            "topic": session.topic,
            "objectives": session.objectives,
            "tools": self._format_tools_for_prompt(),
            "chat_history": session.get_chat_history()
        }
        
        result = await self.agent_executor.ainvoke(agent_input)
        
        return {
            "response": result.get("output", "Sin respuesta"),
            "sources": self._extract_sources_from_reasoning(),
            "reasoning": self._format_reasoning_steps(),
            "confidence": self._calculate_confidence(),
            "session_id": session.session_id,
            "model_used": self.model_name
        }
    
    async def _process_direct_mode(self, session: ResearchSession, query: str) -> Dict[str, Any]:
        """Procesar en modo directo (sin herramientas)"""
        
        # Construir prompt contextual
        context_prompt = f"""
{SMARTDOC_SYSTEM_PROMPT}

INFORMACIÓN DE LA SESIÓN:
- Tema de investigación: {session.topic}
- Objetivos: {', '.join(session.objectives) if session.objectives else 'No especificados'}

HISTORIAL DE CONVERSACIÓN:
{session.get_chat_history()}

PREGUNTA ACTUAL: {query}

Proporciona una respuesta útil y bien estructurada. Si necesitas más información específica o herramientas especializadas para dar una respuesta completa, indícalo claramente.

RESPUESTA:"""
        
        response = await self.llm.ainvoke(context_prompt)
        
        return {
            "response": response,
            "sources": [],
            "reasoning": "Respuesta directa del modelo de lenguaje",
            "confidence": 0.7,  # Confianza media para respuestas directas
            "session_id": session.session_id,
            "model_used": self.model_name,
            "mode": "direct"
        }
    
    def _format_tools_for_prompt(self) -> str:
        """Formatear herramientas para el prompt"""
        if not self.tools:
            return "No hay herramientas disponibles actualmente."
        
        tool_descriptions = []
        for tool in self.tools:
            tool_descriptions.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(tool_descriptions)
    
    def _extract_sources_from_reasoning(self) -> List[Dict]:
        """Extraer fuentes del proceso de razonamiento"""
        sources = []
        
        if self.callback_handler:
            for obs in self.callback_handler.observations:
                sources.append({
                    "type": "tool_output",
                    "content": obs["output"][:200],
                    "step": obs["step"],
                    "timestamp": obs["timestamp"]
                })
        
        return sources
    
    def _format_reasoning_steps(self) -> str:
        """Formatear pasos de razonamiento"""
        if not self.callback_handler:
            return "No hay pasos de razonamiento disponibles"
        
        steps = []
        for i, action in enumerate(self.callback_handler.actions):
            steps.append(f"Paso {i+1}: Usó {action['tool']} con entrada: {action['input']}")
        
        return " | ".join(steps) if steps else "Razonamiento directo"
    
    def _calculate_confidence(self) -> float:
        """Calcular nivel de confianza de la respuesta"""
        # Lógica básica - se puede mejorar
        base_confidence = 0.6
        
        if self.callback_handler:
            # Más herramientas usadas = mayor confianza
            tools_used = len(self.callback_handler.actions)
            confidence_boost = min(tools_used * 0.1, 0.3)
            return min(base_confidence + confidence_boost, 0.95)
        
        return base_confidence
    
    def _error_response(self, error_msg: str, session_id: str) -> Dict[str, Any]:
        """Generar respuesta de error estandardizada"""
        return {
            "response": f"Error: {error_msg}",
            "sources": [],
            "reasoning": "Error en el procesamiento",
            "confidence": 0.0,
            "session_id": session_id,
            "error": True
        }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Obtener estado de una sesión"""
        if session_id not in self.active_sessions:
            return {"error": "Sesión no encontrada"}
        
        session = self.active_sessions[session_id]
        
        return {
            "session_id": session_id,
            "topic": session.topic,
            "objectives": session.objectives,
            "created_at": session.created_at.isoformat(),
            "message_count": len(session.messages),
            "sources_found": len(session.sources_found),
            "last_activity": session.messages[-1]["timestamp"] if session.messages else None
        }
    
    def list_active_sessions(self) -> Dict[str, Any]:
        """Listar todas las sesiones activas"""
        sessions_info = {}
        
        for session_id, session in self.active_sessions.items():
            sessions_info[session_id] = {
                "topic": session.topic,
                "message_count": len(session.messages),
                "created_at": session.created_at.isoformat()
            }
        
        return {
            "active_sessions": list(self.active_sessions.keys()),
            "count": len(self.active_sessions),
            "sessions": sessions_info
        }
    
    async def close(self):
        """Limpiar recursos del agente"""
        logger.info("Cerrando SmartDocAgent...")
        self.active_sessions.clear()
        if self.memory:
            self.memory.clear()
        logger.info("SmartDocAgent cerrado correctamente")

# Factory function para crear agentes
async def create_smartdoc_agent(
    model_name: str = None,
    research_style: str = "general",
    max_iterations: int = 5
) -> SmartDocAgent:
    """Factory para crear y inicializar un SmartDocAgent"""
    
    agent = SmartDocAgent(
        model_name=model_name,
        research_style=research_style,
        max_iterations=max_iterations
    )
    
    success = await agent.initialize()
    
    if not success:
        raise RuntimeError("No se pudo inicializar SmartDocAgent")
    
    return agent