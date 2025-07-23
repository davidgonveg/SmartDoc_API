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

from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.callbacks import AsyncCallbackHandler
from langchain_ollama import OllamaLLM

from app.agents.prompts.react_templates import ReactTemplates, SMARTDOC_SYSTEM_PROMPT
from app.agents.tools.base_tool import BaseTool
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
        """Inicializar el agente y todos sus componentes"""
        try:
            logger.info("Inicializando SmartDocAgent...")
            
            # 1. Configurar LLM
            await self._setup_llm()
            
            # 2. Configurar herramientas (por ahora vacías)
            await self._setup_tools()
            
            # 3. Configurar memoria
            self._setup_memory()
            
            # 4. Configurar callback handler
            self.callback_handler = SmartDocCallbackHandler()
            
            # 5. Crear el agente ReAct
            self._create_react_agent()
            
            # 6. Probar funcionamiento
            test_result = await self._test_agent()
            
            if test_result:
                self.is_initialized = True
                logger.info("✅ SmartDocAgent inicializado correctamente")
                return True
            else:
                logger.error("❌ Falló el test del agente")
                return False
                
        except Exception as e:
            logger.error(f"Error inicializando SmartDocAgent: {e}")
            return False
    
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
    
    async def process_query(self, 
                          session_id: str, 
                          query: str,
                          stream: bool = False) -> Dict[str, Any]:
        """Procesar una consulta del usuario"""
        
        if not self.is_initialized:
            return self._error_response("Agente no inicializado", session_id)
        
        if session_id not in self.active_sessions:
            return self._error_response("Sesión no encontrada", session_id)
        
        session = self.active_sessions[session_id]
        
        try:
            logger.info(f"Procesando query en sesión {session_id}: {query[:100]}...")
            
            # Agregar mensaje del usuario
            session.add_message("user", query)
            
            # Resetear callback handler
            if self.callback_handler:
                self.callback_handler.thoughts = []
                self.callback_handler.actions = []
                self.callback_handler.observations = []
                self.callback_handler.current_step = 0
            
            # Procesar con agente o modo directo
            if self.agent_executor:
                result = await self._process_with_agent_executor(session, query)
            else:
                result = await self._process_direct_mode(session, query)
            
            # Agregar respuesta del agente
            session.add_message("assistant", result["response"], {
                "confidence": result.get("confidence", 0.0),
                "sources_count": len(result.get("sources", [])),
                "reasoning_steps": len(self.callback_handler.actions) if self.callback_handler else 0
            })
            
            logger.info(f"Query procesado exitosamente para sesión {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando query: {e}")
            return self._error_response(f"Error: {str(e)}", session_id)
    
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