# Crear: agent-api/app/agents/core/base_agent.py
"""Agente base usando LangChain"""

from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from typing import List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)

class SmartDocAgent:
    """Agente principal de investigación"""
    
    def __init__(self, model_name: str = "llama3.2:3b", ollama_host: str = "localhost"):
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.llm = None
        self.agent_executor = None
        self.tools = []
        
    async def initialize(self):
        """Inicializar el agente y sus componentes"""
        try:
            # Configurar LLM
            self.llm = OllamaLLM(
                model=self.model_name,
                base_url=f"http://{self.ollama_host}:11434"
            )
            
            # Probar conexión
            test_response = await self.llm.ainvoke("Hello")
            logger.info(f"LLM conectado correctamente: {test_response[:50]}...")
            
            # Configurar herramientas (por ahora vacías)
            self.tools = []
            
            # Crear el agente ReAct
            self._create_agent()
            
            logger.info("Agente inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando agente: {e}")
            return False
    
    def _create_agent(self):
        """Crear el agente con el patrón ReAct"""
        
        # Prompt template para ReAct
        react_prompt = PromptTemplate.from_template("""
Eres un agente de investigación inteligente. Tu trabajo es responder preguntas usando un proceso de razonamiento paso a paso.

Tienes acceso a las siguientes herramientas:
{tools}

Usa el siguiente formato:

Pregunta: la pregunta de entrada que debes responder
Pensamiento: siempre debes pensar sobre qué hacer
Acción: la acción a tomar, debe ser una de [{tool_names}]
Entrada de Acción: la entrada para la acción
Observación: el resultado de la acción
... (este proceso de Pensamiento/Acción/Entrada de Acción/Observación puede repetirse N veces)
Pensamiento: Ahora sé la respuesta final
Respuesta Final: la respuesta final a la pregunta original

Comienza!

Pregunta: {input}
Pensamiento: {agent_scratchpad}
""")
        
        if self.tools:
            # Crear agente con herramientas
            self.agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=react_prompt
            )
            
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=5
            )
        else:
            # Sin herramientas por ahora, solo respuesta directa
            self.agent_executor = None
    
    async def process_query(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Procesar una consulta del usuario"""
        
        try:
            logger.info(f"Procesando query: {query[:100]}...")
            
            if self.agent_executor:
                # Usar agente con herramientas
                result = await self.agent_executor.ainvoke({"input": query})
                response = result.get("output", "Sin respuesta")
            else:
                # Respuesta directa del LLM
                response = await self.llm.ainvoke(f"Responde esta pregunta de manera clara y útil: {query}")
            
            return {
                "response": response,
                "sources": [],
                "reasoning": "Respuesta directa del modelo de lenguaje",
                "confidence": 0.8,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error procesando query: {e}")
            return {
                "response": f"Error: {str(e)}",
                "sources": [],
                "reasoning": "Error en el procesamiento",
                "confidence": 0.0,
                "session_id": session_id
            }