# Actualizar: agent-api/app/api/routes/research.py
"""Research endpoints con agente real"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid
import logging

from app.agents.core.base_agent import SmartDocAgent
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Almacén global de agentes (en producción usar Redis)
active_agents: Dict[str, SmartDocAgent] = {}

class ResearchRequest(BaseModel):
    topic: str
    objectives: Optional[List[str]] = []
    max_sources: int = 10
    research_depth: str = "intermediate"

class ChatMessage(BaseModel):
    message: str
    stream: bool = False

@router.post("/session")
async def create_research_session(request: ResearchRequest) -> Dict[str, Any]:
    """Crear una nueva sesión de investigación"""
    
    try:
        session_id = str(uuid.uuid4())
        
        # Crear y inicializar agente
        agent = SmartDocAgent(
            model_name=settings.default_model,
            ollama_host=settings.ollama_host
        )
        
        initialized = await agent.initialize()
        
        if not initialized:
            raise HTTPException(status_code=500, detail="Error inicializando agente")
        
        # Guardar agente
        active_agents[session_id] = agent
        
        logger.info(f"Sesión creada: {session_id} para tópico: {request.topic}")
        
        return {
            "session_id": session_id,
            "status": "created",
            "topic": request.topic,
            "model": settings.default_model
        }
        
    except Exception as e:
        logger.error(f"Error creando sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{session_id}")
async def chat_with_agent(session_id: str, message: ChatMessage) -> Dict[str, Any]:
    """Chat con el agente de investigación"""
    
    try:
        # Buscar agente
        if session_id not in active_agents:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        agent = active_agents[session_id]
        
        # Procesar mensaje
        result = await agent.process_query(message.message, session_id)
        
        logger.info(f"Respuesta generada para sesión {session_id}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def list_active_sessions() -> Dict[str, Any]:
    """Listar sesiones activas"""
    return {
        "active_sessions": list(active_agents.keys()),
        "count": len(active_agents)
    }