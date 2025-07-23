"""
SmartDoc Agent API - Versión Mínima Funcional
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uuid
import logging

# Setup logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SmartDoc Agent API",
    description="Intelligent research agent",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelos Pydantic
class ResearchRequest(BaseModel):
    topic: str
    objectives: Optional[List[str]] = []
    max_sources: int = 10
    research_depth: str = "intermediate"

class ChatMessage(BaseModel):
    message: str
    stream: bool = False

# Almacén temporal de sesiones
active_sessions = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartDoc Agent API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "gpu_enabled": False
    }

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "smartdoc-agent-api",
        "version": "0.1.0"
    }

@app.post("/research/session")
async def create_research_session(request: ResearchRequest) -> Dict[str, Any]:
    """Create research session"""
    try:
        session_id = str(uuid.uuid4())
        
        # Guardar sesión
        active_sessions[session_id] = {
            "topic": request.topic,
            "objectives": request.objectives,
            "created_at": "now",
            "messages": []
        }
        
        logger.info(f"Sesión creada: {session_id} para tópico: {request.topic}")
        
        return {
            "session_id": session_id,
            "status": "created",
            "topic": request.topic,
            "message": "Sesión creada exitosamente (versión básica)"
        }
        
    except Exception as e:
        logger.error(f"Error creando sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/research/chat/{session_id}")
async def chat_with_agent(session_id: str, message: ChatMessage) -> Dict[str, Any]:
    """Chat with research agent"""
    try:
        # Verificar que la sesión existe
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Sesión no encontrada")
        
        session = active_sessions[session_id]
        
        # Simular respuesta del agente
        response = f"Hola! Recibí tu mensaje: '{message.message}'. Soy el agente de SmartDoc en versión básica. El tópico de tu sesión es: {session['topic']}"
        
        # Guardar mensaje en la sesión
        session["messages"].append({
            "user": message.message,
            "agent": response
        })
        
        logger.info(f"Mensaje procesado para sesión {session_id}")
        
        return {
            "response": response,
            "sources": [],
            "reasoning": "Respuesta simulada - agente básico",
            "confidence": 1.0,
            "session_id": session_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/research/sessions")
async def list_active_sessions():
    """List active sessions"""
    return {
        "active_sessions": list(active_sessions.keys()),
        "count": len(active_sessions),
        "sessions": {k: {"topic": v["topic"], "message_count": len(v["messages"])} for k, v in active_sessions.items()}
    }

@app.post("/upload/{session_id}")
async def upload_files(session_id: str):
    """Upload files placeholder"""
    return {
        "session_id": session_id,
        "message": "Upload functionality - coming soon",
        "status": "placeholder"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
