"""Research endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import uuid
import logging

from app.agents.core.smart_agent import SmartDocAgent
from app.config.settings import get_settings

logger = logging.getLogger(__name__)
router = APIRouter()
settings = get_settings()

# Almacén global de agentes
active_agents: Dict[str, SmartDocAgent] = {}
session_mapping: Dict[str, str] = {}

class ResearchRequest(BaseModel):
    topic: str
    objectives: Optional[List[str]] = []
    max_sources: int = Field(default=10, ge=1, le=20)
    research_depth: str = Field(default="intermediate", pattern="^(basic|intermediate|advanced)$")
    # 🚀 NUEVOS: Parámetros de optimización
    optimization_level: Optional[str] = Field(default=None, pattern="^(performance|balanced|cpu)$")
    max_iterations: Optional[int] = Field(default=None, ge=5, le=30)
    enable_streaming: Optional[bool] = None

class ChatMessage(BaseModel):
    message: str
    stream: bool = False
    # 🚀 NUEVO: Control de profundidad por query
    depth: Optional[str] = Field(default="normal", pattern="^(quick|normal|deep)$")

@router.post("/session")
async def create_research_session(request: ResearchRequest) -> Dict[str, Any]:
    """Crear sesión de investigación optimizada"""
    
    try:
        api_session_id = str(uuid.uuid4())
        
        # 🚀 Determinar parámetros optimizados
        optimization_level = request.optimization_level or settings.agent_optimization_level
        max_iterations = request.max_iterations or settings.agent_max_iterations
        enable_streaming = request.enable_streaming if request.enable_streaming is not None else settings.agent_enable_streaming
        
        # Crear agente con parámetros optimizados
        agent = SmartDocAgent(
            model_name=settings.default_model,
            ollama_host=settings.ollama_host,
            research_style=_map_depth_to_style(request.research_depth),
            max_iterations=max_iterations,
            enable_streaming=enable_streaming,
            optimization_level=optimization_level
        )
        
        # Inicializar agente
        initialized = await agent.initialize()
        if not initialized:
            raise HTTPException(status_code=500, detail="Error inicializando agente optimizado")
        
        # Crear research session interna
        research_session_id = await agent.create_research_session(request.topic, request.objectives)
        
        # Guardar mapeo
        active_agents[api_session_id] = agent
        session_mapping[api_session_id] = research_session_id
        
        logger.info(f"✅ Sesión optimizada creada: {api_session_id}")
        logger.info(f"🎯 Config: {optimization_level} | Iterations: {max_iterations} | Streaming: {enable_streaming}")
        
        return {
            "session_id": api_session_id,
            "status": "created",
            "topic": request.topic,
            "model": settings.default_model,
            "optimization_level": optimization_level,
            "max_iterations": max_iterations,
            "streaming_enabled": enable_streaming,
            "hardware_profile": settings.get_hardware_profile(optimization_level)
        }
        
    except Exception as e:
        logger.error(f"❌ Error creando sesión optimizada: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/{session_id}")
async def chat_with_agent(session_id: str, message: ChatMessage) -> Dict[str, Any]:
    """Chat optimizado con control de profundidad"""
    
    try:
        # Verificar agente
        if session_id not in active_agents:
            raise HTTPException(status_code=404, detail="Session not found")
        
        agent = active_agents[session_id]
        research_session_id = session_mapping[session_id]
        
        # 🚀 Ajustar agente según profundidad de query
        await _adjust_agent_for_query_depth(agent, message.depth)
        
        logger.info(f"🤖 Processing {message.depth} query: {message.message[:100]}...")
        
        # Procesar query
        result = await agent.process_query(research_session_id, message.message)
        
        # 🚀 Agregar métricas de performance
        if hasattr(agent, 'callback_handler') and agent.callback_handler:
            performance_metrics = agent.callback_handler.get_performance_metrics()
            result["performance"] = performance_metrics
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error en chat optimizado: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 🚀 NUEVOS ENDPOINTS DE OPTIMIZACIÓN

@router.post("/session/{session_id}/optimize")
async def optimize_session(session_id: str, optimization_level: str) -> Dict[str, Any]:
    """Cambiar nivel de optimización de una sesión activa"""
    
    if session_id not in active_agents:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = active_agents[session_id]
    
    try:
        # Reconfigurar agente con nuevo nivel
        agent.optimization_level = optimization_level
        await agent._optimize_for_hardware()
        
        logger.info(f"🔧 Sesión {session_id} optimizada a nivel: {optimization_level}")
        
        return {
            "session_id": session_id,
            "optimization_level": optimization_level,
            "status": "optimized",
            "new_params": settings.get_hardware_profile(optimization_level)
        }
        
    except Exception as e:
        logger.error(f"❌ Error optimizando sesión: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/performance")
async def get_session_performance(session_id: str) -> Dict[str, Any]:
    """Obtener métricas de performance de una sesión"""
    
    if session_id not in active_agents:
        raise HTTPException(status_code=404, detail="Session not found")
    
    agent = active_agents[session_id]
    
    metrics = {
        "session_id": session_id,
        "optimization_level": agent.optimization_level,
        "max_iterations": agent.max_iterations,
        "model": agent.model_name,
        "streaming_enabled": agent.enable_streaming
    }
    
    if hasattr(agent, 'callback_handler') and agent.callback_handler:
        metrics["performance"] = agent.callback_handler.get_performance_metrics()
    
    return metrics

# Funciones auxiliares

def _map_depth_to_style(research_depth: str) -> str:
    """Mapear profundidad de investigación a estilo de agente"""
    mapping = {
        "basic": "general",
        "intermediate": "academic", 
        "advanced": "technical"
    }
    return mapping.get(research_depth, "general")

async def _adjust_agent_for_query_depth(agent: SmartDocAgent, depth: str):
    """Ajustar parámetros del agente según profundidad de query"""
    
    if depth == "quick":
        # Query rápida - menos iteraciones
        agent.agent_executor.max_iterations = min(agent.max_iterations, 5)
        agent.agent_executor.max_execution_time = 60  # 1 minuto
        
    elif depth == "deep":
        # Query profunda - más iteraciones
        agent.agent_executor.max_iterations = agent.max_iterations
        agent.agent_executor.max_execution_time = 600  # 10 minutos
        
    else:  # normal
        # Configuración por defecto
        agent.agent_executor.max_iterations = max(agent.max_iterations // 2, 8)
        agent.agent_executor.max_execution_time = 300  # 5 minutos
        
# AGREGAR AL FINAL de agent-api/app/api/routes/research.py

@router.get("/system/hardware")
async def get_hardware_info() -> Dict[str, Any]:
    """Información del hardware disponible"""
    
    try:
        import subprocess
        import psutil
        import platform
        
        # Detectar GPU
        try:
            gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=5)
            has_gpu = gpu_result.returncode == 0
            gpu_info = gpu_result.stdout.strip() if has_gpu else "No GPU detected"
        except:
            has_gpu = False
            gpu_info = "GPU detection failed"
        
        # Info del sistema
        memory_gb = round(psutil.virtual_memory().total / (1024**3), 2)
        
        # Determinar optimización recomendada
        if has_gpu and memory_gb >= 16:
            recommended = "performance"
        elif memory_gb >= 8:
            recommended = "balanced"
        else:
            recommended = "cpu"
        
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_gb": memory_gb,
            "has_gpu": has_gpu,
            "gpu_info": gpu_info,
            "platform": platform.system(),
            "architecture": platform.machine(),
            "recommended_optimization": recommended,
            "available_profiles": ["performance", "balanced", "cpu"]
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo info de hardware: {e}")
        return {
            "cpu_count": None,
            "memory_gb": None,
            "has_gpu": None,
            "gpu_info": "Error detecting hardware",
            "recommended_optimization": "balanced",
            "error": str(e)
        }
