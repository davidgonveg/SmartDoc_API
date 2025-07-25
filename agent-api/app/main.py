"""
SmartDoc Agent API - Versión con LangChain Real
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

# IMPORT DIRECTO DEL ROUTER (evita imports circulares)
from app.api.routes.research import router as research_router

# Setup logging básico
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SmartDoc Agent API",
    description="Intelligent research agent with LangChain",
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

# INCLUIR RUTAS REALES DE LANGCHAIN (eliminar endpoints duplicados de main.py)
app.include_router(research_router, prefix="/research", tags=["research"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SmartDoc Agent API with LangChain",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "langchain_enabled": True
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "service": "smartdoc-agent-api",
        "version": "0.1.0",
        "langchain": "enabled"
    }

@app.get("/test-ollama")
async def test_ollama_connection():
    """Test endpoint para verificar conexión Ollama"""
    try:
        from app.services.ollama_client import get_ollama_client
        
        client = await get_ollama_client()
        is_healthy = await client.health_check()
        
        if is_healthy:
            # Test basic generation
            result = await client.generate(
                model="llama3.2:3b", 
                prompt="Say hello in one sentence.",
                options={"num_predict": 50}
            )
            
            if result["success"]:
                return {
                    "status": "connected",
                    "ollama_healthy": True,
                    "model_test": "success",
                    "test_response": result["response"],
                    "model_used": result.get("model", "unknown")
                }
            else:
                return {
                    "status": "connected_but_generation_failed",
                    "ollama_healthy": True,
                    "model_test": "failed",
                    "error": result.get("error", "Unknown error")
                }
        else:
            return {
                "status": "connection_failed",
                "ollama_healthy": False,
                "error": "Ollama health check failed"
            }
            
    except Exception as e:
        logger.error(f"Ollama test failed: {e}")
        return {
            "status": "error",
            "ollama_healthy": False,
            "error": str(e)
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