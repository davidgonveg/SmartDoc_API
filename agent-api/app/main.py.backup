"""
SmartDoc Agent API - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from app.config.settings import get_settings
from app.api.routes import research, health, upload
from app.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting SmartDoc Agent API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"GPU Support: {settings.use_gpu}")
    
    # Initialize services here
    # await initialize_agent()
    # await initialize_databases()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SmartDoc Agent API")

# Create FastAPI app
app = FastAPI(
    title="SmartDoc Agent API",
    description="Intelligent research agent powered by LangChain",
    version="0.1.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(research.router, prefix="/research", tags=["research"])
app.include_router(upload.router, prefix="/upload", tags=["upload"])

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "SmartDoc Agent API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "gpu_enabled": settings.use_gpu
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
