"""Health check endpoints"""

from fastapi import APIRouter
from typing import Dict, Any
import asyncio

router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "smartdoc-agent-api",
        "version": "0.1.0"
    }

@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with dependencies"""
    # TODO: Check Ollama, ChromaDB, Redis connectivity
    return {
        "status": "healthy",
        "services": {
            "api": "healthy",
            "ollama": "pending",
            "chromadb": "pending", 
            "redis": "pending"
        }
    }
