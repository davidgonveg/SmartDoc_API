# ===============================================
# agent-api/app/api/__init__.py (actualizado)
# ===============================================
"""
SmartDoc API Module
FastAPI routes and endpoints
"""

# Re-export common API components
from .routes import research, health, upload

__all__ = ["research", "health", "upload"]
