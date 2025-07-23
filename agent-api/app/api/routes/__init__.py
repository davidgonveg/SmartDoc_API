
# ===============================================
# agent-api/app/api/routes/__init__.py (actualizado)
# ===============================================
"""
API Routes Module
FastAPI route definitions
"""

from . import health, research, upload

__all__ = ["health", "research", "upload"]