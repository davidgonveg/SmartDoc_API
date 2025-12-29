"""
Services module - External service integrations
"""

from .ollama_client import OllamaClient, get_ollama_client

__all__ = ["OllamaClient", "get_ollama_client"]


