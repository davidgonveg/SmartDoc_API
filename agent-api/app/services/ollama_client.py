"""
Ollama Client - HTTP client for local Ollama inference
"""

import httpx
import asyncio
import logging
from typing import Dict, Any, Optional
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for communicating with Ollama API"""
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = f"http://{self.settings.ollama_host}:{self.settings.ollama_port}"
        self.client = httpx.AsyncClient(timeout=120.0)
        
    async def health_check(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = await self.client.get(f"{self.base_url}/api/version")
            if response.status_code == 200:
                version_info = response.json()
                logger.info(f"Ollama connected - Version: {version_info.get('version', 'unknown')}")
                return True
            else:
                logger.error(f"Ollama health check failed - Status: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False
    
    async def generate(
        self,
        model: str,
        prompt: str,
        system: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate text using Ollama"""
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens
            }
        }
        
        if system:
            payload["system"] = system
            
        try:
            logger.info(f"Generating with model {model}, prompt length: {len(prompt)}")
            
            response = await self.client.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120.0
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Generation successful, response length: {len(result.get('response', ''))}")
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": result.get("model", model),
                    "total_duration": result.get("total_duration", 0)
                }
            else:
                error_msg = f"Ollama API error - Status: {response.status_code}, Body: {response.text}"
                logger.error(error_msg)
                return {
                    "success": False,
                    "error": error_msg,
                    "response": ""
                }
                
        except Exception as e:
            error_msg = f"Ollama generation failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "response": ""
            }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# Global instance
_ollama_client = None

async def get_ollama_client() -> OllamaClient:
    """Get singleton Ollama client"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client
