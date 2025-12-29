# app/config/settings.py - PARÁMETROS OPTIMIZADOS

<<<<<<< HEAD
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional
import os
=======
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
>>>>>>> 6c454ad3e4eb18b9c07646a44fc47615aaba7690

class Settings(BaseSettings):
    """Configuración optimizada para SmartDoc Agent"""
    
    # Configuración base
    app_name: str = "SmartDoc Research Agent"
    environment: str = "development"
    debug: bool = True
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8001
    
    # Ollama Configuration - OPTIMIZADO
    ollama_host: str = "ollama"
    ollama_port: int = 11434
    default_model: str = "llama3.2:3b"
    
    # 🚀 NUEVOS: Parámetros de optimización LLM
    llm_temperature: float = 0.7
    llm_top_p: float = 0.9
    llm_repeat_penalty: float = 1.1
    llm_num_ctx: int = 8192          # Context window
    llm_num_predict: int = 1024      # Max tokens por respuesta
    llm_timeout: int = 300           # 5 minutos
    
    # 🚀 NUEVOS: Configuración del agente
    agent_max_iterations: int = 15    # Aumentado de 5
    agent_optimization_level: str = "balanced"  # performance/balanced/cpu
    agent_enable_streaming: bool = True
    agent_max_execution_time: int = 300
    
    # 🚀 NUEVOS: Configuración de memoria
    memory_window_size: int = 10      # Mensajes en memoria
    memory_max_stored_steps: int = 50 # Pasos máximos en callback
    
    # 🚀 NUEVOS: Configuración de herramientas
    web_search_timeout: int = 30
    web_search_max_results: int = 10
    web_search_rate_limit: int = 5    # requests por minuto
    
    # ChromaDB Configuration
    chromadb_host: str = "chromadb"
    chromadb_port: int = 8000
    chromadb_persist_directory: str = "/data/chromadb"
    
    # Redis Configuration
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # 🚀 NUEVOS: Configuración de performance
    enable_gpu: bool = True           # Auto-detectar GPU
    max_concurrent_sessions: int = 10 # Sesiones concurrentes
    cache_ttl: int = 3600            # 1 hora TTL para cache
    
    # 🚀 NUEVOS: Configuración por hardware
    hardware_profiles: Dict[str, Dict[str, Any]] = {
        "performance": {  # GPU mode
            "llm_num_ctx": 16384,
            "llm_num_predict": 2048,
            "agent_max_iterations": 20,
            "memory_window_size": 20,
            "web_search_max_results": 15
        },
        "balanced": {     # Default
            "llm_num_ctx": 8192,
            "llm_num_predict": 1024,
            "agent_max_iterations": 15,
            "memory_window_size": 10,
            "web_search_max_results": 10
        },
        "cpu": {          # CPU-only mode
            "llm_num_ctx": 4096,
            "llm_num_predict": 512,
            "agent_max_iterations": 8,
            "memory_window_size": 5,
            "web_search_max_results": 5
        }
    }
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_hardware_profile(self, profile_name: str) -> Dict[str, Any]:
        """Obtener configuración para perfil de hardware específico"""
        return self.hardware_profiles.get(profile_name, self.hardware_profiles["balanced"])
    
    def get_optimized_llm_params(self, optimization_level: str = None) -> Dict[str, Any]:
        """Obtener parámetros optimizados del LLM"""
        level = optimization_level or self.agent_optimization_level
        profile = self.get_hardware_profile(level)
        
        return {
            "temperature": self.llm_temperature,
            "top_p": self.llm_top_p,
            "repeat_penalty": self.llm_repeat_penalty,
            "num_ctx": profile.get("llm_num_ctx", self.llm_num_ctx),
            "num_predict": profile.get("llm_num_predict", self.llm_num_predict),
            "timeout": self.llm_timeout
        }

def get_settings() -> Settings:
    """Singleton para obtener configuración"""
    return Settings()