# Crear: agent-api/app/utils/logging_config.py
"""Configuración de logging para el sistema"""

import logging
import sys
from pathlib import Path

def setup_logging():
    """Configurar logging del sistema"""
    # Crear directorio de logs si no existe
    Path("logs").mkdir(exist_ok=True)
    
    # Configurar formato
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Handler para consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # Handler para archivo
    file_handler = logging.FileHandler('logs/agent.log')
    file_handler.setFormatter(formatter)
    
    # Configurar root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler, file_handler]
    )