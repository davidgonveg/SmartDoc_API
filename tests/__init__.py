"""
SmartDoc Research Agent - Test Suite Package

Este paquete contiene todos los tests para el agente de investigación SmartDoc.
Incluye tests unitarios, de integración, end-to-end y de performance.

Estructura del paquete:
    tests/
    ├── __init__.py                 # Este archivo
    ├── fixtures.py                 # Fixtures compartidas
    ├── test_data.py               # Datos de prueba centralizados
    ├── unit/                      # Tests unitarios
    │   ├── __init__.py
    │   ├── test_smart_agent.py
    │   ├── test_web_search_tool.py
    │   └── test_*.py
    ├── integration/               # Tests de integración
    │   ├── __init__.py
    │   ├── test_agent_web_integration.py
    │   ├── test_full_workflow.py
    │   ├── test_research_session.py
    │   └── test_*.py
    └── scripts/
        └── run_tests.sh          # Script ejecutor de tests

Uso:
    # Ejecutar todos los tests
    pytest tests/

    # Tests específicos por tipo
    pytest tests/unit/              # Solo tests unitarios
    pytest tests/integration/       # Solo tests de integración
    
    # Tests con markers específicos
    pytest -m "unit"               # Tests marcados como unitarios
    pytest -m "integration"       # Tests marcados como integración
    pytest -m "slow"              # Tests que tardan más tiempo
    
    # Con coverage
    pytest tests/ --cov=app --cov-report=html
    
    # Script personalizado
    ./run_tests.sh -t unit -v verbose -c
"""

import sys
import os
from pathlib import Path

# Agregar el directorio raíz al path para imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configuración de pytest para este paquete
pytest_plugins = [
    "tests.fixtures",
]

# Variables globales para configuración de tests
TEST_CONFIG = {
    "project_root": PROJECT_ROOT,
    "test_dir": Path(__file__).parent,
    "reports_dir": PROJECT_ROOT / "test_reports",
    "logs_dir": PROJECT_ROOT / "logs" / "tests",
    "default_timeout": 30.0,
    "slow_test_timeout": 120.0
}

# Crear directorios necesarios si no existen
for dir_path in [TEST_CONFIG["reports_dir"], TEST_CONFIG["logs_dir"]]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuración de logging para tests
import logging

def setup_test_logging():
    """Configurar logging para tests"""
    
    # Crear logger específico para tests
    test_logger = logging.getLogger('smartdoc.tests')
    test_logger.setLevel(logging.DEBUG)
    
    # Handler para archivo de log
    log_file = TEST_CONFIG["logs_dir"] / "test_execution.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Handler para consola (solo errores durante tests)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    
    # Formato de logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Agregar handlers si no existen
    if not test_logger.handlers:
        test_logger.addHandler(file_handler)
        test_logger.addHandler(console_handler)
    
    return test_logger

# Logger global para tests
test_logger = setup_test_logging()

# Funciones de utilidad para tests
def get_test_config(key: str = None):
    """Obtener configuración de tests"""
    if key:
        return TEST_CONFIG.get(key)
    return TEST_CONFIG

def get_project_root():
    """Obtener directorio raíz del proyecto"""
    return TEST_CONFIG["project_root"]

def get_test_logger():
    """Obtener logger para tests"""
    return test_logger

# Verificaciones de entorno al importar
def _verify_test_environment():
    """Verificar que el entorno de test está correctamente configurado"""
    
    try:
        # Verificar que estamos en el entorno correcto
        requirements_file = PROJECT_ROOT / "requirements.txt"
        if not requirements_file.exists():
            test_logger.warning(f"requirements.txt not found at {requirements_file}")
        
        # Verificar estructura básica del proyecto
        essential_dirs = [
            PROJECT_ROOT / "app",
            PROJECT_ROOT / "app" / "agents",
            PROJECT_ROOT / "tests"
        ]
        
        for dir_path in essential_dirs:
            if not dir_path.exists():
                test_logger.warning(f"Essential directory not found: {dir_path}")
        
        # Verificar que los módulos principales son importables
        try:
            import app
            test_logger.debug("App module import successful")
        except ImportError as e:
            test_logger.warning(f"Could not import app module: {e}")
        
        test_logger.debug("Test environment verification completed")
        
    except Exception as e:
        test_logger.error(f"Test environment verification failed: {e}")

# Ejecutar verificación al importar
_verify_test_environment()

# Metadata del paquete de tests
__version__ = "1.0.0"
__author__ = "SmartDoc Team"
__email__ = "dev@smartdoc.ai"
__description__ = "Test suite for SmartDoc Research Agent"

# Exports principales
__all__ = [
    "TEST_CONFIG",
    "get_test_config",
    "get_project_root", 
    "get_test_logger",
    "test_logger"
]