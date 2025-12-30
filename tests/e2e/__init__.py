"""
SmartDoc Research Agent - End-to-End Tests Package

Este paquete contiene tests end-to-end que verifican el funcionamiento completo
del sistema SmartDoc desde la perspectiva del usuario final.

Los tests E2E se enfocan en:
- Flujos completos desde UI hasta backend
- Integración con servicios reales (cuando sea apropiado)
- Escenarios de usuario realistas
- Validación de requisitos de negocio
- Performance del sistema completo

Módulos de tests E2E:
    test_complete_user_journey.py    # Journey completo del usuario
    test_api_endpoints_e2e.py        # Tests E2E de API
    test_streamlit_ui_e2e.py        # Tests E2E de UI
    test_docker_deployment_e2e.py   # Tests de deployment
    test_real_world_scenarios.py    # Escenarios del mundo real

Características:
- Tests más lentos (pueden tomar minutos)
- Uso opcional de servicios reales
- Validación de requirements de usuario
- Testing de integración completa
- Verificación de deployment

Convenciones:
- Prefijo test_e2e_ para tests puramente E2E
- Uso de datos realistas y completos
- Validación de experiencia de usuario
- Cleanup completo de recursos

Ejecución:
    pytest tests/e2e/                      # Todos los tests E2E
    pytest tests/e2e/test_complete_user_journey.py # Test específico
    pytest -m e2e                          # Solo tests marcados como e2e
    pytest -m "e2e and not docker"         # E2E excluyendo Docker tests
"""

import pytest
import asyncio
import logging
import time
import os
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import subprocess
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

# Imports del proyecto
try:
    from app.agents.core.smart_agent import SmartDocAgent
    from app.agents.tools.web.web_search_tool import WebSearchTool
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    IMPORT_ERROR = str(e)

# Configuración específica para tests E2E
E2E_TEST_CONFIG = {
    "default_timeout": 60.0,           # Tests E2E toman más tiempo
    "ui_timeout": 30.0,                # Timeout para elementos UI
    "api_timeout": 45.0,               # Timeout para llamadas API
    "deployment_timeout": 300.0,       # Timeout para deployment
    "allow_real_services": True,       # E2E puede usar servicios reales
    "require_docker": False,           # Docker opcional por defecto
    "require_browser": False,          # Browser opcional por defecto
    "cleanup_containers": True,        # Limpiar containers después
    "save_screenshots": True,          # Guardar screenshots en fallos
    "save_logs": True,                 # Guardar logs de servicios
}

# URLs para tests E2E
E2E_ENDPOINTS = {
    "api_base": "http://localhost:8002",
    "ui_base": "http://localhost:8501",
    "health_check": "http://localhost:8002/health",
    "api_docs": "http://localhost:8002/docs",
    "metrics": "http://localhost:8002/metrics"
}

# Logger específico para tests E2E
e2e_logger = logging.getLogger('smartdoc.e2e_tests')
e2e_logger.setLevel(logging.INFO)

@pytest.fixture(scope="session")
def e2e_test_environment():
    """Environment setup para tests E2E"""
    
    env_data = {
        "start_time": time.time(),
        "services_started": [],
        "containers_created": [],
        "test_data_created": [],
        "screenshots_saved": [],
        "cleanup_tasks": []
    }
    
    e2e_logger.info("Setting up E2E test environment")
    
    # Verificar servicios disponibles
    services_health = check_services_health()
    env_data["services_health"] = services_health
    
    yield env_data
    
    # Cleanup global
    cleanup_time = time.time()
    total_duration = cleanup_time - env_data["start_time"]
    
    e2e_logger.info(f"E2E test environment cleanup after {total_duration:.2f}s")
    
    # Ejecutar cleanup tasks
    for task in env_data["cleanup_tasks"]:
        try:
            if asyncio.iscoroutinefunction(task):
                asyncio.get_event_loop().run_until_complete(task())
            else:
                task()
        except Exception as e:
            e2e_logger.warning(f"E2E cleanup task failed: {e}")

@pytest.fixture
def docker_environment():
    """Docker environment para tests E2E que requieren deployment"""
    
    if not E2E_TEST_CONFIG["require_docker"]:
        pytest.skip("Docker tests disabled")
    
    # Verificar Docker disponible
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            pytest.skip("Docker not available")
    except FileNotFoundError:
        pytest.skip("Docker not installed")
    
    containers = []
    
    class DockerManager:
        def __init__(self):
            self.containers = containers
        
        def start_service(self, service_name: str, **kwargs) -> str:
            """Iniciar un servicio en Docker"""
            # Implementación simplificada
            container_id = f"test_{service_name}_{int(time.time())}"
            self.containers.append(container_id)
            return container_id
        
        def stop_service(self, container_id: str):
            """Detener un servicio"""
            if container_id in self.containers:
                # subprocess.run(["docker", "stop", container_id])
                self.containers.remove(container_id)
        
        def wait_for_service(self, url: str, timeout: float = 60.0) -> bool:
            """Esperar a que un servicio esté disponible"""
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        return True
                except:
                    pass
                time.sleep(2)
            return False
    
    manager = DockerManager()
    yield manager
    
    # Cleanup containers
    if E2E_TEST_CONFIG["cleanup_containers"]:
        for container_id in containers:
            try:
                manager.stop_service(container_id)
            except Exception as e:
                e2e_logger.warning(f"Failed to cleanup container {container_id}: {e}")

@pytest.fixture
def browser_driver():
    """Selenium WebDriver para tests de UI"""
    
    if not E2E_TEST_CONFIG["require_browser"]:
        pytest.skip("Browser tests disabled")
    
    # Configurar Chrome en modo headless
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.implicitly_wait(10)
        
        yield driver
        
        # Cleanup
        driver.quit()
        
    except Exception as e:
        pytest.skip(f"Browser setup failed: {e}")

@pytest.fixture
async def e2e_api_client():
    """Cliente API configurado para tests E2E"""
    
    import httpx
    
    class E2EApiClient:
        def __init__(self, base_url: str):
            self.base_url = base_url
            self.client = httpx.AsyncClient(timeout=E2E_TEST_CONFIG["api_timeout"])
        
        async def health_check(self) -> bool:
            """Verificar health del API"""
            try:
                response = await self.client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except:
                return False
        
        async def create_research_session(self, topic: str, objectives: List[str]) -> Dict:
            """Crear sesión de investigación"""
            response = await self.client.post(
                f"{self.base_url}/research/session",
                json={
                    "topic": topic,
                    "objectives": objectives,
                    "research_depth": "standard"
                }
            )
            response.raise_for_status()
            return response.json()
        
        async def send_query(self, session_id: str, query: str) -> Dict:
            """Enviar query al agente"""
            response = await self.client.post(
                f"{self.base_url}/research/chat/{session_id}",
                json={
                    "message": query,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()
        
        async def get_session_status(self, session_id: str) -> Dict:
            """Obtener estado de sesión"""
            response = await self.client.get(f"{self.base_url}/research/status/{session_id}")
            response.raise_for_status()
            return response.json()
        
        async def close(self):
            """Cerrar cliente"""
            await self.client.aclose()
    
    client = E2EApiClient(E2E_ENDPOINTS["api_base"])
    
    # Verificar que API está disponible
    if not await client.health_check():
        await client.close()
        pytest.skip("API not available for E2E tests")
    
    yield client
    
    await client.close()

# Utilidades para tests E2E
class E2ETestHelpers:
    """Utilidades específicas para tests E2E"""
    
    @staticmethod
    def wait_for_service_ready(url: str, timeout: float = 60.0) -> bool:
        """Esperar a que un servicio esté listo"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False
    
    @staticmethod
    def take_screenshot(driver, name: str):
        """Tomar screenshot del browser"""
        if E2E_TEST_CONFIG["save_screenshots"]:
            from tests import get_test_config
            reports_dir = get_test_config("reports_dir")
            screenshot_path = reports_dir / f"screenshot_{name}_{int(time.time())}.png"
            driver.save_screenshot(str(screenshot_path))
            e2e_logger.info(f"Screenshot saved: {screenshot_path}")
    
    @staticmethod
    async def simulate_complete_user_journey(api_client, topic: str) -> Dict:
        """Simular journey completo de usuario"""
        journey_result = {
            "steps_completed": 0,
            "total_time": 0,
            "session_id": None,
            "queries_processed": 0,
            "errors": []
        }
        
        start_time = time.time()
        
        try:
            # 1. Crear sesión
            session_response = await api_client.create_research_session(
                topic=topic,
                objectives=["Understand key concepts", "Find practical examples"]
            )
            journey_result["session_id"] = session_response.get("session_id")
            journey_result["steps_completed"] += 1
            
            # 2. Enviar queries
            queries = [
                f"What is {topic}?",
                f"How to get started with {topic}?",
                f"Best practices for {topic}?"
            ]
            
            for query in queries:
                response = await api_client.send_query(journey_result["session_id"], query)
                if response.get("success", False):
                    journey_result["queries_processed"] += 1
                journey_result["steps_completed"] += 1
            
            # 3. Verificar estado final
            status = await api_client.get_session_status(journey_result["session_id"])
            if status.get("message_count", 0) > 0:
                journey_result["steps_completed"] += 1
            
        except Exception as e:
            journey_result["errors"].append(str(e))
        
        journey_result["total_time"] = time.time() - start_time
        return journey_result
    
    @staticmethod
    def validate_ui_elements(driver, expected_elements: List[str]) -> Dict[str, bool]:
        """Validar que elementos UI están presentes"""
        results = {}
        
        for element_selector in expected_elements:
            try:
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, element_selector))
                )
                results[element_selector] = element is not None
            except:
                results[element_selector] = False
        
        return results

def check_services_health() -> Dict[str, bool]:
    """Verificar health de servicios para E2E"""
    services = {}
    
    for service_name, url in E2E_ENDPOINTS.items():
        try:
            response = requests.get(url, timeout=5)
            services[service_name] = response.status_code == 200
        except:
            services[service_name] = False
    
    return services

# Decoradores para tests E2E
def e2e_test(func):
    """Decorator para marcar tests como E2E"""
    return pytest.mark.e2e(func)

def requires_services(*service_names):
    """Decorator para tests que requieren servicios específicos"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Verificar servicios antes del test
            health = check_services_health()
            for service in service_names:
                if not health.get(service, False):
                    pytest.skip(f"Required service not available: {service}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def requires_docker(func):
    """Decorator para tests que requieren Docker"""
    return pytest.mark.skipif(
        not E2E_TEST_CONFIG["require_docker"],
        reason="Docker tests disabled"
    )(func)

def requires_browser(func):
    """Decorator para tests que requieren browser"""
    return pytest.mark.skipif(
        not E2E_TEST_CONFIG["require_browser"],
        reason="Browser tests disabled"
    )(func)

def slow_e2e_test(func):
    """Decorator para tests E2E lentos"""
    return pytest.mark.e2e(pytest.mark.slow(func))

# Assertions específicas para E2E
class E2EAssertions:
    """Assertions personalizadas para tests E2E"""
    
    @staticmethod
    def assert_user_journey_successful(journey_result: Dict):
        """Verificar que un user journey fue exitoso"""
        assert journey_result["steps_completed"] >= 3, "Insufficient steps completed"
        assert journey_result["queries_processed"] >= 1, "No queries processed successfully"
        assert len(journey_result["errors"]) == 0, f"Journey had errors: {journey_result['errors']}"
        assert journey_result["session_id"] is not None, "No session created"
    
    @staticmethod
    def assert_api_response_valid(response: Dict, expected_fields: List[str]):
        """Verificar que respuesta API es válida"""
        for field in expected_fields:
            assert field in response, f"Missing field in API response: {field}"
        
        if "success" in response:
            assert response["success"] != False, "API returned failure"
    
    @staticmethod
    def assert_ui_functional(validation_results: Dict[str, bool]):
        """Verificar que UI es funcional"""
        failed_elements = [elem for elem, found in validation_results.items() if not found]
        assert len(failed_elements) == 0, f"UI elements not found: {failed_elements}"
    
    @staticmethod
    def assert_performance_acceptable(duration: float, max_duration: float):
        """Verificar que performance es aceptable"""
        assert duration <= max_duration, \
            f"E2E test took {duration:.2f}s, max allowed: {max_duration}s"

# Datos específicos para tests E2E
E2E_TEST_DATA = {
    "realistic_scenarios": [
        {
            "name": "Student Research",
            "topic": "Machine Learning for Beginners",
            "queries": [
                "What is machine learning and how does it work?",
                "What are the different types of machine learning?",
                "How do I start learning machine learning?",
                "What programming languages are best for ML?"
            ],
            "expected_duration": 120.0
        },
        {
            "name": "Business Analysis",
            "topic": "Digital Marketing Trends 2024",
            "queries": [
                "What are the latest digital marketing trends?",
                "How is AI changing digital marketing?",
                "What metrics should I track for digital campaigns?",
                "ROI optimization strategies for digital marketing"
            ],
            "expected_duration": 90.0
        },
        {
            "name": "Technical Research",
            "topic": "Microservices Architecture Best Practices",
            "queries": [
                "What are microservices and their benefits?",
                "How to design microservices architecture?",
                "Common pitfalls in microservices implementation?",
                "Monitoring and observability for microservices"
            ],
            "expected_duration": 150.0
        }
    ],
    "ui_test_scenarios": [
        {
            "name": "Chat Interface",
            "elements": [
                "input[type='text']",  # Chat input
                ".chat-message",       # Chat messages
                ".send-button",        # Send button
                ".session-info"        # Session info
            ]
        },
        {
            "name": "Research Dashboard",
            "elements": [
                ".progress-indicator",
                ".source-list",
                ".objective-tracker",
                ".export-button"
            ]
        }
    ],
    "api_test_scenarios": [
        {
            "endpoint": "/research/session",
            "method": "POST",
            "expected_fields": ["session_id", "status"]
        },
        {
            "endpoint": "/research/chat/{session_id}",
            "method": "POST", 
            "expected_fields": ["response", "success", "sources"]
        },
        {
            "endpoint": "/research/status/{session_id}",
            "method": "GET",
            "expected_fields": ["progress", "message_count", "topic"]
        }
    ]
}

# Setup de logging para E2E
def setup_e2e_logging():
    """Configurar logging específico para tests E2E"""
    from tests import get_test_config
    logs_dir = get_test_config("logs_dir")
    
    log_file = logs_dir / "e2e_tests.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [E2E] - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    if not e2e_logger.handlers:
        e2e_logger.addHandler(file_handler)

# Setup inicial
setup_e2e_logging()

# Exports
__all__ = [
    "E2E_TEST_CONFIG",
    "E2E_ENDPOINTS",
    "E2ETestHelpers",
    "E2EAssertions",
    "E2E_TEST_DATA",
    "e2e_test",
    "requires_services",
    "requires_docker",
    "requires_browser",
    "slow_e2e_test",
    "e2e_logger"
]

__version__ = "1.0.0"
__description__ = "End-to-end tests package for SmartDoc Research Agent"