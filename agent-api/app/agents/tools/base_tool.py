"""
SmartDoc Agent - Base Tool Interface
Clase base para todas las herramientas del agente de investigación
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from enum import Enum

from langchain.tools import BaseTool as LangChainBaseTool
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class ToolCategory(Enum):
    """Categorías de herramientas disponibles"""
    WEB_SEARCH = "web_search"
    DOCUMENT_PROCESSING = "document_processing"
    CALCULATION = "calculation"
    CODE_EXECUTION = "code_execution"
    MEMORY_STORAGE = "memory_storage"
    REPORT_GENERATION = "report_generation"
    DATA_ANALYSIS = "data_analysis"
    VALIDATION = "validation"

class ToolResult(BaseModel):
    """Modelo estándar para resultados de herramientas"""
    
    success: bool = Field(description="Si la operación fue exitosa")
    data: Any = Field(description="Datos principales del resultado")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadatos adicionales")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Fuentes consultadas")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Nivel de confianza (0-1)")
    execution_time: float = Field(default=0.0, description="Tiempo de ejecución en segundos")
    error_message: Optional[str] = Field(default=None, description="Mensaje de error si failed")
    
    @validator('confidence')
    def validate_confidence(cls, v):
        """Validar que la confianza esté en rango válido"""
        return max(0.0, min(1.0, v))

class ToolInput(BaseModel):
    """Modelo base para inputs de herramientas"""
    
    query: str = Field(description="Query o parámetro principal")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Contexto adicional")
    session_id: Optional[str] = Field(default=None, description="ID de sesión para tracking")
    
class BaseTool(LangChainBaseTool, ABC):
    """
    Clase base abstracta para todas las herramientas del SmartDoc Agent
    
    Todas las herramientas deben heredar de esta clase e implementar:
    - _arun(): Lógica asíncrona principal
    - get_tool_info(): Información detallada de la herramienta
    """
    
    # Metadatos de la herramienta
    category: ToolCategory
    version: str = "1.0.0"
    requires_internet: bool = False
    requires_gpu: bool = False
    max_execution_time: int = 60  # segundos
    rate_limit: Optional[int] = None  # requests per minute
    
    # Estado interno
    _last_execution: Optional[datetime] = None
    _execution_count: int = 0
    _rate_limit_tracker: List[datetime] = []
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_tool()
    
    def _setup_tool(self):
        """Configuración inicial de la herramienta"""
        logger.info(f"Inicializando herramienta: {self.name} (v{self.version})")
        
    @abstractmethod
    async def _arun(self, query: str, **kwargs) -> str:
        """
        Implementación asíncrona principal de la herramienta
        
        Args:
            query: Input principal de la herramienta
            **kwargs: Parámetros adicionales
            
        Returns:
            str: Resultado formateado para el agente
        """
        pass
    
    def _run(self, query: str, **kwargs) -> str:
        """Wrapper síncrono - no usar directamente"""
        return asyncio.run(self._arun(query, **kwargs))
    
    async def execute(self, input_data: Union[str, ToolInput, Dict]) -> ToolResult:
        """
        Método principal para ejecutar la herramienta con manejo completo
        
        Args:
            input_data: Input en formato string, ToolInput o diccionario
            
        Returns:
            ToolResult: Resultado estructurado
        """
        
        start_time = time.time()
        
        try:
            # Normalizar input
            normalized_input = self._normalize_input(input_data)
            
            # Validaciones pre-ejecución
            await self._pre_execution_checks(normalized_input)
            
            # Ejecutar herramienta
            logger.info(f"Ejecutando {self.name} con query: {normalized_input.query[:100]}...")
            
            result_data = await self._arun(
                query=normalized_input.query,
                context=normalized_input.context,
                session_id=normalized_input.session_id
            )
            
            # Post-procesamiento
            execution_time = time.time() - start_time
            self._update_execution_stats(execution_time)
            
            # Construir resultado exitoso
            return ToolResult(
                success=True,
                data=result_data,
                metadata=self._get_execution_metadata(normalized_input, execution_time),
                confidence=self._calculate_confidence(result_data),
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Error ejecutando {self.name}: {e}")
            
            return ToolResult(
                success=False,
                data=None,
                metadata=self._get_execution_metadata(input_data, execution_time),
                error_message=str(e),
                execution_time=execution_time
            )
    
    def _normalize_input(self, input_data: Union[str, ToolInput, Dict]) -> ToolInput:
        """Normalizar diferentes tipos de input a ToolInput"""
        
        if isinstance(input_data, str):
            return ToolInput(query=input_data)
        elif isinstance(input_data, ToolInput):
            return input_data
        elif isinstance(input_data, dict):
            return ToolInput(**input_data)
        else:
            raise ValueError(f"Tipo de input no soportado: {type(input_data)}")
    
    async def _pre_execution_checks(self, input_data: ToolInput):
        """Validaciones antes de ejecutar la herramienta"""
        
        # Validar rate limiting
        if self.rate_limit:
            await self._check_rate_limit()
        
        # Validar input
        if not input_data.query or len(input_data.query.strip()) == 0:
            raise ValueError("Query no puede estar vacío")
        
        # Validaciones específicas de la herramienta
        await self._validate_input(input_data)
    
    async def _validate_input(self, input_data: ToolInput):
        """Validaciones específicas de cada herramienta - override en subclases"""
        pass
    
    async def _check_rate_limit(self):
        """Verificar rate limiting"""
        if not self.rate_limit:
            return
        
        now = datetime.now()
        # Limpiar entradas antiguas (más de 1 minuto)
        self._rate_limit_tracker = [
            timestamp for timestamp in self._rate_limit_tracker
            if (now - timestamp).total_seconds() < 60
        ]
        
        if len(self._rate_limit_tracker) >= self.rate_limit:
            wait_time = 60 - (now - self._rate_limit_tracker[0]).total_seconds()
            if wait_time > 0:
                logger.warning(f"Rate limit alcanzado para {self.name}, esperando {wait_time:.1f} segundos")
                await asyncio.sleep(wait_time)
        
        self._rate_limit_tracker.append(now)
    
    def _update_execution_stats(self, execution_time: float):
        """Actualizar estadísticas de ejecución"""
        self._last_execution = datetime.now()
        self._execution_count += 1
        
        if execution_time > self.max_execution_time:
            logger.warning(f"{self.name} tardó {execution_time:.2f}s (límite: {self.max_execution_time}s)")
    
    def _get_execution_metadata(self, input_data: Any, execution_time: float) -> Dict[str, Any]:
        """Generar metadatos de ejecución"""
        return {
            "tool_name": self.name,
            "tool_version": self.version,
            "category": self.category.value,
            "execution_time": execution_time,
            "execution_count": self._execution_count,
            "timestamp": datetime.now().isoformat(),
            "input_length": len(str(input_data)),
            "requires_internet": self.requires_internet,
            "requires_gpu": self.requires_gpu
        }
    
    def _calculate_confidence(self, result_data: Any) -> float:
        """Calcular confianza del resultado - override en subclases"""
        # Lógica básica - las subclases pueden ser más específicas
        if result_data is None:
            return 0.0
        elif isinstance(result_data, str) and len(result_data) > 50:
            return 0.7
        elif isinstance(result_data, (list, dict)) and len(result_data) > 0:
            return 0.8
        else:
            return 0.5
    
    @abstractmethod
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Información detallada de la herramienta
        
        Returns:
            Dict con información completa de la herramienta
        """
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de uso de la herramienta"""
        return {
            "execution_count": self._execution_count,
            "last_execution": self._last_execution.isoformat() if self._last_execution else None,
            "rate_limit": self.rate_limit,
            "max_execution_time": self.max_execution_time,
            "category": self.category.value,
            "version": self.version
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check de la herramienta"""
        try:
            # Test básico según tipo de herramienta
            test_input = self._get_health_check_input()
            start_time = time.time()
            
            result = await self.execute(test_input)
            health_time = time.time() - start_time
            
            return {
                "healthy": result.success,
                "response_time": health_time,
                "error": result.error_message if not result.success else None,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    def _get_health_check_input(self) -> str:
        """Input para health check - override en subclases"""
        return "test"

class ToolRegistry:
    """Registry para gestionar todas las herramientas disponibles"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tools_by_category: Dict[ToolCategory, List[BaseTool]] = {}
        
    def register_tool(self, tool: BaseTool):
        """Registrar una nueva herramienta"""
        self._tools[tool.name] = tool
        
        if tool.category not in self._tools_by_category:
            self._tools_by_category[tool.category] = []
        
        self._tools_by_category[tool.category].append(tool)
        
        logger.info(f"Herramienta registrada: {tool.name} ({tool.category.value})")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Obtener herramienta por nombre"""
        return self._tools.get(name)
    
    def get_tools_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Obtener herramientas por categoría"""
        return self._tools_by_category.get(category, [])
    
    def list_tools(self) -> List[str]:
        """Listar nombres de todas las herramientas"""
        return list(self._tools.keys())
    
    def get_all_tools(self) -> List[BaseTool]:
        """Obtener todas las herramientas registradas"""
        return list(self._tools.values())
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Health check de todas las herramientas"""
        results = {}
        
        for name, tool in self._tools.items():
            results[name] = await tool.health_check()
        
        return results
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Estadísticas del registry"""
        return {
            "total_tools": len(self._tools),
            "tools_by_category": {
                category.value: len(tools) 
                for category, tools in self._tools_by_category.items()
            },
            "tool_names": list(self._tools.keys())
        }

# Instancia global del registry
global_tool_registry = ToolRegistry()

def register_tool(tool: BaseTool):
    """Función helper para registrar herramientas"""
    global_tool_registry.register_tool(tool)

def get_available_tools() -> List[BaseTool]:
    """Función helper para obtener todas las herramientas"""
    return global_tool_registry.get_all_tools()

def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """Función helper para obtener herramienta por nombre"""
    return global_tool_registry.get_tool(name)

# Decorador para auto-registrar herramientas
def tool_registration(category: ToolCategory, auto_register: bool = True):
    """
    Decorador para auto-registrar herramientas
    
    Usage:
        @tool_registration(ToolCategory.WEB_SEARCH)
        class WebSearchTool(BaseTool):
            ...
    """
    def decorator(cls):
        # Asegurar que la clase tenga la categoría
        cls.category = category
        
        # Auto-registrar si está habilitado
        if auto_register:
            # Crear instancia y registrar
            tool_instance = cls()
            register_tool(tool_instance)
        
        return cls
    
    return decorator

# Herramienta de ejemplo para testing
class EchoTool(BaseTool):
    """Herramienta de ejemplo para testing"""
    
    name = "echo_tool"
    description = "Herramienta de prueba que hace echo del input"
    category = ToolCategory.VALIDATION
    
    async def _arun(self, query: str, **kwargs) -> str:
        """Implementación simple de echo"""
        return f"Echo: {query}"
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Información de la herramienta echo"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "capabilities": ["echo", "testing"],
            "limitations": ["Solo para testing"],
            "example_usage": "echo_tool('hello world') -> 'Echo: hello world'"
        }
    
    def _get_health_check_input(self) -> str:
        """Input específico para health check"""
        return "health_check_test"