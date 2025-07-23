"""
Web Search Tool Configuration Module
Exports de configuración para el web search tool
"""

from .search_config import (
    # Enums
    SearchEngine,
    
    # Data classes
    SearchEngineConfig,
    
    # Main configurations
    SEARCH_ENGINES_CONFIG,
    SEARCH_CONFIG,
    QUERY_TYPE_CONFIG,
    DEV_CONFIG,
    
    # Helper functions
    get_engine_config,
    get_enabled_engines,
    get_query_config,
    validate_engine_config,
)

from .extraction_rules import (
    # Enums
    ContentType,
    
    # Data classes
    ExtractionRule,
    
    # Main configurations
    EXTRACTION_RULES,
    DOMAIN_SPECIFIC_RULES,
    REMOVE_ELEMENTS,
    INVALID_CONTENT_PATTERNS,
    TEXT_CLEANING_CONFIG,
    CONTENT_QUALITY_CONFIG,
    PAGE_TYPE_RULES,
    
    # Helper functions
    get_extraction_rule,
    get_domain_from_url,
    is_content_valid,
    calculate_content_quality_score,
)

# Re-export all main components for easy imports
__all__ = [
    # Search Config
    "SearchEngine",
    "SearchEngineConfig", 
    "SEARCH_ENGINES_CONFIG",
    "SEARCH_CONFIG",
    "QUERY_TYPE_CONFIG",
    "DEV_CONFIG",
    "get_engine_config",
    "get_enabled_engines",
    "get_query_config",
    "validate_engine_config",
    
    # Extraction Rules
    "ContentType",
    "ExtractionRule",
    "EXTRACTION_RULES",
    "DOMAIN_SPECIFIC_RULES",
    "REMOVE_ELEMENTS",
    "INVALID_CONTENT_PATTERNS",
    "TEXT_CLEANING_CONFIG",
    "CONTENT_QUALITY_CONFIG",
    "PAGE_TYPE_RULES",
    "get_extraction_rule",
    "get_domain_from_url",
    "is_content_valid",
    "calculate_content_quality_score",
]

# Configuración consolidada para facilitar el acceso
class WebSearchConfig:
    """Clase helper para acceder a toda la configuración"""
    
    # Search engines
    engines = SEARCH_ENGINES_CONFIG
    search = SEARCH_CONFIG
    query_types = QUERY_TYPE_CONFIG
    
    # Content extraction
    extraction = EXTRACTION_RULES
    domain_rules = DOMAIN_SPECIFIC_RULES
    cleaning = TEXT_CLEANING_CONFIG
    quality = CONTENT_QUALITY_CONFIG
    
    # Development
    dev = DEV_CONFIG
    
    @classmethod
    def get_default_engine(cls) -> SearchEngine:
        """Obtener motor de búsqueda por defecto"""
        return cls.search["default_engine"]
    
    @classmethod
    def get_fallback_engines(cls) -> list:
        """Obtener motores de fallback"""
        return cls.search["fallback_engines"]
    
    @classmethod
    def get_max_results(cls, query_type: str = "general") -> int:
        """Obtener máximo número de resultados para un tipo de query"""
        query_config = get_query_config(query_type)
        return query_config.get("max_results", cls.search["default_max_results"])
    
    @classmethod
    def is_domain_blocked(cls, domain: str) -> bool:
        """Verificar si un dominio está bloqueado"""
        blocked = cls.search.get("blocked_domains", [])
        return any(blocked_domain in domain for blocked_domain in blocked)
    
    @classmethod
    def is_domain_preferred(cls, domain: str) -> bool:
        """Verificar si un dominio es preferido"""
        preferred = cls.search.get("preferred_domains", [])
        return any(preferred_domain in domain for preferred_domain in preferred)
    
    @classmethod
    def get_domain_priority(cls, domain: str) -> int:
        """Obtener prioridad de un dominio (mayor número = mayor prioridad)"""
        if cls.is_domain_blocked(domain):
            return 0
        elif cls.is_domain_preferred(domain):
            return 10
        else:
            return 5  # Neutral
    
    @classmethod
    def validate_config(cls) -> dict:
        """Validar toda la configuración"""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "engine_status": {}
        }
        
        # Validar engines
        for engine in SearchEngine:
            is_valid = validate_engine_config(engine)
            config = get_engine_config(engine)
            
            results["engine_status"][engine.value] = {
                "valid": is_valid,
                "enabled": config.enabled if config else False,
                "requires_api_key": config.requires_api_key if config else False,
            }
            
            if not is_valid and config and config.enabled:
                results["errors"].append(f"Engine {engine.value} is enabled but not properly configured")
                results["valid"] = False
        
        # Verificar si hay al menos un engine válido
        enabled_engines = get_enabled_engines()
        if not enabled_engines:
            results["errors"].append("No search engines are enabled")
            results["valid"] = False
        
        # Verificar configuración de rate limiting
        if cls.search["global_rate_limit"] <= 0:
            results["warnings"].append("Global rate limit is disabled")
        
        return results

# Crear instancia global para fácil acceso
config = WebSearchConfig()

# Validation al importar el módulo
_validation_results = config.validate_config()
if not _validation_results["valid"]:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Web search configuration issues found: {_validation_results['errors']}")