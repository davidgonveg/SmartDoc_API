"""
Search Configuration for Web Search Tool
Configuraciones para diferentes motores de búsqueda y parámetros
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class SearchEngine(Enum):
    """Motores de búsqueda disponibles"""
    DUCKDUCKGO = "duckduckgo"
    GOOGLE_CUSTOM = "google_custom"
    BING = "bing"
    SEARX = "searx"

@dataclass
class SearchEngineConfig:
    """Configuración para un motor de búsqueda específico"""
    name: str
    base_url: str
    search_endpoint: str
    params: Dict[str, Any]
    headers: Dict[str, str]
    rate_limit_per_minute: int
    timeout: int
    requires_api_key: bool = False
    api_key_env_var: Optional[str] = None
    max_results: int = 10
    enabled: bool = True

# Configuraciones por motor de búsqueda
SEARCH_ENGINES_CONFIG = {
    SearchEngine.DUCKDUCKGO: SearchEngineConfig(
        name="DuckDuckGo",
        base_url="https://duckduckgo.com",
        search_endpoint="/html/",
        params={
            "q": "",  # Query placeholder
            "kl": "us-en",  # Language/region
            "safe": "moderate",  # Safe search
            "df": "",  # Date filter (empty = any time)
        },
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; SmartDocAgent/1.0)",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        },
        rate_limit_per_minute=30,
        timeout=10,
        requires_api_key=False,
        max_results=10,
        enabled=True
    ),
    
    SearchEngine.GOOGLE_CUSTOM: SearchEngineConfig(
        name="Google Custom Search",
        base_url="https://www.googleapis.com",
        search_endpoint="/customsearch/v1",
        params={
            "q": "",  # Query placeholder
            "cx": "",  # Custom Search Engine ID (from env)
            "key": "",  # API key (from env)
            "num": 10,  # Number of results
            "safe": "medium",  # Safe search
            "lr": "lang_en",  # Language restriction
        },
        headers={
            "User-Agent": "SmartDocAgent/1.0",
            "Accept": "application/json",
        },
        rate_limit_per_minute=100,  # Depends on API quota
        timeout=15,
        requires_api_key=True,
        api_key_env_var="GOOGLE_SEARCH_API_KEY",
        max_results=10,
        enabled=False  # Disabled by default, enable with API key
    ),
    
    SearchEngine.BING: SearchEngineConfig(
        name="Bing Search",
        base_url="https://api.bing.microsoft.com",
        search_endpoint="/v7.0/search",
        params={
            "q": "",  # Query placeholder
            "count": 10,  # Number of results
            "offset": 0,  # Offset for pagination
            "mkt": "en-US",  # Market
            "safesearch": "Moderate",
            "textDecorations": False,
            "textFormat": "Raw",
        },
        headers={
            "Ocp-Apim-Subscription-Key": "",  # API key (from env)
            "User-Agent": "SmartDocAgent/1.0",
            "Accept": "application/json",
        },
        rate_limit_per_minute=1000,  # Depends on subscription
        timeout=15,
        requires_api_key=True,
        api_key_env_var="BING_SEARCH_API_KEY",
        max_results=10,
        enabled=False  # Disabled by default
    ),
    
    SearchEngine.SEARX: SearchEngineConfig(
        name="SearX",
        base_url="https://searx.be",  # Public instance, can be changed
        search_endpoint="/search",
        params={
            "q": "",  # Query placeholder
            "categories": "general",
            "engines": "google,duckduckgo,bing",
            "format": "json",
            "safesearch": "1",
            "language": "en",
        },
        headers={
            "User-Agent": "SmartDocAgent/1.0",
            "Accept": "application/json",
        },
        rate_limit_per_minute=60,
        timeout=20,
        requires_api_key=False,
        max_results=10,
        enabled=True
    )
}

# Configuración general de búsqueda
SEARCH_CONFIG = {
    # Motor de búsqueda por defecto
    "default_engine": SearchEngine.DUCKDUCKGO,
    
    # Engines de fallback si el principal falla
    "fallback_engines": [SearchEngine.SEARX, SearchEngine.DUCKDUCKGO],
    
    # Configuración de reintentos
    "max_retries": 3,
    "retry_delay": 2,  # segundos
    "backoff_multiplier": 2,
    
    # Configuración de timeout
    "default_timeout": 15,
    "max_timeout": 30,
    
    # Configuración de resultados
    "default_max_results": 10,
    "absolute_max_results": 50,
    
    # Configuración de rate limiting
    "global_rate_limit": 100,  # requests per minute across all engines
    "rate_limit_window": 60,   # seconds
    
    # Configuración de contenido
    "min_content_length": 100,  # Minimum content length to consider
    "max_content_length": 50000,  # Maximum content length to process
    
    # Configuración de filtros
    "blocked_domains": [
        "facebook.com",
        "twitter.com", 
        "instagram.com",
        "tiktok.com",
        "reddit.com",  # Opcional, puede tener contenido útil
    ],
    
    "preferred_domains": [
        "wikipedia.org",
        "github.com",
        "stackoverflow.com",
        "medium.com",
        "arxiv.org",
        "scholar.google.com",
    ],
    
    # Configuración de User-Agent rotation
    "rotate_user_agents": True,
    "user_agent_pool_size": 10,
    
    # Configuración de cache
    "cache_enabled": True,
    "cache_ttl": 3600,  # 1 hour
    "cache_max_size": 1000,  # Maximum cached queries
    
    # Configuración de logging
    "log_searches": True,
    "log_results": False,  # Para debugging, normalmente False
    "log_errors": True,
}

# Configuración específica por tipo de query
QUERY_TYPE_CONFIG = {
    "academic": {
        "preferred_engines": [SearchEngine.GOOGLE_CUSTOM, SearchEngine.SEARX],
        "preferred_domains": [
            "scholar.google.com",
            "arxiv.org",
            "pubmed.ncbi.nlm.nih.gov",
            "ieee.org",
            "acm.org",
            "springer.com",
            "elsevier.com",
        ],
        "query_modifiers": ["site:scholar.google.com", "filetype:pdf"],
        "max_results": 15,
    },
    
    "news": {
        "preferred_engines": [SearchEngine.GOOGLE_CUSTOM, SearchEngine.BING],
        "preferred_domains": [
            "reuters.com",
            "bbc.com", 
            "cnn.com",
            "npr.org",
            "apnews.com",
        ],
        "query_modifiers": ["after:2023-01-01"],  # Recent news
        "max_results": 10,
    },
    
    "technical": {
        "preferred_engines": [SearchEngine.GOOGLE_CUSTOM, SearchEngine.SEARX],
        "preferred_domains": [
            "stackoverflow.com",
            "github.com",
            "docs.python.org",
            "developer.mozilla.org",
            "readthedocs.io",
        ],
        "query_modifiers": ["site:stackoverflow.com OR site:github.com"],
        "max_results": 12,
    },
    
    "general": {
        "preferred_engines": [SearchEngine.DUCKDUCKGO, SearchEngine.SEARX],
        "preferred_domains": [
            "wikipedia.org",
            "britannica.com",
            "howstuffworks.com",
        ],
        "query_modifiers": [],
        "max_results": 10,
    }
}

def get_engine_config(engine: SearchEngine) -> SearchEngineConfig:
    """Obtener configuración de un motor de búsqueda"""
    return SEARCH_ENGINES_CONFIG.get(engine)

def get_enabled_engines() -> List[SearchEngine]:
    """Obtener lista de motores habilitados"""
    return [engine for engine, config in SEARCH_ENGINES_CONFIG.items() if config.enabled]

def get_query_config(query_type: str = "general") -> Dict[str, Any]:
    """Obtener configuración específica para tipo de query"""
    return QUERY_TYPE_CONFIG.get(query_type, QUERY_TYPE_CONFIG["general"])

def validate_engine_config(engine: SearchEngine) -> bool:
    """Validar si un motor está correctamente configurado"""
    config = get_engine_config(engine)
    if not config:
        return False
    
    if config.requires_api_key and not config.api_key_env_var:
        return False
    
    return config.enabled

# Configuración de desarrollo/testing
DEV_CONFIG = {
    "mock_responses": True,
    "use_local_cache": True,
    "reduced_rate_limits": True,
    "debug_logging": True,
    "test_mode": False,
}