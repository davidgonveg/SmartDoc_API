"""
Web Search Tools Module
Módulo completo de herramientas de búsqueda web para SmartDoc Agent
"""

# Core components
from .web_search_tool import (
    WebSearchTool,
    WebSearchRequest,
    WebSearchResult,
    web_search,
    quick_search
)

# Search engines
from .search_engines import (
    BaseSearchEngine,
    DuckDuckGoSearchEngine,
    GoogleCustomSearchEngine,
    SearXSearchEngine,
    BingSearchEngine,
    SearchEngineManager,
    SearchResult,
    SearchResponse,
    search_engine_manager
)

# Content extraction
from .content_extractor import (
    ContentExtractor,
    ExtractedContent,
    extract_content_from_url,
    extract_content_from_html,
    global_content_extractor
)

# Rate limiting
from .rate_limiter import (
    RateLimitManager,
    RateLimit,
    RateLimitType,
    RateLimitState,
    create_rate_limiter,
    global_rate_limiter
)

# Web utilities
from .web_utils import (
    WebResponse,
    URLInfo,
    URLNormalizer,
    ContentCleaner,
    WebRequestManager,
    RobotsTxtChecker,
    is_valid_url,
    extract_links_from_html,
    get_content_hash,
    estimate_reading_time,
    detect_language,
    robots_checker
)

# User agents
from .user_agents import (
    UserAgentManager,
    UserAgentInfo,
    BrowserType,
    PlatformType,
    get_random_user_agent,
    get_headers_with_random_user_agent,
    get_academic_user_agent,
    global_user_agent_manager
)

# Configuration
from .config import (
    # Enums
    SearchEngine,
    ContentType,
    
    # Main config classes
    SearchEngineConfig,
    ExtractionRule,
    WebSearchConfig,
    
    # Configuration dictionaries
    SEARCH_ENGINES_CONFIG,
    SEARCH_CONFIG,
    QUERY_TYPE_CONFIG,
    EXTRACTION_RULES,
    DOMAIN_SPECIFIC_RULES,
    
    # Helper functions
    get_engine_config,
    get_enabled_engines,
    get_query_config,
    validate_engine_config,
    get_extraction_rule,
    get_domain_from_url,
    is_content_valid,
    calculate_content_quality_score,
    
    # Global config instance
    config
)

# Export all main components for easy imports
__all__ = [
    # Main tool
    "WebSearchTool",
    "WebSearchRequest", 
    "WebSearchResult",
    "web_search",
    "quick_search",
    
    # Search engines
    "BaseSearchEngine",
    "DuckDuckGoSearchEngine",
    "GoogleCustomSearchEngine", 
    "SearXSearchEngine",
    "BingSearchEngine",
    "SearchEngineManager",
    "SearchResult",
    "SearchResponse",
    "search_engine_manager",
    
    # Content extraction
    "ContentExtractor",
    "ExtractedContent",
    "extract_content_from_url",
    "extract_content_from_html",
    "global_content_extractor",
    
    # Rate limiting
    "RateLimitManager",
    "RateLimit",
    "RateLimitType", 
    "RateLimitState",
    "create_rate_limiter",
    "global_rate_limiter",
    
    # Web utilities
    "WebResponse",
    "URLInfo",
    "URLNormalizer",
    "ContentCleaner",
    "WebRequestManager",
    "RobotsTxtChecker",
    "is_valid_url",
    "extract_links_from_html",
    "get_content_hash",
    "estimate_reading_time",
    "detect_language",
    "robots_checker",
    
    # User agents
    "UserAgentManager",
    "UserAgentInfo",
    "BrowserType",
    "PlatformType", 
    "get_random_user_agent",
    "get_headers_with_random_user_agent",
    "get_academic_user_agent",
    "global_user_agent_manager",
    
    # Configuration
    "SearchEngine",
    "ContentType",
    "SearchEngineConfig",
    "ExtractionRule", 
    "WebSearchConfig",
    "SEARCH_ENGINES_CONFIG",
    "SEARCH_CONFIG",
    "QUERY_TYPE_CONFIG",
    "EXTRACTION_RULES",
    "DOMAIN_SPECIFIC_RULES",
    "get_engine_config",
    "get_enabled_engines",
    "get_query_config",
    "validate_engine_config",
    "get_extraction_rule",
    "get_domain_from_url",
    "is_content_valid", 
    "calculate_content_quality_score",
    "config",
]

# Module metadata
__version__ = "1.0.0"
__author__ = "SmartDoc Team"
__description__ = "Complete web search and content extraction toolkit"

# Global instances for easy access
web_search_tool = WebSearchTool()
web_config = config

# Convenience functions for quick access
async def search_web(query: str, **kwargs) -> str:
    """Quick web search function"""
    return await web_search_tool._arun(query, **kwargs)

async def extract_page_content(url: str, content_type_hint: str = None) -> ExtractedContent:
    """Quick content extraction function"""
    return await global_content_extractor.extract_content(url, content_type_hint=content_type_hint)

def get_search_engines_status() -> dict:
    """Get status of all search engines"""
    import asyncio
    return asyncio.run(search_engine_manager.health_check())

def get_web_tools_info() -> dict:
    """Get comprehensive info about web tools"""
    return {
        "version": __version__,
        "description": __description__,
        "main_tool": web_search_tool.get_tool_info(),
        "available_engines": [engine.value for engine in get_enabled_engines()],
        "supported_domains": list(DOMAIN_SPECIFIC_RULES.keys()),
        "query_types": list(QUERY_TYPE_CONFIG.keys()),
        "rate_limits": {
            "global": SEARCH_CONFIG.get("global_rate_limit", "N/A"),
            "per_engine": {
                engine.value: config.engines[engine].rate_limit_per_minute 
                for engine in get_enabled_engines()
            }
        },
        "features": [
            "Multi-engine search with fallbacks",
            "Intelligent content extraction", 
            "Rate limiting and robots.txt compliance",
            "User-Agent rotation",
            "Domain-specific extraction rules",
            "Content quality scoring",
            "Language detection",
            "Caching for performance",
            "Parallel content processing",
            "Query type optimization"
        ]
    }

# Auto-validation on module import
import logging
logger = logging.getLogger(__name__)

def _validate_module_setup():
    """Validate module setup on import"""
    try:
        # Validate configuration
        validation_results = config.validate_config()
        
        if not validation_results["valid"]:
            logger.warning(f"Web search module configuration issues: {validation_results['errors']}")
        
        # Log available engines
        available_engines = get_enabled_engines()
        logger.info(f"Web search module loaded with {len(available_engines)} engines: {[e.value for e in available_engines]}")
        
        # Log special configurations
        if any(engine.requires_api_key for engine in [get_engine_config(e) for e in available_engines]):
            logger.info("Some search engines require API keys - check configuration for full functionality")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating web search module: {e}")
        return False

# Run validation
_module_valid = _validate_module_setup()

# Export module health status
def is_module_healthy() -> bool:
    """Check if web search module is properly configured"""
    return _module_valid

# Advanced search functions for specific use cases
async def academic_search(query: str, max_results: int = 5) -> str:
    """Search optimized for academic content"""
    return await web_search_tool._arun(query, query_type="academic", max_results=max_results)

async def news_search(query: str, max_results: int = 8) -> str:
    """Search optimized for news content"""
    return await web_search_tool._arun(query, query_type="news", max_results=max_results)

async def technical_search(query: str, max_results: int = 6) -> str:
    """Search optimized for technical documentation"""
    return await web_search_tool._arun(query, query_type="technical", max_results=max_results)

async def domain_specific_search(query: str, domains: list, max_results: int = 5) -> str:
    """Search limited to specific domains"""
    return await web_search_tool._arun(query, filter_domains=domains, max_results=max_results)

# Add specialized search functions to exports
__all__.extend([
    "search_web",
    "extract_page_content", 
    "get_search_engines_status",
    "get_web_tools_info",
    "is_module_healthy",
    "academic_search",
    "news_search", 
    "technical_search",
    "domain_specific_search",
    "web_search_tool",
    "web_config"
])

# Final module setup logging
if _module_valid:
    logger.info(f"✅ Web Search Tools Module v{__version__} loaded successfully")
else:
    logger.warning(f"⚠️ Web Search Tools Module v{__version__} loaded with issues")