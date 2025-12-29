"""
WebSearchTool - Versión Compatible con LangChain Pydantic
Implementación que funciona correctamente con las restricciones de Pydantic
"""

import asyncio
import json
import logging
import re
import time
import hashlib
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urlencode, urlparse, quote_plus
from enum import Enum

# Third party imports
import httpx
from bs4 import BeautifulSoup
from langchain.tools import BaseTool

# Configure logging
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

class WebSearchTool(BaseTool):
    """
    Web Search Tool completamente funcional y compatible con LangChain
    
    Características:
    - Multi-engine support (DuckDuckGo como principal, fallbacks)
    - Parsing robusto y defensivo
    - Rate limiting inteligente
    - Cache con TTL
    - Error handling completo
    - Compatible con LangChain BaseTool
    - Respeta restricciones de Pydantic
    """
    
    # LangChain required attributes
    name: str = "web_search"
    description: str = """Search the web for current information. 
    Input should be a search query string.
    Returns search results with titles, URLs, and snippets."""
    
    def __init__(self, **kwargs):
        # Llamar al __init__ de BaseTool primero
        super().__init__(**kwargs)
        
        # Inicializar atributos privados (Pydantic no los valida)
        self._category = ToolCategory.WEB_SEARCH
        self._version = "2.1.0"
        self._requires_internet = True
        self._requires_gpu = False
        self._max_execution_time = 60
        self._rate_limit = 30
        
        # Cache system
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_request_time = 0
        self._min_request_interval = 2.0  # 2 seconds between requests
        
        # Statistics
        self._stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'cache_hits': 0,
            'engine_failures': 0
        }
        
        # HTTP client configuration
        self._client_timeout = 15
        self._max_retries = 3
        
        # User agents for better success rate
        self._user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        logger.info("WebSearchTool initialized successfully")
    
    # Properties para acceso a atributos privados
    @property
    def category(self) -> ToolCategory:
        return self._category
    
    @property
    def version(self) -> str:
        return self._version
    
    @property
    def requires_internet(self) -> bool:
        return self._requires_internet
    
    @property
    def stats(self) -> Dict[str, int]:
        return self._stats.copy()
    
    def _run(self, query: str, **kwargs) -> str:
        """Synchronous wrapper for async _arun"""
        return asyncio.run(self._arun(query, **kwargs))
    
    async def _arun(self, query: str, **kwargs) -> str:
        """
        Perform web search and return formatted results
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (max_results, etc.)
        
        Returns:
            Formatted search results as string
        """
        max_results = kwargs.get('max_results', 5)
        
        try:
            self._stats['total_searches'] += 1
            
            # Check cache first
            cache_key = self._get_cache_key(query, max_results)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._stats['cache_hits'] += 1
                logger.info(f"Cache hit for query: {query}")
                return cached_result
            
            # Enforce rate limiting
            await self._enforce_rate_limit()
            
            # Try different search engines with fallbacks
            engines = [
                ('duckduckgo', self._search_duckduckgo),
                ('html_duckduckgo', self._search_duckduckgo_html),
                ('searx', self._search_searx),
                ('bing', self._search_bing)
            ]
            
            last_error = None
            
            for engine_name, search_func in engines:
                try:
                    logger.info(f"Trying search engine: {engine_name}")
                    results = await search_func(query, max_results)
                    
                    if results and len(results) > 0:
                        formatted_result = self._format_results(query, results, engine_name)
                        
                        # Cache successful result
                        self._add_to_cache(cache_key, formatted_result)
                        self._stats['successful_searches'] += 1
                        
                        logger.info(f"Search successful with {engine_name}: {len(results)} results")
                        return formatted_result
                    
                except Exception as e:
                    logger.warning(f"Search engine {engine_name} failed: {str(e)}")
                    last_error = e
                    self._stats['engine_failures'] += 1
                    continue
            
            # All engines failed
            error_msg = f"All search engines failed for query '{query}'. Last error: {last_error}"
            logger.error(error_msg)
            return f"Search failed: Unable to retrieve results for '{query}'. Please try a different query."
            
        except Exception as e:
            logger.error(f"Critical error in web search: {str(e)}")
            return f"Search error: {str(e)}"
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[Dict]:
        """Search using DuckDuckGo API (primary method)"""
        
        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1'
        }
        
        async with httpx.AsyncClient(timeout=self._client_timeout) as client:
            response = await client.get(url, params=params)
            
            if response.status_code != 200:
                raise Exception(f"DuckDuckGo API returned {response.status_code}")
            
            data = response.json()
            results = []
            
            # Parse Abstract (main result)
            if data.get('Abstract'):
                results.append({
                    'title': data.get('Heading', query),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('Abstract', '')[:300]
                })
            
            # Parse Related Topics
            for topic in data.get('RelatedTopics', [])[:max_results-1]:
                if isinstance(topic, dict) and topic.get('Text'):
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', '')[:300]
                    })
            
            return results[:max_results]
    
    async def _search_duckduckgo_html(self, query: str, max_results: int) -> List[Dict]:
        """Search using DuckDuckGo HTML scraping (fallback)"""
        
        search_url = f"https://html.duckduckgo.com/html/"
        params = {'q': query}
        
        headers = {
            'User-Agent': self._user_agents[0],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        async with httpx.AsyncClient(timeout=self._client_timeout) as client:
            response = await client.post(search_url, data=params, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"DuckDuckGo HTML returned {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse search results with robust selectors
            result_containers = soup.find_all(['div'], class_=['result', 'web-result', 'result__body'])
            
            for container in result_containers[:max_results]:
                try:
                    # Find title and URL
                    title_link = container.find('a', href=True)
                    if not title_link:
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    # Find snippet
                    snippet_elem = container.find(['div', 'span'], class_=['result__snippet', 'snippet'])
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and url:
                        results.append({
                            'title': title[:200],
                            'url': url,
                            'snippet': snippet[:300]
                        })
                        
                except Exception as e:
                    logger.debug(f"Error parsing result container: {e}")
                    continue
            
            return results
    
    async def _search_searx(self, query: str, max_results: int) -> List[Dict]:
        """Search using SearX instances"""
        
        searx_instances = [
            'https://searx.be',
            'https://search.sapti.me',
            'https://searx.tiekoetter.com'
        ]
        
        for instance in searx_instances:
            try:
                url = f"{instance}/search"
                params = {
                    'q': query,
                    'format': 'json',
                    'engines': 'google,bing,duckduckgo'
                }
                
                headers = {'User-Agent': self._user_agents[1]}
                
                async with httpx.AsyncClient(timeout=self._client_timeout) as client:
                    response = await client.get(url, params=params, headers=headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        results = []
                        
                        for result in data.get('results', [])[:max_results]:
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('url', ''),
                                'snippet': result.get('content', '')[:300]
                            })
                        
                        if results:
                            return results
                            
            except Exception as e:
                logger.debug(f"SearX instance {instance} failed: {e}")
                continue
        
        raise Exception("All SearX instances failed")
    
    async def _search_bing(self, query: str, max_results: int) -> List[Dict]:
        """Search using Bing (last resort)"""
        
        url = "https://www.bing.com/search"
        params = {'q': query, 'count': max_results}
        headers = {
            'User-Agent': self._user_agents[2],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        async with httpx.AsyncClient(timeout=self._client_timeout) as client:
            response = await client.get(url, params=params, headers=headers)
            
            if response.status_code != 200:
                raise Exception(f"Bing returned {response.status_code}")
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Parse Bing results
            for result in soup.find_all('li', class_='b_algo')[:max_results]:
                try:
                    title_elem = result.find('h2')
                    title_link = title_elem.find('a') if title_elem else None
                    
                    if title_link:
                        title = title_link.get_text(strip=True)
                        url = title_link.get('href', '')
                        
                        snippet_elem = result.find('p') or result.find('div', class_='b_caption')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                        
                        results.append({
                            'title': title[:200],
                            'url': url,
                            'snippet': snippet[:300]
                        })
                        
                except Exception as e:
                    logger.debug(f"Error parsing Bing result: {e}")
                    continue
            
            return results
    
    def _format_results(self, query: str, results: List[Dict], engine: str) -> str:
        """Format search results for agent consumption"""
        
        if not results:
            return f"No results found for '{query}'"
        
        lines = [f"Found {len(results)} results for '{query}' (via {engine}):"]
        lines.append("")
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', 'No URL')
            snippet = result.get('snippet', 'No description')
            
            lines.append(f"**{i}. {title}**")
            lines.append(f"URL: {url}")
            lines.append(f"Description: {snippet}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _get_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key for query"""
        key_string = f"{query}:{max_results}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_from_cache(self, key: str) -> Optional[str]:
        """Get result from cache if not expired"""
        if key in self._cache:
            result, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            else:
                del self._cache[key]
        return None
    
    def _add_to_cache(self, key: str, result: str):
        """Add result to cache with cleanup"""
        self._cache[key] = (result, time.time())
        
        # Clean old entries if cache is too large
        if len(self._cache) > 100:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
    
    async def _enforce_rate_limit(self):
        """Enforce rate limiting between requests"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check of the web search tool"""
        
        try:
            # Test search with simple query
            test_result = await self._arun("python programming tutorial", max_results=1)
            
            success = test_result and "error" not in test_result.lower()
            
            return {
                'healthy': success,
                'last_test': time.time(),
                'test_query': 'python programming tutorial',
                'test_successful': success,
                'statistics': self._stats.copy(),
                'cache_size': len(self._cache)
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'statistics': self._stats.copy(),
                'cache_size': len(self._cache)
            }


# Factory function for easy instantiation
def create_web_search_tool() -> WebSearchTool:
    """Create and return a configured WebSearchTool instance"""
    return WebSearchTool()


# Convenience function for direct search
async def web_search(query: str, max_results: int = 5) -> str:
    """Direct web search function"""
    tool = create_web_search_tool()
    return await tool._arun(query, max_results=max_results)


if __name__ == "__main__":
    # Test function
    async def test_web_search_tool():
        print("🧪 Testing LangChain Compatible WebSearchTool...")
        tool = create_web_search_tool()
        
        result = await tool._arun("Python programming", max_results=2)
        print(f"Result: {result[:200]}...")
    
    asyncio.run(test_web_search_tool())