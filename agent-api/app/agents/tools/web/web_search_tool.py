"""
WebSearchTool - Versión Standalone sin dependencias problemáticas
Completamente funcional e independiente
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
    Web Search Tool completamente funcional y standalone
    
    Características:
    - Multi-engine support (DuckDuckGo como principal, fallbacks)
    - Parsing robusto y defensivo
    - Rate limiting inteligente
    - Cache con TTL
    - Error handling completo
    - Compatible con LangChain BaseTool
    - Sin dependencias problemáticas
    """
    
    # LangChain required attributes
    name = "web_search"
    description = """Search the web for current information. 
    Input should be a search query string.
    Returns search results with titles, URLs, and snippets."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Tool metadata
        self.category = ToolCategory.WEB_SEARCH
        self.version = "2.1.0"
        self.requires_internet = True
        self.requires_gpu = False
        self.max_execution_time = 60
        self.rate_limit = 30
        
        # Cache system
        self._cache = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_request_time = 0
        self._min_request_interval = 2.0  # 2 seconds between requests
        
        # Statistics
        self.stats = {
            'total_searches': 0,
            'successful_searches': 0,
            'cache_hits': 0,
            'engine_failures': 0
        }
        
        # HTTP client configuration
        self.client_timeout = 15
        self.max_retries = 3
        
        logger.info("WebSearchTool initialized successfully")
    
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
            Formatted search results string
        """
        
        start_time = time.time()
        self.stats['total_searches'] += 1
        
        try:
            logger.info(f"Starting web search for: {query}")
            
            # Parse and validate input
            max_results = kwargs.get('max_results', 5)
            if max_results > 10:
                max_results = 10  # Reasonable limit
            
            # Check cache first
            cache_key = self._generate_cache_key(query, max_results)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.stats['cache_hits'] += 1
                logger.info("Returning cached search result")
                return cached_result
            
            # Rate limiting
            await self._enforce_rate_limit()
            
            # Perform search with fallback engines
            search_result = await self._perform_multi_engine_search(query, max_results)
            
            if not search_result or not search_result.get('results'):
                error_msg = "No search results found from any search engine"
                logger.warning(error_msg)
                return self._format_error_response(error_msg)
            
            # Format results for output
            formatted_result = self._format_search_results(search_result)
            
            # Cache the result
            self._cache_result(cache_key, formatted_result)
            
            # Update stats
            self.stats['successful_searches'] += 1
            
            search_time = time.time() - start_time
            logger.info(f"Web search completed in {search_time:.2f}s, found {len(search_result['results'])} results")
            
            return formatted_result
            
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.stats['engine_failures'] += 1
            return self._format_error_response(error_msg)
    
    async def _perform_multi_engine_search(self, query: str, max_results: int) -> Optional[Dict]:
        """
        Try multiple search engines with fallback
        """
        
        # Search engines in priority order
        engines = [
            ('duckduckgo', self._search_duckduckgo),
            ('duckduckgo_lite', self._search_duckduckgo_lite),
            ('searx_be', self._search_searx_be),
        ]
        
        for engine_name, search_func in engines:
            try:
                logger.info(f"Trying search engine: {engine_name}")
                result = await search_func(query, max_results)
                
                if result and result.get('results') and len(result['results']) > 0:
                    logger.info(f"Search successful with {engine_name}, found {len(result['results'])} results")
                    result['engine_used'] = engine_name
                    return result
                else:
                    logger.warning(f"No results from {engine_name}")
                    
            except Exception as e:
                logger.warning(f"Search engine {engine_name} failed: {e}")
                continue
        
        logger.error("All search engines failed")
        return None
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> Optional[Dict]:
        """
        Search DuckDuckGo HTML interface (primary method)
        """
        
        url = "https://html.duckduckgo.com/html/"
        params = {
            'q': query,
            'b': '',
            'kl': 'us-en',
            'df': ''
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                return self._parse_duckduckgo_html(response.text, query, max_results)
                
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")
            return None
    
    async def _search_duckduckgo_lite(self, query: str, max_results: int) -> Optional[Dict]:
        """
        Search DuckDuckGo Lite (backup method with different parsing)
        """
        
        url = "https://lite.duckduckgo.com/lite/"
        params = {
            'q': query,
            'kl': 'us-en'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SmartDocBot/1.0)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                return self._parse_duckduckgo_lite_html(response.text, query, max_results)
                
        except Exception as e:
            logger.warning(f"DuckDuckGo Lite search failed: {e}")
            return None
    
    async def _search_searx_be(self, query: str, max_results: int) -> Optional[Dict]:
        """
        Search using searx.be instance
        """
        
        url = "https://searx.be/search"
        params = {
            'q': query,
            'category_general': 'on',
            'language': 'en-US',
            'time_range': '',
            'safesearch': '1',
            'format': 'json'
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; SmartDocBot/1.0)',
            'Accept': 'application/json, text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.client_timeout) as client:
                response = await client.get(url, params=params, headers=headers)
                response.raise_for_status()
                
                if 'application/json' in response.headers.get('content-type', ''):
                    return self._parse_searx_json(response.text, query, max_results)
                else:
                    # Fallback to HTML parsing
                    return self._parse_searx_html(response.text, query, max_results)
                    
        except Exception as e:
            logger.warning(f"SearX.be search failed: {e}")
            return None
    
    def _parse_duckduckgo_html(self, html: str, query: str, max_results: int) -> Optional[Dict]:
        """
        Parse DuckDuckGo HTML results with improved selectors
        """
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # Updated selectors for current DuckDuckGo structure
            result_selectors = [
                'div[class*="web-result"]',
                'div[class*="result__"]',
                'div[data-testid="result"]',
                'div.result',
                'article[data-testid="result"]',
                '.results_links'
            ]
            
            result_elements = []
            for selector in result_selectors:
                try:
                    elements = soup.select(selector)
                    if elements and len(elements) > 0:
                        result_elements = elements
                        logger.debug(f"Found {len(elements)} results using selector: {selector}")
                        break
                except Exception as e:
                    logger.debug(f"Selector {selector} failed: {e}")
                    continue
            
            if not result_elements:
                # Super fallback: find any div or article with links
                result_elements = soup.find_all(['div', 'article'])
                result_elements = [elem for elem in result_elements if elem.find('a', href=True)]
                logger.debug(f"Using super fallback, found {len(result_elements)} potential results")
            
            for i, element in enumerate(result_elements[:max_results * 2]):  # Parse more, filter later
                try:
                    result = self._extract_result_from_element(element, len(results) + 1, 'DuckDuckGo')
                    if result and len(results) < max_results:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Failed to parse result element {i}: {e}")
                    continue
            
            logger.info(f"Parsed {len(results)} results from DuckDuckGo HTML")
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'source': 'DuckDuckGo',
                'success': len(results) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo HTML: {e}")
            return None
    
    def _parse_duckduckgo_lite_html(self, html: str, query: str, max_results: int) -> Optional[Dict]:
        """
        Parse DuckDuckGo Lite HTML results
        """
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # DuckDuckGo Lite structure
            table_rows = soup.find_all('tr')
            
            for i, tr in enumerate(table_rows):
                try:
                    # Find the main link
                    links = tr.find_all('a', href=True)
                    main_link = None
                    
                    for link in links:
                        href = link.get('href', '')
                        if href and not href.startswith('/') and 'http' in href:
                            main_link = link
                            break
                    
                    if not main_link:
                        continue
                    
                    url = main_link['href']
                    title = self._clean_text(main_link.get_text())
                    
                    # Find snippet text in the row
                    all_text = self._clean_text(tr.get_text())
                    snippet = all_text.replace(title, '').strip()[:300]
                    
                    if self._is_valid_result(title, url) and len(results) < max_results:
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'rank': len(results) + 1,
                            'source': 'DuckDuckGo Lite'
                        })
                        
                except Exception as e:
                    logger.debug(f"Failed to parse DuckDuckGo Lite result {i}: {e}")
                    continue
            
            logger.info(f"Parsed {len(results)} results from DuckDuckGo Lite")
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'source': 'DuckDuckGo Lite',
                'success': len(results) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to parse DuckDuckGo Lite HTML: {e}")
            return None
    
    def _parse_searx_json(self, json_text: str, query: str, max_results: int) -> Optional[Dict]:
        """
        Parse SearX JSON results
        """
        
        try:
            data = json.loads(json_text)
            results = []
            
            searx_results = data.get('results', [])
            
            for i, item in enumerate(searx_results[:max_results]):
                try:
                    title = item.get('title', '')
                    url = item.get('url', '')
                    content = item.get('content', '')
                    
                    if self._is_valid_result(title, url):
                        results.append({
                            'title': self._clean_text(title),
                            'url': url,
                            'snippet': self._clean_text(content)[:300],
                            'rank': len(results) + 1,
                            'source': 'SearX'
                        })
                        
                except Exception as e:
                    logger.debug(f"Failed to parse SearX result {i}: {e}")
                    continue
            
            logger.info(f"Parsed {len(results)} results from SearX JSON")
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'source': 'SearX',
                'success': len(results) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to parse SearX JSON: {e}")
            return None
    
    def _parse_searx_html(self, html: str, query: str, max_results: int) -> Optional[Dict]:
        """
        Parse SearX HTML results
        """
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            
            # SearX result selectors
            result_elements = soup.select('div.result, article.result, .result')
            
            for i, element in enumerate(result_elements[:max_results]):
                try:
                    result = self._extract_result_from_element(element, i + 1, 'SearX')
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.debug(f"Failed to parse SearX result {i}: {e}")
                    continue
            
            logger.info(f"Parsed {len(results)} results from SearX HTML")
            
            return {
                'query': query,
                'results': results,
                'total_results': len(results),
                'source': 'SearX',
                'success': len(results) > 0
            }
            
        except Exception as e:
            logger.error(f"Failed to parse SearX HTML: {e}")
            return None
    
    def _extract_result_from_element(self, element, rank: int, source: str) -> Optional[Dict]:
        """
        Extract search result from HTML element (improved parser)
        """
        
        try:
            # Find title and URL with improved selectors
            title_element = None
            url = None
            
            # Enhanced selectors for different engines
            link_selectors = [
                'h2 a[href]', 'h3 a[href]', 'h4 a[href]',
                'a[href].result-title', 'a[href].result__title',
                '.result__title a[href]', '.result-header a[href]',
                'a[href][class*="title"]', 'a[href][class*="link"]',
                'a[href]:first-of-type'
            ]
            
            for selector in link_selectors:
                try:
                    link = element.select_one(selector)
                    if link and link.get('href'):
                        title_element = link
                        url = link['href']
                        break
                except:
                    continue
            
            if not title_element or not url:
                # Super fallback: find any meaningful link
                links = element.find_all('a', href=True)
                for link in links:
                    href = link.get('href', '')
                    text = link.get_text().strip()
                    if href and text and len(text) > 5 and 'http' in href:
                        title_element = link
                        url = href
                        break
            
            if not title_element or not url:
                return None
            
            # Extract title
            title = self._clean_text(title_element.get_text())
            
            # Extract snippet with improved selectors
            snippet = ""
            snippet_selectors = [
                '.result-content', '.result__snippet', '.result-snippet',
                '.content', '.description', '.snippet',
                'p', '.text', '[class*="desc"]'
            ]
            
            for selector in snippet_selectors:
                try:
                    snippet_elem = element.select_one(selector)
                    if snippet_elem:
                        snippet_text = self._clean_text(snippet_elem.get_text())
                        if snippet_text and len(snippet_text) > 10:
                            snippet = snippet_text
                            break
                except:
                    continue
            
            if not snippet:
                # Fallback: extract all text and clean
                all_text = self._clean_text(element.get_text())
                if title in all_text:
                    snippet = all_text.replace(title, '').strip()
                else:
                    snippet = all_text
            
            # Clean up URL
            url = self._clean_url(url)
            
            if self._is_valid_result(title, url):
                return {
                    'title': title[:200],  # Limit title length
                    'url': url,
                    'snippet': snippet[:300],  # Limit snippet length
                    'rank': rank,
                    'source': source
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to extract result from element: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove common unwanted characters but keep essential punctuation
        text = re.sub(r'[^\w\s\-\.,!?:;()[\]{}"\'/&@#%]', '', text)
        
        return text[:1000]  # Reasonable limit
    
    def _clean_url(self, url: str) -> str:
        """Clean and validate URL"""
        if not url:
            return ""
        
        # Handle relative URLs
        if url.startswith('//'):
            url = 'https:' + url
        elif url.startswith('/'):
            return ""  # Skip relative paths without domain
        
        # Handle DuckDuckGo redirects
        if '/l/?uddg=' in url or '/l/?kh=' in url:
            try:
                from urllib.parse import unquote
                if 'uddg=' in url:
                    encoded_url = url.split('uddg=')[1].split('&')[0]
                    url = unquote(encoded_url)
                elif 'kh=' in url:
                    # Different DuckDuckGo redirect format
                    encoded_url = url.split('kh=')[1].split('&')[0]
                    url = unquote(encoded_url)
            except:
                pass
        
        return url[:500]  # Reasonable URL length limit
    
    def _is_valid_result(self, title: str, url: str) -> bool:
        """Validate if result is useful"""
        
        if not title or not url:
            return False
        
        if len(title.strip()) < 3:
            return False
        
        # Check for valid URL scheme
        if not (url.startswith('http://') or url.startswith('https://')):
            return False
        
        # Filter out common non-useful results
        spam_patterns = [
            r'^\d+\s*$',  # Just numbers
            r'^click here',  # Generic click here
            r'^more.*results',  # More results links
            r'^next.*page',  # Next page links
            r'^previous.*page',  # Previous page links
            r'^loading\.*',  # Loading messages
            r'^search.*',  # Search-related links
        ]
        
        title_lower = title.lower()
        for pattern in spam_patterns:
            if re.match(pattern, title_lower):
                return False
        
        # Filter spam domains
        spam_domains = ['javascript:', 'mailto:', 'tel:', 'ftp:']
        for spam in spam_domains:
            if url.lower().startswith(spam):
                return False
        
        return True
    
    def _format_search_results(self, search_result: Dict) -> str:
        """Format search results for agent consumption"""
        
        if not search_result or not search_result.get('results'):
            return "No search results found."
        
        results = search_result['results']
        query = search_result['query']
        source = search_result.get('source', 'Web')
        
        output = f"Search results for '{query}' (Source: {source}):\n\n"
        
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            url = result.get('url', '')
            snippet = result.get('snippet', 'No description available')
            
            output += f"{i}. **{title}**\n"
            output += f"   URL: {url}\n"
            output += f"   Summary: {snippet}\n\n"
        
        output += f"\nFound {len(results)} results. Use these results to answer the user's question."
        
        return output
    
    def _format_error_response(self, error_msg: str) -> str:
        """Format error response for agent"""
        return f"Web search error: {error_msg}. Please try rephrasing your query or ask me something I can answer with my existing knowledge."
    
    def _generate_cache_key(self, query: str, max_results: int) -> str:
        """Generate cache key for request"""
        cache_string = f"{query}:{max_results}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Get result from cache if not expired"""
        if cache_key in self._cache:
            cached_item = self._cache[cache_key]
            if time.time() - cached_item['timestamp'] < self._cache_ttl:
                return cached_item['result']
            else:
                # Remove expired item
                del self._cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: str):
        """Cache result with timestamp"""
        self._cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }
        
        # Basic cache cleanup (remove old items)
        if len(self._cache) > 100:  # Max 100 cached items
            oldest_key = min(self._cache.keys(), 
                           key=lambda k: self._cache[k]['timestamp'])
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
                'statistics': self.stats.copy(),
                'cache_size': len(self._cache)
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'statistics': self.stats.copy(),
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
        print("🧪 Testing Standalone WebSearchTool...")
        tool = create_web_search_tool()
        
        result = await tool._arun("Python programming", max_results=2)
        print(f"Result: {result[:200]}...")
    
    asyncio.run(test_web_search_tool())