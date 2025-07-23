"""
Search Engines Implementations for Web Search Tool
Adaptadores para diferentes motores de búsqueda
"""

import asyncio
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from urllib.parse import urlencode, urlparse, parse_qs
import os

from bs4 import BeautifulSoup
import httpx

from .config import SearchEngine, SearchEngineConfig, get_engine_config
from .web_utils import WebRequestManager, WebResponse, URLNormalizer
from .user_agents import get_headers_with_random_user_agent

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Resultado individual de búsqueda"""
    title: str
    url: str
    snippet: str
    domain: str
    rank: int
    source_engine: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        
        # Normalizar URL y extraer dominio
        url_info = URLNormalizer.get_url_info(self.url)
        self.domain = url_info.domain
        self.url = url_info.normalized_url

@dataclass
class SearchResponse:
    """Respuesta completa de búsqueda"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    engine: str
    success: bool
    error_message: Optional[str] = None
    next_page_token: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseSearchEngine(ABC):
    """Clase base abstracta para motores de búsqueda"""
    
    def __init__(self, config: SearchEngineConfig):
        self.config = config
        self.name = config.name
        self.engine_type = None  # Se define en subclases
        
    @abstractmethod
    async def search(self, 
                    query: str, 
                    max_results: int = 10,
                    **kwargs) -> SearchResponse:
        """Realizar búsqueda"""
        pass
    
    def _clean_text(self, text: str) -> str:
        """Limpiar texto de resultados"""
        if not text:
            return ""
        
        # Remover HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalizar espacios
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Decodificar entidades HTML comunes
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&#39;', "'")
        
        return text
    
    def _validate_url(self, url: str) -> bool:
        """Validar que la URL sea válida"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc and parsed.scheme in ['http', 'https'])
        except Exception:
            return False
    
    async def _make_request(self, 
                           url: str, 
                           headers: Dict[str, str] = None,
                           params: Dict[str, Any] = None) -> WebResponse:
        """Hacer request HTTP con configuración del engine"""
        
        # Usar headers del engine o generar aleatorios
        if not headers:
            headers = get_headers_with_random_user_agent()
        
        # Agregar headers específicos del engine
        headers.update(self.config.headers)
        
        # Construir URL con parámetros
        if params:
            separator = '&' if '?' in url else '?'
            url = f"{url}{separator}{urlencode(params)}"
        
        async with WebRequestManager(
            timeout=self.config.timeout,
            max_retries=3
        ) as request_manager:
            return await request_manager.fetch_url(url, headers)

class DuckDuckGoSearchEngine(BaseSearchEngine):
    """Motor de búsqueda DuckDuckGo"""
    
    def __init__(self):
        super().__init__(get_engine_config(SearchEngine.DUCKDUCKGO))
        self.engine_type = SearchEngine.DUCKDUCKGO
    
    async def search(self, 
                    query: str, 
                    max_results: int = 10,
                    **kwargs) -> SearchResponse:
        """Realizar búsqueda en DuckDuckGo"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Searching DuckDuckGo for: {query}")
            
            # Preparar parámetros de búsqueda
            search_params = self.config.params.copy()
            search_params['q'] = query
            
            # Construir URL de búsqueda
            search_url = f"{self.config.base_url}{self.config.search_endpoint}"
            
            # Hacer request
            response = await self._make_request(search_url, params=search_params)
            
            if not response.success:
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=asyncio.get_event_loop().time() - start_time,
                    engine=self.name,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.error}"
                )
            
            # Parsear resultados
            results = await self._parse_duckduckgo_results(response.content, query, max_results)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                engine=self.name,
                success=True,
                metadata={
                    'final_url': response.final_url,
                    'response_time': response.response_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error searching DuckDuckGo: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                engine=self.name,
                success=False,
                error_message=str(e)
            )
    
    async def _parse_duckduckgo_results(self, 
                                       html: str, 
                                       query: str, 
                                       max_results: int) -> List[SearchResult]:
        """Parsear resultados HTML de DuckDuckGo"""
        
        results = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Buscar divs de resultados
            result_divs = soup.find_all('div', class_=re.compile(r'result'))
            
            if not result_divs:
                # Fallback: buscar por otros selectores comunes
                result_divs = soup.find_all('div', {'data-testid': 'result'})
            
            for i, div in enumerate(result_divs[:max_results]):
                try:
                    # Extraer título
                    title_elem = div.find('h2') or div.find('h3') or div.find('a')
                    title = self._clean_text(title_elem.get_text()) if title_elem else ""
                    
                    # Extraer URL
                    link_elem = div.find('a', href=True)
                    url = link_elem['href'] if link_elem else ""
                    
                    # DuckDuckGo puede usar URLs de redirect
                    if url.startswith('/l/?kh='):
                        # Extraer URL real del parámetro uddg
                        url_match = re.search(r'uddg=([^&]+)', url)
                        if url_match:
                            import urllib.parse
                            url = urllib.parse.unquote(url_match.group(1))
                    
                    # Extraer snippet
                    snippet_elem = div.find('span', class_=re.compile(r'result__snippet'))
                    if not snippet_elem:
                        # Fallback: buscar cualquier span o div con texto
                        snippet_elem = div.find('span') or div.find('div')
                    
                    snippet = self._clean_text(snippet_elem.get_text()) if snippet_elem else ""
                    
                    # Validar resultado
                    if title and url and self._validate_url(url):
                        result = SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            domain="",  # Se calcula en __post_init__
                            rank=i + 1,
                            source_engine=self.name,
                            metadata={
                                'query': query,
                                'extracted_from': 'html_parsing'
                            }
                        )
                        results.append(result)
                
                except Exception as e:
                    logger.warning(f"Error parsing DuckDuckGo result {i}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error parsing DuckDuckGo HTML: {e}")
        
        return results

class GoogleCustomSearchEngine(BaseSearchEngine):
    """Motor de búsqueda Google Custom Search API"""
    
    def __init__(self):
        super().__init__(get_engine_config(SearchEngine.GOOGLE_CUSTOM))
        self.engine_type = SearchEngine.GOOGLE_CUSTOM
        self.api_key = os.getenv('GOOGLE_SEARCH_API_KEY')
        self.search_engine_id = os.getenv('GOOGLE_SEARCH_ENGINE_ID')
    
    async def search(self, 
                    query: str, 
                    max_results: int = 10,
                    **kwargs) -> SearchResponse:
        """Realizar búsqueda usando Google Custom Search API"""
        
        start_time = asyncio.get_event_loop().time()
        
        if not self.api_key or not self.search_engine_id:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0,
                engine=self.name,
                success=False,
                error_message="Google API key or Search Engine ID not configured"
            )
        
        try:
            logger.info(f"Searching Google Custom Search for: {query}")
            
            # Preparar parámetros
            search_params = self.config.params.copy()
            search_params.update({
                'q': query,
                'key': self.api_key,
                'cx': self.search_engine_id,
                'num': min(max_results, 10)  # Google API limita a 10 por request
            })
            
            # Construir URL
            search_url = f"{self.config.base_url}{self.config.search_endpoint}"
            
            # Hacer request
            response = await self._make_request(search_url, params=search_params)
            
            if not response.success:
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=asyncio.get_event_loop().time() - start_time,
                    engine=self.name,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.error}"
                )
            
            # Parsear JSON response
            results = await self._parse_google_json_results(response.content, query)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                engine=self.name,
                success=True,
                metadata={
                    'api_response_time': response.response_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error searching Google Custom Search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                engine=self.name,
                success=False,
                error_message=str(e)
            )
    
    async def _parse_google_json_results(self, 
                                        json_content: str, 
                                        query: str) -> List[SearchResult]:
        """Parsear respuesta JSON de Google Custom Search"""
        
        results = []
        
        try:
            data = json.loads(json_content)
            
            if 'items' not in data:
                logger.warning("No 'items' field in Google API response")
                return results
            
            for i, item in enumerate(data['items']):
                try:
                    title = self._clean_text(item.get('title', ''))
                    url = item.get('link', '')
                    snippet = self._clean_text(item.get('snippet', ''))
                    
                    if title and url and self._validate_url(url):
                        result = SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            domain="",  # Se calcula en __post_init__
                            rank=i + 1,
                            source_engine=self.name,
                            metadata={
                                'query': query,
                                'google_cache_id': item.get('cacheId'),
                                'kind': item.get('kind'),
                                'extracted_from': 'google_api'
                            }
                        )
                        results.append(result)
                
                except Exception as e:
                    logger.warning(f"Error parsing Google result {i}: {e}")
                    continue
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Google JSON response: {e}")
        except Exception as e:
            logger.error(f"Error processing Google results: {e}")
        
        return results

class SearXSearchEngine(BaseSearchEngine):
    """Motor de búsqueda SearX (metabuscador)"""
    
    def __init__(self):
        super().__init__(get_engine_config(SearchEngine.SEARX))
        self.engine_type = SearchEngine.SEARX
    
    async def search(self, 
                    query: str, 
                    max_results: int = 10,
                    **kwargs) -> SearchResponse:
        """Realizar búsqueda en SearX"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            logger.info(f"Searching SearX for: {query}")
            
            # Preparar parámetros
            search_params = self.config.params.copy()
            search_params['q'] = query
            
            # Construir URL
            search_url = f"{self.config.base_url}{self.config.search_endpoint}"
            
            # Hacer request
            response = await self._make_request(search_url, params=search_params)
            
            if not response.success:
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=asyncio.get_event_loop().time() - start_time,
                    engine=self.name,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.error}"
                )
            
            # Parsear resultados JSON
            results = await self._parse_searx_json_results(response.content, query, max_results)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                engine=self.name,
                success=True,
                metadata={
                    'searx_instance': self.config.base_url,
                    'response_time': response.response_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error searching SearX: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                engine=self.name,
                success=False,
                error_message=str(e)
            )
    
    async def _parse_searx_json_results(self, 
                                       json_content: str, 
                                       query: str, 
                                       max_results: int) -> List[SearchResult]:
        """Parsear respuesta JSON de SearX"""
        
        results = []
        
        try:
            data = json.loads(json_content)
            
            if 'results' not in data:
                logger.warning("No 'results' field in SearX response")
                return results
            
            for i, item in enumerate(data['results'][:max_results]):
                try:
                    title = self._clean_text(item.get('title', ''))
                    url = item.get('url', '')
                    content = self._clean_text(item.get('content', ''))
                    
                    if title and url and self._validate_url(url):
                        result = SearchResult(
                            title=title,
                            url=url,
                            snippet=content,
                            domain="",  # Se calcula en __post_init__
                            rank=i + 1,
                            source_engine=self.name,
                            metadata={
                                'query': query,
                                'engines': item.get('engines', []),
                                'score': item.get('score'),
                                'extracted_from': 'searx_api'
                            }
                        )
                        results.append(result)
                
                except Exception as e:
                    logger.warning(f"Error parsing SearX result {i}: {e}")
                    continue
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing SearX JSON response: {e}")
        except Exception as e:
            logger.error(f"Error processing SearX results: {e}")
        
        return results

class BingSearchEngine(BaseSearchEngine):
    """Motor de búsqueda Bing Search API"""
    
    def __init__(self):
        super().__init__(get_engine_config(SearchEngine.BING))
        self.engine_type = SearchEngine.BING
        self.api_key = os.getenv('BING_SEARCH_API_KEY')
    
    async def search(self, 
                    query: str, 
                    max_results: int = 10,
                    **kwargs) -> SearchResponse:
        """Realizar búsqueda usando Bing Search API"""
        
        start_time = asyncio.get_event_loop().time()
        
        if not self.api_key:
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0,
                engine=self.name,
                success=False,
                error_message="Bing API key not configured"
            )
        
        try:
            logger.info(f"Searching Bing for: {query}")
            
            # Preparar parámetros y headers
            search_params = self.config.params.copy()
            search_params.update({
                'q': query,
                'count': min(max_results, 50)  # Bing permite hasta 50
            })
            
            headers = self.config.headers.copy()
            headers['Ocp-Apim-Subscription-Key'] = self.api_key
            
            # Construir URL
            search_url = f"{self.config.base_url}{self.config.search_endpoint}"
            
            # Hacer request
            response = await self._make_request(search_url, headers, search_params)
            
            if not response.success:
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=asyncio.get_event_loop().time() - start_time,
                    engine=self.name,
                    success=False,
                    error_message=f"HTTP {response.status_code}: {response.error}"
                )
            
            # Parsear resultados JSON
            results = await self._parse_bing_json_results(response.content, query)
            
            search_time = asyncio.get_event_loop().time() - start_time
            
            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                engine=self.name,
                success=True,
                metadata={
                    'api_response_time': response.response_time
                }
            )
            
        except Exception as e:
            logger.error(f"Error searching Bing: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=asyncio.get_event_loop().time() - start_time,
                engine=self.name,
                success=False,
                error_message=str(e)
            )
    
    async def _parse_bing_json_results(self, 
                                      json_content: str, 
                                      query: str) -> List[SearchResult]:
        """Parsear respuesta JSON de Bing Search API"""
        
        results = []
        
        try:
            data = json.loads(json_content)
            
            web_pages = data.get('webPages', {})
            if 'value' not in web_pages:
                logger.warning("No 'webPages.value' field in Bing response")
                return results
            
            for i, item in enumerate(web_pages['value']):
                try:
                    title = self._clean_text(item.get('name', ''))
                    url = item.get('url', '')
                    snippet = self._clean_text(item.get('snippet', ''))
                    
                    if title and url and self._validate_url(url):
                        result = SearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            domain="",  # Se calcula en __post_init__
                            rank=i + 1,
                            source_engine=self.name,
                            metadata={
                                'query': query,
                                'last_crawled': item.get('dateLastCrawled'),
                                'display_url': item.get('displayUrl'),
                                'extracted_from': 'bing_api'
                            }
                        )
                        results.append(result)
                
                except Exception as e:
                    logger.warning(f"Error parsing Bing result {i}: {e}")
                    continue
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing Bing JSON response: {e}")
        except Exception as e:
            logger.error(f"Error processing Bing results: {e}")
        
        return results

class SearchEngineManager:
    """Gestor de motores de búsqueda con fallbacks"""
    
    def __init__(self):
        self.engines = {
            SearchEngine.DUCKDUCKGO: DuckDuckGoSearchEngine(),
            SearchEngine.GOOGLE_CUSTOM: GoogleCustomSearchEngine(),
            SearchEngine.SEARX: SearXSearchEngine(),
            SearchEngine.BING: BingSearchEngine(),
        }
        
        self.default_engine = SearchEngine.DUCKDUCKGO
        self.fallback_engines = [SearchEngine.SEARX, SearchEngine.DUCKDUCKGO]
    
    async def search(self, 
                    query: str,
                    engine: SearchEngine = None,
                    max_results: int = 10,
                    use_fallback: bool = True,
                    **kwargs) -> SearchResponse:
        """Realizar búsqueda con fallback automático"""
        
        # Usar engine especificado o el por defecto
        target_engine = engine or self.default_engine
        
        try:
            # Intentar búsqueda principal
            search_engine = self.engines[target_engine]
            result = await search_engine.search(query, max_results, **kwargs)
            
            if result.success and result.results:
                return result
            
            logger.warning(f"Search failed on {target_engine.value}: {result.error_message}")
            
            # Intentar fallbacks si está habilitado
            if use_fallback:
                for fallback_engine in self.fallback_engines:
                    if fallback_engine == target_engine:
                        continue  # Skip el engine que ya falló
                    
                    try:
                        logger.info(f"Trying fallback engine: {fallback_engine.value}")
                        fallback_search = self.engines[fallback_engine]
                        fallback_result = await fallback_search.search(query, max_results, **kwargs)
                        
                        if fallback_result.success and fallback_result.results:
                            # Marcar que vino de fallback
                            fallback_result.metadata = fallback_result.metadata or {}
                            fallback_result.metadata['used_fallback'] = True
                            fallback_result.metadata['original_engine'] = target_engine.value
                            return fallback_result
                    
                    except Exception as e:
                        logger.warning(f"Fallback engine {fallback_engine.value} failed: {e}")
                        continue
            
            # Si todo falló, retornar el resultado original con el error
            return result
            
        except Exception as e:
            logger.error(f"Error in search manager: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0,
                engine=target_engine.value,
                success=False,
                error_message=str(e)
            )
    
    def get_available_engines(self) -> List[SearchEngine]:
        """Obtener lista de engines disponibles"""
        available = []
        
        for engine_type, engine in self.engines.items():
            config = engine.config
            
            # Verificar si el engine está habilitado y configurado
            if config.enabled:
                if config.requires_api_key:
                    # Verificar que tenga API key
                    if engine_type == SearchEngine.GOOGLE_CUSTOM:
                        if hasattr(engine, 'api_key') and engine.api_key:
                            available.append(engine_type)
                    elif engine_type == SearchEngine.BING:
                        if hasattr(engine, 'api_key') and engine.api_key:
                            available.append(engine_type)
                else:
                    available.append(engine_type)
        
        return available
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """Health check de todos los engines"""
        results = {}
        
        for engine_type, engine in self.engines.items():
            try:
                # Test search simple
                test_result = await engine.search("test", max_results=1)
                
                results[engine_type.value] = {
                    'available': True,
                    'healthy': test_result.success,
                    'error': test_result.error_message if not test_result.success else None,
                    'search_time': test_result.search_time,
                    'requires_api_key': engine.config.requires_api_key,
                    'enabled': engine.config.enabled
                }
            
            except Exception as e:
                results[engine_type.value] = {
                    'available': False,
                    'healthy': False,
                    'error': str(e),
                    'search_time': 0,
                    'requires_api_key': engine.config.requires_api_key,
                    'enabled': engine.config.enabled
                }
        
        return results

# Instancia global del manager
search_engine_manager = SearchEngineManager()