"""
Web Search Tool - Herramienta principal de búsqueda web para SmartDoc Agent
Combina búsqueda, extracción de contenido y rate limiting
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Type
from pydantic import Field, PrivateAttr
from dataclasses import dataclass
import json
from langchain.tools import BaseTool

from ..base_tool import ToolCategory, tool_registration
from .config import SearchEngine, WebSearchConfig, get_query_config
from .search_engines import search_engine_manager, SearchResult, SearchResponse
from .content_extractor import global_content_extractor, ExtractedContent
from .rate_limiter import global_rate_limiter
from .web_utils import robots_checker, URLNormalizer, get_content_hash

# Type hints para Pydantic
from typing import Optional, Any

logger = logging.getLogger(__name__)

@dataclass
class WebSearchRequest:
    """Request específico para web search"""
    query: str
    max_results: int = 10
    search_engine: Optional[str] = None
    query_type: str = "general"  # general, academic, news, technical
    extract_content: bool = True
    content_max_length: int = 5000
    include_metadata: bool = True
    filter_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None

@dataclass
class WebSearchResult:
    """Resultado enriquecido de búsqueda web"""
    # Datos de búsqueda básicos
    title: str
    url: str
    snippet: str
    domain: str
    rank: int
    
    # Contenido extraído (opcional)
    extracted_content: Optional[ExtractedContent] = None
    
    # Metadatos
    content_hash: Optional[str] = None
    relevance_score: float = 0.0
    quality_score: float = 0.0
    language: str = "unknown"
    content_type: str = "unknown"
    word_count: int = 0
    reading_time: int = 0
    
    # Metadata adicional
    extraction_time: float = 0.0
    extraction_success: bool = False
    source_engine: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para serialización"""
        result = {
            'title': self.title,
            'url': self.url,
            'snippet': self.snippet,
            'domain': self.domain,
            'rank': self.rank,
            'relevance_score': self.relevance_score,
            'quality_score': self.quality_score,
            'language': self.language,
            'content_type': self.content_type,
            'word_count': self.word_count,
            'reading_time': self.reading_time,
            'source_engine': self.source_engine,
        }
        
        if self.extracted_content:
            result['extracted_content'] = {
                'main_content': self.extracted_content.main_content[:1000],  # Truncar
                'author': self.extracted_content.author,
                'publish_date': self.extracted_content.publish_date,
                'keywords': self.extracted_content.keywords,
                'extraction_success': self.extracted_content.extraction_success
            }
        
        return result


class WebSearchTool(BaseTool):
    """
    Herramienta de búsqueda web inteligente con extracción de contenido
    """
    
    # ATRIBUTOS PÚBLICOS REQUERIDOS POR LANGCHAIN
    name: str = "web_search"
    description: str = """Search the web for current information on any topic. 
                        Use this tool when you need up-to-date information that you don't have in your knowledge base.
                        Input should be a search query string or JSON with parameters like max_results, search_engine, etc.
                        Returns formatted search results with titles, URLs, snippets and extracted content."""
                            
    # CONFIGURACIÓN PYDANTIC - CRÍTICO PARA COMPATIBILIDAD
    class Config:
        """Configuración Pydantic para permitir atributos privados"""
        extra = "allow"  # Permitir atributos adicionales
        arbitrary_types_allowed = True  # Permitir tipos arbitrarios
        validate_assignment = False  # No validar en asignación
    
    def __init__(self, **kwargs):
        """Inicializar WebSearchTool con configuración de compatibilidad"""
        # Inicializar padre PRIMERO
        super().__init__(**kwargs)
        
        # INICIALIZAR ATRIBUTOS DESPUÉS (evita problemas Pydantic)
        # Cache y estadísticas
        object.__setattr__(self, '_results_cache', {})
        object.__setattr__(self, '_cache_ttl', 3600)  # 1 hora
        object.__setattr__(self, '_stats', {
            'total_searches': 0,
            'successful_searches': 0,
            'total_extractions': 0,
            'successful_extractions': 0,
            'cache_hits': 0,
            'robots_blocks': 0,
            'rate_limit_blocks': 0
        })
        
        # Inicializar componentes externos
        try:
            from .config import WebSearchConfig
            from .search_engines import search_engine_manager
            from .content_extractor import global_content_extractor
            from .rate_limiter import global_rate_limiter
            from .web_utils import robots_checker
            
            object.__setattr__(self, '_config', WebSearchConfig())
            object.__setattr__(self, '_search_manager', search_engine_manager)
            object.__setattr__(self, '_content_extractor', global_content_extractor)
            object.__setattr__(self, '_rate_limiter', global_rate_limiter)
            object.__setattr__(self, '_robots_checker', robots_checker)
            
        except ImportError as e:
            # Fallback si algunos módulos no están disponibles
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Some WebSearchTool components not available: {e}")
            
            # Crear mocks básicos para compatibilidad
            object.__setattr__(self, '_config', None)
            object.__setattr__(self, '_search_manager', None)
            object.__setattr__(self, '_content_extractor', None)
            object.__setattr__(self, '_rate_limiter', None)
            object.__setattr__(self, '_robots_checker', None)
    
    @property
    def config(self):
        """Acceso al config"""
        return getattr(self, '_config', None)
    
    @property
    def search_manager(self):
        """Acceso al search manager"""  
        return getattr(self, '_search_manager', None)
    
    @property
    def content_extractor(self):
        """Acceso al content extractor"""
        return getattr(self, '_content_extractor', None)
    
    @property
    def rate_limiter(self):
        """Acceso al rate limiter"""
        return getattr(self, '_rate_limiter', None)
    
    @property
    def robots_checker(self):
        """Acceso al robots checker"""
        return getattr(self, '_robots_checker', None)
    
    @property
    def stats(self):
        """Acceso a estadísticas"""
        return getattr(self, '_stats', {})
    
    async def _arun(self, query: str, **kwargs) -> str:
        """Implementación principal del tool de búsqueda web"""
        
        try:
            # Parsear argumentos adicionales
            search_request = self._parse_search_request(query, kwargs)
            
            logger.info(f"Starting web search for: {search_request.query}")
            
            # Verificar rate limiting
            can_proceed = await self.rate_limiter.acquire(
                identifier=f"web_search_{self.name}",
                engine="web_search"
            )
            
            if not can_proceed:
                self._stats['rate_limit_blocks'] += 1
                return self._format_error_response("Rate limit exceeded. Please try again later.")
            
            # Verificar cache
            cache_key = self._generate_cache_key(search_request)
            cached_result = self._get_cached_result(cache_key)
            
            if cached_result:
                self._stats['cache_hits'] += 1
                logger.info("Returning cached search results")
                return cached_result
            
            # Realizar búsqueda
            search_results = await self._perform_search(search_request)
            
            if not search_results:
                return self._format_error_response("No search results found")
            
            # Procesar y enriquecer resultados
            enriched_results = await self._enrich_search_results(
                search_results, 
                search_request
            )
            
            # Formatear respuesta final
            formatted_response = self._format_search_response(
                enriched_results, 
                search_request
            )
            
            # Guardar en cache
            self._cache_result(cache_key, formatted_response)
            
            # Actualizar estadísticas
            self._stats['total_searches'] += 1
            self._stats['successful_searches'] += 1
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error in web search tool: {e}")
            return self._format_error_response(f"Search failed: {str(e)}")
        
        
    def _run(self, query: str, **kwargs) -> str:
        """
        Versión síncrona del método _arun requerida por LangChain
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            str: Formatted search results
        """
        try:
            # Ejecutar la versión async de forma síncrona
            import asyncio
            
            # Verificar si hay un event loop corriendo
            try:
                loop = asyncio.get_running_loop()
                # Si hay loop corriendo, ejecutar en thread separado
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._arun(query, **kwargs))
                    return future.result(timeout=120)  # 2 minutos timeout
                    
            except RuntimeError:
                # No hay loop corriendo, crear uno nuevo
                return asyncio.run(self._arun(query, **kwargs))
                
        except asyncio.TimeoutError:
            logger.error(f"Web search timeout for query: {query}")
            return self._format_error_response("Search timeout - query took too long to complete")
            
        except Exception as e:
            logger.error(f"Error in synchronous web search: {e}")
            return self._format_error_response(f"Search failed: {str(e)}")    
        
        
    
    def _parse_search_request(self, query: str, kwargs: Dict[str, Any]) -> WebSearchRequest:
        """Parsear request de búsqueda desde argumentos del tool"""
        
        # Si query es JSON, parsearlo
        if query.strip().startswith('{'):
            try:
                params = json.loads(query)
                actual_query = params.get('query', query)
            except json.JSONDecodeError:
                params = {}
                actual_query = query
        else:
            params = {}
            actual_query = query.strip()
        
        # Combinar parámetros de JSON y kwargs
        combined_params = {**params, **kwargs}
        
        return WebSearchRequest(
            query=actual_query,
            max_results=combined_params.get('max_results', 10),
            search_engine=combined_params.get('search_engine'),
            query_type=combined_params.get('query_type', 'general'),
            extract_content=combined_params.get('extract_content', True),
            content_max_length=combined_params.get('content_max_length', 5000),
            include_metadata=combined_params.get('include_metadata', True),
            filter_domains=combined_params.get('filter_domains'),
            exclude_domains=combined_params.get('exclude_domains')
        )
    
    async def _perform_search(self, request: WebSearchRequest) -> List[SearchResult]:
        """Realizar búsqueda usando el motor especificado o por defecto"""
        
        # Determinar motor de búsqueda
        if request.search_engine:
            try:
                engine = SearchEngine(request.search_engine)
            except ValueError:
                logger.warning(f"Unknown search engine: {request.search_engine}, using default")
                engine = None
        else:
            engine = None
        
        # Obtener configuración de query según tipo
        query_config = get_query_config(request.query_type)
        
        # Modificar query según tipo
        enhanced_query = request.query
        if query_config.get('query_modifiers'):
            modifiers = ' '.join(query_config['query_modifiers'])
            enhanced_query = f"{request.query} {modifiers}"
        
        try:
            # Realizar búsqueda
            search_response = await self.search_manager.search(
                query=enhanced_query,
                engine=engine,
                max_results=request.max_results,
                use_fallback=True
            )
            
            if not search_response.success:
                logger.error(f"Search failed: {search_response.error_message}")
                return []
            
            # Filtrar resultados según dominios
            filtered_results = self._filter_results_by_domain(
                search_response.results,
                request.filter_domains,
                request.exclude_domains
            )
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error performing search: {e}")
            return []
    
    def _filter_results_by_domain(self, 
                                 results: List[SearchResult],
                                 include_domains: Optional[List[str]],
                                 exclude_domains: Optional[List[str]]) -> List[SearchResult]:
        """Filtrar resultados por dominio"""
        
        filtered = results
        
        # Filtrar dominios a incluir
        if include_domains:
            filtered = [
                r for r in filtered 
                if any(domain in r.domain for domain in include_domains)
            ]
        
        # Filtrar dominios a excluir
        if exclude_domains:
            filtered = [
                r for r in filtered 
                if not any(domain in r.domain for domain in exclude_domains)
            ]
        
        # Aplicar filtros de configuración global
        filtered = [
            r for r in filtered
            if not self.config.is_domain_blocked(r.domain)
        ]
        
        return filtered
    
    async def _enrich_search_results(self, 
                                   search_results: List[SearchResult],
                                   request: WebSearchRequest) -> List[WebSearchResult]:
        """Enriquecer resultados con extracción de contenido"""
        
        enriched_results = []
        extraction_tasks = []
        
        for search_result in search_results:
            # Crear resultado base
            web_result = WebSearchResult(
                title=search_result.title,
                url=search_result.url,
                snippet=search_result.snippet,
                domain=search_result.domain,
                rank=search_result.rank,
                source_engine=search_result.source_engine,
                relevance_score=self._calculate_relevance_score(search_result, request.query)
            )
            
            enriched_results.append(web_result)
            
            # Programar extracción de contenido si se solicita
            if request.extract_content:
                task = self._extract_content_for_result(web_result, request)
                extraction_tasks.append(task)
        
        # Ejecutar extracciones de contenido en paralelo (máximo 5 simultáneas)
        if extraction_tasks:
            await self._execute_content_extraction_tasks(extraction_tasks, max_concurrent=5)
        
        # Ordenar por puntuación combinada
        enriched_results.sort(
            key=lambda r: (r.relevance_score + r.quality_score) / 2,
            reverse=True
        )
        
        return enriched_results
    
    async def _extract_content_for_result(self, 
                                        web_result: WebSearchResult,
                                        request: WebSearchRequest):
        """Extraer contenido para un resultado específico"""
        
        start_time = time.time()
        
        try:
            # Verificar robots.txt
            can_fetch = await self.robots_checker.can_fetch(web_result.url)
            if not can_fetch:
                logger.info(f"Robots.txt blocks access to {web_result.url}")
                self.stats['robots_blocks'] += 1
                web_result.extraction_success = False
                return
            
            # Extraer contenido
            extracted = await self.content_extractor.extract_content(
                url=web_result.url,
                content_type_hint=self._infer_content_type(web_result, request.query_type)
            )
            
            web_result.extraction_time = time.time() - start_time
            web_result.extraction_success = extracted.extraction_success
            
            if extracted.extraction_success:
                # Truncar contenido si es muy largo
                if len(extracted.main_content) > request.content_max_length:
                    extracted.main_content = extracted.main_content[:request.content_max_length] + "..."
                
                web_result.extracted_content = extracted
                web_result.content_hash = get_content_hash(extracted.main_content)
                web_result.quality_score = extracted.quality_score
                web_result.language = extracted.language
                web_result.content_type = extracted.content_type
                web_result.word_count = extracted.word_count
                web_result.reading_time = extracted.reading_time
                
                self.stats['successful_extractions'] += 1
            
            self.stats['total_extractions'] += 1
            
        except Exception as e:
            logger.warning(f"Content extraction failed for {web_result.url}: {e}")
            web_result.extraction_time = time.time() - start_time
            web_result.extraction_success = False
    
    async def _execute_content_extraction_tasks(self, 
                                              tasks: List[Any],
                                              max_concurrent: int = 5):
        """Ejecutar tareas de extracción con concurrencia limitada"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_task(task):
            async with semaphore:
                await task
        
        # Ejecutar todas las tareas con límite de concurrencia
        await asyncio.gather(*[bounded_task(task) for task in tasks])
    
    def _calculate_relevance_score(self, result: SearchResult, query: str) -> float:
        """Calcular puntuación de relevancia"""
        
        score = 0.0
        query_terms = query.lower().split()
        
        # Puntuación por coincidencias en título
        title_lower = result.title.lower()
        title_matches = sum(1 for term in query_terms if term in title_lower)
        score += (title_matches / len(query_terms)) * 0.4
        
        # Puntuación por coincidencias en snippet
        snippet_lower = result.snippet.lower()
        snippet_matches = sum(1 for term in query_terms if term in snippet_lower)
        score += (snippet_matches / len(query_terms)) * 0.3
        
        # Puntuación por ranking (primeros resultados son más relevantes)
        rank_score = max(0, (10 - result.rank) / 10) * 0.2
        score += rank_score
        
        # Puntuación por dominio preferido
        if self.config.is_domain_preferred(result.domain):
            score += 0.1
        
        return min(score, 1.0)
    
    def _infer_content_type(self, result: WebSearchResult, query_type: str) -> str:
        """Inferir tipo de contenido basado en resultado y query type"""
        
        if query_type == 'academic':
            return 'academic'
        elif query_type == 'technical':
            return 'technical'
        elif 'documentation' in result.url.lower() or 'docs' in result.url.lower():
            return 'documentation'
        elif any(forum in result.domain for forum in ['stackoverflow.com', 'reddit.com']):
            return 'forum_post'
        else:
            return 'article'
    
    def _format_search_response(self, 
                              results: List[WebSearchResult],
                              request: WebSearchRequest) -> str:
        """Formatear respuesta final para el agente"""
        
        if not results:
            return "No search results found for the given query."
        
        response_parts = []
        response_parts.append(f"Found {len(results)} results for '{request.query}':\n")
        
        for i, result in enumerate(results[:request.max_results], 1):
            response_parts.append(f"{i}. **{result.title}**")
            response_parts.append(f"   URL: {result.url}")
            response_parts.append(f"   Domain: {result.domain}")
            
            if result.snippet:
                response_parts.append(f"   Summary: {result.snippet}")
            
            # Incluir contenido extraído si está disponible
            if result.extracted_content and result.extraction_success:
                content = result.extracted_content.main_content
                if len(content) > 300:
                    content = content[:300] + "..."
                response_parts.append(f"   Content: {content}")
                
                if result.extracted_content.author:
                    response_parts.append(f"   Author: {result.extracted_content.author}")
                
                if result.extracted_content.publish_date:
                    response_parts.append(f"   Date: {result.extracted_content.publish_date}")
            
            # Incluir metadatos si se solicita
            if request.include_metadata:
                response_parts.append(f"   Quality: {result.quality_score:.2f}")
                response_parts.append(f"   Language: {result.language}")
                if result.word_count > 0:
                    response_parts.append(f"   Length: {result.word_count} words ({result.reading_time} min read)")
            
            response_parts.append("")  # Línea vacía entre resultados
        
        # Agregar resumen final
        successful_extractions = sum(1 for r in results if r.extraction_success)
        if request.extract_content and successful_extractions > 0:
            response_parts.append(f"\nSuccessfully extracted content from {successful_extractions}/{len(results)} pages.")
        
        return "\n".join(response_parts)
    
    def _format_error_response(self, error_message: str) -> str:
        """Formatear respuesta de error"""
        return f"Web search error: {error_message}"
    
    def _generate_cache_key(self, request: WebSearchRequest) -> str:
        """Generar clave de cache para el request"""
        key_parts = [
            request.query,
            str(request.max_results),
            request.search_engine or "default",
            request.query_type,
            str(request.extract_content),
        ]
        return "|".join(key_parts)
    
    def _get_cached_result(self, cache_key: str) -> Optional[str]:
        """Obtener resultado del cache si es válido"""
        if cache_key in self._results_cache:
            result, timestamp = self._results_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return result
            else:
                # Cache expirado
                del self._results_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, result: str):
        """Guardar resultado en cache"""
        self._results_cache[cache_key] = (result, time.time())
        
        # Limpiar cache si tiene demasiadas entradas
        if len(self._results_cache) > 100:
            # Remover entradas más antiguas
            oldest_keys = sorted(
                self._results_cache.keys(),
                key=lambda k: self._results_cache[k][1]
            )[:20]  # Remover 20 más antiguas
            
            for key in oldest_keys:
                del self._results_cache[key]
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Información detallada del tool"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "version": self.version,
            "capabilities": [
                "Multi-engine search (DuckDuckGo, Google, SearX, Bing)",
                "Intelligent content extraction",
                "Rate limiting and robots.txt compliance",
                "Result deduplication and ranking",
                "Multiple query types (general, academic, technical, news)",
                "Domain filtering and preferences",
                "Content quality scoring",
                "Language detection",
                "Cache for performance"
            ],
            "parameters": {
                "query": "Search query (required)",
                "max_results": "Maximum results to return (default: 10)",
                "search_engine": "Specific engine: duckduckgo, google_custom, searx, bing",
                "query_type": "Type: general, academic, technical, news",
                "extract_content": "Extract full page content (default: true)",
                "content_max_length": "Max content length (default: 5000)",
                "include_metadata": "Include quality scores and metadata (default: true)",
                "filter_domains": "List of domains to include",
                "exclude_domains": "List of domains to exclude"
            },
            "example_usage": {
                "simple": "artificial intelligence applications",
                "advanced": '{"query": "machine learning research", "query_type": "academic", "max_results": 5}',
                "filtered": '{"query": "python tutorials", "filter_domains": ["python.org", "docs.python.org"]}'
            },
            "limitations": [
                "Rate limited to 30 searches per minute",
                "Content extraction may fail on complex sites",
                "Some search engines require API keys",
                "Respects robots.txt (some sites may be blocked)"
            ],
            "statistics": self.stats
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check específico del web search tool"""
        health_info = await super().health_check()
        
        try:
            # Test de búsqueda simple
            test_start = time.time()
            test_result = await self._arun("test search health check")
            test_time = time.time() - test_start
            
            # Health check de motores de búsqueda
            engines_health = await self.search_manager.health_check()
            
            health_info.update({
                "search_test_time": test_time,
                "search_test_success": "error" not in test_result.lower(),
                "available_engines": list(engines_health.keys()),
                "healthy_engines": [
                    engine for engine, status in engines_health.items()
                    if status.get('healthy', False)
                ],
                "cache_size": len(self._results_cache),
                **self.stats
            })
            
        except Exception as e:
            health_info["health_check_error"] = str(e)
            health_info["healthy"] = False
        
        return health_info
    
    def _get_health_check_input(self) -> str:
        """Input específico para health check"""
        return "python programming tutorial"

# Funciones de conveniencia para uso directo
async def web_search(query: str, **kwargs) -> str:
    """Función de conveniencia para búsqueda web"""
    tool = WebSearchTool()
    return await tool._arun(query, **kwargs)

async def quick_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Búsqueda rápida que retorna resultados estructurados"""
    tool = WebSearchTool()
    
    # Realizar búsqueda sin extracción de contenido para mayor velocidad
    request = WebSearchRequest(
        query=query,
        max_results=max_results,
        extract_content=False,
        include_metadata=False
    )
    
    search_results = await tool._perform_search(request)
    enriched_results = await tool._enrich_search_results(search_results, request)
    
    return [result.to_dict() for result in enriched_results]