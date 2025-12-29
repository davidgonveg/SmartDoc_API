"""
Web Utilities for Web Search Tool
Utilidades comunes para web scraping y manejo de URLs/contenido
"""

import re
import time
import asyncio
import hashlib
import urllib.parse
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

import httpx
from bs4 import BeautifulSoup, Comment
import unicodedata

logger = logging.getLogger(__name__)

@dataclass
class WebResponse:
    """Respuesta estandarizada de requests web"""
    url: str
    status_code: int
    content: str
    headers: Dict[str, str]
    encoding: str
    response_time: float
    final_url: str  # URL después de redirects
    error: Optional[str] = None
    success: bool = True

@dataclass
class URLInfo:
    """Información detallada de una URL"""
    original_url: str
    normalized_url: str
    domain: str
    subdomain: str
    path: str
    query_params: Dict[str, str]
    fragment: str
    is_secure: bool
    is_valid: bool
    url_hash: str

class URLNormalizer:
    """Normalizador de URLs para evitar duplicados y estandarizar"""
    
    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalizar URL removiendo parámetros innecesarios y estandarizando formato"""
        if not url or not isinstance(url, str):
            return ""
        
        try:
            # Agregar protocolo si falta
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Parse URL
            parsed = urllib.parse.urlparse(url)
            
            # Normalizar dominio (lowercase, remover www opcional)
            domain = parsed.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Normalizar path (remover trailing slash excepto root)
            path = parsed.path
            if path.endswith('/') and len(path) > 1:
                path = path[:-1]
            
            # Filtrar query parameters (remover tracking, session, etc.)
            query_params = urllib.parse.parse_qs(parsed.query)
            filtered_params = URLNormalizer._filter_query_params(query_params)
            query_string = urllib.parse.urlencode(filtered_params, doseq=True)
            
            # Reconstruir URL normalizada
            normalized = urllib.parse.urlunparse((
                'https',  # Siempre usar HTTPS
                domain,
                path,
                '',  # params
                query_string,
                ''   # fragment removido para normalización
            ))
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Error normalizando URL {url}: {e}")
            return url
    
    @staticmethod
    def _filter_query_params(params: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Filtrar parámetros de query irrelevantes para normalización"""
        # Parámetros a remover (tracking, analytics, session, etc.)
        remove_params = {
            # Google Analytics
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'gclid', 'gclsrc', 'dclid', 'fbclid',
            
            # Social media tracking
            'ref', 'referer', 'referrer', 'source', 'src',
            'share', 'shared', 'via',
            
            # Session/user tracking
            'sessionid', 'session_id', 'sid', 'userid', 'user_id',
            'token', 'auth', 'key', 'signature', 'hash',
            
            # Time/cache busting
            'timestamp', 'ts', 'time', 't', 'cache', 'v', 'version',
            '_', 'rand', 'random',
            
            # Page/view tracking
            'page', 'p', 'tab', 'view', 'display',
        }
        
        # Mantener solo parámetros relevantes
        filtered = {}
        for key, values in params.items():
            if key.lower() not in remove_params and not key.startswith('_'):
                filtered[key] = values
        
        return filtered
    
    @staticmethod
    def get_url_info(url: str) -> URLInfo:
        """Obtener información detallada de una URL"""
        normalized = URLNormalizer.normalize_url(url)
        
        try:
            parsed = urllib.parse.urlparse(normalized)
            
            # Extraer componentes
            domain_parts = parsed.netloc.split('.')
            subdomain = '.'.join(domain_parts[:-2]) if len(domain_parts) > 2 else ''
            domain = '.'.join(domain_parts[-2:]) if len(domain_parts) >= 2 else parsed.netloc
            
            query_params = dict(urllib.parse.parse_qsl(parsed.query))
            
            # Generar hash único para la URL
            url_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]
            
            return URLInfo(
                original_url=url,
                normalized_url=normalized,
                domain=domain,
                subdomain=subdomain,
                path=parsed.path,
                query_params=query_params,
                fragment=parsed.fragment,
                is_secure=parsed.scheme == 'https',
                is_valid=True,
                url_hash=url_hash
            )
            
        except Exception as e:
            logger.warning(f"Error parsing URL {url}: {e}")
            return URLInfo(
                original_url=url,
                normalized_url=url,
                domain="",
                subdomain="",
                path="",
                query_params={},
                fragment="",
                is_secure=False,
                is_valid=False,
                url_hash=""
            )

class ContentCleaner:
    """Limpiador de contenido HTML y texto"""
    
    @staticmethod
    def clean_html(html: str, preserve_links: bool = False) -> str:
        """Limpiar HTML y extraer texto limpio"""
        if not html:
            return ""
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remover elementos no deseados
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'advertisement', 'ads']):
                element.decompose()
            
            # Remover comentarios HTML
            for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
                comment.extract()
            
            # Extraer texto
            if preserve_links:
                # Mantener información de links
                for link in soup.find_all('a', href=True):
                    link.string = f"{link.get_text()} [{link['href']}]"
            
            text = soup.get_text(separator=' ', strip=True)
            
            # Limpiar texto
            return ContentCleaner.clean_text(text)
            
        except Exception as e:
            logger.warning(f"Error cleaning HTML: {e}")
            return html
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Limpiar y normalizar texto"""
        if not text:
            return ""
        
        # Normalizar unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remover caracteres de control excepto newlines y tabs
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Normalizar espacios en blanco
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Limpiar líneas vacías excesivas
        lines = text.split('\n')
        cleaned_lines = []
        empty_line_count = 0
        
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                empty_line_count = 0
            else:
                empty_line_count += 1
                if empty_line_count <= 1:  # Máximo una línea vacía consecutiva
                    cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    @staticmethod
    def extract_main_content(soup: BeautifulSoup) -> str:
        """Extraer contenido principal usando heurísticas"""
        if not soup:
            return ""
        
        # Buscar por selectores comunes de contenido principal
        main_selectors = [
            'article', '[role="main"]', 'main', '#main', '#content',
            '.article-content', '.post-content', '.entry-content',
            '.content', '.main-content', '[itemprop="articleBody"]'
        ]
        
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                return ContentCleaner.clean_html(str(element))
        
        # Fallback: buscar el div con más texto
        text_containers = soup.find_all(['div', 'section', 'article'])
        if text_containers:
            best_container = max(text_containers, 
                               key=lambda x: len(x.get_text(strip=True)))
            return ContentCleaner.clean_html(str(best_container))
        
        # Último recurso: todo el body
        body = soup.find('body')
        if body:
            return ContentCleaner.clean_html(str(body))
        
        return ContentCleaner.clean_html(str(soup))

class WebRequestManager:
    """Gestor de requests HTTP con retry, timeout y error handling"""
    
    def __init__(self, 
                 timeout: int = 15,
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 max_response_size: int = 10 * 1024 * 1024):  # 10MB
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_response_size = max_response_size
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.aclose()
    
    async def fetch_url(self, 
                       url: str, 
                       headers: Dict[str, str] = None,
                       allow_redirects: bool = True) -> WebResponse:
        """Hacer request HTTP con manejo de errores y reintentos"""
        
        start_time = time.time()
        headers = headers or {}
        
        # Validar URL
        url_info = URLNormalizer.get_url_info(url)
        if not url_info.is_valid:
            return WebResponse(
                url=url,
                status_code=0,
                content="",
                headers={},
                encoding="",
                response_time=0,
                final_url=url,
                error="Invalid URL",
                success=False
            )
        
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Fetching URL (attempt {attempt + 1}): {url}")
                
                response = await self.session.get(
                    url,
                    headers=headers,
                    follow_redirects=allow_redirects
                )
                
                response_time = time.time() - start_time
                
                # Verificar tamaño de respuesta
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_response_size:
                    return WebResponse(
                        url=url,
                        status_code=response.status_code,
                        content="",
                        headers=dict(response.headers),
                        encoding=response.encoding or 'utf-8',
                        response_time=response_time,
                        final_url=str(response.url),
                        error=f"Response too large: {content_length} bytes",
                        success=False
                    )
                
                # Leer contenido con límite de tamaño
                content = response.text
                if len(content) > self.max_response_size:
                    content = content[:self.max_response_size]
                    logger.warning(f"Response truncated to {self.max_response_size} bytes")
                
                success = 200 <= response.status_code < 400
                error_msg = None if success else f"HTTP {response.status_code}"
                
                return WebResponse(
                    url=url,
                    status_code=response.status_code,
                    content=content,
                    headers=dict(response.headers),
                    encoding=response.encoding or 'utf-8',
                    response_time=response_time,
                    final_url=str(response.url),
                    error=error_msg,
                    success=success
                )
                
            except httpx.TimeoutException:
                error_msg = f"Timeout after {self.timeout}s"
                logger.warning(f"Timeout fetching {url} (attempt {attempt + 1})")
                
            except httpx.RequestError as e:
                error_msg = f"Request error: {str(e)}"
                logger.warning(f"Request error fetching {url}: {e}")
                
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Unexpected error fetching {url}: {e}")
            
            # Esperar antes del siguiente intento
            if attempt < self.max_retries:
                delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                logger.debug(f"Retrying in {delay}s...")
                await asyncio.sleep(delay)
        
        # Todos los intentos fallaron
        response_time = time.time() - start_time
        return WebResponse(
            url=url,
            status_code=0,
            content="",
            headers={},
            encoding="",
            response_time=response_time,
            final_url=url,
            error=error_msg,
            success=False
        )

class RobotsTxtChecker:
    """Verificador de robots.txt para respetar directivas de sitios web"""
    
    def __init__(self):
        self.cache = {}  # Cache de robots.txt por dominio
        self.cache_ttl = 3600  # 1 hora
        self.user_agent = "SmartDocAgent"
    
    async def can_fetch(self, url: str, user_agent: str = None) -> bool:
        """Verificar si se puede hacer fetch de una URL según robots.txt"""
        user_agent = user_agent or self.user_agent
        
        try:
            url_info = URLNormalizer.get_url_info(url)
            domain = url_info.domain
            
            if not domain:
                return True  # Si no se puede parsear, permitir
            
            # Verificar cache
            cache_key = f"{domain}:{user_agent}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return self._is_path_allowed(cached_data, url_info.path, user_agent)
            
            # Fetch robots.txt
            robots_url = f"https://{domain}/robots.txt"
            async with WebRequestManager(timeout=5, max_retries=1) as request_manager:
                response = await request_manager.fetch_url(robots_url)
                
                if response.success:
                    robots_content = response.content
                else:
                    robots_content = ""  # Si no hay robots.txt, permitir todo
            
            # Cache robots.txt
            self.cache[cache_key] = (robots_content, time.time())
            
            return self._is_path_allowed(robots_content, url_info.path, user_agent)
            
        except Exception as e:
            logger.warning(f"Error checking robots.txt for {url}: {e}")
            return True  # En caso de error, permitir
    
    def _is_path_allowed(self, robots_content: str, path: str, user_agent: str) -> bool:
        """Verificar si un path está permitido según robots.txt"""
        if not robots_content:
            return True
        
        lines = robots_content.split('\n')
        current_user_agent = None
        disallowed_paths = []
        allowed_paths = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.lower().startswith('user-agent:'):
                ua = line.split(':', 1)[1].strip()
                current_user_agent = ua.lower()
            
            elif current_user_agent in [user_agent.lower(), '*']:
                if line.lower().startswith('disallow:'):
                    disallow_path = line.split(':', 1)[1].strip()
                    if disallow_path:
                        disallowed_paths.append(disallow_path)
                
                elif line.lower().startswith('allow:'):
                    allow_path = line.split(':', 1)[1].strip()
                    if allow_path:
                        allowed_paths.append(allow_path)
        
        # Verificar allows primero (más específico)
        for allow_path in allowed_paths:
            if self._path_matches(path, allow_path):
                return True
        
        # Verificar disallows
        for disallow_path in disallowed_paths:
            if self._path_matches(path, disallow_path):
                return False
        
        return True  # Default: permitir
    
    def _path_matches(self, path: str, pattern: str) -> bool:
        """Verificar si un path coincide con un patrón de robots.txt"""
        if pattern == '/':
            return True
        
        if pattern.endswith('*'):
            return path.startswith(pattern[:-1])
        
        return path.startswith(pattern)

# Funciones de utilidad globales
def is_valid_url(url: str) -> bool:
    """Verificar si una URL es válida"""
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def extract_links_from_html(html: str, base_url: str = "") -> List[str]:
    """Extraer todos los links de HTML"""
    if not html:
        return []
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Resolver URLs relativas
            if base_url and not href.startswith(('http://', 'https://')):
                href = urllib.parse.urljoin(base_url, href)
            
            if is_valid_url(href):
                links.append(href)
        
        return list(set(links))  # Remover duplicados
        
    except Exception as e:
        logger.warning(f"Error extracting links: {e}")
        return []

def get_content_hash(content: str) -> str:
    """Obtener hash único del contenido para detección de duplicados"""
    if not content:
        return ""
    
    # Normalizar contenido antes del hash
    normalized = ContentCleaner.clean_text(content)
    return hashlib.md5(normalized.encode()).hexdigest()

def estimate_reading_time(text: str) -> int:
    """Estimar tiempo de lectura en minutos (asumiendo 200 palabras por minuto)"""
    if not text:
        return 0
    
    word_count = len(text.split())
    return max(1, round(word_count / 200))

def detect_language(text: str) -> str:
    """Detección básica de idioma (heurística simple)"""
    if not text or len(text) < 20:
        return "unknown"
    
    # Patrones simples para idiomas comunes
    spanish_indicators = ['el', 'la', 'de', 'que', 'y', 'es', 'en', 'un', 'se', 'no']
    english_indicators = ['the', 'of', 'and', 'to', 'a', 'in', 'is', 'it', 'you', 'that']
    
    words = text.lower().split()[:100]  # Primeras 100 palabras
    
    spanish_count = sum(1 for word in words if word in spanish_indicators)
    english_count = sum(1 for word in words if word in english_indicators)
    
    if spanish_count > english_count:
        return "spanish"
    elif english_count > spanish_count:
        return "english"
    else:
        return "unknown"

# Instancias globales reutilizables
robots_checker = RobotsTxtChecker()