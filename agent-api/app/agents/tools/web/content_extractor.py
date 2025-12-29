"""
Content Extractor for Web Search Tool
Extracción inteligente de contenido principal de páginas web
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import asyncio

from bs4 import BeautifulSoup, NavigableString, Comment
from bs4.element import Tag

from .config import (
    ContentType, get_extraction_rule, get_domain_from_url,
    is_content_valid, calculate_content_quality_score,
    REMOVE_ELEMENTS, TEXT_CLEANING_CONFIG, PAGE_TYPE_RULES
)
from .web_utils import WebRequestManager, ContentCleaner

logger = logging.getLogger(__name__)

@dataclass
class ExtractedContent:
    """Contenido extraído de una página web"""
    url: str
    title: str
    main_content: str
    description: str
    author: str
    publish_date: str
    keywords: List[str]
    images: List[str]
    links: List[str]
    language: str
    content_type: str  # article, documentation, forum_post, etc.
    quality_score: float
    word_count: int
    reading_time: int  # minutos
    metadata: Dict[str, any]
    extraction_success: bool
    extraction_errors: List[str]
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.extraction_errors is None:
            self.extraction_errors = []

class ContentExtractor:
    """Extractor principal de contenido web"""
    
    def __init__(self):
        self.cleaning_config = TEXT_CLEANING_CONFIG
        self.page_type_rules = PAGE_TYPE_RULES
        
        # Cache de reglas por dominio
        self._domain_rules_cache = {}
        
        # Selectores adicionales por heurística
        self.heuristic_selectors = {
            'title': [
                'h1', 'h2.title', '.page-title', '.article-title',
                '.post-title', '.entry-title', '[itemprop="headline"]'
            ],
            'content': [
                '.post-content', '.article-body', '.entry-content',
                '.content-body', '.main-content', '.page-content',
                '[role="main"] p', 'article p', '.text p'
            ],
            'author': [
                '.author-name', '.byline', '.post-author',
                '[rel="author"]', '[itemprop="author"]'
            ],
            'date': [
                '.post-date', '.publish-date', '.article-date',
                'time', '[datetime]', '[itemprop="datePublished"]'
            ]
        }
    
    async def extract_content(self, 
                            url: str, 
                            html: str = None,
                            content_type_hint: str = None) -> ExtractedContent:
        """
        Extraer contenido completo de una URL o HTML
        
        Args:
            url: URL de la página
            html: HTML opcional (si no se provee, se hace fetch)
            content_type_hint: Tipo de contenido esperado (article, documentation, etc.)
        """
        
        extraction_errors = []
        
        try:
            # Obtener HTML si no se provee
            if not html:
                html = await self._fetch_html(url)
                if not html:
                    return self._create_empty_result(url, ["Failed to fetch HTML"])
            
            # Parsear HTML
            soup = BeautifulSoup(html, 'html.parser')
            if not soup:
                return self._create_empty_result(url, ["Failed to parse HTML"])
            
            # Limpiar HTML de elementos no deseados
            soup = self._clean_html_structure(soup)
            
            # Detectar tipo de página si no se especifica
            if not content_type_hint:
                content_type_hint = self._detect_content_type(soup, url)
            
            # Obtener dominio para reglas específicas
            domain = get_domain_from_url(url)
            
            # Extraer cada tipo de contenido
            results = {}
            
            # Título
            try:
                results['title'] = await self._extract_title(soup, domain)
            except Exception as e:
                extraction_errors.append(f"Title extraction failed: {e}")
                results['title'] = ""
            
            # Contenido principal
            try:
                results['main_content'] = await self._extract_main_content(soup, domain, content_type_hint)
            except Exception as e:
                extraction_errors.append(f"Main content extraction failed: {e}")  
                results['main_content'] = ""
            
            # Descripción
            try:
                results['description'] = await self._extract_description(soup, domain)
            except Exception as e:
                extraction_errors.append(f"Description extraction failed: {e}")
                results['description'] = ""
            
            # Autor
            try:
                results['author'] = await self._extract_author(soup, domain)
            except Exception as e:
                extraction_errors.append(f"Author extraction failed: {e}")
                results['author'] = ""
            
            # Fecha
            try:
                results['publish_date'] = await self._extract_date(soup, domain)
            except Exception as e:
                extraction_errors.append(f"Date extraction failed: {e}")
                results['publish_date'] = ""
            
            # Keywords
            try:
                results['keywords'] = await self._extract_keywords(soup, domain)
            except Exception as e:
                extraction_errors.append(f"Keywords extraction failed: {e}")
                results['keywords'] = []
            
            # Links e imágenes
            try:
                results['links'] = self._extract_links(soup, url)
                results['images'] = self._extract_images(soup, url)
            except Exception as e:
                extraction_errors.append(f"Links/images extraction failed: {e}")
                results['links'] = []
                results['images'] = []
            
            # Detectar idioma
            language = self._detect_language(results['main_content'])
            
            # Calcular métricas
            word_count = len(results['main_content'].split()) if results['main_content'] else 0
            reading_time = max(1, round(word_count / 200))  # 200 WPM
            quality_score = calculate_content_quality_score(results['main_content'])
            
            # Verificar si la extracción fue exitosa
            extraction_success = self._validate_extraction(results, content_type_hint)
            
            return ExtractedContent(
                url=url,
                title=results['title'],
                main_content=results['main_content'],
                description=results['description'],
                author=results['author'],
                publish_date=results['publish_date'],
                keywords=results['keywords'],
                images=results['images'],
                links=results['links'],
                language=language,
                content_type=content_type_hint,
                quality_score=quality_score,
                word_count=word_count,
                reading_time=reading_time,
                metadata={
                    'domain': domain,
                    'extraction_method': 'intelligent_extraction',
                    'selectors_used': [],  # Se llena durante extracción
                    'html_length': len(html),
                    'soup_elements': len(soup.find_all()) if soup else 0
                },
                extraction_success=extraction_success,
                extraction_errors=extraction_errors
            )
        
        except Exception as e:
            logger.error(f"Critical error extracting content from {url}: {e}")
            return self._create_empty_result(url, [f"Critical extraction error: {e}"])
    
    async def _fetch_html(self, url: str) -> Optional[str]:
        """Hacer fetch del HTML de una URL"""
        try:
            async with WebRequestManager(timeout=15, max_retries=2) as request_manager:
                response = await request_manager.fetch_url(url)
                return response.content if response.success else None
        except Exception as e:
            logger.error(f"Error fetching HTML from {url}: {e}")
            return None
    
    def _clean_html_structure(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Limpiar estructura HTML removiendo elementos no deseados"""
        
        # Remover elementos por tag
        for tag in REMOVE_ELEMENTS['tags']:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remover elementos por clase
        for class_name in REMOVE_ELEMENTS['classes']:
            for element in soup.find_all(class_=re.compile(class_name, re.I)):
                element.decompose()
        
        # Remover elementos por ID
        for id_name in REMOVE_ELEMENTS['ids']:
            element = soup.find(id=re.compile(id_name, re.I))
            if element:
                element.decompose()
        
        # Remover comentarios HTML
        for comment in soup.find_all(text=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Remover elementos vacíos o con muy poco contenido
        for element in soup.find_all():
            if element.name in ['div', 'span', 'p']:
                text = element.get_text(strip=True)
                if len(text) < 5:  # Muy poco contenido
                    element.decompose()
        
        return soup
    
    def _detect_content_type(self, soup: BeautifulSoup, url: str) -> str:
        """Detectar tipo de contenido de la página"""
        
        # Detectar por URL
        if any(pattern in url.lower() for pattern in ['docs', 'documentation', 'api', 'guide']):
            return 'documentation'
        
        if any(pattern in url.lower() for pattern in ['forum', 'discuss', 'community']):
            return 'forum_post'
        
        if any(pattern in url.lower() for pattern in ['paper', 'journal', 'research', 'scholar']):
            return 'academic'
        
        # Detectar por estructura HTML
        if soup.find('article') or soup.find(attrs={'role': 'article'}):
            return 'article'
        
        if soup.find(class_=re.compile(r'post|article|entry', re.I)):
            return 'article'
        
        if soup.find(class_=re.compile(r'doc|documentation', re.I)):
            return 'documentation'
        
        # Default
        return 'article'
    
    async def _extract_title(self, soup: BeautifulSoup, domain: str) -> str:
        """Extraer título de la página"""
        
        rule = get_extraction_rule(ContentType.TITLE, domain)
        
        # Intentar reglas específicas del dominio/generales
        for selector in rule.selectors:
            element = soup.select_one(selector)
            if element:
                title = self._extract_text_from_element(element, rule.attributes)
                if title and rule.min_length <= len(title) <= rule.max_length:
                    return self._clean_extracted_text(title)
        
        # Fallback heurístico
        for selector in self.heuristic_selectors['title']:
            element = soup.select_one(selector)
            if element:
                title = element.get_text(strip=True)
                if title and len(title) > 5:
                    return self._clean_extracted_text(title)
        
        # Último recurso: title tag
        title_tag = soup.find('title')
        if title_tag:
            return self._clean_extracted_text(title_tag.get_text(strip=True))
        
        return ""
    
    async def _extract_main_content(self, soup: BeautifulSoup, domain: str, content_type: str) -> str:
        """Extraer contenido principal"""
        
        rule = get_extraction_rule(ContentType.MAIN_CONTENT, domain)
        
        # Intentar reglas específicas
        for selector in rule.selectors:
            element = soup.select_one(selector)
            if element:
                content = self._extract_text_from_element(element, rule.attributes)
                if content and len(content) >= rule.min_length:
                    return self._clean_extracted_text(content)
        
        # Fallback: heurística por puntuación de contenido
        best_content = ""
        best_score = 0
        
        # Buscar candidatos de contenido
        content_candidates = []
        
        for selector in self.heuristic_selectors['content']:
            elements = soup.select(selector)
            for element in elements:
                text = element.get_text(strip=True)
                if len(text) > 100:  # Mínimo de contenido útil
                    score = self._score_content_element(element, text)
                    content_candidates.append((element, text, score))
        
        # También considerar divs grandes con mucho texto
        for div in soup.find_all('div'):
            text = div.get_text(strip=True)
            if len(text) > 200:
                score = self._score_content_element(div, text)
                content_candidates.append((div, text, score))
        
        # Ordenar por puntuación y tomar el mejor
        if content_candidates:
            content_candidates.sort(key=lambda x: x[2], reverse=True)
            best_element, best_text, best_score = content_candidates[0]
            
            if best_score > 0.3:  # Umbral mínimo de calidad
                return self._clean_extracted_text(best_text)
        
        # Último recurso: todo el body
        body = soup.find('body')
        if body:
            content = body.get_text(separator=' ', strip=True)
            return self._clean_extracted_text(content)
        
        return ""
    
    def _score_content_element(self, element: Tag, text: str) -> float:
        """Asignar puntuación a un elemento basado en probabilidad de ser contenido principal"""
        
        score = 0.0
        
        # Puntuación por longitud (más texto = mejor, hasta un punto)
        text_length = len(text)
        if text_length > 100:
            score += min(text_length / 1000, 0.4)  # Max 0.4 por longitud
        
        # Puntuación por estructura del texto
        sentences = text.count('. ') + text.count('? ') + text.count('! ')
        if sentences > 3:
            score += min(sentences / 50, 0.2)  # Max 0.2 por estructura
        
        # Puntuación por clases CSS positivas
        class_list = element.get('class', [])
        positive_classes = ['content', 'article', 'post', 'main', 'body', 'text']
        for cls in class_list:
            if any(pos in cls.lower() for pos in positive_classes):
                score += 0.1
        
        # Penalización por clases CSS negativas
        negative_classes = ['sidebar', 'nav', 'menu', 'footer', 'header', 'ad', 'comment']
        for cls in class_list:
            if any(neg in cls.lower() for neg in negative_classes):
                score -= 0.2
        
        # Puntuación por tags semánticos
        if element.name in ['article', 'main', 'section']:
            score += 0.15
        elif element.name in ['div', 'span']:
            score += 0.05
        
        # Puntuación por posición relativa (elementos más al centro/arriba)
        # Esta es una heurística simple - en la práctica sería más compleja
        if element.parent and element.parent.name == 'body':
            score += 0.1
        
        return max(0.0, min(score, 1.0))  # Normalizar entre 0 y 1
    
    async def _extract_description(self, soup: BeautifulSoup, domain: str) -> str:
        """Extraer descripción/resumen"""
        
        rule = get_extraction_rule(ContentType.DESCRIPTION, domain)
        
        for selector in rule.selectors:
            element = soup.select_one(selector)
            if element:
                desc = self._extract_text_from_element(element, rule.attributes)
                if desc and rule.min_length <= len(desc) <= rule.max_length:
                    return self._clean_extracted_text(desc)
        
        return ""
    
    async def _extract_author(self, soup: BeautifulSoup, domain: str) -> str:
        """Extraer autor"""
        
        rule = get_extraction_rule(ContentType.AUTHOR, domain)
        
        for selector in rule.selectors:
            element = soup.select_one(selector)
            if element:
                author = self._extract_text_from_element(element, rule.attributes)
                if author and len(author) <= rule.max_length:
                    return self._clean_extracted_text(author)
        
        return ""
    
    async def _extract_date(self, soup: BeautifulSoup, domain: str) -> str:
        """Extraer fecha de publicación"""
        
        rule = get_extraction_rule(ContentType.DATE, domain)
        
        for selector in rule.selectors:
            element = soup.select_one(selector)
            if element:
                date_text = self._extract_text_from_element(element, rule.attributes)
                if date_text:
                    # Validar formato de fecha con regex
                    if rule.regex_patterns:
                        for pattern in rule.regex_patterns:
                            if re.search(pattern, date_text):
                                return self._clean_extracted_text(date_text)
                    else:
                        return self._clean_extracted_text(date_text)
        
        return ""
    
    async def _extract_keywords(self, soup: BeautifulSoup, domain: str) -> List[str]:
        """Extraer keywords/tags"""
        
        rule = get_extraction_rule(ContentType.KEYWORDS, domain)
        keywords = []
        
        for selector in rule.selectors:
            element = soup.select_one(selector)
            if element:
                keywords_text = self._extract_text_from_element(element, rule.attributes)
                if keywords_text:
                    # Dividir por comas o espacios
                    split_keywords = re.split(r'[,\s]+', keywords_text)
                    keywords.extend([kw.strip() for kw in split_keywords if kw.strip()])
        
        # Remover duplicados y limpiar
        return list(set([self._clean_extracted_text(kw) for kw in keywords if len(kw) > 2]))
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extraer links internos y externos"""
        
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Resolver URLs relativas
            if href.startswith('/') or not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            
            # Validar URL
            try:
                parsed = urlparse(href)
                if parsed.netloc and parsed.scheme in ['http', 'https']:
                    links.append(href)
            except Exception:
                continue
        
        return list(set(links))  # Remover duplicados
    
    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extraer URLs de imágenes"""
        
        images = []
        
        for img in soup.find_all('img', src=True):
            src = img['src']
            
            # Resolver URLs relativas
            if src.startswith('/') or not src.startswith(('http://', 'https://')):
                src = urljoin(base_url, src)
            
            # Validar URL de imagen
            try:
                parsed = urlparse(src)
                if parsed.netloc and parsed.scheme in ['http', 'https']:
                    # Filtrar imágenes muy pequeñas (probablemente decorativas)
                    width = img.get('width')
                    height = img.get('height')
                    
                    if width and height:
                        try:
                            w, h = int(width), int(height)
                            if w >= 100 and h >= 100:  # Mínimo tamaño útil
                                images.append(src)
                        except ValueError:
                            images.append(src)  # Si no se puede parsear, incluir
                    else:
                        images.append(src)
            except Exception:
                continue
        
        return list(set(images))  # Remover duplicados
    
    def _extract_text_from_element(self, element: Tag, attributes: List[str]) -> str:
        """Extraer texto de un elemento según atributos especificados"""
        
        for attr in attributes:
            if attr == 'text':
                return element.get_text(strip=True)
            elif attr in element.attrs:
                return str(element.attrs[attr])
        
        # Fallback: texto del elemento
        return element.get_text(strip=True)
    
    def _clean_extracted_text(self, text: str) -> str:
        """Aplicar limpieza específica al texto extraído"""
        
        if not text:
            return ""
        
        # Usar el limpiador general
        cleaned = ContentCleaner.clean_text(text)
        
        # Aplicar patrones de limpieza específicos
        for pattern, replacement in self.cleaning_config['cleanup_patterns']:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        return cleaned.strip()
    
    def _detect_language(self, text: str) -> str:
        """Detectar idioma del contenido"""
        
        if not text or len(text) < 50:
            return "unknown"
        
        # Detectores simples por palabras comunes
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al']
        english_words = ['the', 'of', 'and', 'to', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'i']
        
        words = text.lower().split()[:200]  # Primeras 200 palabras
        
        spanish_count = sum(1 for word in words if word in spanish_words)
        english_count = sum(1 for word in words if word in english_words)
        
        if spanish_count > english_count and spanish_count > 5:
            return "spanish"
        elif english_count > spanish_count and english_count > 5:
            return "english"
        else:
            return "unknown"
    
    def _validate_extraction(self, results: Dict[str, any], content_type: str) -> bool:
        """Validar si la extracción fue exitosa según reglas del tipo de página"""
        
        if content_type not in self.page_type_rules:
            content_type = 'article'  # Default
        
        rules = self.page_type_rules[content_type]
        
        # Verificar elementos requeridos
        for required_type in rules['required_elements']:
            field_name = required_type.value.replace('_', '_')  # main_content, etc.
            
            if field_name == 'main_content':
                if not results.get('main_content') or len(results['main_content']) < rules['min_content_length']:
                    return False
            elif field_name == 'title':
                if not results.get('title'):
                    return False
        
        # Verificar calidad mínima si hay contenido principal
        if results.get('main_content'):
            quality = calculate_content_quality_score(results['main_content'])
            if quality < rules['quality_threshold']:
                return False
        
        return True
    
    def _create_empty_result(self, url: str, errors: List[str]) -> ExtractedContent:
        """Crear resultado vacío en caso de error"""
        
        return ExtractedContent(
            url=url,
            title="",
            main_content="",
            description="",
            author="",
            publish_date="",
            keywords=[],
            images=[],
            links=[],
            language="unknown",
            content_type="unknown",
            quality_score=0.0,
            word_count=0,
            reading_time=0,
            metadata={},
            extraction_success=False,
            extraction_errors=errors
        )

# Función de conveniencia para uso directo
async def extract_content_from_url(url: str, content_type_hint: str = None) -> ExtractedContent:
    """Función de conveniencia para extraer contenido de una URL"""
    extractor = ContentExtractor()
    return await extractor.extract_content(url, content_type_hint=content_type_hint)

async def extract_content_from_html(url: str, html: str, content_type_hint: str = None) -> ExtractedContent:
    """Función de conveniencia para extraer contenido de HTML"""
    extractor = ContentExtractor()
    return await extractor.extract_content(url, html, content_type_hint)

# Instancia global reutilizable
global_content_extractor = ContentExtractor()