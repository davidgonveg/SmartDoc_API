"""
Content Extraction Rules for Web Search Tool
Reglas para extraer contenido limpio y relevante de páginas web
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re

class ContentType(Enum):
    """Tipos de contenido a extraer"""
    MAIN_CONTENT = "main_content"
    TITLE = "title"
    DESCRIPTION = "description"
    KEYWORDS = "keywords"
    AUTHOR = "author"
    DATE = "date"
    LINKS = "links"
    IMAGES = "images"

@dataclass
class ExtractionRule:
    """Regla de extracción para un elemento específico"""
    selectors: List[str]  # CSS selectors en orden de prioridad
    attributes: List[str]  # Atributos a extraer (text, href, src, etc.)
    required: bool = False  # Si es requerido para considerar válida la extracción
    max_length: Optional[int] = None  # Longitud máxima del contenido
    min_length: Optional[int] = None  # Longitud mínima del contenido
    clean_html: bool = True  # Si limpiar tags HTML
    regex_patterns: Optional[List[str]] = None  # Patrones regex para limpiar/validar

# Reglas generales de extracción por tipo de contenido
EXTRACTION_RULES = {
    ContentType.TITLE: ExtractionRule(
        selectors=[
            "title",
            "h1",
            "[property='og:title']",
            "[name='twitter:title']",
            ".article-title",
            ".post-title",
            ".entry-title",
        ],
        attributes=["text", "content"],
        required=True,
        max_length=200,
        min_length=5,
        clean_html=True,
    ),
    
    ContentType.DESCRIPTION: ExtractionRule(
        selectors=[
            "[name='description']",
            "[property='og:description']",
            "[name='twitter:description']",
            ".article-summary",
            ".post-excerpt",
            ".entry-summary",
            "p:first-of-type",
        ],
        attributes=["content", "text"],
        required=False,
        max_length=500,
        min_length=20,
        clean_html=True,
    ),
    
    ContentType.MAIN_CONTENT: ExtractionRule(
        selectors=[
            "article",
            "[role='main']",
            ".article-content",
            ".post-content",
            ".entry-content",
            ".content",
            "main",
            "#content",
            "#main",
            ".main-content",
            "[itemprop='articleBody']",
        ],
        attributes=["text"],
        required=True,
        max_length=50000,
        min_length=100,
        clean_html=True,
    ),
    
    ContentType.AUTHOR: ExtractionRule(
        selectors=[
            "[name='author']",
            "[rel='author']",
            "[itemprop='author']",
            ".author",
            ".byline",
            ".article-author",
            ".post-author",
        ],
        attributes=["content", "text"],
        required=False,
        max_length=100,
        min_length=2,
        clean_html=True,
    ),
    
    ContentType.DATE: ExtractionRule(
        selectors=[
            "[property='article:published_time']",
            "[name='publishdate']",
            "[itemprop='datePublished']",
            "time[datetime]",
            ".publish-date",
            ".article-date",
            ".post-date",
        ],
        attributes=["content", "datetime", "text"],
        required=False,
        max_length=50,
        min_length=4,
        clean_html=True,
        regex_patterns=[
            r'\d{4}-\d{2}-\d{2}',  # ISO date format
            r'\d{1,2}/\d{1,2}/\d{4}',  # MM/DD/YYYY
            r'\w+ \d{1,2}, \d{4}',  # Month DD, YYYY
        ],
    ),
    
    ContentType.KEYWORDS: ExtractionRule(
        selectors=[
            "[name='keywords']",
            "[property='article:tag']",
            ".tags",
            ".keywords",
            ".article-tags",
        ],
        attributes=["content", "text"],
        required=False,
        max_length=500,
        min_length=3,
        clean_html=True,
    ),
}

# Reglas específicas por dominio/sitio web
DOMAIN_SPECIFIC_RULES = {
    "wikipedia.org": {
        ContentType.MAIN_CONTENT: ExtractionRule(
            selectors=["#mw-content-text", ".mw-parser-output"],
            attributes=["text"],
            required=True,
            max_length=100000,
            min_length=200,
            clean_html=True,
        ),
        ContentType.TITLE: ExtractionRule(
            selectors=["#firstHeading", "h1.firstHeading"],
            attributes=["text"],
            required=True,
            max_length=200,
            min_length=5,
            clean_html=True,
        ),
    },
    
    "github.com": {
        ContentType.MAIN_CONTENT: ExtractionRule(
            selectors=[
                ".repository-content",
                "#readme",
                ".markdown-body",
                ".blob-wrapper",
            ],
            attributes=["text"],
            required=True,
            max_length=50000,
            min_length=50,
            clean_html=True,
        ),
        ContentType.TITLE: ExtractionRule(
            selectors=[".entry-title", "h1", ".repository-name"],
            attributes=["text"],
            required=True,
            max_length=200,
            min_length=3,
            clean_html=True,
        ),
    },
    
    "stackoverflow.com": {
        ContentType.MAIN_CONTENT: ExtractionRule(
            selectors=[
                ".question .s-prose",
                ".answer .s-prose",
                ".post-text",
                "#question",
                ".answercell",
            ],
            attributes=["text"],
            required=True,
            max_length=20000,
            min_length=50,
            clean_html=True,
        ),
        ContentType.TITLE: ExtractionRule(
            selectors=["h1[itemprop='name']", ".question-hyperlink"],
            attributes=["text"],
            required=True,
            max_length=300,
            min_length=10,
            clean_html=True,
        ),
    },
    
    "medium.com": {
        ContentType.MAIN_CONTENT: ExtractionRule(
            selectors=[
                "[data-testid='storyContent']",
                ".postArticle-content",
                "article section",
            ],
            attributes=["text"],
            required=True,
            max_length=50000,
            min_length=200,
            clean_html=True,
        ),
        ContentType.TITLE: ExtractionRule(
            selectors=["h1", "[data-testid='storyTitle']"],
            attributes=["text"],
            required=True,
            max_length=200,
            min_length=10,
            clean_html=True,
        ),
    },
    
    "arxiv.org": {
        ContentType.MAIN_CONTENT: ExtractionRule(
            selectors=[
                ".abstract",
                "#abs",
                ".abstract-full",
            ],
            attributes=["text"],
            required=True,
            max_length=5000,
            min_length=100,
            clean_html=True,
        ),
        ContentType.TITLE: ExtractionRule(
            selectors=[".title", ".descriptor"],
            attributes=["text"],
            required=True,
            max_length=300,
            min_length=10,
            clean_html=True,
        ),
    },
}

# Elementos a remover antes de la extracción
REMOVE_ELEMENTS = {
    "tags": [
        "script", "style", "nav", "header", "footer", "aside",
        "advertisement", "ads", "social-share", "comments",
        "related-articles", "sidebar", "menu", "breadcrumb",
    ],
    
    "classes": [
        "advertisement", "ads", "ad", "social", "share",
        "comment", "comments", "sidebar", "nav", "navigation",
        "menu", "footer", "header", "breadcrumb", "related",
        "popup", "modal", "overlay", "cookie", "gdpr",
    ],
    
    "ids": [
        "ads", "advertisement", "sidebar", "nav", "navigation",
        "footer", "header", "comments", "social", "share",
        "popup", "modal", "cookie-notice",
    ],
}

# Patrones de contenido inválido
INVALID_CONTENT_PATTERNS = [
    r"404 not found",
    r"page not found",
    r"access denied",
    r"javascript required",
    r"cookies required",
    r"please enable javascript",
    r"this site requires cookies",
    r"captcha",
    r"robot",
    r"bot detection",
]

# Configuración de limpieza de texto
TEXT_CLEANING_CONFIG = {
    "remove_extra_whitespace": True,
    "remove_empty_lines": True,
    "normalize_unicode": True,
    "remove_urls": False,  # Mantener URLs pueden ser útiles
    "remove_emails": False,
    "min_sentence_length": 10,
    "max_sentence_length": 1000,
    
    # Patrones regex para limpiar
    "cleanup_patterns": [
        (r'\s+', ' '),  # Multiple spaces to single space
        (r'\n\s*\n\s*\n', '\n\n'),  # Multiple newlines to double
        (r'^\s+|\s+$', ''),  # Trim whitespace
        (r'[^\x00-\x7F]+', ''),  # Remove non-ASCII (opcional)
    ],
    
    # Patrones para detectar contenido de baja calidad
    "low_quality_patterns": [
        r'^(click here|read more|continue reading)\.?$',
        r'^(home|menu|search|login|register)$',
        r'^\d+$',  # Solo números
        r'^[^\w\s]+$',  # Solo símbolos
    ],
}

# Configuración de scoring de calidad del contenido
CONTENT_QUALITY_CONFIG = {
    "min_word_count": 50,
    "max_word_count": 10000,
    "min_sentence_count": 3,
    
    # Factores de scoring (0-1, mayor es mejor)
    "scoring_factors": {
        "length_score_weight": 0.2,  # Penalizar muy corto/largo
        "sentence_structure_weight": 0.3,  # Calidad de oraciones
        "readability_weight": 0.2,  # Facilidad de lectura
        "uniqueness_weight": 0.3,  # Contenido único vs boilerplate
    },
    
    # Palabras que indican contenido de calidad
    "quality_indicators": [
        "research", "study", "analysis", "data", "evidence",
        "conclusion", "methodology", "results", "findings",
        "according", "however", "therefore", "furthermore",
    ],
    
    # Palabras que indican contenido de baja calidad
    "spam_indicators": [
        "click here", "buy now", "limited time", "act fast",
        "amazing deal", "free download", "make money",
        "lose weight", "miracle", "secret",
    ],
}

def get_extraction_rule(content_type: ContentType, domain: Optional[str] = None) -> ExtractionRule:
    """Obtener regla de extracción para un tipo de contenido y dominio"""
    # Verificar si hay regla específica del dominio
    if domain and domain in DOMAIN_SPECIFIC_RULES:
        domain_rules = DOMAIN_SPECIFIC_RULES[domain]
        if content_type in domain_rules:
            return domain_rules[content_type]
    
    # Retornar regla general
    return EXTRACTION_RULES.get(content_type)

def get_domain_from_url(url: str) -> str:
    """Extraer dominio de una URL"""
    import re
    match = re.search(r'https?://(?:www\.)?([^/]+)', url)
    return match.group(1) if match else ""

def is_content_valid(content: str) -> bool:
    """Verificar si el contenido extraído es válido"""
    if not content or len(content.strip()) < 10:
        return False
    
    content_lower = content.lower()
    
    # Verificar patrones de contenido inválido
    for pattern in INVALID_CONTENT_PATTERNS:
        if re.search(pattern, content_lower):
            return False
    
    return True

def calculate_content_quality_score(content: str) -> float:
    """Calcular score de calidad del contenido (0-1)"""
    if not content:
        return 0.0
    
    config = CONTENT_QUALITY_CONFIG
    words = content.split()
    sentences = content.split('.')
    
    # Length score
    word_count = len(words)
    if word_count < config["min_word_count"]:
        length_score = word_count / config["min_word_count"]
    elif word_count > config["max_word_count"]:
        length_score = config["max_word_count"] / word_count
    else:
        length_score = 1.0
    
    # Sentence structure score
    avg_sentence_length = word_count / max(len(sentences), 1)
    sentence_score = min(avg_sentence_length / 20, 1.0)  # Optimal ~20 words/sentence
    
    # Quality indicators
    content_lower = content.lower()
    quality_count = sum(1 for word in config["quality_indicators"] if word in content_lower)
    spam_count = sum(1 for word in config["spam_indicators"] if word in content_lower)
    
    uniqueness_score = max(0, (quality_count - spam_count * 2) / max(quality_count + spam_count, 1))
    
    # Readability (simple heuristic)
    readability_score = min(len(sentences) / max(word_count / 100, 1), 1.0)
    
    # Combine scores
    factors = config["scoring_factors"]
    total_score = (
        length_score * factors["length_score_weight"] +
        sentence_score * factors["sentence_structure_weight"] +
        readability_score * factors["readability_weight"] +
        uniqueness_score * factors["uniqueness_weight"]
    )
    
    return min(max(total_score, 0.0), 1.0)

# Configuración específica para tipos de páginas
PAGE_TYPE_RULES = {
    "article": {
        "required_elements": [ContentType.TITLE, ContentType.MAIN_CONTENT],
        "optional_elements": [ContentType.AUTHOR, ContentType.DATE, ContentType.DESCRIPTION],
        "min_content_length": 200,
        "quality_threshold": 0.6,
    },
    
    "documentation": {
        "required_elements": [ContentType.TITLE, ContentType.MAIN_CONTENT],
        "optional_elements": [ContentType.DESCRIPTION],
        "min_content_length": 100,
        "quality_threshold": 0.5,
    },
    
    "forum_post": {
        "required_elements": [ContentType.TITLE, ContentType.MAIN_CONTENT],
        "optional_elements": [ContentType.AUTHOR, ContentType.DATE],
        "min_content_length": 50,
        "quality_threshold": 0.4,
    },
    
    "academic": {
        "required_elements": [ContentType.TITLE, ContentType.MAIN_CONTENT],
        "optional_elements": [ContentType.AUTHOR, ContentType.DATE, ContentType.KEYWORDS],
        "min_content_length": 300,
        "quality_threshold": 0.7,
    },
}