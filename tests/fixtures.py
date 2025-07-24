"""
Test Fixtures and Mock Data
Datos de prueba, HTML samples, respuestas mock y fixtures específicas
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta

# =============================================================================
# HTML SAMPLES PARA CONTENT EXTRACTION TESTING
# =============================================================================

# HTML básico bien formado
SIMPLE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Simple Test Page</title>
    <meta name="description" content="A simple test page for content extraction">
    <meta name="author" content="Test Author">
</head>
<body>
    <h1>Main Article Title</h1>
    <p>This is the first paragraph of the main content.</p>
    <p>This is the second paragraph with more detailed information.</p>
</body>
</html>
"""

# HTML complejo con elementos a remover
COMPLEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Complex Test Page</title>
    <meta name="description" content="Complex page with various elements">
    <meta name="keywords" content="test, html, extraction">
    <meta property="og:title" content="Complex Test Page OG Title">
    <meta property="og:description" content="OpenGraph description">
</head>
<body>
    <header>
        <nav>Navigation menu to be removed</nav>
    </header>
    
    <main>
        <article>
            <h1>Main Article Title</h1>
            <div class="author">By Test Author</div>
            <time datetime="2024-01-15">January 15, 2024</time>
            
            <p>This is the main content paragraph that should be extracted.</p>
            <p>Another paragraph with <a href="http://example.com">a link</a> inside.</p>
            
            <aside class="sidebar">
                Sidebar content that should be removed
            </aside>
            
            <p>Final paragraph of the main content.</p>
        </article>
    </main>
    
    <footer>
        Footer content to be removed
    </footer>
    
    <script>
        // JavaScript to be removed
        console.log("This should not appear");
    </script>
    
    <style>
        /* CSS to be removed */
        body { margin: 0; }
    </style>
</body>
</html>
"""

# HTML específico de Wikipedia
WIKIPEDIA_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>Python (programming language) - Wikipedia</title>
    <meta name="description" content="Python is a high-level programming language">
</head>
<body>
    <div id="mw-content-text">
        <div class="mw-parser-output">
            <h1 id="firstHeading" class="firstHeading">Python (programming language)</h1>
            <p><strong>Python</strong> is a high-level, general-purpose programming language.</p>
            <p>Its design philosophy emphasizes code readability with the use of significant indentation.</p>
            <div class="infobox">
                <tr><td>Paradigm</td><td>Multi-paradigm</td></tr>
            </div>
        </div>
    </div>
</body>
</html>
"""

# HTML específico de GitHub
GITHUB_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>python/cpython: The Python programming language</title>
    <meta name="description" content="The Python programming language repository">
</head>
<body>
    <div class="repository-content">
        <h1 class="entry-title">
            <strong itemprop="name">cpython</strong>
        </h1>
        
        <div id="readme" class="Box md">
            <div class="markdown-body">
                <h1>Python</h1>
                <p>This is the official repository for the Python programming language.</p>
                <p>Python is an interpreted, high-level programming language.</p>
            </div>
        </div>
    </div>
</body>
</html>
"""

# HTML específico de Stack Overflow
STACKOVERFLOW_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <title>How to learn Python programming? - Stack Overflow</title>
</head>
<body>
    <div id="question">
        <h1 itemprop="name">How to learn Python programming?</h1>
        <div class="s-prose">
            <p>I want to learn Python programming. What are the best resources?</p>
            <p>I'm a complete beginner to programming.</p>
        </div>
    </div>
    
    <div class="answer">
        <div class="s-prose">
            <p>Here are some great resources for learning Python:</p>
            <ul>
                <li>Official Python tutorial</li>
                <li>Python.org documentation</li>
                <li>Online courses and books</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

# HTML mal formado para testing de robustez
MALFORMED_HTML = """
<html>
<head>
<title>Malformed Test Page
<body>
<h1>Missing closing tags
<p>This paragraph has no closing tag
<div>Nested content
<p>Another paragraph
<script>alert('malicious script');</script>
<p>Content after script
</html>
"""

# HTML con caracteres especiales y Unicode
UNICODE_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Página de Prueba con Caracteres Especiales</title>
</head>
<body>
    <h1>Título con Acentos y Ñ</h1>
    <p>Este párrafo contiene caracteres especiales: áéíóú ñÑ ¿¡</p>
    <p>También tiene símbolos: © ® ™ € $ £ ¥</p>
    <p>Y emojis: 🐍 🔥 💻 🚀</p>
    <p>Texto en japonés: こんにちは</p>
    <p>Texto en árabe: مرحبا</p>
</body>
</html>
"""

# =============================================================================
# RESPUESTAS MOCK DE MOTORES DE BÚSQUEDA
# =============================================================================

# Respuesta mock de DuckDuckGo (HTML parsing)
DUCKDUCKGO_MOCK_HTML = """
<!DOCTYPE html>
<html>
<body>
    <div class="result">
        <h2><a href="https://www.python.org/">Python.org</a></h2>
        <span class="result__snippet">
            Welcome to Python.org. Python is a programming language that lets you work quickly.
        </span>
    </div>
    
    <div class="result">
        <h2><a href="https://en.wikipedia.org/wiki/Python_(programming_language)">Python (programming language) - Wikipedia</a></h2>
        <span class="result__snippet">
            Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability.
        </span>
    </div>
    
    <div class="result">
        <h2><a href="https://docs.python.org/3/">Python 3 Documentation</a></h2>
        <span class="result__snippet">
            Welcome! This is the documentation for Python 3.12. Parts of the documentation are available.
        </span>
    </div>
</body>
</html>
"""

# Respuesta mock de Google Custom Search (JSON)
GOOGLE_CUSTOM_SEARCH_MOCK = {
    "kind": "customsearch#search",
    "url": {
        "type": "application/json",
        "template": "https://www.googleapis.com/customsearch/v1?q={searchTerms}"
    },
    "queries": {
        "request": [{
            "title": "Google Custom Search - python programming",
            "totalResults": "145000000",
            "searchTerms": "python programming",
            "count": 10,
            "startIndex": 1
        }]
    },
    "items": [
        {
            "kind": "customsearch#result",
            "title": "Welcome to Python.org",
            "htmlTitle": "Welcome to <b>Python</b>.org",
            "link": "https://www.python.org/",
            "displayLink": "www.python.org",
            "snippet": "The official home of the Python Programming Language. Python is a programming language that lets you work quickly and integrate systems more effectively.",
            "htmlSnippet": "The official home of the <b>Python Programming</b> Language. <b>Python</b> is a <b>programming</b> language that lets you work quickly and integrate systems more effectively.",
            "cacheId": "test_cache_id_1"
        },
        {
            "kind": "customsearch#result",
            "title": "Python (programming language) - Wikipedia",
            "htmlTitle": "<b>Python</b> (<b>programming</b> language) - Wikipedia",
            "link": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "displayLink": "en.wikipedia.org",
            "snippet": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            "htmlSnippet": "<b>Python</b> is a high-level, general-purpose <b>programming</b> language. Its design philosophy emphasizes code readability with the use of significant indentation.",
            "cacheId": "test_cache_id_2"
        }
    ]
}

# Respuesta mock de SearX (JSON)
SEARX_MOCK_RESPONSE = {
    "query": "python programming",
    "number_of_results": 50,
    "results": [
        {
            "url": "https://www.python.org/",
            "title": "Welcome to Python.org",
            "content": "The official home of the Python Programming Language. Python is a programming language that lets you work quickly.",
            "engines": ["google", "duckduckgo"],
            "score": 100.0,
            "category": "general"
        },
        {
            "url": "https://docs.python.org/3/tutorial/",
            "title": "The Python Tutorial — Python 3.12 documentation",
            "content": "Python is an easy to learn, powerful programming language. It has efficient high-level data structures.",
            "engines": ["google", "bing"],
            "score": 95.0,
            "category": "general"
        }
    ],
    "answers": [],
    "corrections": [],
    "infoboxes": [],
    "suggestions": ["python programming tutorial", "python programming language"]
}

# Respuesta mock de Bing Search (JSON)
BING_SEARCH_MOCK = {
    "_type": "SearchResponse",
    "queryContext": {
        "originalQuery": "python programming"
    },
    "webPages": {
        "webSearchUrl": "https://www.bing.com/search?q=python+programming",
        "totalEstimatedMatches": 50800000,
        "value": [
            {
                "id": "https://api.cognitive.microsoft.com/api/v7/#WebPages.0",
                "name": "Welcome to Python.org",
                "url": "https://www.python.org/",
                "displayUrl": "https://www.python.org",
                "snippet": "The official home of the Python Programming Language. Python is a programming language that lets you work quickly and integrate systems more effectively.",
                "dateLastCrawled": "2024-01-15T10:30:00.0000000Z"
            },
            {
                "id": "https://api.cognitive.microsoft.com/api/v7/#WebPages.1",
                "name": "Python Tutorial - W3Schools",
                "url": "https://www.w3schools.com/python/",
                "displayUrl": "https://www.w3schools.com/python",
                "snippet": "Well organized and easy to understand Web building tutorials with lots of examples of how to use HTML, CSS, JavaScript, SQL, Python, PHP, Bootstrap, Java, XML and more.",
                "dateLastCrawled": "2024-01-14T15:45:00.0000000Z"
            }
        ]
    }
}

# =============================================================================
# RESPUESTAS MOCK DE OLLAMA
# =============================================================================

# Respuesta mock exitosa de Ollama
OLLAMA_SUCCESS_RESPONSE = {
    "model": "llama3.2:3b",
    "created_at": "2024-01-15T10:00:00.000Z",
    "response": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991. Python supports multiple programming paradigms including procedural, object-oriented, and functional programming.",
    "done": True,
    "context": [1, 2, 3, 4, 5],
    "total_duration": 2500000000,
    "load_duration": 500000000,
    "prompt_eval_count": 25,
    "prompt_eval_duration": 800000000,
    "eval_count": 45,
    "eval_duration": 1200000000
}

# Respuesta mock de error de Ollama
OLLAMA_ERROR_RESPONSE = {
    "error": "model not found",
    "details": "The requested model 'nonexistent:latest' was not found on this server"
}

# Respuesta mock de health check de Ollama
OLLAMA_HEALTH_RESPONSE = {
    "status": "ok",
    "version": "0.1.17"
}

# =============================================================================
# CONTENIDO EXTRAÍDO MOCK
# =============================================================================

# Contenido extraído exitoso
MOCK_EXTRACTED_CONTENT = {
    "main_content": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented and functional programming.",
    "title": "Python (programming language)",
    "description": "Python is a high-level programming language known for its simplicity.",
    "author": "Wikipedia Contributors",
    "publish_date": "2024-01-15",
    "keywords": ["python", "programming", "language", "development"],
    "language": "english",
    "word_count": 156,
    "reading_time": 1,
    "quality_score": 0.85,
    "extraction_success": True,
    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "domain": "wikipedia.org",
    "content_type": "article"
}

# Contenido con fallo de extracción
MOCK_FAILED_EXTRACTION = {
    "main_content": "",
    "title": "",
    "description": "",
    "author": None,
    "publish_date": None,
    "keywords": [],
    "language": "unknown",
    "word_count": 0,
    "reading_time": 0,
    "quality_score": 0.0,
    "extraction_success": False,
    "url": "https://example.com/blocked",
    "domain": "example.com",
    "content_type": "unknown",
    "error": "Content extraction failed: robots.txt disallowed"
}

# =============================================================================
# DATOS DE AGENT TESTING
# =============================================================================

# Respuesta mock del SmartDoc Agent
MOCK_AGENT_RESPONSE = {
    "response": "Based on my research, Python is a versatile programming language that's excellent for beginners and professionals alike. It's known for its clean syntax and extensive library ecosystem.",
    "sources": [
        {
            "type": "web",
            "url": "https://www.python.org/",
            "title": "Welcome to Python.org",
            "relevance": 0.95,
            "snippet": "Official Python website with documentation and downloads."
        },
        {
            "type": "web", 
            "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
            "title": "Python (programming language) - Wikipedia",
            "relevance": 0.90,
            "snippet": "Comprehensive overview of Python programming language."
        }
    ],
    "reasoning": "I searched for information about Python programming and found authoritative sources including the official website and Wikipedia. These sources provide comprehensive information about Python's features and applications.",
    "confidence": 0.92,
    "session_id": "test-session-123",
    "model_used": "llama3.2:3b",
    "processing_time": 2.5,
    "tool_calls": [
        {
            "tool": "web_search",
            "input": "python programming language overview",
            "output": "Found 10 relevant results",
            "duration": 1.8
        }
    ]
}

# Estado de sesión mock
MOCK_SESSION_STATUS = {
    "session_id": "test-session-123",
    "topic": "Python programming",
    "objectives": ["Learn basics", "Understand syntax", "Find resources"],
    "created_at": "2024-01-15T10:00:00Z",
    "message_count": 3,
    "sources_found": 5,
    "last_activity": "2024-01-15T10:15:00Z",
    "status": "active",
    "progress": 0.75
}

# =============================================================================
# ESCENARIOS DE ERROR PARA TESTING
# =============================================================================

# Escenarios de error HTTP
HTTP_ERROR_SCENARIOS = {
    "not_found": {
        "status_code": 404,
        "content": "<html><body><h1>404 Not Found</h1></body></html>",
        "headers": {"content-type": "text/html"}
    },
    "server_error": {
        "status_code": 500,
        "content": "<html><body><h1>500 Internal Server Error</h1></body></html>",
        "headers": {"content-type": "text/html"}
    },
    "timeout": {
        "exception": "TimeoutException",
        "message": "Request timed out after 30 seconds"
    },
    "connection_error": {
        "exception": "ConnectionError",
        "message": "Failed to establish connection"
    }
}

# Escenarios de error de búsqueda
SEARCH_ERROR_SCENARIOS = {
    "empty_results": {
        "query": "askjdhlaksjdhlaksjdh",
        "results": [],
        "error": None,
        "message": "No results found for query"
    },
    "api_quota_exceeded": {
        "query": "test query",
        "error": "quota_exceeded",
        "message": "API quota exceeded for today"
    },
    "blocked_by_robots": {
        "query": "blocked content",
        "error": "robots_blocked",
        "message": "Access blocked by robots.txt"
    }
}

# =============================================================================
# DATOS PARA PERFORMANCE TESTING
# =============================================================================

# Queries de diferentes complejidades para performance testing
PERFORMANCE_TEST_QUERIES = {
    "simple": [
        "python",
        "javascript",
        "html"
    ],
    "medium": [
        "python programming tutorial",
        "machine learning algorithms",
        "web development best practices"
    ],
    "complex": [
        "advanced python programming techniques for data science and machine learning applications",
        "comparison of modern javascript frameworks react vue angular performance benchmarks",
        "comprehensive guide to microservices architecture patterns and implementation strategies"
    ]
}

# Métricas esperadas de performance
PERFORMANCE_BENCHMARKS = {
    "web_search_tool": {
        "simple_query": {"max_time": 5.0, "expected_results": 5},
        "medium_query": {"max_time": 10.0, "expected_results": 8},
        "complex_query": {"max_time": 15.0, "expected_results": 10}
    },
    "content_extraction": {
        "simple_page": {"max_time": 2.0, "min_content_length": 100},
        "complex_page": {"max_time": 5.0, "min_content_length": 500}
    },
    "agent_response": {
        "simple_query": {"max_time": 30.0, "min_response_length": 50},
        "complex_query": {"max_time": 60.0, "min_response_length": 200}
    }
}

# =============================================================================
# FIXTURES DE CONFIGURACIÓN PARA TESTS ESPECÍFICOS
# =============================================================================

# Configuración para tests de rate limiting
RATE_LIMIT_TEST_CONFIG = {
    "requests_per_minute": 5,
    "test_duration": 70,  # segundos
    "expected_blocks": 3,
    "recovery_time": 60
}

# Configuración para tests de concurrencia
CONCURRENCY_TEST_CONFIG = {
    "concurrent_requests": 10,
    "max_response_time": 30.0,
    "min_success_rate": 0.8
}

# =============================================================================
# DATOS PARA INTEGRATION TESTING
# =============================================================================

# Flujo completo de investigación para e2e testing
E2E_RESEARCH_SCENARIO = {
    "topic": "Python web development",
    "objectives": [
        "Learn about popular Python web frameworks",
        "Understand deployment options",
        "Find tutorial resources"
    ],
    "queries": [
        "What are the best Python web frameworks?",
        "How to deploy Python web applications?",
        "Python web development tutorials for beginners"
    ],
    "expected_sources": 15,
    "expected_frameworks": ["Django", "Flask", "FastAPI"],
    "min_response_length": 500
}

# =============================================================================
# UTILITY FUNCTIONS PARA FIXTURES
# =============================================================================

def get_mock_search_results(query: str, engine: str = "mock", count: int = 5):
    """Generar resultados de búsqueda mock basados en query"""
    base_results = [
        {
            "title": f"Result for '{query}' #{i+1}",
            "url": f"https://example{i+1}.com/{query.replace(' ', '-')}",
            "snippet": f"This is mock search result #{i+1} for the query '{query}'. It contains relevant information about the topic.",
            "domain": f"example{i+1}.com", 
            "rank": i+1,
            "source_engine": engine,
            "relevance_score": max(0.5, 1.0 - (i * 0.1))
        }
        for i in range(count)
    ]
    return base_results

def get_mock_html_for_domain(domain: str) -> str:
    """Generar HTML mock específico para un dominio"""
    domain_templates = {
        "wikipedia.org": WIKIPEDIA_HTML,
        "github.com": GITHUB_HTML,
        "stackoverflow.com": STACKOVERFLOW_HTML
    }
    
    return domain_templates.get(domain, SIMPLE_HTML)

def create_mock_agent_session(topic: str = "test topic") -> Dict[str, Any]:
    """Crear sesión mock del agent"""
    return {
        "session_id": f"test-{hash(topic) % 10000}",
        "topic": topic,
        "created_at": datetime.now().isoformat(),
        "messages": [],
        "sources": [],
        "status": "active"
    }

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # HTML Samples
    "SIMPLE_HTML",
    "COMPLEX_HTML", 
    "WIKIPEDIA_HTML",
    "GITHUB_HTML",
    "STACKOVERFLOW_HTML",
    "MALFORMED_HTML",
    "UNICODE_HTML",
    
    # Search Engine Responses
    "DUCKDUCKGO_MOCK_HTML",
    "GOOGLE_CUSTOM_SEARCH_MOCK",
    "SEARX_MOCK_RESPONSE", 
    "BING_SEARCH_MOCK",
    
    # Ollama Responses
    "OLLAMA_SUCCESS_RESPONSE",
    "OLLAMA_ERROR_RESPONSE",
    "OLLAMA_HEALTH_RESPONSE",
    
    # Content Extraction
    "MOCK_EXTRACTED_CONTENT",
    "MOCK_FAILED_EXTRACTION",
    
    # Agent Responses
    "MOCK_AGENT_RESPONSE",
    "MOCK_SESSION_STATUS",
    
    # Error Scenarios
    "HTTP_ERROR_SCENARIOS",
    "SEARCH_ERROR_SCENARIOS",
    
    # Performance Testing
    "PERFORMANCE_TEST_QUERIES",
    "PERFORMANCE_BENCHMARKS",
    "RATE_LIMIT_TEST_CONFIG",
    "CONCURRENCY_TEST_CONFIG",
    
    # Integration Testing
    "E2E_RESEARCH_SCENARIO",
    
    # Utility Functions
    "get_mock_search_results",
    "get_mock_html_for_domain",
    "create_mock_agent_session"
]



# Missing exports
MOCK_RESEARCH_SESSIONS = []
MOCK_SEARCH_RESULTS = []

