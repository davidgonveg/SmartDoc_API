"""
Datos centralizados y fixtures para todos los tests de SmartDoc

Este módulo contiene todos los datos de prueba, respuestas mock, 
configuraciones y utilities compartidos entre todos los tests.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# CONFIGURACIONES BASE PARA TESTS
# =============================================================================

class TestEnvironment(Enum):
    """Entornos de test disponibles"""
    UNIT = "unit"
    INTEGRATION = "integration"  
    E2E = "e2e"
    PERFORMANCE = "performance"

class ModelType(Enum):
    """Tipos de modelo para testing"""
    MOCK = "mock_model"
    LOCAL = "llama3.1:8b"
    FALLBACK = "gpt-3.5-turbo"

# Configuración global de tests
TEST_CONFIG = {
    "default_timeout": 30.0,
    "slow_test_timeout": 120.0,
    "max_concurrent_sessions": 10,
    "max_session_messages": 100,
    "default_model": ModelType.MOCK.value,
    "enable_real_api_tests": False,
    "enable_gpu_tests": False
}

# URLs y endpoints para testing
TEST_ENDPOINTS = {
    "local_api": "http://localhost:8001",
    "local_ui": "http://localhost:8501", 
    "ollama": "http://localhost:11434",
    "chromadb": "http://localhost:8000"
}


# =============================================================================
# DATOS DE INVESTIGACIÓN Y TOPICS PARA TESTS
# =============================================================================

# Topics diversos para diferentes tipos de tests
RESEARCH_TOPICS = {
    "simple": [
        "Python programming",
        "JavaScript basics", 
        "HTML fundamentals",
        "CSS styling",
        "Git version control"
    ],
    "intermediate": [
        "Machine learning algorithms",
        "Web development best practices",
        "Database design patterns",
        "API development with FastAPI",
        "Docker containerization"
    ],
    "complex": [
        "Advanced neural network architectures for computer vision",
        "Microservices architecture patterns and implementation strategies",
        "Distributed systems design for high-scale applications",
        "Quantum computing applications in cryptography",
        "Blockchain consensus mechanisms and scalability solutions"
    ],
    "academic": [
        "Recent advances in transformer architecture research",
        "Meta-learning approaches in few-shot learning",
        "Federated learning privacy preservation techniques",
        "Graph neural networks for molecular property prediction",
        "Causal inference methods in observational data"
    ],
    "business": [
        "AI startup funding trends 2024",
        "Digital transformation strategies for traditional industries",
        "Competitive analysis of cloud computing providers",
        "Market penetration strategies for SaaS products",
        "Customer retention optimization in e-commerce"
    ]
}

# Objetivos típicos por tipo de investigación
RESEARCH_OBJECTIVES = {
    "learning": [
        "Understand fundamental concepts",
        "Find practical examples and tutorials",
        "Identify common patterns and best practices",
        "Discover tools and resources"
    ],
    "analysis": [
        "Compare different approaches and solutions",
        "Analyze pros and cons of each option",
        "Identify trends and patterns in the market",
        "Generate insights and recommendations"
    ],
    "implementation": [
        "Find step-by-step implementation guides",
        "Discover code examples and templates",
        "Identify potential challenges and solutions",
        "Locate relevant documentation and APIs"
    ],
    "research": [
        "Review recent academic literature",
        "Identify key researchers and institutions",
        "Understand current state of the art",
        "Find gaps in existing knowledge"
    ]
}


# =============================================================================
# RESPUESTAS MOCK Y CONTENIDO SIMULADO
# =============================================================================

# Respuestas de LLM por tipo de query
MOCK_LLM_RESPONSES = {
    "planning": {
        "thought": "I need to search for information about this topic to provide a comprehensive answer.",
        "action": "web_search",
        "action_input": {"query": "machine learning algorithms 2024"},
        "observation": "[This will be filled by the web search tool]",
        "final_thought": "Based on the search results, I can now provide a detailed response."
    },
    "search_response": {
        "simple": "Based on my search, here are the key points about {topic}: {content}",
        "detailed": """Based on my comprehensive research, here's what I found about {topic}:

## Key Findings:
1. **Current State**: {finding1}
2. **Recent Developments**: {finding2}  
3. **Best Practices**: {finding3}

## Sources Consulted:
- Academic papers and research
- Industry reports and analysis
- Expert opinions and case studies

## Recommendations:
Based on this research, I recommend {recommendations}.""",
        "academic": """## Literature Review: {topic}

### Abstract
This research summary provides an overview of recent developments in {topic} based on current academic literature and industry reports.

### Key Findings
1. **Theoretical Advances**: {theory}
2. **Empirical Results**: {results}
3. **Practical Applications**: {applications}

### Methodology
The research was conducted using systematic search across multiple academic databases and industry sources.

### Conclusions
{conclusions}

### References
[Detailed source list would be included here]"""
    },
    "context_aware": {
        "first_response": "I'll start by researching the fundamental concepts of {topic}.",
        "follow_up": "Building on what we discussed about {previous_topic}, let me explore {new_aspect}.",
        "summary": "Based on our conversation covering {topics}, here's a comprehensive summary of what we've learned: {summary_content}"
    },
    "error_handling": {
        "search_failed": "I encountered an issue with the web search, but let me try an alternative approach.",
        "timeout": "The search is taking longer than expected. Let me provide what I can based on my existing knowledge.",
        "no_results": "I couldn't find specific information about that topic. Could you try rephrasing the question?",
        "recovery": "I've resolved the previous issue and can now continue with your research."
    }
}

# Contenido web simulado para diferentes tipos de páginas
MOCK_WEB_CONTENT = {
    "tech_blog": {
        "title": "Latest Developments in {topic}",
        "content": """
        # Latest Developments in {topic}

        ## Introduction
        The field of {topic} has seen significant advances in recent months. This article explores the key developments and their implications.

        ## Key Developments
        1. **Breakthrough Research**: Recent studies have shown promising results in {area1}.
        2. **Industry Applications**: Major tech companies are implementing {topic} in {use_case}.
        3. **Open Source Tools**: New frameworks and libraries have been released for {tool_type}.

        ## Future Outlook
        Experts predict that {topic} will continue to evolve rapidly, with potential applications in {future_areas}.

        ## Conclusion
        The current state of {topic} shows great promise for {conclusion}.
        """,
        "metadata": {
            "author": "Tech Expert",
            "date": "2024-01-15",
            "reading_time": "5 minutes",
            "tags": ["technology", "research", "innovation"]
        }
    },
    "academic_paper": {
        "title": "A Comprehensive Study of {topic}: Methods and Applications",
        "content": """
        ## Abstract
        This paper presents a comprehensive analysis of {topic}, examining current methodologies and practical applications. Our research demonstrates {key_finding}.

        ## 1. Introduction
        The study of {topic} has gained significant attention in recent years due to {motivation}.

        ## 2. Methodology
        We employed {method} to analyze {dataset} and evaluate {metrics}.

        ## 3. Results
        Our experiments show that {result1} and {result2}, indicating {significance}.

        ## 4. Discussion
        These findings suggest that {implication1} and have practical applications in {domain}.

        ## 5. Conclusion
        In conclusion, {topic} shows great potential for {future_work}.

        ## References
        [1] Author et al. (2023). "Related Work in {topic}." Journal of {field}.
        [2] Researcher et al. (2024). "Advanced Methods for {application}." Conference Proceedings.
        """,
        "metadata": {
            "authors": ["Dr. Research Lead", "PhD Student"],
            "journal": "Journal of Advanced Technology",
            "year": 2024,
            "citations": 45,
            "doi": "10.1234/example.2024.001"
        }
    },
    "tutorial": {
        "title": "Complete Guide to {topic}: From Beginner to Advanced",
        "content": """
        # Complete Guide to {topic}

        ## Prerequisites
        Before starting this tutorial, you should have basic knowledge of {prereq}.

        ## Chapter 1: Getting Started
        Let's begin with the fundamentals of {topic}.

        ### What is {topic}?
        {topic} is {definition} and is commonly used for {purpose}.

        ### Installation and Setup
        ```bash
        pip install {package}
        ```

        ## Chapter 2: Basic Concepts
        Understanding the core concepts is essential for mastering {topic}.

        ### Key Concepts
        1. **Concept 1**: {explanation1}
        2. **Concept 2**: {explanation2}
        3. **Concept 3**: {explanation3}

        ## Chapter 3: Practical Examples
        Let's explore some hands-on examples.

        ### Example 1: Basic Implementation
        ```python
        # Example code for {topic}
        import {module}

        def example_function():
            return "Hello, {topic}!"
        ```

        ## Chapter 4: Advanced Topics
        For more experienced users, here are advanced techniques.

        ## Conclusion
        You now have a solid foundation in {topic}. Continue practicing with real projects!
        """,
        "metadata": {
            "difficulty": "Beginner to Advanced",
            "duration": "2-3 hours",
            "last_updated": "2024-01-20",
            "prerequisites": ["Basic programming", "Command line"]
        }
    },
    "news_article": {
        "title": "Breaking: Major Breakthrough in {topic} Research",
        "content": """
        # Breaking: Major Breakthrough in {topic} Research

        **Published**: January 25, 2024 | **Updated**: 2 hours ago

        ## Summary
        Researchers at {institution} have announced a significant breakthrough in {topic} that could revolutionize {industry}.

        ## Key Details
        - **Discovery**: {discovery_detail}
        - **Impact**: Expected to improve {metric} by {percentage}%
        - **Timeline**: Commercial applications expected by {year}
        - **Investment**: $50M in additional funding secured

        ## Expert Opinions
        > "This breakthrough represents a paradigm shift in how we approach {topic}." 
        > — Dr. Expert, {institution}

        ## Market Impact
        Stock prices for companies in the {sector} sector rose by an average of 8% following the announcement.

        ## What's Next
        The research team plans to {next_steps} and begin trials in Q2 2024.

        ## Related Stories
        - [Previous research in {topic}]
        - [Industry analysis of {sector}]
        - [Investment trends in {field}]
        """,
        "metadata": {
            "source": "Tech News Daily",
            "reporter": "Science Correspondent",
            "category": "Technology",
            "breaking": True,
            "shares": 1247,
            "comments": 89
        }
    }
}

# Resultados de búsqueda simulados
MOCK_SEARCH_RESULTS = {
    "general": [
        {
            "title": "Introduction to {topic} - Complete Guide",
            "url": "https://techguide.com/{topic}-guide",
            "snippet": "Learn everything about {topic} with this comprehensive guide covering basics, advanced concepts, and practical examples.",
            "type": "tutorial",
            "relevance": 0.95
        },
        {
            "title": "Latest Research in {topic} (2024)",
            "url": "https://research.edu/papers/{topic}-2024",
            "snippet": "Recent academic research exploring new developments and applications in {topic} with experimental results.",
            "type": "academic",
            "relevance": 0.89
        },
        {
            "title": "{topic} Best Practices and Tools",
            "url": "https://devtools.io/{topic}-best-practices",
            "snippet": "Industry best practices, recommended tools, and common patterns for implementing {topic} in production.",
            "type": "technical",
            "relevance": 0.87
        },
        {
            "title": "Case Studies: {topic} in Industry",
            "url": "https://business.com/{topic}-case-studies",
            "snippet": "Real-world case studies showing how major companies successfully implemented {topic} solutions.",
            "type": "business",
            "relevance": 0.82
        },
        {
            "title": "{topic} News and Updates",
            "url": "https://technews.com/{topic}-updates",
            "snippet": "Latest news, updates, and announcements related to {topic} from industry leaders and research institutions.",
            "type": "news",
            "relevance": 0.78
        }
    ],
    "academic": [
        {
            "title": "A Survey of {topic}: Methods and Applications",
            "url": "https://arxiv.org/abs/2024.{topic}",
            "snippet": "Comprehensive survey paper reviewing current state-of-the-art methods in {topic} with comparative analysis.",
            "type": "paper",
            "relevance": 0.96,
            "citations": 156,
            "year": 2024
        },
        {
            "title": "Novel Approaches to {topic}: Experimental Results",
            "url": "https://journals.ai/{topic}-novel-approaches",
            "snippet": "Experimental study presenting novel approaches to {topic} with empirical validation and performance metrics.",
            "type": "paper",
            "relevance": 0.93,
            "citations": 89,
            "year": 2024
        },
        {
            "title": "Theoretical Foundations of {topic}",
            "url": "https://mathjournal.org/{topic}-theory",
            "snippet": "Mathematical foundations and theoretical analysis of {topic} with formal proofs and algorithmic complexity.",
            "type": "paper",
            "relevance": 0.85,
            "citations": 234,
            "year": 2023
        }
    ],
    "technical": [
        {
            "title": "GitHub - {topic} Implementation Examples",
            "url": "https://github.com/examples/{topic}",
            "snippet": "Open source implementation examples and code repositories for {topic} with documentation and tutorials.",
            "type": "repository",
            "relevance": 0.91,
            "stars": 2847,
            "language": "Python"
        },
        {
            "title": "Stack Overflow - {topic} Questions and Answers",
            "url": "https://stackoverflow.com/questions/tagged/{topic}",
            "snippet": "Community questions, answers, and discussions about {topic} implementation challenges and solutions.",
            "type": "forum",
            "relevance": 0.88,
            "answers": 156,
            "votes": 342
        },
        {
            "title": "API Documentation - {topic} SDK",
            "url": "https://docs.api.com/{topic}",
            "snippet": "Official API documentation for {topic} SDK with code examples, tutorials, and integration guides.",
            "type": "documentation",
            "relevance": 0.86,
            "version": "v2.1",
            "updated": "2024-01-15"
        }
    ]
}


# =============================================================================
# CONFIGURACIONES DE SESIONES DE INVESTIGACIÓN
# =============================================================================

@dataclass
class SessionConfig:
    """Configuración para sesiones de investigación de prueba"""
    topic: str
    objectives: List[str]
    research_depth: str = "standard"
    research_style: str = "casual"
    max_sources: int = 10
    expected_duration: int = 300  # segundos
    complexity_level: str = "intermediate"

# Configuraciones predefinidas para diferentes tipos de tests
SAMPLE_SESSION_CONFIGS = {
    "basic_learning": SessionConfig(
        topic="Python Programming Basics",
        objectives=[
            "Learn Python syntax and data types",
            "Understand control structures and functions",
            "Find beginner-friendly tutorials and resources"
        ],
        research_depth="standard",
        complexity_level="beginner"
    ),
    "technical_deep_dive": SessionConfig(
        topic="Advanced Docker Container Orchestration",
        objectives=[
            "Understand Kubernetes vs Docker Swarm trade-offs",
            "Learn container security best practices",
            "Find production deployment strategies",
            "Research monitoring and logging solutions"
        ],
        research_depth="deep",
        research_style="technical",
        max_sources=20,
        complexity_level="advanced"
    ),
    "business_analysis": SessionConfig(
        topic="AI Startup Market Analysis 2024",
        objectives=[
            "Identify top AI startups by funding received",
            "Analyze market trends and investor preferences", 
            "Understand competitive landscape",
            "Generate investment insights and recommendations"
        ],
        research_depth="deep",
        research_style="business",
        max_sources=25,
        complexity_level="expert"
    ),
    "quick_reference": SessionConfig(
        topic="JavaScript Array Methods",
        objectives=[
            "Get quick reference for common array methods",
            "Find code examples and use cases"
        ],
        research_depth="quick",
        max_sources=5,
        expected_duration=60,
        complexity_level="beginner"
    ),
    "academic_research": SessionConfig(
        topic="Transformer Architecture Innovations 2024",
        objectives=[
            "Review latest academic papers on transformer improvements",
            "Understand efficiency and scaling advances",
            "Identify key research groups and institutions",
            "Compile comprehensive literature review"
        ],
        research_depth="deep",
        research_style="academic",
        max_sources=30,
        complexity_level="expert"
    )
}


# =============================================================================
# DATOS PARA TESTS DE PERFORMANCE Y STRESS
# =============================================================================

# Configuraciones de performance testing
PERFORMANCE_TEST_CONFIG = {
    "response_time_limits": {
        "simple_query": 5.0,      # segundos
        "medium_query": 15.0,
        "complex_query": 30.0,
        "academic_query": 45.0
    },
    "concurrency_limits": {
        "max_concurrent_sessions": 10,
        "max_concurrent_queries": 5,
        "max_session_messages": 100
    },
    "memory_limits": {
        "max_session_memory_mb": 50,
        "max_total_memory_mb": 500,
        "session_cleanup_threshold": 1000
    },
    "rate_limits": {
        "queries_per_minute": 30,
        "web_searches_per_hour": 100,
        "api_calls_per_day": 1000
    }
}

# Queries para stress testing por complejidad
STRESS_TEST_QUERIES = {
    "simple": [
        "What is Python?",
        "How to install Node.js?",
        "Basic Git commands",
        "HTML table syntax",
        "CSS flexbox basics"
    ] * 10,  # 50 queries simples
    
    "medium": [
        "Compare React vs Vue.js for modern web development",
        "Explain machine learning classification algorithms with examples",
        "Best practices for REST API design and implementation",
        "Docker vs virtual machines: comprehensive comparison",
        "Database indexing strategies for query optimization"
    ] * 5,   # 25 queries medianas
    
    "complex": [
        "Comprehensive analysis of microservices architecture patterns including event sourcing, CQRS, and saga patterns with real-world implementation examples and trade-offs",
        "Deep dive into transformer architecture innovations from 2020-2024 including efficiency improvements, scaling strategies, and novel attention mechanisms",
        "Complete guide to implementing secure multi-tenant SaaS applications with data isolation, authentication, authorization, and compliance considerations"
    ] * 2    # 6 queries complejas
}

# Escenarios de error para testing de robustez
ERROR_SCENARIOS = {
    "network_errors": [
        {"type": "timeout", "probability": 0.1, "delay": 30},
        {"type": "connection_refused", "probability": 0.05},
        {"type": "dns_resolution", "probability": 0.02},
        {"type": "rate_limited", "probability": 0.15}
    ],
    "api_errors": [
        {"type": "quota_exceeded", "probability": 0.08},
        {"type": "invalid_request", "probability": 0.03},
        {"type": "server_error", "probability": 0.05},
        {"type": "authentication_failed", "probability": 0.02}
    ],
    "content_errors": [
        {"type": "empty_results", "probability": 0.12},
        {"type": "malformed_html", "probability": 0.07},
        {"type": "blocked_content", "probability": 0.04},
        {"type": "paywall_detected", "probability": 0.09}
    ]
}


# =============================================================================
# UTILITIES PARA GENERACIÓN DE DATOS DE PRUEBA
# =============================================================================

def generate_mock_session_id() -> str:
    """Generar ID de sesión mock para tests"""
    return f"test_session_{uuid.uuid4().hex[:8]}"

def generate_mock_research_topic(complexity: str = "intermediate") -> str:
    """Generar topic de investigación mock basado en complejidad"""
    topics = RESEARCH_TOPICS.get(complexity, RESEARCH_TOPICS["intermediate"])
    import random
    return random.choice(topics)

def generate_mock_objectives(topic: str, count: int = 3) -> List[str]:
    """Generar objetivos mock para un topic dado"""
    objective_templates = [
        f"Learn about {topic} fundamentals and concepts",
        f"Find practical examples and use cases for {topic}",
        f"Understand best practices and common patterns in {topic}",
        f"Discover tools and resources related to {topic}",
        f"Compare different approaches to implementing {topic}",
        f"Analyze recent developments and trends in {topic}"
    ]
    return objective_templates[:count]

def create_mock_web_results(topic: str, result_type: str = "general", count: int = 5) -> List[Dict]:
    """Crear resultados de búsqueda web mock"""
    template_results = MOCK_SEARCH_RESULTS.get(result_type, MOCK_SEARCH_RESULTS["general"])
    
    results = []
    for i, template in enumerate(template_results[:count]):
        result = template.copy()
        result["title"] = template["title"].format(topic=topic)
        result["url"] = template["url"].format(topic=topic.lower().replace(" ", "-"))
        result["snippet"] = template["snippet"].format(topic=topic)
        results.append(result)
    
    return results

def create_mock_llm_response(response_type: str, **kwargs) -> Dict[str, Any]:
    """Crear respuesta mock del LLM"""
    if response_type in MOCK_LLM_RESPONSES:
        template = MOCK_LLM_RESPONSES[response_type]
        
        if isinstance(template, dict):
            response = {}
            for key, value in template.items():
                if isinstance(value, str):
                    response[key] = value.format(**kwargs)
                else:
                    response[key] = value
            return response
        elif isinstance(template, str):
            return {"response": template.format(**kwargs)}
    
    return {"response": f"Mock response for {response_type}"}

def create_performance_benchmark(test_type: str) -> Dict[str, float]:
    """Crear benchmark de performance para un tipo de test"""
    base_benchmarks = {
        "simple": {"max_time": 5.0, "target_time": 2.0},
        "medium": {"max_time": 15.0, "target_time": 8.0},
        "complex": {"max_time": 30.0, "target_time": 20.0},
        "academic": {"max_time": 45.0, "target_time": 30.0}
    }
    
    return base_benchmarks.get(test_type, base_benchmarks["medium"])


# =============================================================================
# VALIDADORES Y ASSERTIONS HELPERS
# =============================================================================

def validate_session_response(response: Dict[str, Any]) -> bool:
    """Validar que una respuesta de sesión tiene la estructura correcta"""
    required_fields = ["success", "response", "session_id"]
    return all(field in response for field in required_fields)

def validate_search_results(results: List[Dict]) -> bool:
    """Validar que los resultados de búsqueda tienen estructura correcta"""
    if not results:
        return True  # Empty results can be valid
    
    required_fields = ["title", "url", "snippet"]
    return all(
        all(field in result for field in required_fields)
        for result in results
    )

def validate_session_status(status: Dict[str, Any]) -> bool:
    """Validar que el estado de sesión tiene estructura correcta"""
    required_fields = ["topic", "objectives", "message_count"]
    return all(field in status for field in required_fields)


# =============================================================================
# CONSTANTES Y CONFIGURACIONES FINALES
# =============================================================================

# Tiempos de espera para diferentes tipos de operaciones
TIMEOUTS = {
    "session_creation": 5.0,
    "simple_query": 10.0,
    "complex_query": 30.0,
    "web_search": 15.0,
    "llm_response": 20.0,
    "full_workflow": 60.0
}

# Límites para validación de respuestas
RESPONSE_LIMITS = {
    "min_response_length": 20,
    "max_response_length": 10000,
    "min_search_results": 0,
    "max_search_results": 50,
    "min_session_objectives": 1,
    "max_session_objectives": 10
}

# Flags para diferentes tipos de tests
TEST_FLAGS = {
    "run_slow_tests": False,
    "run_external_api_tests": False,
    "run_gpu_tests": False,
    "run_stress_tests": False,
    "enable_detailed_logging": True,
    "save_test_artifacts": False
}


if __name__ == "__main__":
    # Verificación básica de los datos
    print("Test data validation:")
    print(f"✅ {len(RESEARCH_TOPICS)} topic categories loaded")
    print(f"✅ {len(MOCK_LLM_RESPONSES)} LLM response templates loaded")
    print(f"✅ {len(MOCK_WEB_CONTENT)} web content templates loaded")
    print(f"✅ {len(SAMPLE_SESSION_CONFIGS)} session configurations loaded")
    print(f"✅ Performance config: {len(PERFORMANCE_TEST_CONFIG)} categories")
    print("All test data loaded successfully!")