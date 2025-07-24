"""
Test Data Constants for SmartDoc Research Agent
"""

from enum import Enum
from typing import Dict, List, Any

# Performance benchmarks
PERFORMANCE_BENCHMARKS = {
    "agent_response": {
        "simple_query": {"max_time": 30.0},
        "complex_query": {"max_time": 60.0},
        "concurrent_queries": {"max_time": 120.0}
    },
    "web_search": {
        "single_query": {"max_time": 10.0},
        "multiple_queries": {"max_time": 25.0}
    },
    "session_management": {
        "create_session": {"max_time": 5.0},
        "delete_session": {"max_time": 2.0}
    }
}

class TestEnvironment(Enum):
    """Test environment enumeration"""
    UNIT = "unit"
    INTEGRATION = "integration"
    E2E = "e2e"

# Test constants
TEST_CONSTANTS = {
    "timeout": 30,
    "max_retries": 3,
    "test_model": "llama3.2:3b",
    "default_session_id": "test-session-12345"
}

# Mock responses for testing
MOCK_RESPONSES = {
    "simple_query": "This is a simple test response.",
    "complex_query": "This is a more complex test response with multiple sentences and detailed information.",
    "error_response": "An error occurred during processing."
}

# Test queries
TEST_QUERIES = {
    "simple": ["hello", "test query", "python programming"],
    "complex": [
        "Explain machine learning algorithms in detail",
        "What are the best practices for web development?",
        "How does artificial intelligence impact society?"
    ],
    "edge_cases": ["", "   ", "a" * 1000, "special chars: !@#$%^&*()"]
}

__all__ = [
    "PERFORMANCE_BENCHMARKS",
    "TestEnvironment", 
    "TEST_CONSTANTS",
    "MOCK_RESPONSES",
    "TEST_QUERIES"
]
