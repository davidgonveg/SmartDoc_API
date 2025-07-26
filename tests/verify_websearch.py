#!/usr/bin/env python3
"""
🔧 Verificación Completa de WebSearchTool + SmartDocAgent
Script para validar la integración funcional end-to-end
"""

import sys
import asyncio
import logging
import time
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"🔧 {title}")
    print(f"{'='*60}")


def print_test_result(test_name: str, success: bool, details: str = ""):
    """Print formatted test result"""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if details:
        print(f"    {details}")


async def test_1_websearch_tool_standalone():
    """Fase 1: Verificar WebSearchTool standalone"""
    print_section("FASE 1: WebSearchTool Standalone")
    
    try:
        # Import the fixed WebSearchTool
        from app.agents.tools.web.web_search_tool import WebSearchTool
        from langchain.tools import BaseTool
        
        # Test instantiation
        tool = WebSearchTool()
        print_test_result("WebSearchTool instantiation", True)
        
        # Test LangChain compatibility
        is_base_tool = isinstance(tool, BaseTool)
        print_test_result("LangChain BaseTool inheritance", is_base_tool)
        
        # Test required attributes
        has_attrs = all(hasattr(tool, attr) for attr in ['name', 'description', '_run', '_arun'])
        print_test_result("Required LangChain attributes", has_attrs)
        
        # Test basic functionality with timeout
        print("⏳ Testing web search functionality...")
        start_time = time.time()
        
        try:
            result = await asyncio.wait_for(
                tool._arun("Python programming tutorial", max_results=2),
                timeout=30.0
            )
            
            success = (
                isinstance(result, str) and 
                len(result) > 50 and 
                "Found" in result and
                "results" in result
            )
            
            elapsed = time.time() - start_time
            print_test_result(
                "Web search execution", 
                success, 
                f"Completed in {elapsed:.1f}s, {len(result)} characters"
            )
            
            if success:
                # Show sample of result
                print(f"📋 Sample result:\n{result[:300]}...")
            
        except asyncio.TimeoutError:
            print_test_result("Web search execution", False, "Timeout after 30s")
        except Exception as e:
            print_test_result("Web search execution", False, str(e))
        
        return tool
        
    except Exception as e:
        print_test_result("WebSearchTool test", False, str(e))
        return None


async def test_2_agent_integration():
    """Fase 2: Verificar integración con SmartDocAgent"""
    print_section("FASE 2: SmartDocAgent Integration")
    
    try:
        from app.agents.core.smart_agent import SmartDocAgent
        
        # Create agent instance
        agent = SmartDocAgent(
            model_name="llama3.2:3b",
            research_style="general",
            max_iterations=3
        )
        print_test_result("Agent instantiation", True)
        
        # Test initialization
        try:
            initialized = await asyncio.wait_for(agent.initialize(), timeout=20.0)
            print_test_result("Agent initialization", initialized)
            
            if initialized:
                # Check if WebSearchTool is available
                has_web_tool = any(tool.name == "web_search" for tool in agent.tools)
                print_test_result("WebSearchTool registered", has_web_tool)
                
                return agent
            
        except Exception as e:
            print_test_result("Agent initialization", False, str(e))
            
    except Exception as e:
        print_test_result("Agent integration test", False, str(e))
    
    return None


async def test_3_end_to_end_workflow():
    """Fase 3: Test end-to-end completo"""
    print_section("FASE 3: End-to-End Workflow")
    
    try:
        # Initialize agent
        agent = await test_2_agent_integration()
        if not agent:
            print_test_result("E2E workflow", False, "Agent initialization failed")
            return
        
        # Test simple query that should trigger web search
        test_queries = [
            "What is Python programming?",
            "Current trends in artificial intelligence",
            "How to learn machine learning"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n🔍 Testing query {i}: {query}")
            
            try:
                start_time = time.time()
                result = await asyncio.wait_for(
                    agent.process_query(query),
                    timeout=60.0
                )
                
                elapsed = time.time() - start_time
                
                success = (
                    isinstance(result, dict) and 
                    'response' in result and
                    len(result['response']) > 100
                )
                
                print_test_result(
                    f"Query {i} processing", 
                    success,
                    f"Completed in {elapsed:.1f}s"
                )
                
                if success:
                    print(f"📋 Response sample:\n{result['response'][:200]}...")
                    
                    # Check if web search was used
                    if 'actions' in result:
                        web_search_used = any(
                            'web_search' in str(action) 
                            for action in result['actions']
                        )
                        print_test_result(
                            f"Query {i} used web search", 
                            web_search_used
                        )
                
                # Only test one query if successful
                if success:
                    break
                    
            except asyncio.TimeoutError:
                print_test_result(f"Query {i} processing", False, "Timeout after 60s")
            except Exception as e:
                print_test_result(f"Query {i} processing", False, str(e))
        
        # Cleanup
        await agent.close()
        
    except Exception as e:
        print_test_result("E2E workflow", False, str(e))


async def test_4_api_integration():
    """Fase 4: Verificar integración con API"""
    print_section("FASE 4: API Integration Test")
    
    try:
        import httpx
        
        # Test API health
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get("http://localhost:8001/health", timeout=10.0)
                api_healthy = response.status_code == 200
                print_test_result("API health check", api_healthy)
                
                if api_healthy:
                    # Test session creation
                    session_response = await client.post(
                        "http://localhost:8001/research/session",
                        json={"research_style": "general"}
                    )
                    
                    session_created = session_response.status_code == 200
                    print_test_result("Session creation", session_created)
                    
                    if session_created:
                        session_data = session_response.json()
                        session_id = session_data.get('session_id')
                        
                        # Test chat endpoint
                        chat_response = await client.post(
                            f"http://localhost:8001/research/chat/{session_id}",
                            json={"message": "What is Python programming?"},
                            timeout=60.0
                        )
                        
                        chat_success = chat_response.status_code == 200
                        print_test_result("Chat endpoint", chat_success)
                        
                        if chat_success:
                            chat_data = chat_response.json()
                            print(f"📋 API Response sample:\n{str(chat_data)[:200]}...")
                
            except httpx.TimeoutError:
                print_test_result("API integration", False, "API timeout")
            except httpx.ConnectError:
                print_test_result("API integration", False, "API not accessible - start with docker-compose up")
            
    except Exception as e:
        print_test_result("API integration test", False, str(e))


async def main():
    """Main verification workflow"""
    print_section("🚀 SmartDoc Agent Integration Verification")
    print("This script will verify the complete WebSearchTool + Agent integration")
    
    start_time = time.time()
    
    # Run all tests
    await test_1_websearch_tool_standalone()
    await test_2_agent_integration()
    await test_3_end_to_end_workflow()
    await test_4_api_integration()
    
    total_time = time.time() - start_time
    
    print_section(f"✅ Verification Complete - Total time: {total_time:.1f}s")
    print("\n🎯 Next Steps:")
    print("1. If tests pass: Your WebSearchTool integration is working!")
    print("2. If tests fail: Check the error messages above")
    print("3. For API tests: Ensure Docker services are running")
    print("4. For agent tests: Verify Ollama is accessible and model is pulled")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Verification interrupted by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        sys.exit(1)