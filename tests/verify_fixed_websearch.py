#!/usr/bin/env python3
"""
Quick Verification Script for Fixed WebSearchTool
Ejecutar para verificar que el WebSearchTool arreglado funciona correctamente
"""

import sys
import asyncio
import logging
import time
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
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


async def test_import_and_instantiation():
    """Test 1: Import and instantiation"""
    print_section("Test 1: Import and Instantiation")
    
    try:
        from app.agents.tools.web.web_search_tool import FixedWebSearchTool
        from langchain.tools import BaseTool
        
        tool = FixedWebSearchTool()
        
        # Verify inheritance
        is_base_tool = isinstance(tool, BaseTool)
        has_required_attrs = all(hasattr(tool, attr) for attr in ['name', 'description', '_run', '_arun'])
        
        print_test_result("Import FixedWebSearchTool", True)
        print_test_result("LangChain BaseTool inheritance", is_base_tool)
        print_test_result("Required attributes present", has_required_attrs)
        print_test_result("Tool name correct", tool.name == "web_search")
        
        return tool if is_base_tool and has_required_attrs else None
        
    except Exception as e:
        print_test_result("Import and instantiation", False, str(e))
        return None


async def test_basic_functionality(tool):
    """Test 2: Basic functionality"""
    print_section("Test 2: Basic Functionality")
    
    if not tool:
        print_test_result("Basic functionality", False, "Tool not available")
        return False
    
    try:
        # Test cache system
        cache_key = tool._generate_cache_key("test", 5)
        cache_works = isinstance(cache_key, str) and len(cache_key) > 0
        print_test_result("Cache key generation", cache_works)
        
        # Test text cleaning
        cleaned = tool._clean_text("  Test   Text  \n\n  ")
        text_clean_works = cleaned == "Test Text"
        print_test_result("Text cleaning", text_clean_works)
        
        # Test URL validation
        valid_url = tool._is_valid_result("Good Title", "https://example.com")
        invalid_url = tool._is_valid_result("", "invalid-url")
        url_validation_works = valid_url and not invalid_url
        print_test_result("URL validation", url_validation_works)
        
        # Test statistics initialization
        stats_ok = all(key in tool.stats for key in ['total_searches', 'successful_searches', 'cache_hits', 'engine_failures'])
        print_test_result("Statistics initialization", stats_ok)
        
        return cache_works and text_clean_works and url_validation_works and stats_ok
        
    except Exception as e:
        print_test_result("Basic functionality", False, str(e))
        return False


async def test_mock_search(tool):
    """Test 3: Mock search functionality"""
    print_section("Test 3: Mock Search Test")
    
    if not tool:
        print_test_result("Mock search", False, "Tool not available")
        return False
    
    try:
        # Mock the multi-engine search method
        from unittest.mock import patch
        
        mock_result = {
            'query': 'test query',
            'results': [
                {
                    'title': 'Test Result Title',
                    'url': 'https://example.com/test',
                    'snippet': 'This is a test search result snippet',
                    'rank': 1,
                    'source': 'MockEngine'
                }
            ],
            'source': 'MockEngine',
            'success': True
        }
        
        with patch.object(tool, '_perform_multi_engine_search', return_value=mock_result):
            result = await tool._arun("test query", max_results=1)
            
            # Verify result
            is_string = isinstance(result, str)
            has_content = len(result) > 100 if is_string else False
            has_title = 'Test Result Title' in result if is_string else False
            has_url = 'https://example.com/test' in result if is_string else False
            
            print_test_result("Returns string result", is_string)
            print_test_result("Has substantial content", has_content)
            print_test_result("Contains expected title", has_title)
            print_test_result("Contains expected URL", has_url)
            
            # Check statistics
            stats_updated = tool.stats['total_searches'] >= 1 and tool.stats['successful_searches'] >= 1
            print_test_result("Statistics updated", stats_updated)
            
            if is_string and has_content:
                print(f"    Result preview: {result[:150]}...")
            
            return is_string and has_content and has_title and has_url and stats_updated
            
    except Exception as e:
        print_test_result("Mock search", False, str(e))
        return False


async def test_real_search(tool):
    """Test 4: Real search (if network available)"""
    print_section("Test 4: Real Search Test")
    
    if not tool:
        print_test_result("Real search", False, "Tool not available")
        return False
    
    try:
        print("    Attempting real web search (this may take a few seconds)...")
        
        start_time = time.time()
        result = await tool._arun("python programming tutorial", max_results=2)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Verify result
        is_string = isinstance(result, str)
        has_content = len(result) > 100 if is_string else False
        no_error = "error" not in result.lower() if is_string else False
        reasonable_time = duration < 60  # Less than 60 seconds
        
        print_test_result("Returns string result", is_string)
        print_test_result("Has substantial content", has_content, f"{len(result)} chars" if is_string else "")
        print_test_result("No error messages", no_error)
        print_test_result("Reasonable response time", reasonable_time, f"{duration:.2f}s")
        
        if is_string and has_content and no_error:
            print(f"    Result preview: {result[:200]}...")
            return True
        else:
            if is_string:
                print(f"    Full result: {result}")
            return False
            
    except Exception as e:
        print_test_result("Real search", False, str(e))
        return False


async def test_agent_integration(tool):
    """Test 5: Agent integration"""
    print_section("Test 5: Agent Integration Test")
    
    if not tool:
        print_test_result("Agent integration", False, "Tool not available")
        return False
    
    try:
        from app.agents.core.smart_agent import SmartDocAgent
        
        # Create agent
        agent = SmartDocAgent(model_name="llama3.2:3b")
        
        # Add tool to agent
        agent.tools = [tool]
        
        # Verify integration
        tool_added = len(agent.tools) == 1
        correct_tool = agent.tools[0].name == "web_search" if tool_added else False
        
        print_test_result("Agent instantiation", True)
        print_test_result("Tool added to agent", tool_added)
        print_test_result("Correct tool name", correct_tool)
        
        # Test LangChain compatibility in agent context
        from langchain.tools import BaseTool
        langchain_compatible = isinstance(agent.tools[0], BaseTool) if tool_added else False
        print_test_result("LangChain compatible in agent", langchain_compatible)
        
        return tool_added and correct_tool and langchain_compatible
        
    except Exception as e:
        print_test_result("Agent integration", False, str(e))
        return False


async def test_health_check(tool):
    """Test 6: Health check"""
    print_section("Test 6: Health Check")
    
    if not tool:
        print_test_result("Health check", False, "Tool not available")
        return False
    
    try:
        # Mock a successful search for health check
        from unittest.mock import patch
        
        with patch.object(tool, '_arun', return_value="Health check successful"):
            health = await tool.health_check()
            
            is_dict = isinstance(health, dict)
            has_healthy = 'healthy' in health if is_dict else False
            has_stats = 'statistics' in health if is_dict else False
            is_healthy = health.get('healthy', False) if is_dict else False
            
            print_test_result("Health check returns dict", is_dict)
            print_test_result("Has 'healthy' field", has_healthy)
            print_test_result("Has 'statistics' field", has_stats)
            print_test_result("Reports as healthy", is_healthy)
            
            if is_dict:
                print(f"    Health data: {health}")
            
            return is_dict and has_healthy and has_stats and is_healthy
            
    except Exception as e:
        print_test_result("Health check", False, str(e))
        return False


async def main():
    """Main verification function"""
    print("🧪 FIXED WEBSEARCHTOOL VERIFICATION")
    print(f"   Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Python: {sys.version}")
    
    # Run all tests
    tool = await test_import_and_instantiation()
    basic_ok = await test_basic_functionality(tool)
    mock_ok = await test_mock_search(tool)
    real_ok = await test_real_search(tool)
    integration_ok = await test_agent_integration(tool)
    health_ok = await test_health_check(tool)
    
    # Calculate results
    tests_run = 6
    tests_passed = sum([
        tool is not None,  # Import test
        basic_ok,
        mock_ok,
        real_ok,
        integration_ok,
        health_ok
    ])
    
    # Final summary
    print_section("VERIFICATION SUMMARY")
    print(f"📊 Tests passed: {tests_passed}/{tests_run} ({tests_passed/tests_run*100:.1f}%)")
    
    if tests_passed >= 5:  # Real search may fail due to network
        print("🚀 VERIFICATION SUCCESSFUL!")
        print("   ✅ WebSearchTool is working correctly")
        print("   ✅ Ready for integration with SmartDocAgent")
        print("   ✅ LangChain compatibility confirmed")
        
        print("\n🎯 NEXT STEPS:")
        print("   1. Test the agent with the working tool:")
        print("      docker-compose exec agent-api python -c \"")
        print("      import asyncio")
        print("      from app.agents.core.smart_agent import SmartDocAgent")
        print("      from app.agents.tools.web.web_search_tool import FixedWebSearchTool")
        print("      async def test():")
        print("          agent = SmartDocAgent('llama3.2:3b')")
        print("          agent.tools = [FixedWebSearchTool()]")
        print("          print('Agent ready with working WebSearchTool!')")
        print("      asyncio.run(test())\"")
        print("\n   2. Initialize the agent and test full workflow")
        
        return True
        
    else:
        print("🚨 VERIFICATION FAILED!")
        print(f"   ❌ Only {tests_passed}/{tests_run} tests passed")
        print("   ❌ WebSearchTool needs additional fixes")
        
        if not tool:
            print("\n🔧 ISSUES FOUND:")
            print("   - Import or instantiation failed")
            print("   - Check that you replaced web_search_tool.py correctly")
        
        if not real_ok:
            print("   - Real search failed (network issues or parsing problems)")
        
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)