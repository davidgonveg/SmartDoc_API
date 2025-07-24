#!/usr/bin/env python3
"""
Script para verificar que todas las importaciones del Web Search Tool funcionan correctamente
"""

import sys
import traceback
from pathlib import Path

# Agregar el directorio app al path
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "app"))

def test_import(module_name, description):
    """Test individual import with error handling"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError as e:
        print(f"❌ {description}: {module_name}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"⚠️  {description}: {module_name}")
        print(f"   Unexpected error: {e}")
        return False

def main():
    """Verificar todas las importaciones del Web Search Tool"""
    
    print("🔧 Verificando importaciones del SmartDoc Web Search Tool...")
    print("=" * 60)
    
    failed_imports = []
    
    # Core dependencies
    imports_to_test = [
        # Python standard library
        ("asyncio", "Async support"),
        ("json", "JSON processing"),
        ("logging", "Logging"),
        ("re", "Regular expressions"),
        ("time", "Time utilities"),
        ("typing", "Type hints"),
        ("urllib.parse", "URL parsing"),
        ("hashlib", "Hashing"),
        ("dataclasses", "Data classes"),
        ("enum", "Enumerations"),
        
        # Third party dependencies
        ("httpx", "HTTP client"),
        ("aiohttp", "Async HTTP client"),
        ("requests", "HTTP client (sync)"),
        ("bs4", "HTML parsing (BeautifulSoup)"),
        ("lxml", "XML/HTML parser"),
        ("selenium", "Web automation"),
        ("chromadb", "Vector database"),
        ("sentence_transformers", "Embeddings"),
        ("pandas", "Data processing"),
        ("numpy", "Numerical computing"),
        ("redis", "Redis client"),
        ("pydantic", "Data validation"),
        ("pydantic_settings", "Settings management"),
        ("fastapi", "Web framework"),
        ("uvicorn", "ASGI server"),
        ("langchain", "LangChain framework"),
        ("langchain_community", "LangChain community"),
        ("langchain_ollama", "LangChain Ollama"),
        
        # Testing dependencies
        ("pytest", "Testing framework"),
        ("pytest_asyncio", "Async testing"),
        ("responses", "HTTP mocking"),
        ("aioresponses", "Async HTTP mocking"),
    ]
    
    # Test standard imports
    for module, description in imports_to_test:
        success = test_import(module, description)
        if not success:
            failed_imports.append((module, description))
    
    print("\n" + "=" * 60)
    print("🧪 Verificando importaciones específicas del Web Search Tool...")
    print("=" * 60)
    
    # SmartDoc specific imports
    smartdoc_imports = [
        # Configuration
        ("app.config.settings", "Settings configuration"),
        
        # Base tool system
        ("app.agents.tools.base_tool", "Base tool system"),
        
        # Web search components
        ("app.agents.tools.web.config", "Web search configuration"),
        ("app.agents.tools.web.user_agents", "User agent management"),
        ("app.agents.tools.web.web_utils", "Web utilities"),
        ("app.agents.tools.web.rate_limiter", "Rate limiting"),
        ("app.agents.tools.web.search_engines", "Search engines"),
        ("app.agents.tools.web.content_extractor", "Content extraction"),
        ("app.agents.tools.web.web_search_tool", "Main web search tool"),
        
        # Services
        ("app.services.ollama_client", "Ollama client"),
        
        # Agent core
        ("app.agents.core.smart_agent", "SmartDoc agent"),
        ("app.agents.prompts.react_templates", "ReAct templates"),
    ]
    
    for module, description in smartdoc_imports:
        success = test_import(module, description)
        if not success:
            failed_imports.append((module, description))
    
    print("\n" + "=" * 60)
    print("🔍 Verificando funcionalidad específica...")
    print("=" * 60)
    
    # Test specific functionality
    try:
        from app.agents.tools.web import WebSearchTool, get_web_tools_info
        print("✅ WebSearchTool class import successful")
        
        # Test tool instantiation
        tool = WebSearchTool()
        print("✅ WebSearchTool instantiation successful")
        
        # Test tool info
        info = get_web_tools_info()
        print(f"✅ Web tools info: {len(info['available_engines'])} engines available")
        
    except Exception as e:
        print(f"❌ WebSearchTool functionality test failed: {e}")
        failed_imports.append(("WebSearchTool", "Main tool functionality"))
    
    # Test agent imports
    try:
        from app.agents.core.smart_agent import SmartDocAgent
        print("✅ SmartDocAgent import successful")
    except Exception as e:
        print(f"❌ SmartDocAgent import failed: {e}")
        failed_imports.append(("SmartDocAgent", "Main agent class"))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE VERIFICACIÓN")
    print("=" * 60)
    
    total_tests = len(imports_to_test) + len(smartdoc_imports) + 2  # +2 for functionality tests
    failed_count = len(failed_imports)
    success_count = total_tests - failed_count
    
    print(f"Total tests: {total_tests}")
    print(f"✅ Exitosos: {success_count}")
    print(f"❌ Fallidos: {failed_count}")
    
    if failed_imports:
        print(f"\n⚠️  IMPORTACIONES FALLIDAS:")
        for module, description in failed_imports:
            print(f"   - {module} ({description})")
        
        print(f"\n💡 SOLUCIONES RECOMENDADAS:")
        print("   1. Verificar que el virtual environment está activado")
        print("   2. Ejecutar: pip install -r requirements.txt")
        print("   3. Instalar dependencias de testing mencionadas arriba")
        print("   4. Verificar que estás en el directorio agent-api/")
        
        return False
    else:
        print(f"\n🎉 ¡TODAS LAS IMPORTACIONES EXITOSAS!")
        print("   El entorno está listo para ejecutar tests.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
