"""
Test de compatibilidad del WebSearchTool con LangChain BaseTool
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, Mock

def test_websearchtool_imports():
    """Test que WebSearchTool puede importar LangChain BaseTool"""
    try:
        from langchain.tools import BaseTool
        from app.agents.tools.web.web_search_tool import WebSearchTool
        
        # Verificar que WebSearchTool hereda de LangChain BaseTool
        assert issubclass(WebSearchTool, BaseTool)
        print("✅ WebSearchTool hereda correctamente de LangChain BaseTool")
        
    except ImportError as e:
        pytest.fail(f"Error importando LangChain o WebSearchTool: {e}")

def test_websearchtool_required_attributes():
    """Test que WebSearchTool tiene atributos requeridos por LangChain"""
    from app.agents.tools.web.web_search_tool import WebSearchTool
    
    # Crear instancia
    tool = WebSearchTool()
    
    # Verificar atributos requeridos
    assert hasattr(tool, 'name'), "WebSearchTool debe tener atributo 'name'"
    assert hasattr(tool, 'description'), "WebSearchTool debe tener atributo 'description'"
    
    # Verificar que los atributos son strings y no vacíos
    assert isinstance(tool.name, str) and len(tool.name) > 0
    assert isinstance(tool.description, str) and len(tool.description) > 0
    
    print(f"✅ Tool name: {tool.name}")
    print(f"✅ Tool description: {tool.description[:100]}...")

def test_websearchtool_required_methods():
    """Test que WebSearchTool tiene métodos requeridos por LangChain"""
    from app.agents.tools.web.web_search_tool import WebSearchTool
    
    tool = WebSearchTool()
    
    # Verificar métodos requeridos
    assert hasattr(tool, '_run'), "WebSearchTool debe tener método '_run'"
    assert hasattr(tool, '_arun'), "WebSearchTool debe tener método '_arun'"
    
    # Verificar que son callable
    assert callable(tool._run), "Método '_run' debe ser callable"
    assert callable(tool._arun), "Método '_arun' debe ser callable"
    
    print("✅ WebSearchTool tiene métodos _run y _arun")

@pytest.mark.asyncio
async def test_websearchtool_basic_functionality():
    """Test básico de funcionalidad del WebSearchTool"""
    from app.agents.tools.web.web_search_tool import WebSearchTool
    
    tool = WebSearchTool()
    
    # Mock de los atributos privados directamente en lugar de las propiedades
    mock_rate_limiter = AsyncMock()
    mock_rate_limiter.acquire = AsyncMock(return_value=True)
    
    mock_search_manager = AsyncMock()
    mock_search_manager.search = AsyncMock(return_value=Mock(
        success=True,
        results=[Mock(title="Test Result", url="https://test.com", snippet="Test snippet")]
    ))
    
    mock_content_extractor = AsyncMock()
    mock_content_extractor.extract_content = AsyncMock(return_value=Mock(
        extraction_success=True,
        main_content="Test content"
    ))
    
    # Asignar mocks directamente a los atributos privados
    object.__setattr__(tool, '_rate_limiter', mock_rate_limiter)
    object.__setattr__(tool, '_search_manager', mock_search_manager)
    object.__setattr__(tool, '_content_extractor', mock_content_extractor)
    
    # Test método async
    result = await tool._arun("test query")
    assert isinstance(result, str)
    assert len(result) > 0
    print(f"✅ _arun result preview: {result[:100]}...")
    
    # Test método sync
    sync_result = tool._run("test query")
    assert isinstance(sync_result, str)
    assert len(sync_result) > 0
    print(f"✅ _run result preview: {sync_result[:100]}...")

def test_langchain_tool_creation():
    """Test que LangChain puede crear el tool correctamente"""
    from app.agents.tools.web.web_search_tool import WebSearchTool
    
    # Crear tool
    tool = WebSearchTool()
    
    # Test que LangChain puede trabajar con el tool
    try:
        # Simular lo que haría LangChain
        tool_name = tool.name
        tool_desc = tool.description
        
        # Verificar estructura básica
        assert tool_name == "web_search"
        assert "search" in tool_desc.lower()
        
        print(f"✅ LangChain compatibility verified")
        print(f"   - Name: {tool_name}")
        print(f"   - Description: {tool_desc[:80]}...")
        
    except Exception as e:
        pytest.fail(f"Error en compatibilidad LangChain: {e}")

if __name__ == "__main__":
    """Ejecutar tests básicos"""
    print("🧪 Testing WebSearchTool compatibility with LangChain...")
    
    test_websearchtool_imports()
    test_websearchtool_required_attributes() 
    test_websearchtool_required_methods()
    
    # Test async
    asyncio.run(test_websearchtool_basic_functionality())
    
    test_langchain_tool_creation()
    
    print("✅ All compatibility tests passed!")