#!/bin/bash
echo "🔍 Verificación rápida de SmartDoc Agent..."

# Verificar servicios Docker
echo "🐳 Estado de servicios Docker:"
docker-compose ps

echo ""
echo "🩺 Health checks:"

# API Health
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ API (8001) - OK"
else
    echo "❌ API (8001) - FAIL"
fi

# UI Check
if curl -s http://localhost:8501 > /dev/null; then
    echo "✅ UI (8501) - OK"
else
    echo "❌ UI (8501) - FAIL"
fi

# Ollama Check
if curl -s http://localhost:11434/api/version > /dev/null; then
    echo "✅ Ollama (11434) - OK"
else
    echo "❌ Ollama (11434) - FAIL"
fi

# ChromaDB Check
if curl -s http://localhost:8000/api/v1/heartbeat > /dev/null; then
    echo "✅ ChromaDB (8000) - OK"
else
    echo "❌ ChromaDB (8000) - FAIL"
fi

# Redis Check
if docker exec smartdoc-redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis (6379) - OK"
else
    echo "❌ Redis (6379) - FAIL"
fi

echo ""
echo "📊 Hardware info:"
curl -s http://localhost:8001/research/system/hardware 2>/dev/null | jq -r '{cpu_count, memory_gb, has_gpu, recommended_optimization}' 2>/dev/null || echo "API no disponible"

echo ""
echo "🔗 Accesos:"
echo "   UI: http://localhost:8501"
echo "   API: http://localhost:8001"
echo "   Docs: http://localhost:8001/docs"
