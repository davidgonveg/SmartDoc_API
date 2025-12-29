#!/bin/bash
echo "🚀 Iniciando SmartDoc Agent (Modo Optimizado)..."

# Detectar GPU
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    GPU_AVAILABLE=true
    echo "🎮 GPU detectada"
fi

# Verificar memoria disponible
TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
echo "💾 Memoria disponible: ${TOTAL_MEM}GB"

# Configurar según hardware
if [ "$GPU_AVAILABLE" = true ] && [ "$TOTAL_MEM" -ge 16 ]; then
    export AGENT_OPTIMIZATION_LEVEL="performance"
    export DEFAULT_MODEL="llama3.1:8b"
    export LLM_NUM_CTX=16384
    export LLM_NUM_PREDICT=2048
    echo "🔥 Modo Performance activado"
elif [ "$TOTAL_MEM" -lt 8 ]; then
    export AGENT_OPTIMIZATION_LEVEL="cpu"
    export DEFAULT_MODEL="llama3.2:3b"
    export LLM_NUM_CTX=4096
    export LLM_NUM_PREDICT=512
    echo "💻 Modo CPU activado"
else
    export AGENT_OPTIMIZATION_LEVEL="balanced"
    export DEFAULT_MODEL="llama3.2:3b"
    export LLM_NUM_CTX=8192
    export LLM_NUM_PREDICT=1024
    echo "⚖️ Modo Balanced activado"
fi

echo "🎯 Configuración:"
echo "   Modelo: $DEFAULT_MODEL"
echo "   Contexto: $LLM_NUM_CTX tokens"
echo "   Max predict: $LLM_NUM_PREDICT tokens"

# Verificar modelo en Ollama
echo "🤖 Verificando modelo..."
if ! ollama list | grep -q "$DEFAULT_MODEL"; then
    echo "�� Descargando $DEFAULT_MODEL..."
    ollama pull "$DEFAULT_MODEL"
fi

# Iniciar servicios
echo "🐳 Iniciando servicios..."
docker-compose up -d

# Esperar servicios
echo "⏳ Esperando servicios..."
sleep 15

# Health check
for i in {1..30}; do
    if curl -s http://localhost:8001/health > /dev/null; then
        echo "✅ API lista"
        break
    fi
    echo "⏳ Esperando API... ($i/30)"
    sleep 2
done

echo ""
echo "🎉 SmartDoc Agent iniciado exitosamente!"
echo "🌐 UI: http://localhost:8501"
echo "🚀 API: http://localhost:8001"
echo "📚 Docs: http://localhost:8001/docs"
