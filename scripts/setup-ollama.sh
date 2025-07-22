#!/bin/bash
echo "🤖 Setting up Ollama models..."

# Check if Ollama is running
if ! command -v ollama &> /dev/null; then
    echo "❌ Ollama not found. Please install Ollama first."
    echo "Visit: https://ollama.ai"
    exit 1
fi

# Pull CPU-friendly model
echo "📥 Pulling Llama 3.2 3B (CPU-friendly)..."
ollama pull llama3.2:3b

# Pull GPU model if GPU available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU detected, pulling Llama 3.1 8B..."
    ollama pull llama3.1:8b
fi

echo "✅ Ollama models ready!"
