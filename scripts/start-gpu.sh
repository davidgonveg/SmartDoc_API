#!/bin/bash
echo "🚀 Starting SmartDoc with GPU support..."
cp .env.gpu .env
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
echo "✅ Services started with GPU support"
echo "🌐 Streamlit UI: http://localhost:8501"
echo "🤖 Agent API: http://localhost:8001"
echo "📚 API Docs: http://localhost:8001/docs"
