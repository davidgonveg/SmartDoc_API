#!/bin/bash
echo "🚀 Starting SmartDoc in CPU mode..."
cp .env.cpu .env
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
echo "✅ Services started in CPU mode"
echo "🌐 Streamlit UI: http://localhost:8501"  
echo "🤖 Agent API: http://localhost:8001"
echo "📚 API Docs: http://localhost:8001/docs"
