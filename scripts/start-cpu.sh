#!/bin/bash
echo "🚀 Starting SmartDoc in CPU mode..."


# Detect python command
if command -v python3 &>/dev/null; then
    PY_CMD=python3
elif command -v python &>/dev/null; then
    PY_CMD=python
else
    echo "❌ Python not found. Please install Python to use the port checker."
    # Fallback: warning only, proceed to try docker
    echo "⚠️ Skipping port check..."
    PY_CMD=""
fi

if [ ! -z "$PY_CMD" ]; then
    # Check for psutil
    $PY_CMD -c "import psutil" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "📦 Installing psutil for port checker..."
        $PY_CMD -m pip install psutil
    fi


    # Run check
    $PY_CMD scripts/check_ports.py
    if [ $? -ne 0 ]; then
        echo "❌ Port verification failed. Aborting."
        exit 1
    fi
fi

cp .env.cpu .env
# Force build to ensure latest code is used
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d --build
echo "✅ Services started in CPU mode"
echo "🌐 Streamlit UI: http://localhost:8501"  
echo "🤖 Agent API: http://localhost:8002"
echo "📚 API Docs: http://localhost:8002/docs"
