#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Setting up SmartDoc API Project Structure${NC}"

# Create main project directory
PROJECT_NAME="smartdoc-api"
mkdir -p $PROJECT_NAME
cd $PROJECT_NAME

echo -e "${GREEN}✅ Created main project directory: $PROJECT_NAME${NC}"

# Create main directory structure
echo -e "${YELLOW}📁 Creating directory structure...${NC}"

# Main directories
mkdir -p {agent-api,streamlit-ui,data,logs,scripts,docs,tests}

# Agent API structure
mkdir -p agent-api/{app,tests}
mkdir -p agent-api/app/{config,agents,tools,memory,models,utils,api}
mkdir -p agent-api/app/agents/{core,prompts,validators}
mkdir -p agent-api/app/tools/{web,pdf,calculator,code,memory_store,report}
mkdir -p agent-api/app/memory/{working,longterm,session}

# Streamlit UI structure  
mkdir -p streamlit-ui/{pages,components,utils,assets}

# Data directories
mkdir -p data/{chromadb,redis,uploads,reports,cache}

# Logs directory structure
mkdir -p logs/{agent,api,ui,system}

# Tests structure
mkdir -p tests/{unit,integration,e2e}

echo -e "${GREEN}✅ Directory structure created${NC}"

# Create Docker files
echo -e "${YELLOW}🐳 Creating Docker configuration files...${NC}"

# Main docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Redis for caching and session storage
  redis:
    image: redis:7-alpine
    container_name: smartdoc-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

  # ChromaDB for vector storage
  chromadb:
    image: chromadb/chroma:latest
    container_name: smartdoc-chromadb
    ports:
      - "8000:8000"
    volumes:
      - chromadb_data:/chroma/chroma
    environment:
      - CHROMA_SERVER_HOST=0.0.0.0
      - CHROMA_SERVER_HTTP_PORT=8000
    restart: unless-stopped

  # Agent API (FastAPI + LangChain)
  agent-api:
    build: 
      context: ./agent-api
      dockerfile: Dockerfile
    container_name: smartdoc-agent-api
    ports:
      - "8001:8001"
    volumes:
      - ./data/uploads:/app/uploads
      - ./data/reports:/app/reports
      - ./logs/agent:/app/logs
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-development}
      - CHROMADB_HOST=chromadb
      - CHROMADB_PORT=8000
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - OLLAMA_HOST=${OLLAMA_HOST:-host.docker.internal}
      - OLLAMA_PORT=${OLLAMA_PORT:-11434}
    depends_on:
      - redis
      - chromadb
    restart: unless-stopped

  # Streamlit UI
  streamlit-ui:
    build:
      context: ./streamlit-ui
      dockerfile: Dockerfile
    container_name: smartdoc-streamlit-ui
    ports:
      - "8501:8501"
    volumes:
      - ./data/uploads:/app/uploads
      - ./data/reports:/app/reports
    environment:
      - AGENT_API_URL=http://agent-api:8001
    depends_on:
      - agent-api
    restart: unless-stopped

volumes:
  redis_data:
  chromadb_data:

networks:
  default:
    name: smartdoc-network
EOF

# GPU override for development machine
cat > docker-compose.gpu.yml << 'EOF'
version: '3.8'

services:
  # Ollama with GPU support
  ollama:
    image: ollama/ollama:latest
    container_name: smartdoc-ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_models:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  # Agent API with GPU-optimized settings
  agent-api:
    environment:
      - USE_GPU=true
      - OLLAMA_HOST=ollama
      - EMBEDDING_DEVICE=cuda
    depends_on:
      - ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_models:
EOF

# CPU-only override for laptop
cat > docker-compose.cpu.yml << 'EOF'
version: '3.8'

services:
  agent-api:
    environment:
      - USE_GPU=false
      - OLLAMA_HOST=${OLLAMA_HOST:-host.docker.internal}
      - EMBEDDING_DEVICE=cpu
      - USE_API_FALLBACK=true
      - OPENAI_API_KEY=${OPENAI_API_KEY:-}
EOF

echo -e "${GREEN}✅ Docker configuration files created${NC}"

# Create environment files
echo -e "${YELLOW}⚙️ Creating environment configuration...${NC}"

cat > .env << 'EOF'
# Environment
ENVIRONMENT=development

# GPU/CPU Configuration
USE_GPU=false
EMBEDDING_DEVICE=cpu

# Ollama Configuration (for external Ollama instance)
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# API Keys (optional, for fallback)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Agent Configuration
DEFAULT_MODEL=llama3.2:3b
EMBEDDING_MODEL=all-MiniLM-L6-v2
MAX_TOKENS=2048
TEMPERATURE=0.7

# Database Configuration
CHROMADB_PERSIST_DIRECTORY=./data/chromadb
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/system/app.log

# Rate Limiting
REQUESTS_PER_MINUTE=60
WEB_SCRAPING_DELAY=1
EOF

cat > .env.gpu << 'EOF'
# GPU Environment Settings
USE_GPU=true
EMBEDDING_DEVICE=cuda
DEFAULT_MODEL=llama3.1:8b
OLLAMA_HOST=ollama
USE_API_FALLBACK=false
EOF

cat > .env.cpu << 'EOF'
# CPU Environment Settings  
USE_GPU=false
EMBEDDING_DEVICE=cpu
DEFAULT_MODEL=llama3.2:3b
OLLAMA_HOST=localhost
USE_API_FALLBACK=true
EOF

echo -e "${GREEN}✅ Environment files created${NC}"

# Create Agent API files
echo -e "${YELLOW}🤖 Creating Agent API structure...${NC}"

# Agent API Dockerfile
cat > agent-api/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p uploads reports logs

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
EOF

# Agent API requirements
cat > agent-api/requirements.txt << 'EOF'
# Core Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
python-multipart==0.0.6

# LangChain ecosystem
langchain==0.0.340
langchain-community==0.0.10
langsmith==0.0.66

# Vector Database & Embeddings
chromadb==0.4.18
sentence-transformers==2.2.2

# Document Processing
PyPDF2==3.0.1
python-docx==1.1.0
openpyxl==3.1.2

# Web Scraping
requests==2.31.0
beautifulsoup4==4.12.2
selenium==4.15.0

# Data Processing
pandas==2.1.4
numpy==1.24.4

# Caching & Session
redis==5.0.1
diskcache==5.6.3

# Utilities
python-dotenv==1.0.0
aiofiles==23.2.1
Jinja2==3.1.2

# Monitoring & Logging
structlog==23.2.0
rich==13.7.0

# Math & Calculations
sympy==1.12
matplotlib==3.8.2
plotly==5.17.0

# Optional GPU support
torch>=2.0.0; sys_platform != "darwin"
torchvision>=0.15.0; sys_platform != "darwin"

# Development
pytest==7.4.3
pytest-asyncio==0.21.1
black==23.11.0
flake8==6.1.0
EOF

# Main FastAPI app
cat > agent-api/app/main.py << 'EOF'
"""
SmartDoc Agent API - Main FastAPI Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict, Any

from app.config.settings import get_settings
from app.api.routes import research, health, upload
from app.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting SmartDoc Agent API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"GPU Support: {settings.use_gpu}")
    
    # Initialize services here
    # await initialize_agent()
    # await initialize_databases()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SmartDoc Agent API")

# Create FastAPI app
app = FastAPI(
    title="SmartDoc Agent API",
    description="Intelligent research agent powered by LangChain",
    version="0.1.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit UI
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(research.router, prefix="/research", tags=["research"])
app.include_router(upload.router, prefix="/upload", tags=["upload"])

@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint"""
    return {
        "message": "SmartDoc Agent API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
        "gpu_enabled": settings.use_gpu
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
EOF

# Create basic config
cat > agent-api/app/config/settings.py << 'EOF'
"""Application settings and configuration"""

from pydantic import BaseSettings, Field
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # GPU/CPU Configuration
    use_gpu: bool = Field(default=False, env="USE_GPU")
    embedding_device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    
    # Ollama Configuration
    ollama_host: str = Field(default="localhost", env="OLLAMA_HOST")
    ollama_port: int = Field(default=11434, env="OLLAMA_PORT")
    default_model: str = Field(default="llama3.2:3b", env="DEFAULT_MODEL")
    
    # API Fallbacks
    use_api_fallback: bool = Field(default=True, env="USE_API_FALLBACK")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Database
    chromadb_host: str = Field(default="localhost", env="CHROMADB_HOST")
    chromadb_port: int = Field(default=8000, env="CHROMADB_PORT")
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    
    # Agent Configuration
    max_tokens: int = Field(default=2048, env="MAX_TOKENS")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    
    class Config:
        env_file = ".env"

_settings = None

def get_settings() -> Settings:
    """Get application settings (singleton)"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
EOF

# Create basic API routes
mkdir -p agent-api/app/api
cat > agent-api/app/api/routes/health.py << 'EOF'
"""Health check endpoints"""

from fastapi import APIRouter
from typing import Dict, Any
import asyncio

router = APIRouter()

@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "smartdoc-agent-api",
        "version": "0.1.0"
    }

@router.get("/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with dependencies"""
    # TODO: Check Ollama, ChromaDB, Redis connectivity
    return {
        "status": "healthy",
        "services": {
            "api": "healthy",
            "ollama": "pending",
            "chromadb": "pending", 
            "redis": "pending"
        }
    }
EOF

cat > agent-api/app/api/routes/research.py << 'EOF'
"""Research endpoints"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

router = APIRouter()

class ResearchRequest(BaseModel):
    topic: str
    objectives: Optional[List[str]] = []
    max_sources: int = 10
    research_depth: str = "intermediate"

class ChatMessage(BaseModel):
    message: str
    stream: bool = False

@router.post("/session")
async def create_research_session(request: ResearchRequest) -> Dict[str, Any]:
    """Create a new research session"""
    # TODO: Initialize agent session
    return {
        "session_id": "temp-session-123",
        "status": "created",
        "topic": request.topic
    }

@router.post("/chat/{session_id}")
async def chat_with_agent(session_id: str, message: ChatMessage) -> Dict[str, Any]:
    """Chat with the research agent"""
    # TODO: Implement agent chat
    return {
        "response": f"Echo: {message.message}",
        "sources": [],
        "reasoning": "Basic echo response",
        "confidence": 0.5
    }
EOF

cat > agent-api/app/api/routes/upload.py << 'EOF'
"""File upload endpoints"""

from fastapi import APIRouter, UploadFile, File
from typing import Dict, Any, List

router = APIRouter()

@router.post("/{session_id}")
async def upload_files(
    session_id: str, 
    files: List[UploadFile] = File(...)
) -> Dict[str, Any]:
    """Upload files for research session"""
    # TODO: Process uploaded files
    filenames = [file.filename for file in files]
    return {
        "session_id": session_id,
        "uploaded_files": filenames,
        "status": "uploaded"
    }
EOF

# Create __init__.py files
touch agent-api/app/__init__.py
touch agent-api/app/config/__init__.py
touch agent-api/app/api/__init__.py
touch agent-api/app/api/routes/__init__.py

echo -e "${GREEN}✅ Agent API structure created${NC}"

# Create Streamlit UI files
echo -e "${YELLOW}🎨 Creating Streamlit UI structure...${NC}"

cat > streamlit-ui/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

cat > streamlit-ui/requirements.txt << 'EOF'
streamlit==1.28.2
requests==2.31.0
pandas==2.1.4
plotly==5.17.0
streamlit-chat==0.1.1
python-dotenv==1.0.0
aiofiles==23.2.1
EOF

cat > streamlit-ui/app.py << 'EOF'
"""
SmartDoc Research Agent - Streamlit UI
"""

import streamlit as st
import requests
from typing import Dict, Any
import os

# Page config
st.set_page_config(
    page_title="SmartDoc Research Agent",
    page_icon="🔬",
    layout="wide"
)

# API Configuration
API_BASE_URL = os.getenv("AGENT_API_URL", "http://localhost:8001")

def main():
    st.title("🔬 SmartDoc Research Agent")
    st.markdown("Intelligent research assistant powered by AI")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Check API health
        try:
            response = requests.get(f"{API_BASE_URL}/health")
            if response.status_code == 200:
                st.success("✅ API Connected")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Unavailable")
    
    # Main interface
    st.header("Start Research")
    
    topic = st.text_input("Research Topic", placeholder="Enter your research topic...")
    
    if st.button("Start Research Session"):
        if topic:
            # TODO: Create research session
            st.success(f"Research session started for: {topic}")
        else:
            st.error("Please enter a research topic")
    
    # Chat interface (placeholder)
    st.header("Chat with Agent")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your research..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # TODO: Send to agent API
        with st.chat_message("assistant"):
            response = f"Echo: {prompt}"
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
EOF

echo -e "${GREEN}✅ Streamlit UI created${NC}"

# Create utility scripts
echo -e "${YELLOW}🔧 Creating utility scripts...${NC}"

cat > scripts/start-gpu.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting SmartDoc with GPU support..."
cp .env.gpu .env
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
echo "✅ Services started with GPU support"
echo "🌐 Streamlit UI: http://localhost:8501"
echo "🤖 Agent API: http://localhost:8001"
echo "📚 API Docs: http://localhost:8001/docs"
EOF

cat > scripts/start-cpu.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting SmartDoc in CPU mode..."
cp .env.cpu .env
docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
echo "✅ Services started in CPU mode"
echo "🌐 Streamlit UI: http://localhost:8501"  
echo "🤖 Agent API: http://localhost:8001"
echo "📚 API Docs: http://localhost:8001/docs"
EOF

cat > scripts/stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping SmartDoc services..."
docker-compose down
echo "✅ All services stopped"
EOF

cat > scripts/logs.sh << 'EOF'
#!/bin/bash
echo "📋 SmartDoc Service Logs"
echo "========================"
docker-compose logs -f
EOF

cat > scripts/setup-ollama.sh << 'EOF'
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
EOF

# Make scripts executable
chmod +x scripts/*.sh

echo -e "${GREEN}✅ Utility scripts created${NC}"

# Create documentation
echo -e "${YELLOW}📚 Creating documentation...${NC}"

cat > README.md << 'EOF'
# SmartDoc Research Agent

Intelligent research agent powered by LangChain that can investigate any topic using multiple sources, synthesize information, and generate structured reports.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Ollama (for local models)

### Setup

1. **Clone and setup:**
   ```bash
   git clone <your-repo>
   cd smartdoc-api
   ./scripts/setup-ollama.sh
   ```

2. **Start services:**
   - **GPU mode:** `./scripts/start-gpu.sh`
   - **CPU mode:** `./scripts/start-cpu.sh`

3. **Access:**
   - UI: http://localhost:8501
   - API: http://localhost:8001
   - Docs: http://localhost:8001/docs

### Environment Modes

- **GPU Mode**: Full local processing with GPU acceleration
- **CPU Mode**: Lighter models + optional API fallbacks

### Stopping Services
```bash
./scripts/stop.sh
```

## 🏗️ Architecture

- **FastAPI**: Agent API backend
- **Streamlit**: User interface  
- **LangChain**: Agent framework
- **ChromaDB**: Vector database
- **Redis**: Caching & sessions
- **Ollama**: Local LLM inference

## 📊 Development

```bash
# View logs
./scripts/logs.sh

# Rebuild services
docker-compose build

# Development mode
docker-compose up --build
```
EOF

cat > docs/DEVELOPMENT.md << 'EOF'
# Development Guide

## Project Structure
- `agent-api/`: FastAPI backend with LangChain agent
- `streamlit-ui/`: Streamlit frontend
- `data/`: Persistent data (ChromaDB, uploads, reports)
- `scripts/`: Utility scripts

## Environment Configuration
- `.env`: Current environment settings  
- `.env.gpu`: GPU-optimized settings
- `.env.cpu`: CPU-only settings

## Docker Services
- `agent-api`: Main agent API
- `streamlit-ui`: Web interface
- `chromadb`: Vector database
- `redis`: Caching layer
- `ollama`: Local LLM (GPU mode only)

## Development Workflow
1. Make changes to code
2. Rebuild affected services
3. Test through UI or API docs
4. Check logs for issues
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Environment files
.env
.env.local
.env.*.local

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Data directories
data/chromadb/
data/uploads/
data/reports/
data/cache/

# Logs
logs/
*.log

# Docker
.docker/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Temporary files
tmp/
temp/
EOF

echo -e "${GREEN}✅ Documentation created${NC}"

# Final summary
echo ""
echo -e "${BLUE}🎉 SmartDoc API Project Setup Complete!${NC}"
echo ""
echo -e "${YELLOW}📁 Project Structure:${NC}"
find . -type d -name "__pycache__" -prune -o -type d -print | head -20
echo ""
echo -e "${YELLOW}🚀 Next Steps:${NC}"
echo "1. cd smartdoc-api"
echo "2. ./scripts/setup-ollama.sh    # Setup local models"
echo "3. ./scripts/start-cpu.sh       # Start in CPU mode (laptop)"
echo "4. ./scripts/start-gpu.sh       # Start in GPU mode (desktop)"
echo ""
echo -e "${YELLOW}🌐 URLs:${NC}"
echo "- UI: http://localhost:8501"
echo "- API: http://localhost:8001"
echo "- Docs: http://localhost:8001/docs"
echo ""
echo -e "${GREEN}✨ Ready to develop your AI research agent!${NC}"
EOF