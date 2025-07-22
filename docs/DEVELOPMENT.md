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
