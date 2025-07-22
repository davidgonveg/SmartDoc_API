# SmartDoc Research Agent

Intelligent research agent powered by LangChain that can investigate any topic using multiple sources, synthesize information, and generate structured reports.

## üöÄ Quick Start

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

## üèóÔ∏è Architecture

- **FastAPI**: Agent API backend
- **Streamlit**: User interface  
- **LangChain**: Agent framework
- **ChromaDB**: Vector database
- **Redis**: Caching & sessions
- **Ollama**: Local LLM inference

## üìä Development

```bash
# View logs
./scripts/logs.sh

# Rebuild services
docker-compose build

# Development mode
docker-compose up --build
```

## ÌæÆ GPU Environment Notes

### GPU Setup Status
- [ ] Docker GPU access configured
- [ ] Ollama with GPU support tested
- [ ] Larger models (Llama3.1 8B) downloaded
- [ ] Performance benchmarks completed

### GPU vs CPU Performance Comparison
TBD after GPU environment setup
