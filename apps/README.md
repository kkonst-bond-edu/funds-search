# Apps Directory

This directory contains application-level components that orchestrate the microservices and provide user-facing interfaces.

## Overview

The apps directory contains:

- **api**: FastAPI REST API with LangGraph orchestrator
- **orchestrator**: LangGraph state machine for search and matching workflows
- **web_ui**: Streamlit web interface for user interactions

## Architecture

### API Service (`apps/api/`)

**Purpose**: Main REST API that orchestrates search and matching operations.

**Key Features**:
- FastAPI REST endpoints (`/search`, `/match`)
- Integrates with LangGraph orchestrator
- Lightweight (no ML libraries) - reduces image size to <500MB
- Uses multi-stage Docker build for optimization

**Endpoints**:
- `POST /search` - Search for job openings
- `POST /match` - Match candidate with vacancies
- `GET /health` - Health check

**Dependencies**: 
- Uses `requirements/api.txt` (langchain, langgraph, fastapi)
- **Excludes**: torch, transformers, docling (keeps image small)

**Port**: 8000

### Orchestrator (`apps/orchestrator/`)

**Purpose**: LangGraph state machine that coordinates retrieval and analysis.

**Key Features**:
- **Retrieval Node**: Fetches embeddings from embedding-service, searches Pinecone
- **Analysis Node**: Uses Gemini AI to analyze matches and generate reasoning
- **State Management**: TypedDict for state passing between nodes
- **Lazy Initialization**: Services initialized on first use

**Workflows**:
1. **Search Workflow**: Query → Embedding → Pinecone Search → AI Analysis → Results
2. **Match Workflow**: Candidate ID → Fetch CV → Search Vacancies → AI Rerank → Results

**Integration**:
- Calls `embedding-service` for embeddings
- Uses `shared.pinecone_client` for vector search
- Uses `langchain_google_genai` for AI analysis

### Web UI (`apps/web_ui/`)

**Purpose**: Streamlit dashboard for user interactions.

**Key Features**:
- **CV Upload**: Upload PDF resumes for processing
- **Vacancy Processing**: Paste vacancy descriptions
- **Matching Dashboard**: Find and display candidate-vacancy matches
- **Progress Bars**: Visual feedback for long-running operations
- **Health Monitoring**: Real-time service status indicators
- **Retry Logic**: Exponential backoff for cold start handling

**User Flows**:
1. Upload CV → Process → Store in Pinecone
2. Process Vacancy → Generate embeddings → Store in Pinecone
3. Find Matches → Search → Display with AI reasoning

**Dependencies**: streamlit, httpx, pydantic

**Port**: 8501

## Legacy Rules Established

### 1. Path Setup Pattern
All apps add project root to Python path:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
```

### 2. Shared Module Imports
Apps import from `shared/` directory:
- `shared.schemas` - Pydantic models (Job, Resume, Vacancy, MatchResult, etc.)
- `shared.pinecone_client` - VectorStore client

### 3. Dockerfile Standards
- **PYTHONPATH**: Always set to `/app`
- **Multi-stage builds**: Used for API service to reduce image size
- **Base image**: `python:3.10-slim` for consistency

### 4. Service Communication
- Use environment variables for service URLs:
  - `BACKEND_API_URL` (default: `http://api:8000`)
  - `CV_PROCESSOR_URL` (default: `http://cv-processor:8001`)
  - `EMBEDDING_SERVICE_URL` (default: `http://embedding-service:8001`)

### 5. Error Handling
- Retry logic with exponential backoff for cold starts
- Timeout configuration: 300 seconds (5 minutes) for long operations
- User-friendly error messages

### 6. Logging
Consistent logging pattern:
```python
import logging
logger = logging.getLogger(__name__)
```

## Data Flow

### Search Flow
```
User → Web UI → API /search → Orchestrator
  → Embedding Service (query embedding)
  → Pinecone (vector search)
  → Gemini AI (analysis)
  → Results → Web UI → User
```

### Match Flow
```
User → Web UI → API /match → Orchestrator
  → Pinecone (fetch candidate CV)
  → Pinecone (search vacancies)
  → Gemini AI (rerank & reasoning)
  → Results → Web UI → User
```

### CV Processing Flow
```
User → Web UI → CV Processor
  → Docling (PDF → Markdown)
  → Embedding Service (chunk embeddings)
  → Pinecone (store in "cvs" namespace)
  → Response → Web UI → User
```

## Deployment

### API Service
- **Workflow**: `.github/workflows/deploy-api.yml`
- **Image**: `fundssearchregistry.azurecr.io/api`
- **Container App**: `api`
- **Optimization**: Multi-stage build reduces image from 4GB to <500MB

### Web UI
- **Workflow**: `.github/workflows/deploy-web-ui.yml`
- **Image**: `fundssearchregistry.azurecr.io/web-ui`
- **Container App**: `web-ui`
- **Port**: 8501

## Development

### Running Locally

```bash
# Start API
docker-compose up api

# Start Web UI
docker-compose up web-ui

# Start all
docker-compose up
```

### Testing

```bash
# API tests
cd apps/api
pytest tests/

# Test API endpoints
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "user123", "top_k": 10}'
```

## Dependencies

### API Service
- Uses `requirements/api.txt` (lightweight, no ML libraries)
- LangChain, LangGraph for orchestration
- FastAPI, Uvicorn for web framework

### Web UI
- Uses `apps/web_ui/requirements.txt`
- Streamlit for UI
- httpx for HTTP requests

## Status

- ✅ **API**: Production ready, optimized with multi-stage build
- ✅ **Orchestrator**: Production ready, LangGraph workflows functional
- ✅ **Web UI**: Production ready, with progress bars and health monitoring

