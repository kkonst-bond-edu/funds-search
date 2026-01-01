# Services Directory

This directory contains all microservices that form the backend infrastructure of the funds-search system. Each service is independently deployable and follows microservices architecture principles.

## Overview

The services directory contains specialized microservices that handle specific responsibilities:

- **embedding-service**: Generates semantic embeddings using BGE-M3 model
- **cv-processor**: Processes CV/resume documents and stores them in vector database
- **vc-worker**: (Future) Scrapes and processes VC fund job postings

## Architecture Principles

### Service Independence
- Each service has its own Dockerfile and requirements.txt
- Services communicate via HTTP/REST APIs
- No direct code dependencies between services
- Shared code lives in `../shared/` directory

### Port Configuration
- **embedding-service**: Port 8001 (internal), 8001 (external)
- **cv-processor**: Port 8001 (internal), 8002 (external)
- **vc-worker**: Port 8003 (internal), 8003 (external)

### Environment Variables
All services use `PYTHONPATH=/app` to ensure consistent module resolution. This allows services to import from `shared/` directory.

### Legacy Rules Established

1. **Path Setup Pattern**: All services add project root to Python path:
   ```python
   import sys
   from pathlib import Path
   sys.path.append(str(Path(__file__).parent.parent.parent))
   ```

2. **Shared Modules**: Services import from `shared/` directory:
   - `shared.schemas` - Pydantic data models
   - `shared.pinecone_client` - Vector database client

3. **Dockerfile Standards**:
   - Base image: `python:3.10-slim`
   - Working directory: `/app`
   - PYTHONPATH: `/app` (set via ENV)
   - Copy entire project: `COPY . .` (for shared modules)

4. **Service Communication**:
   - Use service names from docker-compose (e.g., `http://embedding-service:8001`)
   - Use internal ports for inter-service communication
   - Use external ports for host access

## Service Details

### embedding-service
**Purpose**: Generates 1024-dimensional embeddings using BAAI/bge-m3 model.

**Key Features**:
- Loads model once on startup (lifespan pattern)
- Supports batch processing
- GPU/CPU auto-detection
- Normalized vectors for cosine similarity

**Dependencies**: torch, transformers (heavy ML libraries)

**See**: `embedding-service/README.md` for full documentation

### cv-processor
**Purpose**: Processes PDF/DOCX resumes, generates embeddings, stores in Pinecone.

**Key Features**:
- Document conversion (PDF → Markdown via Docling)
- Text chunking (1000 chars, 800 overlap)
- Calls embedding-service for vector generation
- Stores in Pinecone namespace "cvs"

**Dependencies**: docling, torch, transformers (heavy ML libraries)

**See**: `cv-processor/README.md` for full documentation

### vc-worker
**Purpose**: (Future) Scrapes VC fund websites and processes job postings.

**Status**: Placeholder service, not yet implemented

## Deployment

Each service can be deployed independently to Azure Container Apps:

- **embedding-service**: `.github/workflows/deploy-embedding.yml`
- **cv-processor**: `.github/workflows/deploy-cv-processor.yml`
- **vc-worker**: (Future deployment workflow)

## Development

### Running Locally

```bash
# Start all services
docker-compose up

# Start specific service
docker-compose up embedding-service
docker-compose up cv-processor

# View logs
docker-compose logs -f embedding-service
```

### Adding a New Service

1. Create service directory: `services/new-service/`
2. Create `main.py` with FastAPI app
3. Create `Dockerfile` with `PYTHONPATH=/app`
4. Create `requirements.txt`
5. Add service to `docker-compose.yml`
6. Add deployment workflow (if needed)

### Testing

Each service should have its own test suite:
- `services/embedding-service/tests/` - Example test structure
- Use pytest for testing
- Mock external dependencies (Pinecone, other services)

## Dependencies Management

Services use different dependency sets:

- **Lightweight services** (api): Use `requirements/api.txt` (no ML libraries)
- **ML services** (embedding, cv-processor): Use `requirements/ml.txt` (includes torch, transformers)
- **Base dependencies**: Common in `requirements/base.txt`

See `../requirements/` directory for dependency organization.

## Status

- ✅ **embedding-service**: Production ready
- ✅ **cv-processor**: Production ready
- 🚧 **vc-worker**: Placeholder, not implemented

