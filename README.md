# Funds Search: Multi-Agent RAG Matching System

Production-ready candidate-vacancy matching system using LangGraph orchestration, semantic embeddings (BGE-M3), and AI reasoning (Gemini).

## Project Overview & Goal

**Goal**: Match candidates with job vacancies at VC funds using semantic similarity and AI-powered reasoning.

**Architecture**: Multi-Agent RAG (Retrieval-Augmented Generation) system that:
- Processes and indexes CV/resume documents in Pinecone namespace `"cvs"`
- Processes and indexes vacancy descriptions in Pinecone namespace `"vacancies"`
- Matches candidates with vacancies using semantic similarity (BGE-M3 embeddings)
- Generates AI explanations (Gemini) explaining why each vacancy fits the candidate
- Provides a user-friendly web interface for recruiters

**Current Working State**: 
- ✅ CV processing and storage in namespace `"cvs"` with metadata `type: 'cv'`
- ✅ Vacancy processing and storage in namespace `"vacancies"` with metadata `type: 'vacancy'`
- ✅ Candidate-vacancy matching with AI reasoning using Gemini 2.5 Flash
- ✅ LangGraph orchestrator with 3-node matching workflow (fetch_candidate → search_vacancies → rerank_and_explain)

## System Architecture

```mermaid
graph TB
    subgraph "Client"
        User[👤 Recruiter] --> WebUI[📊 Streamlit UI<br/>:8501]
    end
    
    subgraph "API Gateway"
        WebUI --> API[🚀 FastAPI<br/>:8000]
        WebUI -->|Upload CV| CVProc[📄 CV Processor<br/>:8002]
    end
    
    subgraph "Orchestration"
        API -->|Search/Match| Orch[🔄 LangGraph<br/>Orchestrator]
        Orch -->|1. Retrieve| RetNode[🔍 Retrieval Node]
        Orch -->|2. Analyze| AnaNode[🤖 Analysis Node]
    end
    
    subgraph "Services"
        RetNode -->|Embed Query| EmbSvc[🧮 Embedding Service<br/>BGE-M3 :8001]
        RetNode -->|Vector Search| PC[(📦 Pinecone<br/>Vector DB)]
        CVProc -->|Get Embeddings| EmbSvc
        CVProc -->|Store Vectors| PC
    end
    
    subgraph "AI"
        AnaNode -->|Reasoning| Gemini[🧠 Gemini 2.5 Flash<br/>LLM]
    end
    
    subgraph "Data Models"
        Schemas[📋 Pydantic Schemas<br/>shared/schemas.py]
        Schemas -.->|Used by| API
        Schemas -.->|Used by| Orch
        Schemas -.->|Used by| CVProc
    end
    
    style User fill:#e1f5ff
    style WebUI fill:#c8e6c9
    style API fill:#c8e6c9
    style Orch fill:#fff9c4
    style EmbSvc fill:#f3e5f5
    style CVProc fill:#f3e5f5
    style PC fill:#ffccbc
    style Gemini fill:#ffccbc
    style Schemas fill:#e8f5e9
```

## Component Descriptions

### Apps (`apps/`)

| Component | Port | Description |
|-----------|------|-------------|
| **api** | 8000 | FastAPI REST API + LangGraph orchestrator. Multi-stage Docker build (<500MB image). |
| **web_ui** | 8501 | Streamlit dashboard for CV upload and match viewing. Requires `BACKEND_API_URL` and `CV_PROCESSOR_URL` env vars. |
| **orchestrator** | - | LangGraph state machines for search and matching workflows. Two graphs: search workflow and matching workflow. |

### Services (`services/`)

| Component | Port | Description |
|-----------|------|-------------|
| **embedding-service** | 8001 | BGE-M3 embedding model service (1024-dim vectors). Requires 2-4GB RAM. |
| **cv-processor** | 8002 | PDF→Markdown (Docling), chunking, vectorization. Uses `run_in_threadpool` for async. Processes CVs and vacancies. |
| **vc-worker** | 8003 | Placeholder for future job scraping functionality. |

**Architecture Rule**: Services must **not** import from `apps/`. They may only import from `shared/` (schemas, pinecone_client).

### Shared (`shared/`)

- **schemas.py**: Pydantic v2 models (single source of truth) - Job, Resume, Vacancy, MatchResult, VacancyMatchResult, etc.
- **pinecone_client.py**: Pinecone client wrapper with namespace management (`"cvs"` for candidates, `"vacancies"` for job postings)

## Data Schemas

All schemas in `shared/schemas.py` (Pydantic v2, single source of truth):

### Core Models

**`DocumentChunk`** - Semantic text chunks with embeddings
```python
{
  "text": str,              # Chunk content
  "metadata": Dict,         # Additional metadata (type: 'cv' | 'vacancy')
  "embedding": List[float]  # 1024-dim BGE-M3 vector
}
```

**`Resume`** - Candidate CV/resume
```python
{
  "id": str,                    # Unique resume ID
  "user_id": str,               # Candidate identifier
  "raw_text": str,              # Full CV text
  "chunks": List[DocumentChunk], # Processed chunks (type: 'cv')
  "processed_at": datetime,
  "created_at": datetime
}
```

**`Vacancy`** - Job posting
```python
{
  "id": str,                    # Unique vacancy ID
  "raw_text": str,              # Full job description
  "chunks": List[DocumentChunk], # Processed chunks (type: 'vacancy')
  "processed_at": datetime,
  "created_at": datetime
}
```

**`Job`** - Job opening (search results)
```python
{
  "id": str,
  "company": str,
  "title": str | None,
  "raw_text": str,
  "vector": List[float] | None,
  "url": str | None,
  "location": str | None,
  "remote": bool | None,
  "created_at": datetime
}
```

### Request/Response Models

**`SearchRequest`** - Job search query
```python
{
  "query": str,              # Required: search text
  "location": str | None,    # Optional filter
  "role": str | None,        # Optional job title filter
  "remote": bool | None,     # Optional remote filter
  "user_id": str | None      # Optional personalization
}
```

**`MatchRequest`** - Candidate-vacancy matching
```python
{
  "candidate_id": str,       # Required: user_id
  "top_k": int = 10          # Number of matches
}
```

**`MatchResult`** - Search/match result
```python
{
  "score": float,            # Cosine similarity (0-1)
  "reasoning": str,          # AI-generated explanation
  "job": Job,                # Matched job posting
  "resume": Resume | None    # Candidate resume (if applicable)
}
```

**`VacancyMatchResult`** - Candidate-vacancy match
```python
{
  "score": float,            # Similarity score
  "reasoning": str,          # Why vacancy fits candidate
  "vacancy_id": str,
  "vacancy_text": str,
  "candidate_id": str
}
```

## Setup & Environment Variables

### Prerequisites
- Docker & Docker Compose
- `.env` file with required API keys

### Environment Variables

**Root `.env` file:**
```bash
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=funds-search
GOOGLE_API_KEY=your_key
```

**Service-Specific Variables:**

| Service | Required Variables |
|---------|-------------------|
| **API** | `PINECONE_API_KEY`, `GOOGLE_API_KEY`, `EMBEDDING_SERVICE_URL` (default: `http://embedding-service:8001`) |
| **Web UI** | `BACKEND_API_URL` (default: `http://api:8000`), `CV_PROCESSOR_URL` (default: `http://cv-processor:8001`) |
| **CV Processor** | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `EMBEDDING_SERVICE_URL` |
| **Embedding** | `CUDA_VISIBLE_DEVICES` (optional, for GPU) |

### Quick Start

```bash
git clone <repo-url>
cd funds-search
cp .env.example .env  # Add your API keys
docker-compose up --build
```

**Access:**
- 🌐 Web UI: http://localhost:8501
- 🔌 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

## API & UI Usage Guide

### Main API (`:8000`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/search` | POST | Search jobs (returns `List[MatchResult]`) |
| `/match` | POST | Match candidate→vacancies (returns `List[VacancyMatchResult]`) |

**`POST /search`** - Job search with filters
```json
{
  "query": "software engineer",
  "location": "San Francisco",
  "role": "engineer",
  "remote": true,
  "user_id": "optional"
}
```

**`POST /match`** - Candidate-vacancy matching
```json
{
  "candidate_id": "user123",
  "top_k": 10
}
```

**Response Example:**
```json
[
  {
    "score": 0.85,
    "reasoning": "This vacancy is a great fit because the candidate has 5+ years of Python experience matching the job requirements...",
    "vacancy_id": "vacancy_1",
    "vacancy_text": "We are looking for a Python developer...",
    "candidate_id": "user123"
  }
]
```

### CV Processor (`:8002`)

- **`POST /process-cv`** - Upload CV (multipart: `user_id`, `file`)
  ```bash
  curl -X POST "http://localhost:8002/process-cv?user_id=user123" \
    -F "file=@resume.pdf"
  ```

- **`POST /process-vacancy`** - Process vacancy (JSON: `vacancy_id`, `text`)
  ```bash
  curl -X POST http://localhost:8002/process-vacancy \
    -H "Content-Type: application/json" \
    -d '{"vacancy_id": "vac1", "text": "We are looking for..."}'
  ```

### Embedding Service (`:8001`)

- **`POST /embed`** - Generate embeddings
```json
{
  "texts": ["text to embed"]
}
```

### Testing Steps

1. **Start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Upload a CV via Web UI:**
   - Navigate to http://localhost:8501
   - Upload a PDF resume
   - Enter user_id
   - Verify CV is processed

3. **Process a vacancy:**
   ```bash
   curl -X POST http://localhost:8002/process-vacancy \
     -H "Content-Type: application/json" \
     -d '{
       "vacancy_id": "vac1",
       "text": "We are looking for a Python developer with 5+ years of experience in backend development..."
     }'
   ```

4. **Match candidate with vacancies:**
   ```bash
   curl -X POST http://localhost:8000/match \
     -H "Content-Type: application/json" \
     -d '{"candidate_id": "user123", "top_k": 5}'
   ```

5. **Verify results:**
   - Check response contains `VacancyMatchResult` objects
   - Verify scores are between 0-1
   - Verify reasoning text is generated by Gemini

## Data Flow

### CV Processing Pipeline

```mermaid
sequenceDiagram
    participant U as User
    participant W as Web UI
    participant C as CV Processor
    participant E as Embedding Service
    participant P as Pinecone
    
    U->>W: Upload PDF
    W->>C: POST /process-cv
    C->>C: PDF → Markdown (Docling)
    C->>C: Chunk (1000 chars, 800 overlap)
    C->>E: POST /embed
    E-->>C: Embeddings (1024-dim)
    C->>P: Store vectors (namespace: "cvs", type: "cv")
    C-->>W: Success
    W-->>U: CV Processed
```

### Vacancy Processing Pipeline

```mermaid
sequenceDiagram
    participant Admin
    participant C as CV Processor
    participant E as Embedding Service
    participant P as Pinecone
    
    Admin->>C: POST /process-vacancy {text}
    C->>C: Chunk (1000 chars, 800 overlap)
    C->>E: POST /embed
    E-->>C: Embeddings (1024-dim)
    C->>P: Store vectors (namespace: "vacancies", type: "vacancy")
    C-->>Admin: Success
```

### Matching Pipeline (LangGraph Workflow)

```mermaid
sequenceDiagram
    participant U as User
    participant A as API
    participant O as Orchestrator
    participant R as Retrieval Node
    participant An as Analysis Node
    participant P as Pinecone
    participant G as Gemini
    
    U->>A: POST /match {candidate_id}
    A->>O: Run matching graph
    O->>R: fetch_candidate_node
    R->>P: Get from "cvs" namespace
    P-->>R: Candidate vector (avg of chunks)
    R->>O: candidate_embedding
    O->>R: search_vacancies_node
    R->>P: Search "vacancies" namespace (filter: type='vacancy')
    P-->>R: Top K vacancies + scores
    R->>O: retrieved_vacancies
    O->>An: rerank_and_explain_node
    An->>G: Generate reasoning (why vacancy fits)
    G-->>An: AI explanations
    An->>O: match_results
    O-->>A: List[VacancyMatchResult]
    A-->>U: Ranked matches + scores + reasoning
```

## Orchestration Workflows

### Search Workflow (LangGraph)
1. **Retrieval Node**: Embed query → Search Pinecone → Get top 10 jobs
2. **Analysis Node**: Gemini analyzes each match → Generate reasoning → Return ranked results

### Matching Workflow (LangGraph)
**Graph Structure**: `Entry → fetch_candidate → search_vacancies → rerank_and_explain → End`

**State Schema (`MatchingState`):**
- `candidate_id`: Input candidate identifier
- `candidate_embedding`: Retrieved embedding vector (average of CV chunks)
- `retrieved_vacancies`: List of vacancy search results
- `vacancy_scores`: Similarity scores from Pinecone
- `match_results`: Final list of `VacancyMatchResult` objects
- `top_k`: Number of results to return

**Nodes:**
1. **`fetch_candidate_node`**: Fetches candidate embedding from Pinecone namespace `"cvs"`. Computes average of all CV chunks and normalizes. Raises error if candidate not found.
2. **`search_vacancies_node`**: Uses candidate embedding to search Pinecone namespace `"vacancies"` with filter `type: 'vacancy'`. Retrieves top-k results with similarity scores.
3. **`rerank_and_explain_node`**: For each vacancy, uses Gemini AI to:
   - Analyze why the vacancy fits the candidate
   - Generate detailed reasoning
   - Explain skills alignment and career benefits
   - Creates `VacancyMatchResult` objects with scores and reasoning

**AI Prompting**: Uses `langchain-google-genai` (Gemini 2.5 Flash) with system prompt focused on recruiter-style analysis explaining skills match, experience alignment, career growth, and potential gaps.

## Key Implementation Details

| Aspect | Implementation |
|--------|----------------|
| **Schemas** | Single source of truth: `shared/schemas.py` (Pydantic v2) |
| **Ports** | CV Processor: 8002 (external) → 8001 (internal) |
| **CV Processing** | `run_in_threadpool` for Docling (non-blocking async) |
| **Image Size** | API: multi-stage build (4GB → <500MB) |
| **Pinecone** | Namespaces: `"cvs"` (resumes), `"vacancies"` (jobs)<br/>Metadata filter: `type: 'cv'` or `type: 'vacancy'` |
| **Chunking** | 1000 chars, 800 overlap |
| **Embeddings** | BGE-M3 (1024 dimensions), L2-normalized |
| **LLM** | Gemini 2.5 Flash (reasoning & reranking) |
| **Candidate Embedding** | Average of all CV chunks, normalized |
| **Vector Search** | Cosine similarity with metadata filters |

## Tech Stack

- **Orchestration**: LangGraph state machines (search & matching workflows)
- **LLM**: Google Gemini 2.5 Flash (reasoning)
- **Embeddings**: BAAI/bge-m3 (1024-dim)
- **Vector DB**: Pinecone (namespaces: `cvs`, `vacancies`; metadata filter: `type`)
- **API**: FastAPI/Uvicorn
- **UI**: Streamlit
- **Document Parser**: Docling (PDF→Markdown, async via `run_in_threadpool`)
- **Deployment**: Docker, Azure Container Apps, GitHub Actions

## Deployment

### Azure Container Apps

**Registry**: `fundssearchregistry.azurecr.io`

| Service | Container App | Port | Workflow |
|---------|---------------|------|----------|
| API | `api` | 8000 | `deploy-api.yml` (<500MB image) |
| Web UI | `web-ui` | 8501 | `deploy-web-ui.yml` |
| CV Processor | `cv-processor` | 8001 | `deploy-cv-processor.yml` |
| Embedding | `embedding-service` | 8001 | `deploy-embedding.yml` |

### CI/CD

GitHub Actions workflows (`.github/workflows/`):
- **`ci.yml`**: Runs `pytest apps/` (includes matching tests), validates with flake8, builds Docker images
- **`deploy-*.yml`**: Individual deployment workflows for each service

**Optimizations:**
- API service uses multi-stage Docker build to reduce image size from 4GB to <500MB
- Embedding service pre-downloads model during build
- All services use dependency caching

## Project Structure

```
funds-search/
├── apps/
│   ├── api/              # FastAPI REST API + LangGraph
│   ├── orchestrator/     # LangGraph state machines
│   └── web_ui/           # Streamlit dashboard
├── services/
│   ├── cv-processor/     # PDF→Markdown, chunking, vectorization
│   ├── embedding-service/ # BGE-M3 model service
│   └── vc-worker/        # (Placeholder)
├── shared/
│   ├── schemas.py        # Pydantic v2 models (SSOT)
│   └── pinecone_client.py # Vector DB wrapper
├── requirements/         # Dependency management
│   ├── base.txt       # Common deps
│   ├── ml.txt         # ML (torch, transformers)
│   └── api.txt        # API (no ML)
└── .github/workflows/ # CI/CD (Azure deployments)
```

## Documentation

- **[Services README](services/README.md)** - Microservices architecture
- **[Apps README](apps/README.md)** - Application components
- **[Shared README](shared/README.md)** - Shared modules
