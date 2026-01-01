# Vacancy Matching Implementation Summary

## Overview

This document summarizes the implementation of the vacancy matching system for the recruitment platform. The system enables matching candidates with job vacancies using semantic search and AI-powered reasoning.

## Architecture

The implementation follows the existing microservices architecture with the following components:

```
┌─────────────────┐
│   API Service   │  ← Entry point for matching requests
│  (apps/api/)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Orchestrator   │  ← LangGraph state machine for matching
│(apps/orchestrator)│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│Pinecone │ │  Gemini AI   │
│ Vector  │ │  (Reranking) │
│  Store  │ └──────────────┘
└─────────┘
```

## Implemented Modules

### 1. Schema Updates (`shared/schemas.py`)

**New Schemas:**
- **`Vacancy`**: Represents a job vacancy posting, similar to `Resume`
  - `id`: Unique identifier
  - `raw_text`: Full text description
  - `chunks`: List of `DocumentChunk` objects with embeddings
  - `processed_at`, `created_at`: Timestamps

- **`VacancyMatchResult`**: Result of candidate-vacancy matching
  - `score`: Similarity score (cosine similarity)
  - `reasoning`: AI-generated explanation of why the vacancy fits
  - `vacancy_id`: ID of the matched vacancy
  - `vacancy_text`: Text content of the vacancy
  - `candidate_id`: ID of the candidate

- **`MatchRequest`**: Request schema for matching endpoint
  - `candidate_id`: Required candidate identifier (user_id)
  - `top_k`: Optional number of results (default: 10)

**Enhanced:**
- `DocumentChunk.metadata`: Already supports `type` field to distinguish between CVs (`type: 'cv'`) and vacancies (`type: 'vacancy'`)

### 2. CV Processor Service (`services/cv-processor/main.py`)

**New Endpoint: `POST /process-vacancy`**

**Functionality:**
- Accepts a text description of a vacancy
- Splits the text into chunks (1000 chars, 800 char overlap)
- Generates embeddings using the embedding service
- Saves to Pinecone with metadata `{'type': 'vacancy'}`

**Request:**
```json
{
  "text": "We are looking for a Python developer with 5+ years of experience..."
}
```

**Response:**
```json
{
  "status": "success",
  "vacancy_id": "uuid",
  "chunks_processed": 3
}
```

**Integration:**
- Uses `VectorStore.upsert_vacancy()` to persist vacancies
- Reuses existing embedding service integration

### 3. Vector Store (`shared/pinecone_client.py`)

**New Methods:**

#### `upsert_vacancy(vacancy: Vacancy)`
- Saves vacancy chunks to Pinecone
- Each chunk includes metadata: `vacancy_id`, `text`, `type: 'vacancy'`
- Uses namespace: `"resumes"` (shared with CVs)

#### `get_candidate_embedding(candidate_id: str) -> List[float]`
- Retrieves all resume chunks for a candidate
- Computes average embedding across all chunks
- Normalizes the result for cosine similarity
- Returns `None` if candidate not found
- Uses 1024-dimensional vectors (BGE-M3 model)

#### `search_vacancies(query_vector: List[float], top_k: int = 10) -> List[Dict]`
- Searches Pinecone for vacancies matching the query vector
- Filters by `{'type': {'$eq': 'vacancy'}}`
- Returns list of results with `id`, `metadata`, and `score`

**Technical Details:**
- Uses Pinecone's query API with metadata filters
- Handles vector dimension (1024 for BGE-M3)
- Normalizes embeddings for optimal cosine similarity

### 4. Orchestrator (`apps/orchestrator/graph.py`)

**New LangGraph Implementation: `create_matching_graph()`**

**Graph Structure:**
```
Entry → fetch_candidate → search_vacancies → rerank_and_explain → End
```

**State Schema (`MatchingState`):**
- `candidate_id`: Input candidate identifier
- `candidate_embedding`: Retrieved embedding vector
- `retrieved_vacancies`: List of vacancy search results
- `vacancy_scores`: Similarity scores from Pinecone
- `match_results`: Final list of `VacancyMatchResult` objects
- `top_k`: Number of results to return

**Nodes:**

1. **`fetch_candidate_node`**
   - Fetches candidate embedding from Pinecone
   - Raises error if candidate not found
   - Updates state with `candidate_embedding`

2. **`search_vacancies_node`**
   - Uses candidate embedding to search Pinecone
   - Filters for vacancies only (`type: 'vacancy'`)
   - Retrieves top-k results
   - Updates state with `retrieved_vacancies` and `vacancy_scores`

3. **`rerank_and_explain_node`**
   - For each vacancy, uses Gemini AI to:
     - Analyze why the vacancy fits the candidate
     - Generate detailed reasoning
     - Explain skills alignment and career benefits
   - Creates `VacancyMatchResult` objects with scores and reasoning
   - Updates state with final `match_results`

**AI Prompting:**
- Uses `langchain-google-genai` (Gemini Pro)
- System prompt focuses on recruiter-style analysis
- Explains: skills match, experience alignment, career growth, potential gaps

**Entry Point: `run_match(match_request: MatchRequest)`**
- Initializes state from request
- Executes the graph
- Returns list of `VacancyMatchResult` objects

### 5. API Service (`apps/api/main.py`)

**New Endpoint: `POST /match`**

**Functionality:**
- Accepts `MatchRequest` with `candidate_id` and optional `top_k`
- Triggers the LangGraph orchestrator
- Returns ranked list of vacancy matches with AI explanations

**Request:**
```json
{
  "candidate_id": "user_123",
  "top_k": 10
}
```

**Response:**
```json
[
  {
    "score": 0.85,
    "reasoning": "This vacancy is a great fit because...",
    "vacancy_id": "vacancy_1",
    "vacancy_text": "We are looking for...",
    "candidate_id": "user_123"
  },
  ...
]
```

**Error Handling:**
- `404`: Candidate not found in Pinecone
- `500`: Internal server errors (Pinecone, Gemini, etc.)

### 6. Tests (`apps/api/tests/test_matching.py`)

**Test Coverage:**

1. **`test_match_endpoint_success`**
   - Tests successful matching flow
   - Mocks Pinecone and Gemini responses
   - Validates response structure and data

2. **`test_match_endpoint_candidate_not_found`**
   - Tests 404 error when candidate doesn't exist
   - Validates error message

3. **`test_match_endpoint_missing_candidate_id`**
   - Tests validation error (422) for missing required field

4. **`test_match_endpoint_default_top_k`**
   - Tests default `top_k=10` behavior
   - Validates that search is called correctly

5. **`test_health_endpoint`**
   - Basic health check test

**Testing Approach:**
- Uses `unittest.mock` to mock external dependencies
- Mocks Pinecone client and Gemini LLM
- Uses FastAPI `TestClient` for endpoint testing
- No actual API calls or model loading required

## Data Flow

### Vacancy Processing Flow
```
1. POST /process-vacancy
   ↓
2. Split text into chunks
   ↓
3. Call embedding service → Get embeddings
   ↓
4. Create Vacancy object with chunks (metadata: type='vacancy')
   ↓
5. Save to Pinecone via VectorStore.upsert_vacancy()
```

### Matching Flow
```
1. POST /match (candidate_id)
   ↓
2. Orchestrator: fetch_candidate_node
   → Get candidate embedding from Pinecone
   ↓
3. Orchestrator: search_vacancies_node
   → Search Pinecone with filter type='vacancy'
   → Get top-k vacancies with similarity scores
   ↓
4. Orchestrator: rerank_and_explain_node
   → For each vacancy:
     → Call Gemini AI with candidate + vacancy context
     → Generate reasoning explanation
     → Create VacancyMatchResult
   ↓
5. Return ranked list of matches with AI explanations
```

## Environment Variables

All services use environment variables for configuration:

- `PINECONE_API_KEY`: Pinecone API key (required)
- `PINECONE_INDEX_NAME`: Pinecone index name (default: "funds-search")
- `EMBEDDING_SERVICE_URL`: URL of embedding service (default: "http://embedding-service:8001")
- `GOOGLE_API_KEY`: Google API key for Gemini (required for matching)

## Integration Points

### With Existing System
- **Reuses** `shared.pinecone_client.VectorStore` for all Pinecone operations
- **Extends** existing CV processing pipeline
- **Follows** same folder structure (services/, apps/, shared/)
- **Compatible** with existing Dockerfiles and deployment scripts

### Namespace Strategy
- Both CVs and vacancies stored in `"resumes"` namespace
- Distinguished by metadata `type` field:
  - CVs: `type: 'cv'` (or no type field)
  - Vacancies: `type: 'vacancy'`

## Key Features

1. **Semantic Matching**: Uses BGE-M3 embeddings for semantic similarity
2. **AI-Powered Reasoning**: Gemini explains why each vacancy fits the candidate
3. **Scalable Architecture**: Microservices with clear separation of concerns
4. **Type Safety**: Pydantic schemas for request/response validation
5. **Error Handling**: Comprehensive error handling at all layers
6. **Test Coverage**: Unit tests with mocked dependencies

## Future Enhancements

Potential improvements:
- Add candidate resume text to Gemini context for better reasoning
- Implement caching for candidate embeddings
- Add filtering by location, salary, etc.
- Support batch matching for multiple candidates
- Add metrics and monitoring for matching quality

## Files Modified/Created

**Modified:**
- `shared/schemas.py` - Added Vacancy, VacancyMatchResult, MatchRequest
- `services/cv-processor/main.py` - Added /process-vacancy endpoint
- `shared/pinecone_client.py` - Added vacancy and candidate methods
- `apps/orchestrator/graph.py` - Added matching graph implementation
- `apps/orchestrator/__init__.py` - Exported new functions
- `apps/api/main.py` - Added /match endpoint

**Created:**
- `apps/api/tests/__init__.py` - Test package init
- `apps/api/tests/test_matching.py` - Comprehensive test suite

## CI/CD Integration

The existing `.github/workflows/ci.yml` automatically:
- Runs `pytest apps/` which includes the new tests
- Validates code with flake8
- Builds Docker images on push

No additional CI configuration needed - tests are automatically included.

