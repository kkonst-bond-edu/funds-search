# API Service

FastAPI service for vacancy search, matching, and orchestration functionality.

## Endpoints

### Vacancy Search

#### POST `/api/v1/vacancies/search`

Search for vacancies using structured filter parameters.

**Request Body:**
```json
{
  "role": "Software Engineer",
  "skills": ["Python", "FastAPI"],
  "location": "San Francisco",
  "is_remote": true,
  "company_stages": ["Seed", "Series A"],
  "industry": "AI",
  "min_salary": 120000
}
```

**Response:**
```json
[
  {
    "title": "Senior Backend Engineer",
    "company_name": "Example Corp",
    "company_stage": "Series A",
    "location": "San Francisco, CA",
    "industry": "AI",
    "salary_range": "$150k-$200k",
    "description_url": "https://example.com/job",
    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
    "remote_option": true
  }
]
```

**Query Parameters:**
- `use_firecrawl` (bool, default: false): Use Firecrawl for real-time search
- `use_mock` (bool, default: false): Use mock data for testing

#### POST `/api/v1/vacancies/chat`

Conversational vacancy search using natural language. Delegates to the **LangGraph Orchestrator** to perform a multi-agent search.

**Request Body:**
```json
{
  "message": "I want to work as a Python engineer in a series A AI startup",
  "persona": {
    "technical_skills": ["Python", "Django"],
    "preferred_company_stages": ["Series A"]
  },
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

**Response:**
```json
{
  "vacancies": [...],
  "summary": "I found 3 matching vacancies...",
  "updated_persona": {
    "technical_skills": ["Python", "Django", "FastAPI"],
    ...
  },
  "search_stats": {
    "total_after_filters": 3,
    ...
  },
  "debug_info": { ... }
}
```

**How it works:**
1. **Talent Strategist**: Updates the `persona` based on the `message` and `history`.
2. **Job Scout**: Generates a hybrid search query (Semantic + Metadata Filters) from the updated persona.
3. **Orchestrator**: Executes the search in Pinecone.
4. **Response**: Returns vacancies, summary, and the updated persona (for the frontend to persist).

#### POST `/match`

Run the matching orchestrator for a candidate-vacancy match request. Used when a candidate ID is known.

**Request Body:**
```json
{
  "candidate_id": "user_123",
  "top_k": 10
}
```

**Response:**
List of `VacancyMatchResult` objects with scores and reasoning.

#### GET `/api/v1/vacancies/health`

Health check endpoint for the vacancy search service.

## Environment Variables

- `DEEPSEEK_API_KEY`: Required for chat search functionality
- `PINECONE_API_KEY`: Required for vector search
- `EMBEDDING_SERVICE_URL`: URL of the embedding service (default: `http://embedding-service:8001`)
- `CV_PROCESSOR_URL`: URL of the CV processor service (default: `http://cv-processor:8001`)

## Architecture

The API service acts as a gateway to the **Orchestrator**. It doesn't contain heavy business logic itself but routes requests to the appropriate LangGraph workflows.

- **FastAPI**: Handles HTTP requests and validation.
- **LangGraph**: Manages state and agent execution.
- **Agents**: Talent Strategist, Job Scout, Matchmaker.
