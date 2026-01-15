# API Service

FastAPI service for vacancy search, matching, and orchestration functionality.

## Endpoints

### POST `/api/v1/vacancies/search`

Structured vacancy search against the Pinecone index.

**Request Body (VacancyFilter):**
```json
{
  "role": "Software Engineer",
  "skills": ["Python", "FastAPI"],
  "location": "San Francisco",
  "is_remote": true,
  "company_stages": ["Seed", "Series A"],
  "industry": "AI",
  "min_salary": 120000,
  "category": "Engineering",
  "experience_level": "Mid"
}
```

**Query Parameters:**
- `use_firecrawl` (bool, default: false): Use Firecrawl for live search
- `use_mock` (bool, default: false): Use mock data for testing
- `required_keywords` (list, optional): Required keywords that must appear

**Response (VacancySearchResponse):**
```json
{
  "vacancies": [
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
  ],
  "total_in_db": 2500,
  "initial_vector_matches": 50,
  "total_after_filters": 12
}
```

### POST `/chat/stream`

Streaming conversational search powered by LangGraph. Uses server-sent events (SSE)
to emit workflow steps and intermediate state updates.

**Request Body:**
```json
{
  "message": "I want a remote Python role",
  "history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ],
  "persona": {
    "technical_skills": ["Python", "Django"]
  },
  "user_profile": null,
  "skip_questions": false
}
```

**Streaming Notes:**
- Emits `on_chain_start/on_chain_end` for nodes: `strategist`, `job_scout`, `matchmaker`, `validator`.
- Emits `on_tool_start/on_tool_end` for `search_vacancies_tool`.
- State updates include:
  - `candidate_pool`: list of vacancies
  - `match_results`: matchmaker summaries and scores (0–10)
  - `missing_info`: missing field ids
  - `missing_questions`: human-friendly questions

**Skip Flow:**
- Send `skip_questions=true` to bypass clarification and proceed to search.

### POST `/api/v1/vacancies/chat`

Non-streaming conversational search. Updates persona, interprets message, and returns
vacancies with a summary and debug payload.

**Request Body:**
```json
{
  "message": "I want to work as a Python engineer in a Series A AI startup",
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
    "technical_skills": ["Python", "Django", "FastAPI"]
  },
  "debug_info": {
    "friendly_reasoning": "...",
    "user_persona": {...},
    "search_filters": {...}
  },
  "persona_applied": false
}
```

### POST `/search`

Legacy search endpoint (simple query-based matching).

**Request Body:**
```json
{
  "query": "remote python backend",
  "location": "London",
  "role": "Backend Engineer",
  "remote": true,
  "user_id": "user_123"
}
```

### POST `/match`

Candidate-to-vacancy matching using a processed CV.

**Request Body:**
```json
{
  "candidate_id": "user_123",
  "top_k": 10
}
```

**Response:** List of `VacancyMatchResult` objects with scores and reasoning.

### Diagnostics & Health

- `GET /api/v1/system/diagnostics` (also `/system/diagnostics` fallback)
- `GET /health` and `GET /api/v1/health`
- `GET /api/v1/vacancies/health`

## Environment Variables

- `PINECONE_API_KEY`: Required for vector search
- `PINECONE_INDEX_NAME`: Pinecone index name (default: `funds-search`)
- `EMBEDDING_SERVICE_URL`: Embedding service URL (default: `http://embedding-service:8001`)
- `CV_PROCESSOR_URL`: CV processor URL (default: `http://cv-processor:8001`)
- `ACTIVE_AGENT`: LLM provider selector (e.g., `deepseek`, `openai`, `anthropic`)
- Provider keys as needed: `DEEPSEEK_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`

## Architecture

The API service acts as a gateway to the **Orchestrator** and vacancy search system.

- **FastAPI**: Handles HTTP requests and validation
- **LangGraph**: Manages stateful agent execution for `/chat/stream`
- **Agents**: Talent Strategist, Job Scout, Matchmaker
