# API Reference

## Base URLs

- **External (Host)**: `http://localhost:8000` (for local development)
- **Internal (Docker Network)**: `http://api:8000` (for service-to-service communication)

## Health & Diagnostics

### `GET /health`

Simple health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### `GET /api/v1/system/diagnostics`

Comprehensive system diagnostics checking all services.

**Response:**
```json
{
  "status": "ok",
  "services": {
    "cv_processor": {
      "status": "ok",
      "latency": 45.2,
      "error": null,
      "error_type": null
    },
    "embedding_service": {
      "status": "ok",
      "latency": 12.5,
      "error": null,
      "error_type": null
    },
    "pinecone": {
      "status": "ok",
      "latency": 234.1,
      "error": null,
      "error_type": null
    },
    "llm_provider": {
      "status": "ok",
      "latency": 567.8,
      "error": null,
      "error_type": null
    }
  },
  "timestamp": "2024-12-XXT..."
}
```

**Status Values:**
- `"ok"`: Service is healthy
- `"error"`: Service failed
- `"timeout"`: Service did not respond in time

**Example:**
```bash
curl http://localhost:8000/api/v1/system/diagnostics
```

## Vacancy Search Endpoints

### `POST /api/v1/vacancies/search`

Search for vacancies using structured filter parameters.

**Request Body:**
```json
{
  "role": "Software Engineer",
  "skills": ["Python", "FastAPI", "PostgreSQL"],
  "location": "San Francisco",
  "is_remote": true,
  "company_stages": ["Seed", "SeriesA"],
  "industry": "Logistics",
  "min_salary": 120000
}
```

**Query Parameters:**
- `use_firecrawl` (bool, default: false): Use Firecrawl for real-time search
- `use_mock` (bool, default: false): Use mock data for testing

**Response:**
```json
[
  {
    "title": "Senior Backend Engineer",
    "company_name": "LogiTech AI",
    "company_stage": "SeriesA",
    "location": "San Francisco, CA",
    "industry": "Logistics",
    "salary_range": "$150k-$200k",
    "description_url": "https://logitech-ai.com/careers/backend-engineer",
    "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
    "remote_option": true
  },
  {
    "title": "ML Engineer - Supply Chain Optimization",
    "company_name": "RouteOptima",
    "company_stage": "Seed",
    "location": "New York, NY",
    "industry": "Logistics",
    "salary_range": "$130k-$170k",
    "description_url": "https://routeoptima.com/jobs/ml-engineer",
    "required_skills": ["Python", "TensorFlow", "PyTorch", "Kubernetes", "GCP"],
    "remote_option": false
  }
]
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/vacancies/search?use_firecrawl=false" \
  -H "Content-Type: application/json" \
  -d '{
    "role": "Engineer",
    "skills": ["Python", "FastAPI"],
    "location": "San Francisco",
    "is_remote": true,
    "company_stages": ["Seed", "SeriesA"],
    "industry": "Logistics",
    "min_salary": 120000
  }'
```

### `POST /api/v1/vacancies/chat`

Conversational vacancy search using natural language. The AI agent interprets your message and extracts search parameters automatically.

**Request Body:**
```json
{
  "message": "I want to work as a Python engineer in a series A AI startup"
}
```

**Response:**
```json
{
  "vacancies": [
    {
      "title": "Senior Python Engineer",
      "company_name": "AI Startup Inc",
      "company_stage": "Series A",
      "location": "San Francisco, CA",
      "industry": "AI",
      "salary_range": "$150k-$200k",
      "description_url": "https://example.com/job",
      "required_skills": ["Python", "FastAPI", "PostgreSQL"],
      "remote_option": true
    }
  ],
  "summary": "I found 3 matching vacancies for your search. These Python engineering roles at Series A AI startups match your criteria, with opportunities in San Francisco and remote positions available."
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/vacancies/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to work as a Python engineer in a series A AI startup"
  }'
```

### `GET /api/v1/vacancies/health`

Health check for vacancy search service. Includes Firecrawl configuration status.

**Response:**
```json
{
  "status": "ok",
  "service": "vacancy-search",
  "version": "1.0.0",
  "firecrawl_configured": true
}
```

## Matching Endpoints

### `POST /search`

Job search with filters (legacy endpoint).

**Request Body:**
```json
{
  "query": "software engineer",
  "location": "San Francisco",
  "role": "engineer",
  "remote": true,
  "user_id": "optional"
}
```

**Response:**
```json
[
  {
    "score": 0.85,
    "reasoning": "This job posting matches your profile because...",
    "job": {
      "id": "job_1",
      "company": "Example Corp",
      "title": "Senior Software Engineer",
      "raw_text": "We are looking for...",
      "url": "https://example.com/job",
      "location": "San Francisco, CA",
      "remote": true
    },
    "resume": null
  }
]
```

### `POST /match`

Match candidate with vacancies using their CV.

**Request Body:**
```json
{
  "candidate_id": "user123",
  "top_k": 10
}
```

**Response:**
```json
[
  {
    "score": 0.85,
    "reasoning": "This vacancy is a great fit because the candidate has 5+ years of Python experience matching the job requirements...",
    "vacancy_id": "vacancy_1",
    "vacancy_text": "We are looking for a Python developer...",
    "candidate_id": "user123"
  },
  {
    "score": 0.78,
    "reasoning": "The candidate's experience in backend development aligns well with this role...",
    "vacancy_id": "vacancy_2",
    "vacancy_text": "Backend engineer position...",
    "candidate_id": "user123"
  }
]
```

**Example:**
```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "user123",
    "top_k": 10
  }'
```

## CV Processor Endpoints

**Base URL**: `http://localhost:8002` (external) / `http://cv-processor:8001` (internal)

### `POST /process-cv`

Upload and process a CV/resume PDF.

**Query Parameters:**
- `user_id` (required): Unique identifier for the candidate

**Request:** Multipart form data with `file` field

**Example:**
```bash
curl -X POST "http://localhost:8002/process-cv?user_id=user123" \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "status": "success",
  "user_id": "user123",
  "chunks_processed": 5,
  "message": "CV processed and stored in Pinecone"
}
```

### `POST /process-vacancy`

Process a vacancy description text.

**Request Body:**
```json
{
  "vacancy_id": "vac1",
  "text": "We are looking for a Python developer with 5+ years of experience in backend development..."
}
```

**Example:**
```bash
curl -X POST http://localhost:8002/process-vacancy \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy_id": "vac1",
    "text": "We are looking for a Python developer with 5+ years of experience in backend development..."
  }'
```

**Response:**
```json
{
  "status": "success",
  "vacancy_id": "vac1",
  "chunks_processed": 3,
  "message": "Vacancy processed and stored in Pinecone"
}
```

### `GET /health`

Health check for CV processor service.

**Response:**
```json
{
  "status": "ok"
}
```

## Embedding Service Endpoints

**Base URL**: `http://localhost:8001` (external) / `http://embedding-service:8001` (internal)

### `POST /embed`

Generate embeddings for text(s) using BGE-M3 model.

**Request Body:**
```json
{
  "texts": [
    "software engineer with Python experience",
    "data scientist with ML background"
  ]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.123, 0.456, ...],  // 1024-dimensional vector
    [0.789, 0.012, ...]   // 1024-dimensional vector
  ]
}
```

**Example:**
```bash
curl -X POST http://localhost:8001/embed \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "software engineer with Python experience",
      "data scientist with ML background"
    ]
  }'
```

### `GET /health`

Health check for embedding service.

**Response:**
```json
{
  "status": "ok"
}
```

## Error Responses

All endpoints may return standard HTTP error codes:

- **400 Bad Request**: Invalid request body or parameters
- **404 Not Found**: Resource not found
- **500 Internal Server Error**: Server error

**Error Response Format:**
```json
{
  "detail": "Error message describing what went wrong"
}
```

---

[← Back to README](../README.md)

