# API Service

FastAPI service for vacancy search and matching functionality.

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

**Example Requests:**
```bash
# Search for Go developer in fintech
curl -X POST "http://localhost:8000/api/v1/vacancies/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Looking for a Go dev in fintech"}'

# Search for Python engineer in series A AI startup
curl -X POST "http://localhost:8000/api/v1/vacancies/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to work as a Python engineer in a series A AI startup"}'

# Search for remote React developer
curl -X POST "http://localhost:8000/api/v1/vacancies/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Remote React developer positions"}'
```

**How it works:**
1. The AI agent (`ChatSearchAgent`) interprets your natural language message
2. Extracts search parameters: role, skills, industry, location, company_stage
3. Performs vector search in Pinecone using embeddings
4. Returns matching vacancies with an AI-generated summary explaining why they match

**Extracted Parameters:**
- `role`: Job title or role (e.g., "Software Engineer", "Python Developer")
- `skills`: List of technical skills (e.g., ["Python", "Go", "React"])
- `industry`: Industry sector (e.g., "Fintech", "AI", "Healthcare")
- `location`: Job location (e.g., "San Francisco", "Remote", "New York")
- `company_stage`: Company funding stage (e.g., "Seed", "Series A", "Growth")

If a parameter is not mentioned in your message, it will be `null` and won't be used as a filter.

#### GET `/api/v1/vacancies/health`

Health check endpoint for the vacancy search service.

**Response:**
```json
{
  "status": "ok",
  "service": "vacancy-search",
  "version": "1.0.0",
  "firecrawl_configured": true
}
```

## Environment Variables

- `DEEPSEEK_API_KEY`: Required for chat search functionality
- `PINECONE_API_KEY`: Required for vector search
- `EMBEDDING_SERVICE_URL`: URL of the embedding service (default: `http://embedding-service:8001`)
- `FIRECRAWL_API_KEY`: Optional, for real-time vacancy fetching

## Running Locally

```bash
# Start the API service
cd apps/api
uvicorn main:app --reload --port 8000

# Test the chat endpoint
curl -X POST "http://localhost:8000/api/v1/vacancies/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to work as a Python engineer in a series A AI startup"}'
```

## Architecture

The chat search endpoint uses:
- **ChatSearchAgent**: Interprets natural language and generates summaries
- **DeepSeek LLM**: Via `LLMProviderFactory` for AI processing
- **VectorStore**: Pinecone for semantic search
- **Embedding Service**: Generates query embeddings



