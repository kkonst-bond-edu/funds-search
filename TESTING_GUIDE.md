# Testing Guide

This guide provides curl commands for testing all endpoints locally and remotely (Azure).

## Prerequisites

- All services running (via `docker-compose up` or deployed to Azure)
- API keys configured in `.env` file
- Test PDF file for CV processing

## Local Testing

### Health Checks

```bash
# API Service
curl http://localhost:8000/health

# Embedding Service
curl http://localhost:8001/health

# CV Processor
curl http://localhost:8002/health

# Web UI (Streamlit doesn't have /health, but check http://localhost:8501)
```

### API Endpoints

#### Search for Job Openings

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "software engineer",
    "location": "San Francisco",
    "role": "engineer",
    "remote": true,
    "user_id": "test_user"
  }'
```

#### Match Candidate with Vacancies

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "user123",
    "top_k": 10
  }'
```

### CV Processor Endpoints

#### Process CV/Resume

```bash
curl -X POST "http://localhost:8002/process-cv?user_id=test_user" \
  -F "file=@resume.pdf"
```

#### Process Vacancy

```bash
curl -X POST http://localhost:8002/process-vacancy \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy_id": "vacancy_001",
    "text": "We are looking for a software engineer with Python experience..."
  }'
```

### Embedding Service

#### Generate Embeddings

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

## Remote Testing (Azure Container Apps)

Replace `localhost` with your Azure Container App URLs:

```bash
# Example: API service at https://api-xxx.azurecontainerapps.io
API_URL="https://api-xxx.azurecontainerapps.io"
CV_PROCESSOR_URL="https://cv-processor-xxx.azurecontainerapps.io"
EMBEDDING_URL="https://embedding-xxx.azurecontainerapps.io"
```

### Health Checks

```bash
curl ${API_URL}/health
curl ${EMBEDDING_URL}/health
curl ${CV_PROCESSOR_URL}/health
```

### API Endpoints

```bash
# Search
curl -X POST ${API_URL}/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "software engineer",
    "location": "San Francisco"
  }'

# Match
curl -X POST ${API_URL}/match \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "user123",
    "top_k": 10
  }'
```

### CV Processing

```bash
# Process CV
curl -X POST "${CV_PROCESSOR_URL}/process-cv?user_id=test_user" \
  -F "file=@resume.pdf"

# Process Vacancy
curl -X POST ${CV_PROCESSOR_URL}/process-vacancy \
  -H "Content-Type: application/json" \
  -d '{
    "vacancy_id": "vacancy_001",
    "text": "Software engineer position..."
  }'
```

## Web UI Testing

1. **Access Web UI**: http://localhost:8501 (local) or your Azure Container App URL
2. **Upload CV**: Use the "Upload CV" tab to upload a PDF resume
3. **Process Vacancy**: Use the "Process Vacancy" tab to paste vacancy text
4. **Find Matches**: Use the "Find Matches" tab with a candidate ID

## Expected Responses

### Successful CV Processing

```json
{
  "status": "success",
  "resume_id": "uuid-here",
  "chunks_processed": 5
}
```

### Successful Matching

```json
[
  {
    "score": 0.85,
    "reasoning": "The candidate's Python experience matches...",
    "vacancy_id": "vacancy_001",
    "vacancy_text": "Software engineer position...",
    "candidate_id": "user123"
  }
]
```

### Error Responses

- **404**: Candidate not found
- **500**: Server error (check logs)
- **503**: Service unavailable (cold start or dependency issue)

## Troubleshooting

1. **Connection Refused**: Ensure services are running
2. **Timeout**: Services may be in cold start (wait 30-60 seconds)
3. **404 on Match**: Ensure CV has been processed first
4. **500 Errors**: Check service logs for details

## Discovery Mode Testing

The Autonomous Job Hunter includes a Discovery mode powered by the Web Hunter agent (Firecrawl integration). This mode discovers job opportunities from VC fund websites.

### Testing Discovery Mode

**Note**: Discovery mode is currently in placeholder state. Full Firecrawl integration will be implemented in upcoming roadmap phases.

#### Expected Workflow

1. **Talent Strategist Interview**: Complete the AI Talent Strategist interview in the Web UI to build your UserPersona
2. **Discovery Trigger**: The Web Hunter agent will automatically discover jobs based on your persona
3. **Matching Analysis**: The Deep Match Analyst will generate structured MatchingReport objects

#### Testing Steps (Placeholder)

```bash
# 1. Access Web UI
# Navigate to http://localhost:8501

# 2. Complete Talent Strategist Interview
# - Go to "AI Talent Strategist" tab
# - Have a conversation to build your persona
# - Click "Complete Interview"

# 3. Trigger Discovery (when implemented)
# The Web Hunter will automatically discover jobs from VC fund websites

# 4. Review Matching Reports
# View structured reports with strengths, weaknesses, and value propositions
```

#### Expected Discovery Mode Features (Upcoming)

- **Firecrawl Integration**: Automatic job discovery from VC fund websites
- **UserPersona-Based Discovery**: Jobs matched to your technical skills, career goals, and startup stage preferences
- **Structured Matching Reports**: Detailed analysis with strengths, weaknesses, value propositions, and suggested actions
- **Real-time Updates**: New job opportunities discovered and matched automatically

## Performance Notes

- First request may be slow (cold start)
- CV processing takes 30-120 seconds depending on document size
- Matching takes 10-30 seconds (includes AI reasoning)
- Embedding generation: ~50-200ms per text
- Discovery mode (when implemented): Expected 1-5 minutes for full discovery cycle

