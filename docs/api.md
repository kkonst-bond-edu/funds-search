# API Reference

Base URL (local): `http://localhost:8000`  
OpenAPI: `http://localhost:8000/docs`

> Tip: Treat OpenAPI as the canonical source of endpoint paths. This doc describes the intended contract.

---

## Health & Diagnostics

### GET `/health`
Simple API health check (API container only).

### GET `/api/v1/system/diagnostics`
Cross-service diagnostics (API, embedding-service, cv-processor) and basic configuration sanity checks.

Common uses:
- verify service-to-service DNS inside Docker network
- verify required env vars are present
- verify Pinecone connectivity

---

## Conversational Vacancy Chat

### POST `/api/v1/vacancies/chat`

Purpose:
- Primary conversational interface.
- Accepts a user message (and optional history/persona).
- Returns a synthesized answer plus optional structured items.

**CV Missing State:**
- If `persona` is not provided or is empty, the system operates in "Broad Search Mode".
- All vacancies will have `persona_applied: false` and `match_score: 0`.
- Each vacancy will include `ai_insight: "CV missing: Upload your resume in the 'Career & Match Hub' to enable AI matching."`
- The response includes `persona_applied: false` flag at the top level.
- The system logs a warning event: `chat_search_without_persona` when persona is missing.

Example request:
```json
{
  "message": "I want something like Google, but in crypto. Remote is preferred.",
  "history": [
    {"role": "user", "content": "I'm a senior ML leader in B2B SaaS."}
  ],
  "persona": {
    "technical_skills": ["Python", "ML", "TensorFlow"],
    "experience_years": 8,
    "career_goals": "ML Leadership"
  },
  "limit": 10
}
```

Typical response (high level):
```json
{
  "vacancies": [
    {
      "vacancy_id": "vac_001",
      "title": "Head of ML",
      "company": "ExampleCo",
      "url": "https://...",
      "score": 95,
      "match_score": 95,
      "ai_insight": "Strong match: Your ML leadership experience...",
      "persona_applied": true
    }
  ],
  "summary": "Based on your preference for remote crypto roles and senior ML leadership...",
  "persona_applied": true,
  "debug_info": {
    "friendly_reasoning": "Searching for remote crypto roles matching your ML expertise."
  }
}
```

**Response without CV (persona missing):**
```json
{
  "vacancies": [
    {
      "vacancy_id": "vac_001",
      "title": "Head of ML",
      "company": "ExampleCo",
      "score": 0,
      "match_score": 0,
      "ai_insight": "CV missing: Upload your resume in the 'Career & Match Hub' to enable AI matching.",
      "persona_applied": false
    }
  ],
  "summary": "Found matching vacancies...",
  "persona_applied": false
}
```

---

## Vacancy Search (Structured)

### POST `/api/v1/vacancies/search`

Search over:
- Local cache (`vacancies_dump.json`) OR
- Pinecone namespace `vacancies` OR
- Firecrawl (if enabled)

Example request:
```json
{
  "query": "Founding AI / Head of ML",
  "filters": {
    "location": "US",
    "remote": true,
    "company_stage": ["Seed", "SeriesA"]
  },
  "limit": 20,
  "mode": "cache"
}
```

---

## Matching (Persona → Vacancies)

### POST `/match` (if enabled)
Produces an explainable match report between a candidate persona and top vacancies.

Example request:
```json
{
  "candidate_id": "cand_123",
  "top_k": 10,
  "constraints": {
    "remote": true,
    "industry": ["B2B SaaS", "Logistics"],
    "company_stage": ["Seed", "SeriesA"]
  }
}
```

Response (high level):
- ranked matches
- structured `MatchingReport` per match (score, strengths, gaps, recommended next step)

---

## Internal Service URLs (Docker network)

These are typical defaults (verify in `docker-compose.yml`):

- Embeddings: `http://embedding-service:8001`
- CV Processor: `http://cv-processor:8002`

---

## Troubleshooting API Calls

- DNS errors like `Name or service not known` usually mean the target container crashed.
- Validate:
  - `docker compose ps`
  - `docker compose logs -f <service>`


