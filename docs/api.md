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

Example request:
```json
{
  "message": "I want something like Google, but in crypto. Remote is preferred.",
  "history": [
    {"role": "user", "content": "I'm a senior ML leader in B2B SaaS."}
  ],
  "persona_id": "cand_123",
  "limit": 10
}
```

Typical response (high level):
```json
{
  "answer": "Based on your preference for remote crypto roles and senior ML leadership...",
  "filters": {"industry": ["crypto"], "remote": true},
  "items": [
    {
      "vacancy_id": "vac_001",
      "title": "Head of ML",
      "company": "ExampleCo",
      "url": "https://...",
      "score": 95,
      "highlights": ["pricing/marketplace ML", "team leadership"]
    }
  ]
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

