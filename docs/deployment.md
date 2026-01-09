# Deployment

This repo is designed to deploy services independently:
- lightweight API images
- heavier ML services separated

> Treat `docker-compose.yml` as the canonical source for service names and ports.

---

## Azure Container Apps (ACA) — High Level

Typical apps:
- `api` (FastAPI gateway)
- `web-ui` (Streamlit)
- `embedding-service` (BGE-M3)
- `cv-processor` (Docling + chunking)

Recommended:
- configure each service with its own secrets/env vars
- keep provider keys only where needed

---

## Secrets / Environment Variables

Required (core):
- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `DEEPSEEK_API_KEY` (if DeepSeek is your primary LLM)

Optional (only if enabled):
- `OPENAI_API_KEY` (Matchmaker on GPT-4o)
- `ANTHROPIC_API_KEY` (Matchmaker on Claude)
- `FIRECRAWL_API_KEY` (Hunter Agent real-time scraping)

---

## CI/CD (GitHub Actions)

Workflows live under `.github/workflows/` and typically:
- build and push images to a registry (ACR)
- deploy/update ACA revisions
- manage secrets via ACA configuration (never commit secrets)

---

## Post-deploy Checks

- Open API docs: `/docs`
- Run diagnostics: `GET /api/v1/system/diagnostics`
- Verify Pinecone connectivity and namespaces (`cvs`, `vacancies`)
- Verify the LLM provider env vars are present in ACA

---

## Fast iteration commands (local)

Rebuild only one service:
```bash
docker compose up -d --build api
```

Follow logs:
```bash
docker compose logs -f api
```

One-off import check (without bringing everything up):
```bash
docker compose build api
docker compose run --rm api python -c "import langchain_openai; print('ok')"
```



