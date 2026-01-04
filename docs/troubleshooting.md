# Troubleshooting

This system often fails in “errors on errors” mode:
a container crashes → Docker DNS can’t resolve it → downstream services show network failures.

---

## 1) Docker DNS: `Name or service not known`

Most common cause: the target container crashed on startup.

Fix workflow:
1) `docker compose ps`
2) `docker compose logs -f <service>`
3) fix the crash
4) `docker compose up -d --build <service>`

---

## 2) Dependency / import crashes

### Missing packages (ModuleNotFoundError)
Example: missing `langchain_openai` or provider SDKs.

Fast check:
```bash
docker compose build api
docker compose run --rm api python -c "import langchain_openai; print('ok')"
```

Rebuild only the broken service:
```bash
docker compose up -d --build api
```

### Pinecone rename conflict (`pinecone-client` vs `pinecone`)
Standardize on the modern `pinecone` package.
If code expects `from pinecone import Pinecone`, uninstall/replace legacy packages and rebuild.

---

## 3) “It runs old code” (Docker cache / old containers)

Rebuild the target service:
```bash
docker compose up -d --build api
```

If service names changed (orphans exist):
```bash
docker compose down --remove-orphans
docker compose up -d --build
```

---

## 4) Service-to-service connectivity checks (inside Docker network)

From `web-ui` to `api`:
```bash
docker compose exec web-ui sh -lc "wget -qO- http://api:8000/health || true"
```

From `api` to embedding service:
```bash
docker compose exec api sh -lc "wget -qO- http://embedding-service:8001/health || true"
```

---

## 5) Quick “minimal reset” without full rebuild

Stop and remove containers but keep images:
```bash
docker compose down
```

Remove orphans too (recommended after compose edits):
```bash
docker compose down --remove-orphans
```

