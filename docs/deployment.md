# Deployment Guide

## Overview

The system is designed to run in Docker containers and can be deployed to Azure Container Apps (ACA) or any container orchestration platform.

## Local Deployment (Docker Compose)

### Prerequisites

- Docker & Docker Compose
- `.env` file with required API keys

### Environment Variables

Create a `.env` file in the project root:

```bash
PINECONE_API_KEY=your_key
PINECONE_INDEX_NAME=funds-search
GOOGLE_API_KEY=your_key
FIRECRAWL_API_KEY=your_key  # Optional: For real vacancy search via Firecrawl
```

### Quick Start

```bash
git clone <repo-url>
cd funds-search
cp .env.example .env  # Add your API keys
docker-compose up --build
```

**Access:**
- 🌐 Web UI: http://localhost:8501
- 🔌 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

### Service URLs

| Service | External (Host) | Internal (Docker Network) |
|---------|----------------|---------------------------|
| **API** | `http://localhost:8000` | `http://api:8000` |
| **Web UI** | `http://localhost:8501` | `http://web-ui:8501` |
| **CV Processor** | `http://localhost:8002` | `http://cv-processor:8001` |
| **Embedding Service** | `http://localhost:8001` | `http://embedding-service:8001` |
| **VC Worker** | `http://localhost:8003` | `http://vc-worker:8003` |

**Note**: CV Processor uses port mapping `8002:8001` (external:internal).

### Service-Specific Environment Variables

| Service | Required Variables |
|---------|-------------------|
| **API** | `PINECONE_API_KEY`, `GOOGLE_API_KEY`, `FIRECRAWL_API_KEY` (optional), `EMBEDDING_SERVICE_URL` (default: `http://embedding-service:8001`) |
| **Web UI** | `BACKEND_API_URL` (default: `http://api:8000`), `CV_PROCESSOR_URL` (default: `http://cv-processor:8001`), `FIRECRAWL_API_KEY` (optional) |
| **CV Processor** | `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `EMBEDDING_SERVICE_URL` |
| **Embedding Service** | `CUDA_VISIBLE_DEVICES` (optional, for GPU) |

### Firecrawl Setup (Optional)

To enable real vacancy search via Firecrawl:

1. **Get Firecrawl API Key:**
   - Sign up at https://firecrawl.dev
   - Get your API key from the dashboard

2. **Add to `.env`:**
   ```bash
   FIRECRAWL_API_KEY=fc-your_api_key_here
   ```

3. **Verify Configuration:**
   ```bash
   curl http://localhost:8000/api/v1/vacancies/health
   ```
   Check that `firecrawl_configured: true` in the response.

**Note:** Without `FIRECRAWL_API_KEY`, the system will use mock data for vacancy search.

## Production Deployment (Azure Container Apps)

### Registry

**Container Registry**: `fundssearchregistry.azurecr.io`

### Service Configuration

| Service | Container App | Port | Workflow |
|---------|---------------|------|----------|
| **API** | `api` | 8000 | `deploy-api.yml` (<500MB image) |
| **Web UI** | `web-ui` | 8501 | `deploy-web-ui.yml` |
| **CV Processor** | `cv-processor` | 8001 | `deploy-cv-processor.yml` |
| **Embedding** | `embedding-service` | 8001 | `deploy-embedding.yml` |

### CI/CD Workflows

GitHub Actions workflows are located in `.github/workflows/`:

- **`ci.yml`**: Runs `pytest apps/` (includes matching tests), validates with flake8, builds Docker images
- **`deploy-*.yml`**: Individual deployment workflows for each service

### Build Optimizations

- **API service**: Multi-stage Docker build reduces image size from 4GB to <500MB
- **Embedding service**: Pre-downloads model during build
- **All services**: Use dependency caching

### Environment Variables / Secrets

**Important**: Never commit secrets to the repository. Use Azure Container Apps secrets or environment variables:

1. **Azure Portal**: Configure secrets in Container App settings
2. **GitHub Actions**: Use GitHub Secrets for CI/CD workflows
3. **Local Development**: Use `.env` file (gitignored)

**Required Secrets:**
- `PINECONE_API_KEY`
- `GOOGLE_API_KEY`
- `FIRECRAWL_API_KEY` (optional)
- `PINECONE_INDEX_NAME`

### Deployment Steps

1. **Build and push Docker images:**
   ```bash
   # Tag images with version
   docker tag funds-search-api:latest funds-search-api:v2.0.0
   docker tag funds-search-cv-processor:latest funds-search-cv-processor:v2.0.0
   docker tag funds-search-embedding-service:latest funds-search-embedding-service:v2.0.0
   docker tag funds-search-web-ui:latest funds-search-web-ui:v2.0.0
   
   # Push to container registry
   # (Azure Container Registry commands)
   ```

2. **Update Container Apps:**
   - Update each container app to use v2.0.0 images
   - Ensure environment variables are configured
   - Set up service discovery/routing

3. **Verify deployment:**
   ```bash
   # Test diagnostics endpoint
   curl https://your-api-url.azurecontainerapps.io/api/v1/system/diagnostics
   ```

## Post-Deployment Verification

### 1. System Diagnostics Check

Run the diagnostics endpoint and verify all services are healthy:

```bash
curl https://your-api-url/api/v1/system/diagnostics | jq
```

**Expected Response:**
- `status: "ok"` - All services healthy
- `status: "partial"` - Some services down (investigate)
- `status: "error"` - Critical services down (rollback)

### 2. UI Verification

1. Access web UI
2. Navigate to "System Diagnostics" tab
3. Click "Run Full System Check"
4. Verify all services show ✅ status

### 3. Functional Tests

- [ ] CV upload and processing
- [ ] Vacancy processing
- [ ] Candidate-vacancy matching
- [ ] System diagnostics endpoint
- [ ] Vacancy search (mock and Firecrawl modes)

## Version Management

### Creating a New Version

1. Update `VERSION` file
2. Update `apps/api/main.py` version string
3. Update `CHANGELOG.md` with new version entry
4. Update `README.md` version reference
5. Commit and tag:
   ```bash
   git add VERSION CHANGELOG.md apps/api/main.py README.md
   git commit -m "Release v2.0.0 - System Diagnostics Feature"
   git tag -a v2.0.0 -m "MVP v2.0.0: System Diagnostics"
   git push origin main --tags
   ```

## Rollback Procedure

If issues are detected:

1. **Revert to previous version:**
   ```bash
   # Update VERSION file
   echo "1.0.0" > VERSION
   
   # Revert API version
   # Edit apps/api/main.py: version="1.0.0"
   ```

2. **Redeploy previous container images**

3. **Verify rollback:**
   ```bash
   curl https://your-api-url/health
   ```

## Monitoring

### Key Metrics to Monitor

- System diagnostics endpoint response time
- Service health check success rate
- Cold start frequency (via diagnostics)
- Error rates by service type

### Alerts

Set up alerts for:
- Diagnostics endpoint returning `status: "error"`
- Individual service failures
- High latency in diagnostics checks

## Troubleshooting

For common deployment issues, see [Troubleshooting Guide](troubleshooting.md).

---

[← Back to README](../README.md)

