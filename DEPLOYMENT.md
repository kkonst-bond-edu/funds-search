# Deployment Guide - MVP v2.0.0

## Version Information

- **Current Version**: 2.0.0 (MVP)
- **Release Date**: 2024-12-XX
- **Version File**: `VERSION`
- **Changelog**: `CHANGELOG.md`

## Pre-Deployment Checklist

### 1. Version Verification
- [ ] Verify `VERSION` file contains `2.0.0`
- [ ] Verify `apps/api/main.py` has `version="2.0.0"`
- [ ] Review `CHANGELOG.md` for v2.0.0 changes

### 2. New Features in v2.0.0
- [x] System Diagnostics endpoint (`GET /api/v1/system/diagnostics`)
- [x] System Diagnostics UI tab
- [x] CV Processor health endpoint
- [x] Enhanced error handling and logging

### 3. Environment Variables
Ensure all required environment variables are set:
```bash
PINECONE_API_KEY=<your-key>
GOOGLE_API_KEY=<your-key>
EMBEDDING_SERVICE_URL=http://embedding-service:8001
CV_PROCESSOR_URL=http://cv-processor:8001
BACKEND_API_URL=http://api:8000
```

### 4. Service Health Checks
All services should have `/health` endpoints:
- ✅ API: `http://api:8000/health`
- ✅ CV Processor: `http://cv-processor:8001/health`
- ✅ Embedding Service: `http://embedding-service:8001/health`

## Deployment Steps

### Local Deployment (Docker Compose)

1. **Build and start services:**
   ```bash
   docker-compose up --build
   ```

2. **Verify services are running:**
   ```bash
   # Check API
   curl http://localhost:8000/health
   
   # Check System Diagnostics
   curl http://localhost:8000/api/v1/system/diagnostics
   
   # Check CV Processor
   curl http://localhost:8002/health
   
   # Check Embedding Service
   curl http://localhost:8001/health
   ```

3. **Access Web UI:**
   - Navigate to http://localhost:8501
   - Go to "System Diagnostics" tab
   - Run "Full System Check"

### Production Deployment (Azure Container Apps)

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

Expected response:
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

## Support

For issues or questions:
- Check `CHANGELOG.md` for version-specific changes
- Review service logs for diagnostic errors
- Use System Diagnostics UI for real-time health checks

