# Troubleshooting Guide

## Common Issues

### "Errors on Errors" Scenario

**Problem**: Container crash causes Docker DNS failure, making it impossible to diagnose the issue because health checks also fail.

**Symptoms:**
- All services show as unhealthy
- Cannot reach diagnostics endpoint
- Docker network DNS resolution fails

**Solution:**
1. Check container logs directly:
   ```bash
   docker compose logs -f <service-name>
   ```
2. Verify containers are running:
   ```bash
   docker compose ps
   ```
3. Restart the failing service:
   ```bash
   docker compose up -d --build <service-name>
   ```

### Missing Python Packages in Container

**Problem**: `ModuleNotFoundError` when running services.

**Symptoms:**
```
ModuleNotFoundError: No module named 'X'
```

**Solution:**
1. Verify package is in requirements file
2. Rebuild the service:
   ```bash
   docker compose up -d --build <service-name>
   ```
3. Check if package is installed in container:
   ```bash
   docker compose run --rm <service-name> python -c "import X"
   ```

### Pinecone Package Rename Conflicts

**Problem**: Conflicts between `pinecone-client` and `pinecone` packages.

**Symptoms:**
```
ImportError: cannot import name 'X' from 'pinecone'
```

**Solution:**
1. Check which package is installed:
   ```bash
   docker compose run --rm api pip list | grep pinecone
   ```
2. Ensure only one package is in requirements:
   - Use `pinecone` (newer) or `pinecone-client` (legacy), not both
3. Rebuild service:
   ```bash
   docker compose up -d --build <service-name>
   ```

### Docker Cache Issues

**Problem**: Changes to code or dependencies not reflected after rebuild.

**Symptoms:**
- Code changes not visible
- Old dependencies still in use

**Solution:**
1. Rebuild without cache:
   ```bash
   docker compose build --no-cache <service-name>
   docker compose up -d <service-name>
   ```
2. Or rebuild all services:
   ```bash
   docker compose build --no-cache
   docker compose up -d
   ```

### Service-to-Service Connectivity Issues

**Problem**: Services cannot communicate within Docker network.

**Symptoms:**
- Connection refused errors
- Timeout errors when calling other services

**Solution:**
1. Verify services are on the same network:
   ```bash
   docker network inspect funds-search_job-hunter-net
   ```
2. Test connectivity from within a container:
   ```bash
   docker compose run --rm api curl http://embedding-service:8001/health
   ```
3. Check service URLs in environment variables:
   - Internal URLs should use service names: `http://embedding-service:8001`
   - Not `localhost` or external URLs

### Service Names Changed

**Problem**: After renaming services in `docker-compose.yml`, old containers remain.

**Symptoms:**
- Multiple containers with similar names
- Port conflicts

**Solution:**
1. Remove orphaned containers:
   ```bash
   docker compose down --remove-orphans
   ```
2. Clean up unused containers:
   ```bash
   docker container prune -f
   ```

## Fast Debug Commands

### Check Service Status

```bash
# List all services and their status
docker compose ps

# Check logs for a specific service
docker compose logs -f <service-name>

# Check logs for all services
docker compose logs -f
```

### Rebuild a Single Service

```bash
# Rebuild and restart a specific service
docker compose up -d --build <service-name>

# Examples:
docker compose up -d --build api
docker compose up -d --build cv-processor
docker compose up -d --build embedding-service
```

### Test Service Health

```bash
# Test API health
curl http://localhost:8000/health

# Test system diagnostics
curl http://localhost:8000/api/v1/system/diagnostics

# Test CV processor
curl http://localhost:8002/health

# Test embedding service
curl http://localhost:8001/health
```

### Run Commands Inside Containers

```bash
# Test Python imports
docker compose run --rm api python -c "import shared.schemas"

# Check installed packages
docker compose run --rm api pip list

# Access shell in container
docker compose run --rm api /bin/bash
```

### Clean Up Docker Resources

```bash
# Stop and remove all containers
docker compose down

# Remove orphaned containers
docker compose down --remove-orphans

# Remove volumes (WARNING: deletes data)
docker compose down -v

# Clean up unused Docker resources
docker system prune -a
```

### Check Environment Variables

```bash
# View environment variables for a service
docker compose run --rm api env | grep PINECONE

# Check if .env file is loaded
docker compose config
```

## Service-Specific Issues

### API Service

**Issue**: API cannot connect to embedding service.

**Debug:**
```bash
# Check embedding service URL
docker compose run --rm api env | grep EMBEDDING_SERVICE_URL

# Test connectivity
docker compose run --rm api curl http://embedding-service:8001/health
```

**Fix**: Ensure `EMBEDDING_SERVICE_URL=http://embedding-service:8001` is set.

### CV Processor

**Issue**: CV processing fails with "Pinecone connection error".

**Debug:**
```bash
# Check Pinecone credentials
docker compose run --rm cv-processor env | grep PINECONE

# Test Pinecone connection
docker compose run --rm cv-processor python -c "from shared.pinecone_client import VectorStore; VectorStore()"
```

**Fix**: Ensure `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` are set in `.env`.

### Embedding Service

**Issue**: Embedding service fails to load model.

**Debug:**
```bash
# Check service logs
docker compose logs -f embedding-service

# Check available memory
docker stats embedding-service
```

**Fix**: Ensure container has 2-4GB RAM allocated. Model requires significant memory.

### Web UI

**Issue**: Web UI cannot connect to API.

**Debug:**
```bash
# Check backend URL
docker compose run --rm web-ui env | grep BACKEND_API_URL

# Test API connectivity
docker compose run --rm web-ui curl http://api:8000/health
```

**Fix**: Ensure `BACKEND_API_URL=http://api:8000` is set.

## Getting Help

If issues persist:

1. **Check logs**: `docker compose logs -f <service-name>`
2. **Verify environment**: Ensure all required environment variables are set
3. **Check network**: Verify services are on the same Docker network
4. **Review documentation**: See [Architecture Guide](architecture.md) and [API Reference](api.md)
5. **Check version**: Ensure you're using the correct version (see `VERSION` file)

---

[← Back to README](../README.md)

