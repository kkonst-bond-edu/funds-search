# Embedding Service

A FastAPI microservice that provides semantic text embeddings using the BAAI/bge-m3 model. This service is a critical component of the funds-search RAG system, converting text queries and documents into high-dimensional vector representations for similarity search.

## Overview

The Embedding Service is a production-ready microservice that:
- Loads and serves the BGE-M3 (BAAI General Embedding) model from Hugging Face
- Generates dense vector embeddings (1024 dimensions) for text inputs
- Supports batch processing of multiple texts simultaneously
- Normalizes embeddings for optimal cosine similarity calculations
- Handles long texts up to 8192 tokens per input
- Automatically detects and utilizes GPU when available

## Features

- **High-Quality Embeddings**: Uses BGE-M3, a state-of-the-art multilingual embedding model
- **Batch Processing**: Process multiple texts in a single request for efficiency
- **GPU Support**: Automatic GPU detection and utilization for faster inference
- **Lifespan Management**: Model loaded once on startup, stays in memory for fast requests
- **Normalized Vectors**: All embeddings are L2-normalized for cosine similarity optimization
- **Long Text Support**: Handles texts up to 8192 tokens with automatic truncation
- **Multilingual**: Supports multiple languages out of the box

## Architecture

### Service Components

```
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  - /health endpoint                      │
│  - /embed endpoint                       │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      Lifespan Manager                   │
│  - Loads model on startup               │
│  - Manages model lifecycle              │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      Embedding Generator                │
│  - Tokenization                         │
│  - Model inference                      │
│  - Vector normalization                 │
└─────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────┐
│      BGE-M3 Model                       │
│  - Tokenizer                            │
│  - Transformer Model                    │
│  - GPU/CPU Device                       │
└─────────────────────────────────────────┘
```

### Processing Pipeline

1. **Request Received**: FastAPI receives POST request with list of texts
2. **Tokenization**: Each text is tokenized (max 8192 tokens, padding/truncation)
3. **Model Inference**: Tokens are passed through BGE-M3 model
4. **Embedding Extraction**: Dense embeddings extracted from [CLS] token
5. **Normalization**: Vectors are L2-normalized (unit vectors)
6. **Response**: Returns list of 1024-dimensional embedding vectors

## API Endpoints

### `GET /health`

Health check endpoint to verify service status and model loading.

**Request:**
```bash
curl http://localhost:8001/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true
}
```

### `POST /embed`

Generate embeddings for one or more text strings.

**Request:**
```json
{
  "texts": [
    "software engineer with Python experience",
    "looking for a backend developer role"
  ]
}
```

**Response:**
```json
{
  "embeddings": [
    [0.023, -0.045, 0.123, ...],  // 1024-dimensional vector
    [0.012, 0.034, -0.056, ...]   // 1024-dimensional vector
  ]
}
```

**Example Usage:**
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

**Error Responses:**
- `500`: Error generating embeddings (model not loaded, processing error, etc.)

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES` (optional): Control GPU visibility (e.g., `"0"` for first GPU, `""` for CPU only)
- No API keys required (model is open-source from Hugging Face)

### Model Configuration

- **Model Name**: `BAAI/bge-m3` (hardcoded)
- **Embedding Dimension**: 1024
- **Max Sequence Length**: 8192 tokens
- **Normalization**: L2 normalization (cosine similarity optimized)
- **Device**: Auto-detected (CUDA if available, else CPU)

### Port Configuration

- **Host Port**: 8001 (accessible from your machine)
- **Container Port**: 8001 (internal service port)
- **Port Mapping**: `8001:8001` (maps host port 8001 to container port 8001)

## Dependencies

### Python Packages

- `fastapi>=0.104.1` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `transformers>=4.44.0` - Hugging Face transformers library
- `torch>=2.2.0` - PyTorch for model inference
- `numpy>=1.26.4` - Numerical operations
- `pydantic>=2.10.6` - Request/response validation
- `sentencepiece` - Tokenization support
- `pytest>=7.4.0` - Testing framework
- `httpx==0.27.2` - HTTP client for tests

### System Dependencies

- `build-essential` - Compilation tools (for building PyTorch extensions)
- `curl` - Utility for health checks

## Running the Service

### Using Docker Compose (Recommended)

1. **Start the service:**
   ```bash
   docker-compose up embedding-service
   ```

2. **Check service health:**
   ```bash
   curl http://localhost:8001/health
   ```

3. **Test embedding generation:**
   ```bash
   curl -X POST http://localhost:8001/embed \
     -H "Content-Type: application/json" \
     -d '{"texts": ["test text"]}'
   ```

### Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r services/embedding-service/requirements.txt
   ```

2. **Start the service:**
   ```bash
   uvicorn services.embedding-service.main:app --host 0.0.0.0 --port 8001
   ```

   **Note**: On first run, the model will be downloaded from Hugging Face (~1.5 GB).

3. **With GPU support:**
   ```bash
   CUDA_VISIBLE_DEVICES=0 uvicorn services.embedding-service.main:app --host 0.0.0.0 --port 8001
   ```

### Using Docker

1. **Build the image:**
   ```bash
   docker build -f services/embedding-service/Dockerfile -t embedding-service .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8001:8000 embedding-service
   ```

   **Note**: The Dockerfile pre-downloads the model during build, so first build may take longer.

## Resource Requirements

### Memory
- **RAM**: ~2-4 GB for model loading
- **Model Size**: ~1.5 GB (downloaded on first run)

### Compute
- **GPU**: Optional but recommended for faster inference
  - NVIDIA GPU with CUDA support
  - ~2-4 GB VRAM recommended
- **CPU**: Works on CPU but slower (~200ms vs ~50ms per request)

### Disk
- **Model Files**: ~1.5 GB (cached after first download)
- **Dependencies**: ~2-3 GB

## Performance Characteristics

- **Latency**: 
  - GPU: ~50-100ms per request
  - CPU: ~150-300ms per request
- **Throughput**: Can process multiple texts in a single batch request
- **Memory**: Model stays loaded in memory for fast subsequent requests
- **Scalability**: Stateless service, can be horizontally scaled

## Integration with System

The Embedding Service is used by multiple components:

### 1. Orchestrator (Retrieval Node)
- Generates query embeddings for search requests
- Called during job search operations

### 2. CV Processor
- Generates embeddings for CV/resume chunks
- Called during CV ingestion pipeline

### 3. VC Worker (Future)
- Will generate embeddings for job postings
- Called during job scraping operations

## Testing

The service includes comprehensive tests in `tests/test_main.py`:

- Health endpoint tests
- Embedding generation tests
- Request/response schema validation
- Error handling tests
- Normalization verification
- Batch processing tests

**Run tests:**
```bash
cd services/embedding-service
pytest tests/
```

**Run with coverage:**
```bash
pytest tests/ --cov=. --cov-report=html
```

## Deployment

### Azure Container Apps

The service can be deployed to Azure Container Apps via CI/CD:

1. **GitHub Actions**: See `.github/workflows/deploy-embedding.yml`
2. **Manual Deployment**: Use Azure CLI or Portal

### Production Considerations

- **GPU Support**: Enable GPU in deployment configuration for better performance
- **Scaling**: Service is stateless, can scale horizontally
- **Health Checks**: Use `/health` endpoint for load balancer health checks
- **Monitoring**: Monitor model loading time and request latency
- **Caching**: Model is cached in memory, no additional caching needed

## Troubleshooting

### Model Not Loading

**Issue**: Service starts but `model_loaded: false` in health check

**Solutions**:
- Check logs for download errors
- Verify internet connection (first run downloads model)
- Check disk space (~1.5 GB needed)
- Verify Hugging Face access

### Out of Memory Errors

**Issue**: Service crashes with OOM errors

**Solutions**:
- Reduce batch size in requests
- Use CPU instead of GPU if VRAM is limited
- Increase container memory limits
- Process texts in smaller batches

### Slow Performance

**Issue**: Embedding generation is slow

**Solutions**:
- Enable GPU support (`CUDA_VISIBLE_DEVICES=0`)
- Use batch processing for multiple texts
- Check if model is loaded (health endpoint)
- Monitor system resources (CPU/GPU utilization)

### Connection Errors

**Issue**: Cannot connect to service

**Solutions**:
- Verify service is running: `docker-compose ps embedding-service`
- Check port mapping: `8001:8001` in docker-compose.yml
- Check firewall settings
- Verify service logs: `docker-compose logs embedding-service`

## Development

### Project Structure

```
services/embedding-service/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
├── tests/              # Test suite
│   └── test_main.py    # Unit and integration tests
└── README.md           # This file
```

### Code Structure

- **Lifespan Pattern**: Uses FastAPI lifespan to load model once on startup
- **Global State**: Model and tokenizer stored as global variables
- **Async Support**: FastAPI async endpoints for concurrent requests
- **Error Handling**: Comprehensive error handling with HTTPException

### Adding Features

1. **New Endpoints**: Add to `main.py` with FastAPI decorators
2. **Model Changes**: Update model loading in `lifespan` function
3. **Preprocessing**: Modify tokenization in `generate_embeddings`
4. **Post-processing**: Modify normalization or add additional processing

## Model Details

### BGE-M3 Model

- **Full Name**: BAAI General Embedding Model M3
- **Provider**: Beijing Academy of Artificial Intelligence (BAAI)
- **License**: MIT License (open-source)
- **Hugging Face**: https://huggingface.co/BAAI/bge-m3

### Model Capabilities

- **Multilingual**: Supports 100+ languages
- **Dense Embeddings**: 1024-dimensional vectors
- **Long Context**: Up to 8192 tokens
- **Semantic Understanding**: Captures semantic meaning and context
- **Similarity Search**: Optimized for cosine similarity

## Status

**Current Status**: ✅ Production Ready

The embedding service is fully functional and deployed. It successfully:
- Loads and serves the BGE-M3 model
- Generates high-quality embeddings
- Integrates with CV processor and orchestrator
- Handles production workloads

## License

See LICENSE file in project root for details.




