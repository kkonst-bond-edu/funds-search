# CV Processor Service

A FastAPI microservice for processing CV/resume documents (PDF and DOCX), extracting text, generating embeddings, and storing them in Pinecone vector database.

## Overview

The CV Processor is part of the funds-search RAG (Retrieval-Augmented Generation) system. It handles the data ingestion pipeline for resumes, converting documents to structured data that can be searched and matched against job postings.

## Features

- **Document Processing**: Converts PDF and DOCX files to Markdown using IBM's Docling library
- **Text Chunking**: Splits documents into semantic chunks (1000 characters with 800 character overlap)
- **Embedding Generation**: Calls the embedding-service to generate 1024-dimensional vectors using BGE-M3 model
- **Vector Storage**: Stores processed resumes in Pinecone vector database with metadata
- **Persona Persistence**: Stores raw CV persona data in SQLite for cross-session restore
- **Targeted Deletion**: Deletes CV vectors by user_id without affecting other data
- **Error Handling**: Comprehensive error handling for service failures, invalid files, and processing errors

## Architecture

The service integrates with the following components:

1. **CV Processor** (Port 8002 on host, 8001 in container): FastAPI service that processes CVs
2. **Embedding Service** (Port 8001): Generates embeddings using BGE-M3 model
3. **Pinecone Vector Store**: Cloud vector database (Index: `funds-search`, Namespace: `cvs`)

## API Endpoints

### `POST /process-cv`

Process a CV/resume file and store it in Pinecone.

**Request:**
- `user_id` (query parameter or form field): User identifier
- `file` (multipart/form-data): PDF or DOCX file

**Example using curl:**
```bash
curl -X POST "http://localhost:8002/process-cv?user_id=user123" \
  -F "file=@resume.pdf"
```

**Response:**
```json
{
  "status": "success",
  "resume_id": "uuid-here",
  "chunks_processed": 5
}
```

**Error Responses:**
- `400`: Invalid file format, empty file, or missing user_id
- `503`: Embedding service unavailable
- `500`: Processing error or Pinecone connection error

### `DELETE /delete-cv`

Delete all CV vectors for a user_id and mark the persona as deleted.

**Request:**
- `user_id` (query parameter): User identifier

**Example using curl:**
```bash
curl -X DELETE "http://localhost:8002/delete-cv?user_id=user123"
```

**Response:**
```json
{
  "status": "success",
  "user_id": "user123"
}
```

### `GET /cv/persona`

Fetch the persisted persona for a user_id (if not deleted).

**Request:**
- `user_id` (query parameter): User identifier

**Example using curl:**
```bash
curl "http://localhost:8002/cv/persona?user_id=user123"
```

**Response:**
```json
{
  "status": "success",
  "persona": {
    "user_id": "user123",
    "resume_id": "uuid-here",
    "cv_text": "...",
    "is_deleted": false
  }
}
```

## Configuration

### Environment Variables

- `PINECONE_API_KEY` (required): Pinecone API key for vector database access
- `PINECONE_INDEX_NAME` (optional): Pinecone index name (default: `funds-search`)
- `EMBEDDING_SERVICE_URL` (optional): Embedding service URL (default: `http://embedding-service:8001`)

### Port Configuration

- **Host Port**: 8002 (accessible from your machine)
- **Container Port**: 8001 (internal service port)
- **Port Mapping**: `8002:8001` (maps host port 8002 to container port 8001)

## Dependencies

The service requires the following Python packages:

- `fastapi>=0.104.1` - Web framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `python-multipart>=0.0.6` - File upload support
- `docling>=1.0.0` - Document conversion (PDF/DOCX to Markdown)
- `pinecone>=3.0.0` - Vector database client
- `httpx>=0.27.0` - HTTP client for embedding service
- `pydantic>=2.10.6` - Data validation

### System Dependencies

The Dockerfile includes:
- `libgomp1` - OpenMP support for Docling
- `libgl1`, `libglib2.0-0` - Graphics libraries for document processing
- `build-essential` - Compilation tools

## Running the Service

### Using Docker Compose (Recommended)

1. **Create a `.env` file** in the project root:
   ```bash
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX_NAME=funds-search
   EMBEDDING_SERVICE_URL=http://embedding-service:8001
   ```

2. **Build and start the service:**
   ```bash
   docker-compose up --build cv-processor
   ```

3. **Check service health:**
   ```bash
   curl http://localhost:8002/health
   ```

### Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r services/cv-processor/requirements.txt
   ```

2. **Set environment variables:**
   ```bash
   export PINECONE_API_KEY=your_pinecone_api_key
   export PINECONE_INDEX_NAME=funds-search
   export EMBEDDING_SERVICE_URL=http://localhost:8001
   ```

3. **Start the service:**
   ```bash
   uvicorn services.cv-processor.main:app --host 0.0.0.0 --port 8002
   ```

## Processing Pipeline

1. **File Upload**: Receives PDF or DOCX file via multipart form data
2. **Document Conversion**: Uses Docling to convert file to Markdown format
3. **Text Chunking**: Splits Markdown text into chunks (1000 chars, 800 char overlap)
4. **Embedding Generation**: Calls embedding-service to generate 1024-dim vectors for each chunk
5. **Resume Object Creation**: Creates a `Resume` object with `DocumentChunk` objects
6. **Vector Storage**: Stores all chunks in Pinecone with metadata (user_id, resume_id, chunk_index)

## Data Models

### Resume
- `id`: Unique identifier (UUID)
- `user_id`: User identifier
- `raw_text`: Full markdown text from document
- `chunks`: List of `DocumentChunk` objects
- `processed_at`: Timestamp of processing

### DocumentChunk
- `text`: Chunk text content
- `metadata`: Additional metadata (source filename, chunk_index)
- `embedding`: 1024-dimensional embedding vector

## Storage in Pinecone

Each resume chunk is stored as a separate vector in Pinecone:
- **Index**: `funds-search` (configurable via `PINECONE_INDEX_NAME`)
- **Namespace**: `cvs`
- **Vector ID**: `{user_id}_{resume_id}_chunk_{index}`
- **Metadata**: Includes user_id, resume_id, text (truncated), and chunk metadata

## Error Handling

The service handles various error scenarios:

- **Invalid file format**: Returns 400 with error message
- **Empty files**: Returns 400 with validation error
- **Embedding service unavailable**: Returns 503 with service error details
- **Pinecone connection errors**: Returns 500 with connection error details
- **Processing errors**: Returns 500 with detailed error message

## Development

### Project Structure

```
services/cv-processor/
├── main.py              # FastAPI application
├── requirements.txt     # Python dependencies
├── Dockerfile          # Container configuration
└── README.md           # This file
```

### Code Structure

- **Path Setup**: Adds project root to Python path for shared module imports
- **Lazy Initialization**: DocumentConverter and VectorStore are initialized at module level
- **Async Processing**: Uses async/await for HTTP calls to embedding service
- **Temporary Files**: Creates temporary files for document processing, cleaned up after processing

## Integration with Other Services

- **Embedding Service**: Must be running on port 8001 (or configured via `EMBEDDING_SERVICE_URL`)
- **Pinecone**: Requires valid API key and index name
- **Shared Modules**: Uses `shared.schemas` and `shared.pinecone_client` from the project root

## Troubleshooting

### Connection Reset Errors

If you encounter "Connection reset by peer" errors:

1. **Check if the service is running:**
   ```bash
   docker-compose ps cv-processor
   ```

2. **Check service logs:**
   ```bash
   docker-compose logs cv-processor
   ```

3. **Verify environment variables:**
   ```bash
   docker-compose exec cv-processor env | grep PINECONE
   ```

4. **Check embedding service availability:**
   ```bash
   curl http://localhost:8001/health
   ```

### Common Issues

- **Import errors**: Ensure the project root is in Python path (handled automatically in Docker)
- **Pinecone connection failures**: Verify API key and index name are correct
- **Embedding service timeouts**: Check if embedding-service is running and accessible
- **File processing errors**: Ensure files are valid PDF or DOCX format

## Status

**Current Status**: ✅ Production Ready

The data ingestion pipeline is complete and functional. The service can process CVs, generate embeddings, and store them in Pinecone for similarity search.
