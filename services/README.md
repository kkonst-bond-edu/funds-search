# Services Directory

Heavy compute microservices for document processing and embeddings.

## Services

- **cv-processor** (Port 8002): Docling-based PDF→Markdown conversion; async via `run_in_threadpool` to avoid blocking event loop. Processes CVs and stores in Pinecone namespace "cvs".
- **embedding-service** (Port 8001): BGE-M3 embedding model service. Requires 2-4GB RAM for Azure deployment.
- **vc-worker** (Port 8003): Placeholder for future job scraping functionality.

## Architecture Rule

Services must **not** import from `apps/`. They may only import from `shared/` (schemas, pinecone_client).
