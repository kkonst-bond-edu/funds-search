# Shared Directory

This directory contains shared code modules used across all services and applications in the funds-search system.

## Overview

The `shared/` directory provides common functionality that multiple components need:

- **schemas.py**: Pydantic data models for type safety and validation
- **pinecone_client.py**: Vector database client wrapper for Pinecone operations

## Architecture Principles

### Shared Code Pattern

All services and apps import from `shared/` to ensure:
- **Consistency**: Same data models across the system
- **Type Safety**: Pydantic validation ensures data integrity
- **DRY Principle**: No code duplication

### Import Pattern

All modules use this pattern to access shared code:
```python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from shared.schemas import Resume, Vacancy, MatchResult
from shared.pinecone_client import VectorStore
```

### PYTHONPATH Configuration

All Dockerfiles set `ENV PYTHONPATH=/app` to ensure Python can find the `shared/` directory when running in containers.

## Modules

### schemas.py

**Purpose**: Pydantic BaseModel definitions for all data structures.

**Key Models**:

1. **DocumentChunk**
   - Text content, metadata, and embedding vector
   - Used for chunked documents (CVs, vacancies)

2. **Job**
   - Job posting schema with company, title, location, etc.
   - Used in search results

3. **Resume**
   - CV/resume schema with user_id, raw_text, chunks
   - Stored in Pinecone namespace "cvs"

4. **Vacancy**
   - Vacancy description schema with raw_text and chunks
   - Stored in Pinecone namespace "vacancies"

5. **MatchResult**
   - Search result with score, reasoning, and job details
   - Used in `/search` endpoint

6. **VacancyMatchResult**
   - Candidate-vacancy match with score, reasoning, vacancy_id
   - Used in `/match` endpoint

7. **MatchRequest**
   - Request schema for matching (candidate_id, top_k)
   - Used in `/match` endpoint

8. **SearchRequest**
   - Request schema for search (query, location, role, remote)
   - Used in `/search` endpoint

**Usage Example**:
```python
from shared.schemas import Resume, DocumentChunk

resume = Resume(
    id="uuid",
    user_id="user123",
    raw_text="...",
    chunks=[DocumentChunk(...)]
)
```

### pinecone_client.py

**Purpose**: Wrapper around Pinecone client for vector database operations.

**Key Features**:
- **Lazy Initialization**: Client initialized on first use
- **Namespace Management**: Separate namespaces for CVs and vacancies
- **Error Handling**: Comprehensive error handling for connection issues
- **Metadata Management**: Consistent metadata structure across operations

**Key Methods**:

1. **`upsert_resume(resume: Resume, namespace: str = "cvs")`**
   - Stores resume chunks in Pinecone
   - Creates vector IDs: `{user_id}_{resume_id}_chunk_{index}`
   - Includes metadata: user_id, resume_id, text (truncated)

2. **`upsert_vacancy(vacancy: Vacancy, namespace: str = "vacancies")`**
   - Stores vacancy chunks in Pinecone
   - Creates vector IDs: `{vacancy_id}_chunk_{index}`
   - Includes metadata: type="vacancy", source

3. **`search_vectors(query_vector: List[float], top_k: int, namespace: str, filter: Dict = None)`**
   - Searches for similar vectors using cosine similarity
   - Returns matches with scores and metadata

4. **`get_candidate_embedding(user_id: str, namespace: str = "cvs")`**
   - Retrieves candidate's CV embedding from Pinecone
   - Used in matching workflow

**Configuration**:
- Environment variables:
  - `PINECONE_API_KEY` (required)
  - `PINECONE_INDEX_NAME` (default: "funds-search")

**Usage Example**:
```python
from shared.pinecone_client import VectorStore

vector_store = VectorStore()
vector_store.upsert_resume(resume, namespace="cvs")
results = vector_store.search_vectors(query_vector, top_k=10, namespace="vacancies")
```

## Legacy Rules Established

### 1. No Business Logic
The `shared/` directory contains **only**:
- Data models (schemas)
- Infrastructure code (database clients)
- **NOT** business logic or orchestration

### 2. Stateless Design
- `VectorStore` is a stateless wrapper
- No global state or singletons
- Each service creates its own instance

### 3. Error Handling
- All methods raise exceptions on failure
- Services handle errors appropriately
- No silent failures

### 4. Type Safety
- All data structures use Pydantic models
- Type hints throughout
- Validation on input/output

### 5. Namespace Convention
- **"cvs"**: Candidate resumes/CVs
- **"vacancies"**: Job vacancy descriptions
- Namespaces are hardcoded strings (not configurable)

## Integration

### Services Using Shared Code

1. **cv-processor**: Uses `schemas.py` and `pinecone_client.py`
2. **embedding-service**: (Future) May use schemas for validation
3. **api/orchestrator**: Uses `schemas.py` and `pinecone_client.py`
4. **web_ui**: Uses `schemas.py` for data validation

### Import Pattern in Services

```python
# At top of service file
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Then import shared modules
from shared.schemas import Resume, Vacancy
from shared.pinecone_client import VectorStore
```

## Testing

Shared modules should be tested independently:

```python
# Test schemas
from shared.schemas import Resume
resume = Resume(id="test", user_id="user", raw_text="...")
assert resume.id == "test"

# Test pinecone client (with mocked Pinecone)
from shared.pinecone_client import VectorStore
# ... test operations
```

## Adding New Shared Code

### When to Add to Shared

✅ **Add to shared/**:
- Data models used by multiple services
- Database clients/utilities
- Common validation logic
- Configuration helpers

❌ **Don't add to shared/**:
- Service-specific business logic
- API route handlers
- UI components
- Service-specific utilities

### Adding a New Schema

1. Define Pydantic model in `schemas.py`
2. Add docstring explaining purpose
3. Use Field() for validation rules
4. Update this README with model description

### Adding a New Client

1. Create new file in `shared/` (e.g., `shared/new_client.py`)
2. Follow existing patterns (lazy init, error handling)
3. Add to this README
4. Update services to use new client

## Status

- ✅ **schemas.py**: Production ready, comprehensive data models
- ✅ **pinecone_client.py**: Production ready, handles all vector operations

## Future Enhancements

- [ ] Add caching layer for frequently accessed vectors
- [ ] Add batch operations for better performance
- [ ] Add metrics/monitoring hooks
- [ ] Support for multiple Pinecone indexes
- [ ] Add schema versioning for migrations

