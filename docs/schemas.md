# Data Schemas

## Single Source of Truth

All schemas are defined in `shared/schemas.py` using **Pydantic v2**. This file is the authoritative source for all data structures used across services.

**Location**: [`shared/schemas.py`](../shared/schemas.py)

## Core Models

### `DocumentChunk`

Semantic text chunks with embeddings for vector storage.

**Fields:**
- `text` (str): Chunk text content
- `metadata` (Dict): Additional metadata (e.g., `type: 'cv'` or `type: 'vacancy'`)
- `embedding` (List[float]): 1024-dimensional BGE-M3 embedding vector

**Usage**: Used by `Resume` and `Vacancy` to store processed chunks.

### `Resume`

Candidate CV/resume schema.

**Fields:**
- `id` (str): Unique resume identifier
- `user_id` (str): Candidate identifier
- `raw_text` (str): Full CV text content
- `chunks` (List[DocumentChunk]): Processed chunks with embeddings (metadata: `type: 'cv'`)
- `processed_at` (datetime): Processing timestamp
- `created_at` (datetime): Creation timestamp

**Storage**: Chunks stored in Pinecone namespace `"cvs"` with metadata `type: 'cv'`.

### `Vacancy`

Job posting schema.

**Fields:**
- `id` (str): Unique vacancy identifier
- `raw_text` (str): Full job description text
- `chunks` (List[DocumentChunk]): Processed chunks with embeddings (metadata: `type: 'vacancy'`)
- `processed_at` (datetime): Processing timestamp
- `created_at` (datetime): Creation timestamp

**Storage**: Chunks stored in Pinecone namespace `"vacancies"` with metadata `type: 'vacancy'`.

### `Job`

Job opening schema (search results).

**Fields:**
- `id` (str): Unique job identifier
- `company` (str): Company name
- `title` (Optional[str]): Job title
- `raw_text` (str): Full job posting text
- `vector` (Optional[List[float]]): Embedding vector for similarity search
- `url` (Optional[str]): URL of the job posting
- `source_url` (Optional[str]): Original source URL where job was discovered
- `location` (Optional[str]): Job location
- `remote` (Optional[bool]): Whether position is remote
- `vc_fund` (Optional[str]): VC fund or investor associated with the company
- `created_at` (Optional[datetime]): Creation timestamp

## Request/Response Models

### `SearchRequest`

Job search query schema.

**Fields:**
- `query` (str, required): Search query string
- `location` (Optional[str]): Location filter
- `role` (Optional[str]): Job title filter
- `remote` (Optional[bool]): Remote work filter
- `user_id` (Optional[str]): User ID for personalization

**Usage**: Used by `POST /search` endpoint.

### `MatchRequest`

Candidate-vacancy matching request schema.

**Fields:**
- `candidate_id` (str, required): Unique candidate identifier (user_id)
- `top_k` (Optional[int], default: 10): Number of top matches to return

**Usage**: Used by `POST /match` endpoint.

### `MatchResult`

Search/match result schema.

**Fields:**
- `score` (float): Cosine similarity score (0-1)
- `reasoning` (str): AI-generated explanation for the match
- `job` (Job): Matched job posting
- `resume` (Optional[Resume]): Matched resume (if applicable)

**Usage**: Returned by `POST /search` endpoint.

### `VacancyMatchResult`

Candidate-vacancy match result schema.

**Fields:**
- `score` (float): Similarity score (0-1)
- `reasoning` (str): AI-generated explanation explaining why vacancy fits candidate
- `vacancy_id` (str): ID of the matched vacancy
- `vacancy_text` (str): Text content of the vacancy
- `candidate_id` (str): ID of the candidate

**Usage**: Returned by `POST /match` endpoint.

## Agentic Workflow Models

### `UserPersona`

Candidate profile built by Talent Strategist agent through conversational interviews.

**Fields:**
- `technical_skills` (List[str]): List of technical skills
- `career_goals` (List[str]): Career goals and aspirations
- `preferred_startup_stage` (Optional[str]): Preferred startup stage (Seed, Series A, Series B, Series C, Series D, Series E, IPO, or Public)
- `cultural_preferences` (List[str]): Cultural preferences and values
- `user_id` (Optional[str]): User ID associated with this persona

**Usage**: Used in LangGraph matching workflow state (`MatchingState.user_persona`).

### `MatchingReport`

Structured matching analysis replacing simple reasoning strings.

**Fields:**
- `match_score` (int): Match score (0-100)
- `strengths` (List[str]): List of strengths/positive matches
- `weaknesses` (List[str]): List of weaknesses/gaps
- `value_proposition` (str): Value proposition explaining why this match is valuable
- `suggested_action` (str): Suggested action for the candidate
- `job_id` (Optional[str]): Associated job ID
- `vacancy_id` (Optional[str]): Associated vacancy ID
- `candidate_id` (Optional[str]): Associated candidate ID

**Usage**: Future format for Deep Match Analyst agent output (replacing `VacancyMatchResult.reasoning`).

## Diagnostic Models

### `ServiceDiagnostic`

Service diagnostic result schema.

**Fields:**
- `status` (str): Service status (`"ok"`, `"error"`, or `"timeout"`)
- `latency` (Optional[float]): Response latency in milliseconds
- `error` (Optional[str]): Error message if status is `"error"`
- `error_type` (Optional[str]): Error type (`"404"`, `"timeout"`, `"connection"`, etc.)

**Usage**: Used by system diagnostics endpoint.

### `SystemDiagnosticsResponse`

System diagnostics response schema.

**Fields:**
- `status` (str): Overall system status (`"ok"`, `"error"`, or `"partial"`)
- `services` (Dict[str, ServiceDiagnostic]): Diagnostic results for each service
- `timestamp` (Optional[str]): Timestamp of the diagnostic check

**Usage**: Returned by `GET /api/v1/system/diagnostics` endpoint.

## Vacancy Search Models

These models are defined in `src/schemas/vacancy.py` (separate from shared schemas):

### `VacancyFilter`

Vacancy search filter schema.

**Fields:**
- `role` (Optional[str]): Job role or title filter
- `skills` (Optional[List[str]]): Required skills list
- `location` (Optional[str]): Job location filter
- `is_remote` (Optional[bool]): Remote work option filter
- `company_stages` (Optional[List[CompanyStage]]): Company funding stages filter
- `industry` (Optional[str]): Industry filter
- `min_salary` (Optional[int]): Minimum salary requirement (>= 0)

### `Vacancy` (Search Response)

Vacancy search result schema (different from `shared/schemas.Vacancy`).

**Fields:**
- `title` (str): Job title
- `company_name` (str): Company name
- `company_stage` (CompanyStage): Company funding stage (Seed, SeriesA, Growth, ScaleUp)
- `location` (str): Job location
- `industry` (str): Industry sector
- `salary_range` (Optional[str]): Salary range (e.g., "$120k-$180k")
- `description_url` (str): URL to full job description
- `required_skills` (List[str]): Required skills list
- `remote_option` (bool): Whether remote work is available

### `CompanyStage`

Enum for company funding stages.

**Values**: `SEED`, `SERIES_A`, `GROWTH`, `SCALE_UP`

## Schema Validation

All schemas use Pydantic v2 validation:

- **Type checking**: Automatic type coercion and validation
- **Required fields**: Fields without `Optional` are required
- **Default values**: Fields with `default` or `default_factory` are optional
- **Field descriptions**: All fields have descriptive docstrings

## Usage Across Services

| Service | Uses Schemas From |
|---------|-------------------|
| **API** | `shared/schemas.py`, `src/schemas/vacancy.py` |
| **Orchestrator** | `shared/schemas.py` |
| **CV Processor** | `shared/schemas.py` |
| **Embedding Service** | None (stateless) |
| **Web UI** | `shared/schemas.py` (via API responses) |

**Architecture Rule**: Services must **not** import from `apps/`. They may only import from `shared/` (schemas, pinecone_client).

---

[← Back to README](../README.md)

