# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-01-03

### Added
- **Pinecone Vacancy Search**: Fast vector search using pre-indexed Pinecone database
  - New `upload_to_pinecone.py` script for vectorizing and uploading vacancies
  - Search API now uses Pinecone vector search by default (replaces Firecrawl)
  - Instant search results (< 1 second response time)
  - Support for semantic search based on role, skills, and other criteria

- **Vacancy Upload Script**: Automated vacancy vectorization and upload
  - Loads vacancies from JSON dump
  - Generates embeddings via embedding-service
  - Uploads to Pinecone in batches of 10
  - Filters out 'Unknown' vacancies automatically
  - Clears existing index before upload for clean data

- **VectorStore Enhancements**: 
  - Added `delete_all()` method for clearing namespaces
  - Improved namespace management

### Changed
- **Search API**: Default search mode changed from Firecrawl to Pinecone
  - `POST /api/v1/vacancies/search` now uses Pinecone by default
  - Firecrawl mode still available via `use_firecrawl=true` parameter
  - Mock mode available via `use_mock=true` parameter

- **VacancyFilter Schema**: 
  - Changed `company_stages` from `List[CompanyStage]` to `List[str]` to prevent 422 errors
  - Added normalization of company_stages using `CompanyStage.get_stage_value()`

- **Web UI**:
  - Removed "Search Source" radio buttons (Mock/Firecrawl)
  - Added "Search Mode: Database (Verified)" status indicator
  - Updated company stage options to match exact enum values:
    - 'Seed'
    - 'Series A'
    - 'Growth (Series B or later)'
    - '1-10 employees'
    - '10-100 employees'

### Fixed
- Fixed 422 validation errors for company_stages by accepting strings instead of strict Enum
- Fixed company stage normalization to handle variations like 'SeriesA' → 'Series A'
- Fixed filtering to use case-insensitive substring matching for industry and location
- Fixed search to return results even when role/skills are empty (filters by other criteria)

## [2.0.0] - 2024-12-XX

### Added
- **System Diagnostics Feature**: Comprehensive system health monitoring and diagnostics
  - New endpoint `GET /api/v1/system/diagnostics` for deep health checks
  - Health checks for all backend services:
    - CV Processor service
    - Embedding Service
    - Pinecone Vector Store (database connectivity)
    - LLM Provider (Google Gemini API connectivity)
  - Retry logic (3 attempts with 2s delay) for cold start handling
  - Detailed error classification (404, timeout, connection errors)
  - Structured JSON response with service status, latency, and error details
  
- **System Diagnostics UI**: New "System Diagnostics" tab in web interface
  - "Run Full System Check" button with loading states
  - Service-by-service status display with icons (✅/❌)
  - Latency metrics for each service
  - Error details with specific error type classification
  - Special handling for 404 errors (Routing/Configuration Error)
  - "Force Wake Up" button for services experiencing timeouts
  - Summary statistics (total/healthy/unhealthy services)
  - Partial success state handling

- **CV Processor Health Endpoint**: Added `/health` endpoint to CV Processor service

- **Diagnostic Schemas**: New Pydantic schemas for diagnostics
  - `ServiceDiagnostic`: Individual service diagnostic result
  - `SystemDiagnosticsResponse`: Complete system diagnostics response

### Improved
- Enhanced logging throughout diagnostic flow for better debugging
- Better error messages for service discovery issues
- Improved cold start detection and handling

### Technical Details
- All diagnostic checks run in parallel using `asyncio.gather()` for efficiency
- Comprehensive retry logic with exponential backoff for service pings
- Detailed error type classification for better troubleshooting

## [1.0.0] - Previous Release

### Features
- CV processing and storage in Pinecone namespace `"cvs"`
- Vacancy processing and storage in Pinecone namespace `"vacancies"`
- Candidate-vacancy matching with AI reasoning using Gemini 2.5 Flash
- LangGraph orchestrator with multi-agent workflow
- UserPersona and MatchingReport schemas
- Web UI with CV upload, vacancy processing, and matching functionality
- AI Talent Strategist chat interface (placeholder)

