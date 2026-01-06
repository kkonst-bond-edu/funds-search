# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-01-XX

### Added
- **Matchmaker Agent**: Specialized AI agent for analyzing vacancy-candidate matches
  - New `MatchmakerAgent` class that evaluates matches between candidate personas and vacancies
  - Provides match scores (0-100) with detailed analysis
  - Highlights connections (shared tech stack, experience) and gaps (missing skills, domain mismatches)
  - Uses dedicated `matchmaker.txt` prompt for consistent match analysis
  - Gracefully handles CV missing state with fallback responses
  - JSON response parsing with robust error handling

- **Short-term Memory (Conversation History)**: Context-aware chat search
  - Chat search agent now maintains conversation history across messages
  - `interpret_message()` accepts `history` parameter with previous messages
  - Enables natural follow-up questions and context-aware search refinement
  - History is passed to LLM for better understanding of user intent
  - Supports both persona-based and explicit search modes with history context

- **Improved Agent Architecture**: Standardized agent infrastructure
  - New `BaseAgent` class providing unified interface for all AI agents
  - Agent configuration loaded from `agents.yaml` (model, temperature, prompt file)
  - Each agent initializes its own LLM provider with agent-specific settings
  - Standardized logging with structlog for structured JSON logs
  - Easy to add new agents by extending `BaseAgent` and adding config

- **Hybrid Filtering & Persona Enrichment**: Enhanced search intelligence
  - Persona enrichment: Role queries enriched with technical skills and experience from CV
  - Hybrid mode: Combines user-provided skills with persona role when appropriate
  - High-density search queries for better embedding matching
  - Smart fallback: Explicit mode respects user's specific queries without persona interference
  - Persona mode uses CV data as base, filling missing fields intelligently

- **CV Missing State Handling**: Graceful degradation when CV not uploaded
  - System performs broad search without personalized matching when persona is missing
  - All vacancies include `persona_applied: false` and `match_score: 0` flags
  - User guidance messages: "CV missing: Upload your resume in the 'Career & Match Hub' to enable AI matching"
  - Web UI displays warning banners and "Resume Required" badges
  - Matchmaker agent returns appropriate fallback responses when no persona available

### Changed
- **Agent System**: Refactored to use BaseAgent infrastructure
  - `ChatSearchAgent` (Job Scout) now extends `BaseAgent`
  - `MatchmakerAgent` extends `BaseAgent` for consistent behavior
  - All agents load configuration from centralized `agents.yaml`
  - Prompts stored in separate `.txt` files for easy editing

- **Search Modes**: Enhanced persona vs explicit search logic
  - Explicit mode: Uses ONLY user's current message, ignores persona and history
  - Persona mode: Uses CV data as base, fills missing fields from persona
  - Better detection of user intent (e.g., "for me" triggers persona mode)
  - Improved validation and fallback logic for search mode selection

### Improved
- **Logging**: Structured JSON logging throughout agent system
  - All agents use structlog for consistent log format
  - Better debugging with agent-specific context in logs
  - Logs include agent name, model, temperature, and operation details

- **Error Handling**: More robust error handling in agents
  - JSON parsing with markdown code block removal
  - Graceful fallbacks when persona data is missing
  - Better error messages for debugging

### Technical Details
- Agents are configured via `apps/orchestrator/settings/agents.yaml`
- Each agent has its own prompt file in `apps/orchestrator/prompts/`
- LLM providers are cached per agent configuration (model + temperature)
- Conversation history format: `[{"role": "user/assistant", "content": "..."}]`

## [2.1.1] - 2025-01-03

### Added
- **Integration Tests for Chat API**: Comprehensive test suite for `/api/v1/vacancies/chat` endpoint
  - Created `tests/integration/test_chat_api.py` with full endpoint coverage
  - Tests verify response structure (summary and vacancies list)
  - Mocked ChatSearchAgent methods to avoid LLM calls during CI
  - Tests validate data flow through the endpoint
  - Added `scripts/run_tests.sh` for easy test execution

### Testing
- Integration tests can be run via `pytest tests/integration` or `docker-compose exec api pytest tests/integration/test_chat_api.py`
- All tests use mocks to avoid external API calls during CI/CD

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

