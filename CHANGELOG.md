# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

