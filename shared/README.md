# Shared Directory

Single source of truth (SSOT) for cross-service data structures and utilities.

## Modules

- **schemas.py**: Pydantic v2 models for cross-service exchange (Job, Resume, Vacancy, MatchResult, VacancyMatchResult, etc.)
- **pinecone_client.py**: Pinecone client wrapper with namespace management ("cvs" for candidates, "vacancies" for job postings)

## Usage

All services and apps import from `shared/` to ensure consistency. Pydantic models in `shared/schemas.py` are the authoritative data structures.
