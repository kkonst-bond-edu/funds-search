# Schemas (SSOT)

All schemas are defined in **`shared/schemas.py`** using **Pydantic v2**.  
This document is a human-readable catalog (not a duplication of code).

---

## Core Objects

### `UserPersona`
Produced by **Talent Strategist**.
Represents:
- skills / seniority / past roles
- preferences (location, remote, stage, industries)
- constraints (must-haves, deal-breakers)
- optional: summary statement / positioning

### `Vacancy`
Normalized vacancy record.
Typical fields:
- title, company, location, remote flag
- description / responsibilities
- URL + source (site / integration)
- tags (industry, stage, function, seniority)

### `DocumentChunk`
A chunked text unit stored in Pinecone.
Typical fields:
- id / chunk_id
- text
- metadata: doc_id, source_type, chunk_index, offsets (optional)

### `MatchingReport`
Produced by **Matchmaker**.
Should include:
- overall match score (0–100)
- strengths (why it matches)
- gaps / risks (what’s missing)
- recommended next step (apply / reach out / upskill)
- references to `candidate_id` and `vacancy_id`

---

## Versioning Rules

- Prefer backward-compatible changes (add fields, avoid renames).
- If breaking changes are unavoidable:
  - version the API or the schema
  - add migration notes in `docs/api.md`
