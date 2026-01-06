# Architecture Deep Dive

This document expands the root README into a detailed, implementation-oriented view of the system.

## Goals

- Keep the UI conversational while keeping backend behavior **traceable** (schemas, explicit agent boundaries).
- Use a **vector DB** for retrieval and an LLM for **intent extraction + synthesis**.
- Support both **offline/cache** operation and **real-time discovery** (Firecrawl) as a fallback.

---

## System Responsibilities

### Client (Streamlit)
- Chat UI + session history
- CV upload (optional)
- Displays matched roles + explanations

### API (FastAPI)
- Public entrypoint: chat/search/matching endpoints
- Validates requests/responses (Pydantic v2)
- Orchestrates agent flow (LangGraph)

### Orchestrator (LangGraph)
- Converts requests into a **state machine** (retrieval → analysis)
- Calls agents in sequence and stores intermediate artifacts in state:
  - persona (UserPersona)
  - intent filters
  - retrieved context
  - final response / report

### Internal Services
- **Embedding Service (BGE-M3)**: text → vector
- **CV Processor**: PDF → text → chunks (+ optional upsert to Pinecone)

### Data Layer
- **Pinecone**: vector retrieval across namespaces (recommended: `cvs`, `vacancies`)
- **Local Cache**: `vacancies_dump.json` for fast local testing

---

## End-to-End Flow: Chat → Search → Response

```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant API as FastAPI (Gateway)
    participant ORC as Orchestrator (LangGraph)
    participant LLM as Job Scout (DeepSeek R1)
    participant EMB as Embedding Service
    participant PC as Pinecone
    participant MM as Matchmaker Agent
    participant SYN as Response Synthesizer

    UI->>API: POST /api/v1/vacancies/chat (message + optional persona)
    
    alt Persona Available
        API->>ORC: create state (message, history, persona)
        ORC->>LLM: intent extraction prompt (with persona context)
        LLM-->>ORC: structured filters + query text
        ORC->>EMB: /embed (query text)
        EMB-->>ORC: query vector
        ORC->>PC: query(namespace="vacancies", top_k=K, filters=…)
        PC-->>ORC: top matches (chunks + metadata)
        ORC->>MM: analyze_match(vacancy, persona)
        MM-->>ORC: match score + analysis
        ORC->>SYN: synthesize response with personalized scores
        SYN-->>ORC: final answer (persona_applied: true)
    else Persona Missing (CV Missing State)
        API->>API: Log warning: chat_search_without_persona
        API->>ORC: create state (message, history, persona=null)
        ORC->>LLM: intent extraction prompt (broad search)
        LLM-->>ORC: structured filters + query text (role may be null for "all")
        ORC->>EMB: /embed (query text)
        EMB-->>ORC: query vector
        ORC->>PC: query(namespace="vacancies", top_k=K, filters=…)
        PC-->>ORC: top matches (chunks + metadata)
        ORC->>ORC: Set all scores to 0, ai_insight to "CV missing..."
        ORC->>SYN: synthesize response (broad search)
        SYN-->>ORC: final answer (persona_applied: false)
    end
    
    ORC-->>API: response payload
    API-->>UI: display answer + items (with persona_applied flag)
```

---

## CV Processing Flow: Upload → Persona/Index

There are two common modes:
1) **Persona-only**: parse CV → build UserPersona without indexing
2) **Index CV**: parse CV → chunk → embed → upsert to Pinecone (`cvs` namespace)

```mermaid
sequenceDiagram
    participant UI as Streamlit UI
    participant CV as CV Processor
    participant EMB as Embedding Service
    participant PC as Pinecone
    participant ORC as Orchestrator

    UI->>CV: Upload PDF
    CV->>CV: Parse PDF (Docling) → text/markdown
    CV->>CV: Chunk + metadata (doc_id, section, offsets)
    CV-->>ORC: Extracted text (and/or chunks)

    alt Index CV chunks
        CV->>EMB: /embed (chunk texts)
        EMB-->>CV: embeddings
        CV->>PC: upsert(namespace="cvs")
    end
```

---

## Vacancy Indexing Flow: Cache / Firecrawl → Index

Vacancy acquisition sources:
- Local JSON cache (`vacancies_dump.json`)
- Real-time fetching via Firecrawl (Hunter Agent)

```mermaid
sequenceDiagram
    participant API as API / Orchestrator
    participant JSON as vacancies_dump.json
    participant H as Hunter Agent (Firecrawl)
    participant EMB as Embedding Service
    participant PC as Pinecone

    API->>JSON: load cached vacancies (optional)
    alt cache missing/empty/stale
        API->>H: fetch fresh vacancies from target sites
        H-->>API: raw vacancies (normalized fields)
    end
    API->>API: chunk + normalize + metadata
    API->>EMB: /embed vacancy chunks
    EMB-->>API: embeddings
    API->>PC: upsert(namespace="vacancies")
```

---

## Orchestration Notes (LangGraph)

Recommended node boundaries:
- **intent_node**: call Job Scout (DeepSeek R1) → filters + query text
- **retrieval_node**: embed query → Pinecone query → collect top-K
- **analysis_node**: Matchmaker (optional stronger model) → MatchingReport / explanations
- **synthesis_node**: Response Synthesizer → UI-ready response

Trigger rules:
- Hunter Agent triggers when:
  - cache is empty OR
  - Pinecone returns near-zero results OR
  - user explicitly asks for “latest / today / this week”

---

## Namespaces & Metadata (Pinecone)

Recommended:
- Single index (e.g., `funds-search`)
- Namespaces:
  - `cvs` for CV/resume chunks
  - `vacancies` for vacancy chunks

Minimum metadata per vector:
- `source_type`: `cv` | `vacancy`
- `doc_id` / `candidate_id` / `vacancy_id`
- `title`, `company`, `url` (for vacancies)
- `chunk_id`, `chunk_index`, `text_offset` (optional but helpful)

---

## Source of Truth

- Schemas: `shared/schemas.py`
- Pinecone client wrapper: `shared/pinecone_client.py`
- Orchestration logic: `apps/orchestrator/`


