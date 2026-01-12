# Orchestrator (LangGraph)

This directory contains the **intelligence core** of the Funds Search system. It uses **LangGraph** to manage the state and workflow of multiple specialized AI agents.

## 🧠 The Agent Fleet

The system is composed of specialized agents, each with a distinct responsibility and prompt.

### 1. Talent Strategist 🕵️‍♂️
- **Role**: Memory & Profiler
- **File**: `agents/talent_strategist.py`
- **Prompt**: `prompts/talent_strategist.txt`
- **Function**: Analyzes chat history and user messages to build and update a `UserPersona`. It performs **incremental memory updates**, meaning it merges new information (e.g., "actually, I want remote only") into the existing profile without forgetting previous details (e.g., "skills: Python").

### 2. Job Scout 🛰️
- **Role**: Search Architect
- **File**: `agents/job_scout.py`
- **Prompt**: `prompts/job_scout.txt`
- **Function**: Translates the human-readable `UserPersona` into a precise database query. It generates a **Hybrid Search Query** consisting of:
  - **Semantic Query**: A rich text description for vector search (e.g., "Senior Backend Engineer with Python and high-load experience in Fintech").
  - **Metadata Filters**: Structured Pinecone filters (e.g., `{"remote_available": true, "company_stage": {"$in": ["Series A", "Series B"]}}`).

### 3. Matchmaker 🤝
- **Role**: Analyst & Critic
- **File**: `agents/matchmaker.py`
- **Prompt**: `prompts/matchmaker.txt`
- **Function**: Takes the top candidates found by the search and performs a deep analysis. It reads the full job description and the candidate's profile to assign a relevance score (0-100) and generate a "Why this match?" explanation.

### 4. Vacancy Analyst 🧠
- **Role**: Enrichment
- **File**: `agents/vacancy_analyst.py`
- **Prompt**: `prompts/classification.txt`, `prompts/enrichment.txt`
- **Function**: Used during the *ingestion* phase (scraping) to turn raw HTML into structured data. It classifies jobs into taxonomies and extracts entities like tech stack and benefits.

## ⚙️ Workflows (Graphs)

### Search Graph
Used for the chat interface.
`Start` -> `Talent Strategist` -> `Job Scout` -> `Search Vacancies` -> `End`

1.  **Talent Strategist** updates the `UserPersona` based on the latest message.
2.  **Job Scout** generates search parameters from the updated persona.
3.  **Search Vacancies** executes the query against Pinecone.

### Matching Graph
Used for the "Find Matches" feature (candidate ID based).
`Start` -> `Talent Strategist` (Optional) -> `Job Scout` -> `Search Vacancies` -> `Rerank & Explain` -> `End`

1.  Fetches candidate profile (CV).
2.  Searches for relevant vacancies.
3.  **Matchmaker** reranks and explains the results.

## 📁 Directory Structure

- `agents/`: Python classes for each agent (inheriting from `BaseAgent`).
- `prompts/`: Text files containing the system prompts for LLMs.
- `settings/`: Configuration files (e.g., `agents.yaml` for model selection).
- `workflow.py`: The LangGraph definitions wiring the agents together.
- `llm.py`: LLM provider factory (supports DeepSeek, OpenAI, etc.).
