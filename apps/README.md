# Apps Directory

Business logic and user-facing applications.

## Components

- **api** (Port 8000): FastAPI gateway with LangGraph orchestration and streaming `/chat/stream` endpoint.
- **web_ui** (Port 8501): Streamlit recruiter UI. Uses SSE streaming for chat (`/chat/stream`). Requires `BACKEND_API_URL` and `CV_PROCESSOR_URL` environment variables.
- **orchestrator**: LangGraph state machine for search and matching workflows.

## Deployment

Each app has its own GitHub Actions workflow for Azure Container Apps deployment.
