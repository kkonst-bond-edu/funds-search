# Apps Directory

Business logic and user-facing applications.

## Components

- **api** (Port 8000): Lightweight FastAPI container with LangGraph matching logic. Uses multi-stage build (<500MB image).
- **web_ui** (Port 8501): Streamlit recruiter UI. Communicates via HTTP only. Requires `BACKEND_API_URL` and `CV_PROCESSOR_URL` environment variables.
- **orchestrator**: LangGraph state machine for search and matching workflows.

## Deployment

Each app has its own GitHub Actions workflow for Azure Container Apps deployment.
