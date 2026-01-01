"""
FastAPI main application for funds-search.
Refactored to use LangGraph orchestrator.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from shared.schemas import SearchRequest, MatchResult
from apps.orchestrator import run_search


app = FastAPI(
    title="Funds Search API",
    description="Search and match job openings at VC funds using Multi-Agent RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/search", response_model=List[MatchResult])
async def search(request: SearchRequest):
    """
    Search for job openings at VC funds using the LangGraph orchestrator.
    
    The orchestrator:
    1. Retrieves top 10 matches from Pinecone using BGE-M3 embeddings
    2. Analyzes each match using Gemini AI agent
    
    Accepts a JSON payload with:
    - query: search query string (required)
    - location: optional location filter
    - role: optional role/job title filter
    - remote: optional boolean for remote positions
    - user_id: optional user ID for personalized search
    
    Returns:
        List of MatchResult objects with scores and AI-generated reasoning
    """
    try:
        match_results = await run_search(request)
        return match_results
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing search: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

