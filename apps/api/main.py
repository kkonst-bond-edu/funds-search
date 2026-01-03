"""
FastAPI main application for funds-search.
Refactored to use LangGraph orchestrator.
"""
import logging
import time
import asyncio
import httpx
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from shared.schemas import (
    SearchRequest, MatchResult, MatchRequest, VacancyMatchResult,
    SystemDiagnosticsResponse, ServiceDiagnostic
)
from apps.orchestrator import run_search, run_match, get_pinecone_client, get_llm_provider, LLMProviderFactory
from langchain_core.messages import HumanMessage

# Import vacancies router
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.api.v1.vacancies import router as vacancies_router

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Funds Search API",
    description="Search and match job openings at VC funds using Multi-Agent RAG",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(vacancies_router)


@app.get("/")
async def root():
    """Root endpoint for health checks."""
    return {"status": "ok"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/v1/health")
async def health_v1():
    """Health check endpoint with API v1 prefix."""
    return {"status": "ok", "version": "2.0.0"}


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


@app.post("/match", response_model=List[VacancyMatchResult])
async def match(request: MatchRequest):
    """
    Match a candidate with vacancies using the LangGraph orchestrator.
    
    The orchestrator:
    1. Fetches the candidate's embedding from Pinecone namespace "cvs" (based on their processed CV)
    2. Searches for vacancies in Pinecone namespace "vacancies"
    3. Uses Gemini to rerank results and explain WHY the vacancy fits the candidate
    
    Note: This endpoint has sufficient timeout to handle:
    - Graph's retry logic (5 second sleep for Pinecone eventual consistency)
    - Azure Container Apps cold starts (handled by client-side retries in web_ui)
    - LLM processing time for multiple vacancies
    
    Accepts a JSON payload with:
    - candidate_id: unique identifier for the candidate (user_id) (required)
    - top_k: number of top matches to return (optional, default: 10)
    
    Returns:
        List of VacancyMatchResult objects with scores and AI-generated reasoning
        explaining why each vacancy fits the candidate
    """
    try:
        match_results = await run_match(request)
        return match_results
    except ValueError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error performing match: {str(e)}"
        )


async def check_service_with_retry(
    service_url: str,
    service_name: str,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    timeout: float = 5.0
) -> ServiceDiagnostic:
    """
    Check a service health with retry logic for cold starts.
    
    Args:
        service_url: Base URL of the service
        service_name: Name of the service for logging
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
        timeout: Request timeout in seconds
        
    Returns:
        ServiceDiagnostic object with status, latency, and error details
    """
    last_error = None
    last_error_type = None
    health_url = f"{service_url}/health"
    
    logger.info(f"Checking {service_name} at URL: {health_url}")
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=timeout) as client:
                logger.debug(f"{service_name} health check attempt {attempt + 1}/{max_retries}: {health_url}")
                response = await client.get(health_url)
                elapsed_ms = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    logger.info(f"Service {service_name} is healthy (attempt {attempt + 1}/{max_retries}, latency: {elapsed_ms:.2f}ms)")
                    return ServiceDiagnostic(
                        status="ok",
                        latency=round(elapsed_ms, 2),
                        error=None,
                        error_type=None
                    )
                elif response.status_code == 404:
                    error_msg = f"Service endpoint not found (404) at {health_url}"
                    error_type = "404"
                    logger.warning(f"Service {service_name} returned 404 (attempt {attempt + 1}/{max_retries}): {error_msg}")
                    logger.warning(f"Full URL attempted: {health_url}, Response: {response.text[:200]}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        return ServiceDiagnostic(
                            status="error",
                            latency=None,
                            error=error_msg,
                            error_type=error_type
                        )
                else:
                    error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                    error_type = f"http_{response.status_code}"
                    logger.warning(f"Service {service_name} returned {response.status_code} (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        continue
                    else:
                        return ServiceDiagnostic(
                            status="error",
                            latency=None,
                            error=error_msg,
                            error_type=error_type
                        )
                        
        except httpx.TimeoutException as e:
            last_error = f"Timeout after {timeout}s at {health_url}"
            last_error_type = "timeout"
            logger.warning(f"Service {service_name} timeout (attempt {attempt + 1}/{max_retries}): {last_error}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
        except httpx.ConnectError as e:
            last_error = f"Connection refused to {health_url}: {str(e)[:100]}"
            last_error_type = "connection"
            logger.warning(f"Service {service_name} connection error (attempt {attempt + 1}/{max_retries}): {last_error}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
        except Exception as e:
            last_error = f"Unexpected error calling {health_url}: {str(e)[:100]}"
            last_error_type = "unknown"
            logger.error(f"Service {service_name} unexpected error (attempt {attempt + 1}/{max_retries}): {last_error}")
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                continue
    
    # All retries failed
    logger.error(f"Service {service_name} failed after {max_retries} attempts. URL: {health_url}, Error: {last_error}")
    return ServiceDiagnostic(
        status="error",
        latency=None,
        error=last_error,
        error_type=last_error_type
    )


async def check_pinecone() -> ServiceDiagnostic:
    """
    Check Pinecone vector store connectivity.
    
    Returns:
        ServiceDiagnostic object with status, latency, and error details
    """
    try:
        start_time = time.time()
        pc_client = get_pinecone_client()
        # Execute a simple query to test connectivity
        # Use a dummy vector for the query
        dummy_vector = [0.0] * 1024  # BGE-M3 uses 1024 dimensions
        result = pc_client.index.query(
            vector=dummy_vector,
            top_k=1,
            namespace="cvs"
        )
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Pinecone is healthy (latency: {elapsed_ms:.2f}ms)")
        return ServiceDiagnostic(
            status="ok",
            latency=round(elapsed_ms, 2),
            error=None,
            error_type=None
        )
    except Exception as e:
        error_msg = str(e)[:200]
        logger.error(f"Pinecone check failed: {error_msg}")
        return ServiceDiagnostic(
            status="error",
            latency=None,
            error=error_msg,
            error_type="database_error"
        )


async def check_llm_provider() -> ServiceDiagnostic:
    """
    Check LLM provider connectivity (DeepSeek or other active agent).
    
    Returns:
        ServiceDiagnostic object with status, latency, and error details
    """
    try:
        start_time = time.time()
        llm_provider = get_llm_provider()
        
        # Get provider info for logging
        provider_info = LLMProviderFactory.get_provider_info()
        provider_name = provider_info.get("name", "Unknown")
        
        # Make a simple test call using the provider's health check
        is_healthy = await llm_provider.health_check()
        elapsed_ms = (time.time() - start_time) * 1000
        
        if is_healthy:
            logger.info(f"LLM provider ({provider_name}) is healthy (latency: {elapsed_ms:.2f}ms)")
            return ServiceDiagnostic(
                status="ok",
                latency=round(elapsed_ms, 2),
                error=None,
                error_type=None
            )
        else:
            raise Exception("Health check returned False")
    except Exception as e:
        error_msg = str(e)[:200]
        provider_info = LLMProviderFactory.get_provider_info()
        provider_name = provider_info.get("name", "Unknown")
        logger.error(f"LLM provider ({provider_name}) check failed: {error_msg}")
        return ServiceDiagnostic(
            status="error",
            latency=None,
            error=error_msg,
            error_type="llm_error"
        )


@app.get("/api/v1/system/diagnostics", response_model=SystemDiagnosticsResponse)
@app.get("/system/diagnostics", response_model=SystemDiagnosticsResponse)  # Fallback route
async def system_diagnostics():
    """
    Perform a comprehensive system diagnostics check.
    
    Checks the health and connectivity of:
    - CV Processor service
    - Embedding Service
    - Pinecone Vector Store
    - LLM Provider (DeepSeek or active agent)
    
    Implements retry logic (3 attempts with 2s delay) for service pings to trigger warm-up.
    
    Returns:
        SystemDiagnosticsResponse with overall status and individual service diagnostics
    """
    logger.info("Starting system diagnostics check...")
    
    # Get service URLs from environment with detailed logging
    cv_processor_url = os.getenv("CV_PROCESSOR_URL", "http://cv-processor:8001")
    embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
    
    logger.info(f"CV Processor URL: {cv_processor_url}")
    logger.info(f"Embedding Service URL: {embedding_service_url}")
    
    # Check all services in parallel
    import asyncio
    cv_result, embedding_result, pinecone_result, llm_result = await asyncio.gather(
        check_service_with_retry(cv_processor_url, "CV Processor"),
        check_service_with_retry(embedding_service_url, "Embedding Service"),
        check_pinecone(),
        check_llm_provider(),
        return_exceptions=True
    )
    
    # Handle exceptions from gather
    if isinstance(cv_result, Exception):
        cv_result = ServiceDiagnostic(
            status="error",
            latency=None,
            error=str(cv_result)[:200],
            error_type="exception"
        )
    if isinstance(embedding_result, Exception):
        embedding_result = ServiceDiagnostic(
            status="error",
            latency=None,
            error=str(embedding_result)[:200],
            error_type="exception"
        )
    if isinstance(pinecone_result, Exception):
        pinecone_result = ServiceDiagnostic(
            status="error",
            latency=None,
            error=str(pinecone_result)[:200],
            error_type="exception"
        )
    if isinstance(llm_result, Exception):
        llm_result = ServiceDiagnostic(
            status="error",
            latency=None,
            error=str(llm_result)[:200],
            error_type="exception"
        )
    
    # Get LLM provider info for display
    try:
        provider_info = LLMProviderFactory.get_provider_info()
        provider_name = provider_info.get("name", "Unknown")
        provider_status = provider_info.get("status", "unknown")
        provider_model = provider_info.get("model", "unknown")
    except Exception as e:
        logger.warning(f"Could not get provider info: {str(e)}")
        provider_name = "Unknown"
        provider_status = "unknown"
        provider_model = None
    
    # Build services dict with detailed connectivity information
    services = {
        "cv_processor": cv_result,
        "embedding_service": embedding_result,
        "pinecone": pinecone_result,
        "llm_provider": llm_result,
        # Detailed connectivity diagnostics
        "api_to_cv_processor": cv_result,
        "api_to_db": pinecone_result,
        "api_to_llm": llm_result,
        # Agent information for UI display
        "agent": {
            "name": provider_name,
            "status": "online" if llm_result.status == "ok" else "offline",
            "model": provider_model if llm_result.status == "ok" else None
        }
    }
    
    # Determine overall status
    all_ok = all(s.status == "ok" for s in [cv_result, embedding_result, pinecone_result, llm_result])
    any_ok = any(s.status == "ok" for s in [cv_result, embedding_result, pinecone_result, llm_result])
    
    if all_ok:
        overall_status = "ok"
    elif any_ok:
        overall_status = "partial"
    else:
        overall_status = "error"
    
    logger.info(f"System diagnostics complete. Overall status: {overall_status}")
    logger.info(f"CV Processor: {cv_result.status}, Embedding: {embedding_result.status}, "
                f"Pinecone: {pinecone_result.status}, LLM: {llm_result.status}")
    
    return SystemDiagnosticsResponse(
        status=overall_status,
        services=services,
        timestamp=datetime.utcnow().isoformat()
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

