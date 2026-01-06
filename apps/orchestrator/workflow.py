"""
LangGraph orchestrator for funds-search matching.
Implements a state machine with Retrieval and Analysis nodes.
"""
import logging
import time
import numpy as np
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
import httpx
import os
from shared.schemas import Job, MatchResult, SearchRequest, VacancyMatchResult, MatchRequest, UserPersona, MatchingReport
from shared.pinecone_client import VectorStore
from apps.orchestrator.llm import LLMProviderFactory

logger = logging.getLogger(__name__)


class OrchestratorState(TypedDict):
    """State for the orchestrator graph."""
    query: str
    query_vector: List[float]
    search_request: SearchRequest
    retrieved_jobs: List[Job]
    job_scores: List[float]  # Store similarity scores from Pinecone
    match_results: List[MatchResult]
    location: str
    role: str
    remote: bool


# Initialize services (lazy initialization)
embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
pinecone_client = None


def get_pinecone_client() -> VectorStore:
    """Get or create Pinecone client instance."""
    global pinecone_client
    if pinecone_client is None:
        try:
            logger.info("Initializing Pinecone client...")
            pinecone_client = VectorStore()
            logger.info("Pinecone client initialized successfully")
        except ValueError as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            logger.error("Please check that PINECONE_API_KEY environment variable is set")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Pinecone client: {str(e)}")
            raise
    return pinecone_client


def get_llm_provider():
    """
    Get the active LLM provider instance.

    Returns:
        LLMProvider instance (e.g., DeepSeekProvider)
    """
    return LLMProviderFactory.get_active_provider()


# System prompt for candidate matching agent (provider-agnostic)
SYSTEM_PROMPT = """You are an expert AI analyst specializing in matching job openings at VC funds with candidate profiles.

Your task is to analyze job postings and provide detailed reasoning about:
1. The key requirements and responsibilities of the position
2. The type of candidate profile that would be a good fit
3. Specific skills, experience, and qualifications needed
4. Cultural fit and company values alignment

Provide clear, structured reasoning that explains why a candidate would or would not be a good match for each position.
Be specific about technical skills, years of experience, industry knowledge, and soft skills required."""


async def retrieval_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 1: Retrieval - Query Pinecone for top 10 matches.

    Args:
        state: Current orchestrator state

    Returns:
        Updated state with retrieved_jobs
    """
    query = state["query"]

    # Get query embedding from embedding service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{embedding_service_url}/embed",
                json={"texts": [query]}
            )
            response.raise_for_status()
            embeddings = response.json()["embeddings"]
            query_vector = embeddings[0]
    except Exception as e:
        raise RuntimeError(f"Error getting query embedding: {str(e)}")

    # Build filter for Pinecone search
    filter_dict = {}
    if state.get("location"):
        filter_dict["location"] = state["location"]
    if state.get("remote") is not None:
        filter_dict["remote"] = state["remote"]

    # Search Pinecone
    pc_client = get_pinecone_client()
    search_results = pc_client.search_similar(
        query_vector=query_vector,
        top_k=10,
        filter_dict=filter_dict if filter_dict else None
    )

    # Convert results to Job objects and store scores
    retrieved_jobs = []
    job_scores = []
    for result in search_results:
        metadata = result["metadata"]
        job = Job(
            id=metadata.get("job_id", metadata.get("id", "unknown")),
            url=metadata.get("url", ""),
            company=metadata.get("company", ""),
            raw_text=metadata.get("text", metadata.get("raw_text", "")),
            vector=None,  # Don't store vector in response
            title=metadata.get("title"),
            location=metadata.get("location"),
            remote=metadata.get("remote", False)
        )
        retrieved_jobs.append(job)
        job_scores.append(result["score"])  # Store Pinecone similarity score

    return {
        **state,
        "query_vector": query_vector,
        "retrieved_jobs": retrieved_jobs,
        "job_scores": job_scores
    }


async def analysis_node(state: OrchestratorState) -> OrchestratorState:
    """
    Node 2: Analysis - LLM Agent analyzes matches and generates reasoning.

    Args:
        state: Current orchestrator state

    Returns:
        Updated state with match_results
    """
    query = state["query"]
    retrieved_jobs = state["retrieved_jobs"]
    job_scores = state.get("job_scores", [])

    match_results = []

    # Get LLM provider (supports multi-agent architecture)
    llm_provider = get_llm_provider()

    for idx, job in enumerate(retrieved_jobs):
        # Get similarity score from Pinecone (cosine similarity)
        similarity_score = job_scores[idx] if idx < len(job_scores) else 0.0

        # Prepare context for LLM (prompt management separated from provider)
        job_context = f"""
Job Title: {job.title or 'N/A'}
Company: {job.company}
Location: {job.location or 'N/A'}
Remote: {job.remote}
URL: {job.url}

Job Description:
{job.raw_text[:2000]}  # Limit context size
"""

        # Create messages (provider-agnostic prompt structure)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"""
Query: {query}

Job Posting:
{job_context}

Please analyze this job posting in the context of the search query and provide:
1. A relevance score (0-1) indicating how well this job matches the query
2. Detailed reasoning explaining the match quality
3. Key factors that make this a good or poor match
""")
        ]

        # Get analysis from LLM provider (with built-in retry logic)
        try:
            response = await llm_provider.ainvoke(messages)
            reasoning = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            reasoning = f"Error generating analysis: {str(e)}"

        # Create match result with actual similarity score from Pinecone
        match_result = MatchResult(
            score=similarity_score,  # Cosine similarity from Pinecone
            reasoning=reasoning,
            job=job
        )

        match_results.append(match_result)

    return {
        **state,
        "match_results": match_results
    }


def create_orchestrator_graph() -> StateGraph:
    """
    Create and compile the LangGraph orchestrator.

    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(OrchestratorState)

    # Add nodes
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("analysis", analysis_node)

    # Define edges
    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "analysis")
    workflow.add_edge("analysis", END)

    # Compile graph
    app = workflow.compile()

    return app


# Global orchestrator instance
orchestrator = create_orchestrator_graph()


async def run_search(search_request: SearchRequest) -> List[MatchResult]:
    """
    Run the orchestrator for a search request.

    Args:
        search_request: SearchRequest object

    Returns:
        List of MatchResult objects
    """
    # Initialize state
    initial_state: OrchestratorState = {
        "query": search_request.query,
        "query_vector": [],
        "search_request": search_request,
        "retrieved_jobs": [],
        "job_scores": [],
        "match_results": [],
        "location": search_request.location or "",
        "role": search_request.role or "",
        "remote": search_request.remote or False
    }

    # Run orchestrator
    final_state = await orchestrator.ainvoke(initial_state)

    return final_state["match_results"]


# ============================================================================
# Candidate-Vacancy Matching Graph
# ============================================================================

class MatchingState(TypedDict):
    """State for the candidate-vacancy matching graph."""
    candidate_id: str
    candidate_embedding: List[float]
    user_persona: Optional[UserPersona]  # User persona from interview/context
    raw_scraped_data: List[Dict[str, Any]]  # Raw scraped job data from web discovery
    retrieved_vacancies: List[Dict[str, Any]]  # List of vacancy search results
    vacancy_scores: List[float]  # Store similarity scores from Pinecone
    match_results: List[VacancyMatchResult]  # Legacy format (for backward compatibility)
    final_reports: List[MatchingReport]  # New structured matching reports
    top_k: int


# System prompt for candidate-vacancy matching agent (provider-agnostic)
MATCHING_SYSTEM_PROMPT = """You are an expert AI recruiter specializing in matching candidates with job vacancies.

Your task is to analyze why a specific vacancy is a good fit for a candidate based on:
1. The candidate's skills, experience, and background (from their CV/resume)
2. The vacancy's requirements and responsibilities
3. Alignment between candidate capabilities and job needs
4. Cultural fit and career growth opportunities

Provide clear, detailed reasoning that explains WHY this vacancy fits the candidate.
Be specific about:
- How the candidate's skills match the job requirements
- Relevant experience that makes them suitable
- Potential gaps and how they might be addressed
- Why this role would be a good career move for the candidate

Format your response as a structured explanation that a recruiter would use to present the match to both the candidate and the hiring manager."""


async def talent_strategist_node(state: MatchingState) -> MatchingState:
    """
    Placeholder node: Talent Strategist - Process interview context to build user persona.

    This node will:
    - Process interview/conversation context
    - Extract technical skills, career goals, preferences
    - Build UserPersona object

    Args:
        state: Current matching state

    Returns:
        Updated state with user_persona
    """
    # TODO: Implement interview processing logic
    # For now, return state unchanged (placeholder)
    logger.info("Talent Strategist node: Processing interview context (placeholder)")

    # Placeholder: If user_persona is not set, create a minimal one
    if state.get("user_persona") is None:
        user_persona = UserPersona(
            technical_skills=[],
            career_goals=[],
            preferred_startup_stage=None,
            cultural_preferences=[],
            user_id=state.get("candidate_id")
        )
        return {
            **state,
            "user_persona": user_persona
        }

    return state


async def web_hunter_node(state: MatchingState) -> MatchingState:
    """
    Placeholder node: Web Hunter - Firecrawl discovery logic for job discovery.

    This node will:
    - Use Firecrawl to discover job postings from VC fund websites
    - Scrape and process job listings
    - Store raw_scraped_data for further processing

    Args:
        state: Current matching state

    Returns:
        Updated state with raw_scraped_data
    """
    # TODO: Implement Firecrawl discovery logic
    # For now, return state unchanged (placeholder)
    logger.info("Web Hunter node: Discovering jobs via Firecrawl (placeholder)")

    # Placeholder: Initialize raw_scraped_data if not present
    if state.get("raw_scraped_data") is None:
        return {
            **state,
            "raw_scraped_data": []
        }

    return state


async def fetch_candidate_node(state: MatchingState) -> MatchingState:
    """
    Node 1: Fetch candidate embedding from Pinecone.

    Implements retry logic for Azure cold starts and Pinecone eventual consistency:
    - Retries up to 5 times if candidate is not found
    - Waits 5 seconds between retries
    - Ensures Pinecone client is initialized during each retry (handles wake-up scenarios)
    - Only raises ValueError after all retries have failed

    Args:
        state: Current matching state

    Returns:
        Updated state with candidate_embedding
    """
    candidate_id = state["candidate_id"]
    namespace = "cvs"
    max_retries = 5
    retry_delay = 5  # seconds

    # BGE-M3 uses 1024 dimensions
    EMBEDDING_DIM = 1024

    # Retry loop for Azure cold starts and eventual consistency
    for attempt in range(max_retries):
        try:
            # Get Pinecone client inside the loop to ensure it's initialized during wake-up
            pc_client = get_pinecone_client()

            # Query Pinecone directly
            logger.info(f"Fetching candidate embedding for {candidate_id} from namespace: {namespace} (attempt {attempt + 1}/{max_retries})")
            query_result = pc_client.index.query(
                vector=[0.0] * EMBEDDING_DIM,  # Dummy vector for BGE-M3 (1024 dims)
                top_k=1,  # Get only the first match
                include_metadata=True,
                include_values=True,  # CRITICAL: Include values to get embedding
                filter={"user_id": {"$eq": candidate_id}},
                namespace=namespace
            )

            # Check if matches are found
            if not query_result.matches:
                if attempt < max_retries - 1:
                    logger.info(f"Candidate not found yet, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    # All retries failed
                    raise ValueError(f"Candidate with ID {candidate_id} not found in Pinecone after {max_retries} attempts. Please ensure the CV has been processed.")

            # Extract 'values' from the first match
            first_match = query_result.matches[0]
            if not hasattr(first_match, 'values') or first_match.values is None:
                if attempt < max_retries - 1:
                    logger.info(f"Candidate found but missing embedding values, retrying... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                    continue
                else:
                    raise ValueError(f"Candidate {candidate_id} found but embedding values are missing after {max_retries} attempts.")

            # Get the embedding values from the first match
            candidate_embedding = first_match.values

            # Normalize for cosine similarity
            embedding_array = np.array(candidate_embedding)
            norm = np.linalg.norm(embedding_array)
            if norm > 0:
                embedding_array = embedding_array / norm

            logger.info(f"Successfully extracted embedding for candidate {candidate_id} from first match (dim={len(embedding_array)})")

            return {
                **state,
                "candidate_embedding": embedding_array.tolist()
            }

        except ValueError:
            # Re-raise ValueError (candidate not found after all retries)
            raise
        except Exception as e:
            # For other exceptions, retry if we have attempts left
            if attempt < max_retries - 1:
                logger.warning(f"Error fetching candidate embedding (attempt {attempt + 1}/{max_retries}): {str(e)}. Retrying...")
                time.sleep(retry_delay)
                continue
            else:
                # All retries failed
                logger.error(f"Failed to fetch candidate embedding after {max_retries} attempts: {str(e)}")
                raise ValueError(f"Candidate with ID {candidate_id} not found in Pinecone after {max_retries} attempts. Error: {str(e)}")

    # Should not reach here, but just in case
    raise ValueError(f"Candidate with ID {candidate_id} not found in Pinecone after {max_retries} attempts.")


async def search_vacancies_node(state: MatchingState) -> MatchingState:
    """
    Node: Search for vacancies in Pinecone using filter {'type': 'vacancy'}.

    If candidate_embedding is not available, fetches it first.

    Args:
        state: Current matching state

    Returns:
        Updated state with retrieved_vacancies and vacancy_scores
    """
    # If candidate_embedding is not available, fetch it first
    if not state.get("candidate_embedding"):
        logger.info("Candidate embedding not found, fetching candidate first...")
        state = await fetch_candidate_node(state)

    candidate_embedding = state["candidate_embedding"]
    top_k = state.get("top_k", 10)

    # Search for vacancies using namespace "vacancies"
    pc_client = get_pinecone_client()
    search_results = pc_client.search_vacancies(
        query_vector=candidate_embedding,
        top_k=top_k,
        namespace="vacancies"
    )

    # Extract vacancies and scores
    # search_results is already a list of dicts with 'id', 'metadata', 'score'
    retrieved_vacancies = []
    vacancy_scores = []
    for result in search_results:
        # Ensure result is a dict (not a Pydantic model or other object)
        if isinstance(result, dict):
            retrieved_vacancies.append(result)
            vacancy_scores.append(result.get("score", 0.0))
        else:
            # Convert to dict if needed
            retrieved_vacancies.append({
                "id": getattr(result, "id", "unknown"),
                "metadata": getattr(result, "metadata", {}),
                "score": getattr(result, "score", 0.0)
            })
            vacancy_scores.append(getattr(result, "score", 0.0))

    return {
        **state,
        "retrieved_vacancies": retrieved_vacancies,
        "vacancy_scores": vacancy_scores
    }


async def rerank_and_explain_node(state: MatchingState) -> MatchingState:
    """
    Node 3: Use LLM to rerank results and explain WHY the vacancy fits the candidate.

    Args:
        state: Current matching state

    Returns:
        Updated state with match_results
    """
    candidate_id = state["candidate_id"]
    retrieved_vacancies = state["retrieved_vacancies"]
    vacancy_scores = state.get("vacancy_scores", [])

    # Get candidate's resume text for context (optional, can be enhanced)
    # For now, we'll use the candidate_id in the prompt

    # Get LLM provider (supports multi-agent architecture)
    llm_provider = get_llm_provider()

    match_results = []

    for idx, vacancy_result in enumerate(retrieved_vacancies):
        similarity_score = vacancy_scores[idx] if idx < len(vacancy_scores) else 0.0

        # Ensure vacancy_result is a dict
        if not isinstance(vacancy_result, dict):
            # Convert to dict if it's not already
            vacancy_result = {
                "id": str(getattr(vacancy_result, "id", "unknown")),
                "metadata": getattr(vacancy_result, "metadata", {}),
                "score": float(getattr(vacancy_result, "score", 0.0))
            }

        # Safely extract metadata - ensure it's a dict
        vacancy_metadata = vacancy_result.get("metadata", {})
        if not isinstance(vacancy_metadata, dict):
            vacancy_metadata = {}

        # Extract vacancy_id - ensure it's a clean string
        vacancy_id = str(vacancy_metadata.get("vacancy_id", vacancy_result.get("id", "unknown")))

        # Extract vacancy_text - ensure it's a clean string (not a list or object)
        vacancy_text_raw = vacancy_metadata.get("text", "")
        if isinstance(vacancy_text_raw, str):
            vacancy_text = vacancy_text_raw
        elif isinstance(vacancy_text_raw, list):
            # If text is a list, join it
            vacancy_text = " ".join(str(item) for item in vacancy_text_raw)
        else:
            # Convert to string if it's something else
            vacancy_text = str(vacancy_text_raw) if vacancy_text_raw else ""

        # Ensure vacancy_text is not empty, try to get from raw_text or other fields
        if not vacancy_text:
            vacancy_text = str(vacancy_metadata.get("raw_text", ""))

        # Limit text length for LLM context
        vacancy_text = vacancy_text[:2000] if len(vacancy_text) > 2000 else vacancy_text

        # Prepare context for LLM - ensure all values are clean strings (prompt management separated)
        vacancy_context = f"""
Vacancy ID: {vacancy_id}
Vacancy Description:
{vacancy_text}
"""

        # Create messages (provider-agnostic prompt structure)
        messages = [
            SystemMessage(content=MATCHING_SYSTEM_PROMPT),
            HumanMessage(content=f"""
Candidate ID: {candidate_id}
Similarity Score: {similarity_score:.4f}

Vacancy:
{vacancy_context}

Please analyze why this vacancy is a good fit for this candidate and provide:
1. Detailed reasoning explaining WHY the vacancy fits the candidate
2. Specific skills and experience that align
3. Potential benefits for the candidate's career
4. Any concerns or gaps that should be considered

Provide a comprehensive explanation that would help a recruiter present this match.
""")
        ]

        # Get analysis from LLM provider (with built-in retry logic)
        try:
            response = await llm_provider.ainvoke(messages)
            reasoning = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            reasoning = f"Error generating analysis: {str(e)}"

        # Create match result - ensure all values are clean types
        match_result = VacancyMatchResult(
            score=float(similarity_score),
            reasoning=str(reasoning),
            vacancy_id=str(vacancy_id),
            vacancy_text=str(vacancy_text),
            candidate_id=str(candidate_id)
        )

        match_results.append(match_result)

    return {
        **state,
        "match_results": match_results
    }


def create_matching_graph() -> StateGraph:
    """
    Create and compile the LangGraph orchestrator for candidate-vacancy matching.

    New flow: Entry -> talent_strategist -> web_hunter -> search_vacancies -> rerank_and_explain -> END

    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(MatchingState)

    # Add nodes
    workflow.add_node("talent_strategist", talent_strategist_node)
    workflow.add_node("web_hunter", web_hunter_node)
    workflow.add_node("fetch_candidate", fetch_candidate_node)
    workflow.add_node("search_vacancies", search_vacancies_node)
    workflow.add_node("rerank_and_explain", rerank_and_explain_node)

    # Define edges - new roadmap flow
    # Entry -> talent_strategist -> web_hunter -> search_vacancies -> rerank_and_explain -> END
    # Note: fetch_candidate_node is called conditionally within search_vacancies_node if candidate_embedding is not available
    workflow.set_entry_point("talent_strategist")
    workflow.add_edge("talent_strategist", "web_hunter")
    workflow.add_edge("web_hunter", "search_vacancies")
    workflow.add_edge("search_vacancies", "rerank_and_explain")
    workflow.add_edge("rerank_and_explain", END)

    # Compile graph
    app = workflow.compile()

    return app


# Global matching orchestrator instance
matching_orchestrator = create_matching_graph()


async def run_match(match_request: MatchRequest) -> List[VacancyMatchResult]:
    """
    Run the matching orchestrator for a candidate-vacancy match request.

    Args:
        match_request: MatchRequest object with candidate_id

    Returns:
        List of VacancyMatchResult objects
    """
    # Initialize state
    initial_state: MatchingState = {
        "candidate_id": match_request.candidate_id,
        "candidate_embedding": [],
        "user_persona": None,
        "raw_scraped_data": [],
        "retrieved_vacancies": [],
        "vacancy_scores": [],
        "match_results": [],
        "final_reports": [],
        "top_k": match_request.top_k or 10
    }

    # Run orchestrator
    final_state = await matching_orchestrator.ainvoke(initial_state)

    return final_state["match_results"]
