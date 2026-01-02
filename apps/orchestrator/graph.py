"""
LangGraph orchestrator for funds-search matching.
Implements a state machine with Retrieval and Analysis nodes.
"""
import logging
import time
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import httpx
import os
from shared.schemas import Job, MatchResult, SearchRequest, VacancyMatchResult, MatchRequest
from shared.pinecone_client import VectorStore

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
llm = None


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


def get_llm() -> ChatGoogleGenerativeAI:
    """Get or create LLM instance."""
    global llm
    if llm is None:
        try:
            logger.info("Initializing Gemini LLM client...")
            google_api_key = os.getenv("GOOGLE_API_KEY", "")
            if not google_api_key:
                logger.error("GOOGLE_API_KEY environment variable is not set")
                raise ValueError("GOOGLE_API_KEY environment variable is required")
            logger.info("GOOGLE_API_KEY found, creating ChatGoogleGenerativeAI instance...")
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                google_api_key=google_api_key
            )
            logger.info("Gemini LLM client initialized successfully")
        except ValueError as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            logger.error("Please check that GOOGLE_API_KEY environment variable is set")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing LLM client: {str(e)}")
            raise
    return llm

# System prompt for Gemini agent
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
    Node 2: Analysis - Gemini Agent analyzes matches and generates reasoning.
    
    Args:
        state: Current orchestrator state
        
    Returns:
        Updated state with match_results
    """
    query = state["query"]
    retrieved_jobs = state["retrieved_jobs"]
    job_scores = state.get("job_scores", [])
    
    match_results = []
    
    for idx, job in enumerate(retrieved_jobs):
        # Get similarity score from Pinecone (cosine similarity)
        similarity_score = job_scores[idx] if idx < len(job_scores) else 0.0
        # Calculate similarity score (cosine similarity from Pinecone)
        # We'll use a placeholder score for now, as Pinecone already provides scores
        # In a real implementation, you'd retrieve the score from the search results
        
        # Prepare context for Gemini
        job_context = f"""
Job Title: {job.title or 'N/A'}
Company: {job.company}
Location: {job.location or 'N/A'}
Remote: {job.remote}
URL: {job.url}

Job Description:
{job.raw_text[:2000]}  # Limit context size
"""
        
        # Create messages for Gemini
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
        
        # Get analysis from Gemini
        try:
            gemini_llm = get_llm()
            response = await gemini_llm.ainvoke(messages)
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
    retrieved_vacancies: List[Dict[str, Any]]  # List of vacancy search results
    vacancy_scores: List[float]  # Store similarity scores from Pinecone
    match_results: List[VacancyMatchResult]
    top_k: int


# System prompt for Gemini agent in candidate-vacancy matching
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


async def fetch_candidate_node(state: MatchingState) -> MatchingState:
    """
    Node 1: Fetch candidate embedding from Pinecone.
    
    Handles eventual consistency in Pinecone by retrying once if candidate is not found initially.
    This handles the case where the CV was just uploaded and Pinecone index is still updating.
    
    Args:
        state: Current matching state
        
    Returns:
        Updated state with candidate_embedding
    """
    candidate_id = state["candidate_id"]
    
    # Get candidate embedding from Pinecone using namespace "cvs"
    pc_client = get_pinecone_client()
    candidate_embedding = pc_client.get_candidate_embedding(candidate_id, namespace="cvs")
    
    # If not found initially, wait 5 seconds and try once more (handles eventual consistency)
    if candidate_embedding is None:
        logger.info(f"Candidate {candidate_id} not found initially, waiting 5 seconds for Pinecone eventual consistency...")
        time.sleep(5)
        candidate_embedding = pc_client.get_candidate_embedding(candidate_id, namespace="cvs")
        
        if candidate_embedding is None:
            raise ValueError(f"Candidate with ID {candidate_id} not found in Pinecone. Please ensure the CV has been processed.")
    
    return {
        **state,
        "candidate_embedding": candidate_embedding
    }


async def search_vacancies_node(state: MatchingState) -> MatchingState:
    """
    Node 2: Search for vacancies in Pinecone using filter {'type': 'vacancy'}.
    
    Args:
        state: Current matching state
        
    Returns:
        Updated state with retrieved_vacancies and vacancy_scores
    """
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
    Node 3: Use Gemini to rerank results and explain WHY the vacancy fits the candidate.
    
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
    pc_client = get_pinecone_client()
    
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
        
        # Prepare context for Gemini - ensure all values are clean strings
        vacancy_context = f"""
Vacancy ID: {vacancy_id}
Vacancy Description:
{vacancy_text}
"""
        
        # Create messages for Gemini
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
        
        # Get analysis from Gemini
        try:
            gemini_llm = get_llm()
            response = await gemini_llm.ainvoke(messages)
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
    
    Returns:
        Compiled StateGraph
    """
    # Create graph
    workflow = StateGraph(MatchingState)
    
    # Add nodes
    workflow.add_node("fetch_candidate", fetch_candidate_node)
    workflow.add_node("search_vacancies", search_vacancies_node)
    workflow.add_node("rerank_and_explain", rerank_and_explain_node)
    
    # Define edges
    workflow.set_entry_point("fetch_candidate")
    workflow.add_edge("fetch_candidate", "search_vacancies")
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
        "retrieved_vacancies": [],
        "vacancy_scores": [],
        "match_results": [],
        "top_k": match_request.top_k or 10
    }
    
    # Run orchestrator
    final_state = await matching_orchestrator.ainvoke(initial_state)
    
    return final_state["match_results"]

