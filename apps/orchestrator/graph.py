"""
LangGraph orchestrator for funds-search matching.
Implements a state machine with Retrieval and Analysis nodes.
"""
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
import httpx
import os
from shared.schemas import Job, MatchResult, SearchRequest
from shared.pinecone_client import PineconeClient


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
embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL", "http://localhost:8001")
pinecone_client = None
llm = None


def get_pinecone_client() -> PineconeClient:
    """Get or create Pinecone client instance."""
    global pinecone_client
    if pinecone_client is None:
        pinecone_client = PineconeClient()
    return pinecone_client


def get_llm() -> ChatGoogleGenerativeAI:
    """Get or create LLM instance."""
    global llm
    if llm is None:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
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
            url=metadata.get("url", ""),
            company=metadata.get("company", ""),
            text=metadata.get("text", ""),
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
{job.text[:2000]}  # Limit context size
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

