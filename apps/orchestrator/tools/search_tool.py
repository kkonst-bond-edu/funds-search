"""
Search tool for Job Scout agent.

This tool allows the agent to search for vacancies in Pinecone vector database.
"""
import os
import asyncio
import logging
import concurrent.futures
from typing import Dict, Any, List, Optional, Literal
import structlog
import httpx
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from shared.pinecone_client import VectorStore
from src.schemas.vacancy import CompanyStage

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Embedding request settings (override via env for slow services)
EMBEDDING_REQUEST_TIMEOUT = float(os.getenv("EMBEDDING_REQUEST_TIMEOUT", "60"))
EMBEDDING_CONNECT_TIMEOUT = float(os.getenv("EMBEDDING_CONNECT_TIMEOUT", "10"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
EMBEDDING_TIMEOUT_BACKOFF_BASE = 2
EMBEDDING_ERROR_BACKOFF_STEP = 1
EMBEDDING_THREAD_TIMEOUT_BUFFER = int(os.getenv("EMBEDDING_THREAD_TIMEOUT_BUFFER", "10"))


class SearchSchema(BaseModel):
    """Strict schema for search parameters matching Pinecone metadata fields."""
    
    query: str = Field(..., description="Core role + top skills only. No conversational filler.")
    category: Optional[Literal["Engineering", "Product", "Design", "Data & Analytics", "Sales & Business Development", "Marketing", "Operations", "Finance", "Legal", "People & HR", "Other"]] = Field(None, description="Job category filter")
    experience_level: Optional[Literal["Junior", "Mid", "Senior", "Lead", "Executive", "Unknown"]] = Field(None, description="Experience level filter")
    company_stage: Optional[List[Literal["Seed", "Series A", "Series B", "Series C", "Growth", "Unknown"]]] = Field(None, description="Company funding stage filter (list)")
    remote_option: Optional[Literal["remote", "office", "hybrid"]] = Field(None, description="Remote work option: 'remote' (fully remote), 'office' (on-site only), 'hybrid' (hybrid)")
    location: Optional[str] = Field(None, description="Job location filter (e.g., 'London', 'San Francisco', 'New York'). Case-insensitive matching is supported.")
    industry: Optional[str] = Field(
        None, 
        description=(
            "Industry sector filter. Supported values: "
            "'AI', 'Bio + Health', 'Consumer', 'Enterprise', 'Fintech', "
            "'American Dynamism', 'Logistics', 'Marketing', 'Other'. "
            "Multiple industries can be comma-separated (e.g., 'AI, Enterprise'). "
            "Case-insensitive matching is supported."
        )
    )
    salary_min: Optional[int] = Field(None, ge=0, description="Minimum salary in USD")
    employee_count: Optional[List[str]] = Field(None, description="Employee count filter (e.g., ['1-10', '11-50', '51-200'])")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True


def _build_filter_dict(schema: SearchSchema) -> Dict[str, Any]:
    """
    Convert SearchSchema to Pinecone filter_dict format.
    
    Maps schema fields to Pinecone metadata filters:
    - Single values use $eq
    - Lists use $in
    - remote_option string values are converted to boolean
    - location uses $in with case variations for case-insensitive matching
    """
    filter_dict = {}
    
    if schema.category:
        filter_dict["category"] = {"$eq": schema.category}
    
    if schema.experience_level:
        filter_dict["experience_level"] = {"$eq": schema.experience_level}
    
    if schema.company_stage:
        # Normalize and expand company_stage values
        # Growth includes Series B, Series C, and Growth (same logic as in vacancies.py)
        normalized_stages = []
        for stage in schema.company_stage:
            normalized = CompanyStage.get_stage_value(stage)
            if normalized == "Growth":
                # Growth includes Series B, Series C, and Growth
                normalized_stages.extend(["Series B", "Series C", "Growth"])
            else:
                normalized_stages.append(normalized)
        
        # Remove duplicates
        normalized_stages = list(set(normalized_stages))
        filter_dict["company_stage"] = {"$in": normalized_stages}
        
        logger.debug(
            "company_stage_filter_normalized",
            original_stages=schema.company_stage,
            normalized_stages=normalized_stages,
        )
    
    if schema.remote_option:
        # Map string values to boolean for Pinecone
        # Note: Pinecone stores remote_option as boolean
        # "remote" -> True, "office" -> False, "hybrid" -> False (but could also check is_hybrid)
        if schema.remote_option == "remote":
            filter_dict["remote_option"] = {"$eq": True}
        elif schema.remote_option == "office":
            filter_dict["remote_option"] = {"$eq": False}
        elif schema.remote_option == "hybrid":
            # Hybrid roles have remote_option=False, but we might want to filter differently
            # For now, we'll set remote_option=False (hybrid is not fully remote)
            filter_dict["remote_option"] = {"$eq": False}
            # Note: If we need to specifically filter for hybrid, we'd need an is_hybrid field
    
    if schema.location:
        location_lower = schema.location.lower().strip()
        
        # Location synonyms mapping (same as in vacancies.py)
        LOCATION_SYNONYMS = {
            "us": ["united states", "usa", "america", "u.s.", "u.s.a."],
            "uk": ["united kingdom", "london", "england", "britain", "great britain"],
            "united states": ["us", "usa", "america", "u.s.", "u.s.a."],
            "united kingdom": ["uk", "london", "england", "britain", "great britain"],
            "usa": ["us", "united states", "america", "u.s.", "u.s.a."],
        }
        
        # Start with case variations
        location_variants = [
            schema.location,
            schema.location.title(),
            schema.location.lower(),
            schema.location.upper()
        ]
        
        # Add common location format variations based on actual data patterns
        # Examples: "London" -> ["London", "London, UK", "London, England", "London, United Kingdom"]
        # This handles formats found in the database: "London, UK", "London, England", etc.
        common_suffixes = {
            "london": ["London, UK", "London, England", "London, United Kingdom", "London, UK, United Kingdom, London"],
            "new york": ["New York, NY", "New York, New York", "New York, United States"],
            "san francisco": ["San Francisco, CA", "San Francisco, California", "San Francisco, United States"],
            "berlin": ["Berlin, Germany", "Berlin, DE"],
            "paris": ["Paris, France"],
            "tokyo": ["Tokyo, Japan"],
            "sydney": ["Sydney, Australia"],
        }
        
        # Check if location matches any common city pattern
        location_key = location_lower
        if location_key in common_suffixes:
            location_variants.extend(common_suffixes[location_key])
        
        # Add synonyms-based variations
        for key, synonyms in LOCATION_SYNONYMS.items():
            if key in location_lower:
                # If user searches for "UK", also search for "United Kingdom", "London", etc.
                location_variants.extend([s.title() for s in synonyms])
                # Add combinations: "London, UK", "London, United Kingdom"
                for syn in synonyms:
                    if syn != location_lower:  # Avoid duplicates
                        location_variants.append(f"{schema.location.title()}, {syn.title()}")
            elif any(syn in location_lower for syn in synonyms):
                # If user searches for "London" and it's in UK synonyms, add UK variants
                location_variants.append(key.title())
                location_variants.append(f"{schema.location.title()}, {key.title()}")
        
        # Remove duplicates while preserving order
        location_variants = list(dict.fromkeys(location_variants))
        
        logger.debug(
            "location_filter_variants_generated",
            original_location=schema.location,
            variants_count=len(location_variants),
            variants=location_variants[:10]  # Log first 10
        )
        
        filter_dict["location"] = {"$in": location_variants}
    
    if schema.industry:
        # Generate case variations for industry (same approach as location)
        industry_variants = list(set([
            schema.industry,
            schema.industry.title(),
            schema.industry.lower(),
            schema.industry.upper()
        ]))
        
        # Handle comma-separated industries (e.g., "AI, Enterprise")
        if "," in schema.industry:
            # Split by comma and add each industry separately
            industries = [ind.strip() for ind in schema.industry.split(",")]
            all_variants = []
            for ind in industries:
                all_variants.extend([
                    ind,
                    ind.title(),
                    ind.lower(),
                    ind.upper()
                ])
            industry_variants = list(set(all_variants))
        
        filter_dict["industry"] = {"$in": industry_variants}
        
        logger.debug(
            "industry_filter_variants_generated",
            original_industry=schema.industry,
            variants_count=len(industry_variants),
            variants=industry_variants[:10]
        )
    
    if schema.salary_min is not None:
        filter_dict["min_salary"] = {"$gte": schema.salary_min}
    
    if schema.employee_count:
        filter_dict["employee_count"] = {"$in": schema.employee_count}
    
    return filter_dict if filter_dict else None


async def _get_query_embedding_async(
    query_text: str,
    embedding_service_url: str,
    max_retries: int = EMBEDDING_MAX_RETRIES
) -> List[float]:
    """
    Async function for getting query embedding with retry logic.
    This is used by the tool which must handle async calls in sync context.
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text is empty or None")
    
    # Increased timeout for slow embedding service (defaults to 60 seconds)
    timeout = httpx.Timeout(EMBEDDING_REQUEST_TIMEOUT, connect=EMBEDDING_CONNECT_TIMEOUT)
    
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "requesting_embedding",
                query_text=query_text[:100],
                service_url=embedding_service_url,
                attempt=attempt,
                max_retries=max_retries
            )
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{embedding_service_url}/embed",
                    json={"texts": [query_text]}
                )
                response.raise_for_status()
                result = response.json()
                
                if "embeddings" not in result or not result["embeddings"]:
                    error_msg = f"Invalid response from embedding service: missing 'embeddings' key"
                    logger.error("embedding_service_invalid_response", response_keys=list(result.keys()) if isinstance(result, dict) else "not_a_dict")
                    raise ValueError(error_msg)
                
                embedding = result["embeddings"][0]
                if not embedding or not isinstance(embedding, list):
                    error_msg = f"Invalid embedding format: expected list, got {type(embedding)}"
                    logger.error("embedding_service_invalid_format", embedding_type=type(embedding).__name__)
                    raise ValueError(error_msg)
                
                logger.info("embedding_received", dim=len(embedding), attempt=attempt)
                return embedding
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff: 2, 4, 8 seconds
                logger.warning(
                    "embedding_service_timeout_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e),
                    error_type=type(e).__name__
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "embedding_service_timeout_failed",
                    attempts=max_retries,
                    error=str(e),
                    error_type=type(e).__name__
                )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                wait_time = 1 * attempt  # Linear backoff for other errors: 1, 2, 3 seconds
                logger.warning(
                    "embedding_service_error_retry",
                    attempt=attempt,
                    max_retries=max_retries,
                    wait_time=wait_time,
                    error=str(e),
                    error_type=type(e).__name__
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(
                    "embedding_service_error_failed",
                    attempts=max_retries,
                    error=str(e),
                    error_type=type(e).__name__
                )
    
    # All retries failed
    error_msg = f"Embedding service error after {max_retries} attempts: {str(last_error)}"
    raise Exception(error_msg) from last_error


@tool
def search_vacancies_tool(
    query: str,
    category: Optional[str] = None,
    experience_level: Optional[str] = None,
    company_stage: Optional[List[str]] = None,
    remote_option: Optional[str] = None,
    location: Optional[str] = None,
    industry: Optional[str] = None,
    salary_min: Optional[int] = None,
    employee_count: Optional[List[str]] = None,
    top_k: int = 50
) -> Dict[str, Any]:
    """
    Search for job vacancies in the Pinecone vector database.
    
    This tool performs semantic vector search combined with metadata filtering.
    It generates an embedding for the query text and searches for similar vacancies.
    
    Args:
        query: Core role + top skills only. No conversational filler.
               Example: "Senior Python Backend Engineer"
        category: Job category. Valid values: "Engineering", "Product", "Design", 
                  "Data & Analytics", "Sales & Business Development", "Marketing", 
                  "Operations", "Finance", "Legal", "People & HR", "Other"
        experience_level: Experience level. Valid values: "Junior", "Mid", "Senior", 
                          "Lead", "Executive", "Unknown"
        company_stage: Company funding stage(s). Valid values: "Seed", "Series A", 
                       "Series B", "Series C", "Growth", "Unknown"
        remote_option: Remote work option. Valid values: "remote" (fully remote), 
                       "office" (on-site only), "hybrid" (hybrid work)
        location: Job location filter. Examples: "London", "San Francisco", "New York".
                  Case-insensitive matching is supported.
        industry: Industry sector filter. Supported values: 'AI', 'Bio + Health', 'Consumer', 
                  'Enterprise', 'Fintech', 'American Dynamism', 'Logistics', 'Marketing', 'Other'.
                  Multiple industries can be comma-separated (e.g., 'AI, Enterprise').
                  Case-insensitive matching is supported.
        salary_min: Minimum salary in USD (integer, e.g., 120000)
        employee_count: Employee count filter(s). Examples: ["1-10", "11-50", "51-200"]
        top_k: Number of results to return (default: 50, same as Manual Search to get more results before filtering)
    
    Returns:
        Dictionary with:
        - results: List of vacancy results, each containing:
          - id: Vector ID
          - metadata: Vacancy metadata (title, company_name, location, etc.)
          - score: Similarity score (0.0 to 1.0, higher is better)
        - count: Number of results found
        - query: The original query text
        - filters_applied: The filters that were applied (in Pinecone format)
    
    Example:
        >>> result = search_vacancies_tool(
        ...     query="Senior Python Backend Engineer",
        ...     query="Senior Python Backend Engineer",
        ...     category="Engineering",
        ...     experience_level="Senior",
        ...     remote_option="remote",
        ...     location="London",
        ...     industry="Bio + Health",
        ...     salary_min=120000
        ... )
        >>> print(f"Found {result['count']} vacancies")
    """
    embedding_service_url = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")
    
    # Build SearchSchema from arguments
    try:
        search_schema = SearchSchema(
            query=query,
            category=category,
            experience_level=experience_level,
            company_stage=company_stage,
            remote_option=remote_option,
            location=location,
            industry=industry,
            salary_min=salary_min,
            employee_count=employee_count,
        )
    except Exception as e:
        logger.error(
            "search_schema_validation_error",
            query=query,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "results": [],
            "count": 0,
            "query": query,
            "filters_applied": None,
            "error": f"Invalid search parameters: {str(e)}"
        }
    
    # Convert schema to Pinecone filter_dict
    filter_dict = _build_filter_dict(search_schema)
    
    # Log the generated filter_dict
    logger.info(
        "search_vacancies_tool_called",
        query=query,
        schema_category=category,
        schema_experience_level=experience_level,
        schema_company_stage=company_stage,
        schema_remote_option=remote_option,
        schema_location=location,
        schema_industry=industry,
        schema_salary_min=salary_min,
        schema_employee_count=employee_count,
        filter_dict=filter_dict,
        top_k=top_k,
    )
    
    try:
        # Generate embedding for query text
        # Tool must be synchronous, but embedding function is async
        # When called from LangGraph (which uses uvloop), we need to run in a separate thread
        def _run_in_thread():
            """Run async function in a new thread with its own event loop."""
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            try:
                return new_loop.run_until_complete(_get_query_embedding_async(query, embedding_service_url))
            finally:
                new_loop.close()
        
        # Always use ThreadPoolExecutor to avoid conflicts with uvloop or running event loops
        # Increased timeout to 120 seconds to account for retries and slow embedding service
        # Thread timeout must cover all retries + backoff.
        timeout_backoff = sum(
            EMBEDDING_TIMEOUT_BACKOFF_BASE ** attempt
            for attempt in range(1, EMBEDDING_MAX_RETRIES)
        )
        error_backoff = sum(
            EMBEDDING_ERROR_BACKOFF_STEP * attempt
            for attempt in range(1, EMBEDDING_MAX_RETRIES)
        )
        worst_case_backoff = max(timeout_backoff, error_backoff)
        embedding_thread_timeout = (
            EMBEDDING_REQUEST_TIMEOUT * EMBEDDING_MAX_RETRIES
            + worst_case_backoff
            + EMBEDDING_THREAD_TIMEOUT_BUFFER
        )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_run_in_thread)
            try:
                query_vector = future.result(timeout=embedding_thread_timeout)
            except concurrent.futures.TimeoutError:
                logger.error(
                    "embedding_generation_timeout",
                    query=query,
                    timeout=embedding_thread_timeout,
                    max_retries=EMBEDDING_MAX_RETRIES,
                )
                return {
                    "results": [],
                    "count": 0,
                    "query": query,
                    "filters_applied": filter_dict,
                    "error": f"Embedding generation timed out after {embedding_thread_timeout} seconds"
                }
        
        if not query_vector or not isinstance(query_vector, list) or len(query_vector) == 0:
            logger.error("invalid_embedding_generated", query=query)
            return {
                "results": [],
                "count": 0,
                "query": query,
                "filters_applied": filter_dict,
                "error": "Failed to generate embedding for query"
            }
        
        # Initialize VectorStore and perform search
        # VectorStore initialization and query are synchronous and block the event loop
        # Run in a separate thread to avoid blocking
        def _run_pinecone_query():
            """Run Pinecone query in a separate thread to avoid blocking event loop."""
            try:
                logger.info(
                    "pinecone_query_starting_in_thread",
                    top_k=top_k,
                    filter_dict=filter_dict,
                    namespace="vacancies",
                    vector_dim=len(query_vector) if query_vector else 0,
                )
                pc_client = VectorStore()
                results = pc_client.query(
                    query_vector=query_vector,
                    top_k=top_k,
                    filter_dict=filter_dict,
                    namespace="vacancies",
                    include_values=False,
                )
                logger.info(
                    "pinecone_query_completed_in_thread",
                    results_count=len(results),
                    top_scores=[r.get("score", 0.0) for r in results[:5]] if results else [],
                )
                return results
            except Exception as e:
                logger.error(
                    "pinecone_query_internal_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    error_traceback=str(e.__traceback__) if hasattr(e, '__traceback__') else None,
                )
                raise
        
        # Use ThreadPoolExecutor to run blocking Pinecone query
        # Increased timeout to 120 seconds to account for slow Pinecone queries
        logger.info(
            "starting_pinecone_query",
            query=query,
            filter_dict=filter_dict,
            top_k=top_k,
            filter_details={
                "category": filter_dict.get("category") if filter_dict else None,
                "experience_level": filter_dict.get("experience_level") if filter_dict else None,
                "company_stage": filter_dict.get("company_stage") if filter_dict else None,
                "remote_option": filter_dict.get("remote_option") if filter_dict else None,
                "location": filter_dict.get("location") if filter_dict else None,
                "industry": filter_dict.get("industry") if filter_dict else None,
                "min_salary": filter_dict.get("min_salary") if filter_dict else None,
                "employee_count": filter_dict.get("employee_count") if filter_dict else None,
            },
        )
        
        search_results = []
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(_run_pinecone_query)
                try:
                    logger.info("waiting_for_pinecone_query_result", timeout=120)
                    search_results = future.result(timeout=120)  # 120 second timeout for Pinecone query
                    logger.info(
                        "pinecone_query_result_received",
                        results_count=len(search_results) if search_results else 0,
                        search_results_type=type(search_results).__name__,
                        is_list=isinstance(search_results, list),
                        first_result_sample={
                            "id": search_results[0].get("id") if search_results and len(search_results) > 0 else None,
                            "score": search_results[0].get("score") if search_results and len(search_results) > 0 else None,
                            "metadata_keys": list(search_results[0].get("metadata", {}).keys()) if search_results and len(search_results) > 0 else None,
                            "metadata_sample": {
                                k: str(v)[:100] for k, v in (search_results[0].get("metadata", {}).items() if search_results and len(search_results) > 0 else {})
                            } if search_results and len(search_results) > 0 else None,
                        } if search_results and len(search_results) > 0 else None,
                    )
                except concurrent.futures.TimeoutError:
                    logger.error(
                        "pinecone_query_timeout",
                        query=query,
                        filter_dict=filter_dict,
                        timeout=120,
                    )
                    return {
                        "results": [],
                        "count": 0,
                        "query": query,
                        "filters_applied": filter_dict,
                        "error": "Pinecone query timed out after 120 seconds"
                    }
                except Exception as e:
                    logger.error(
                        "pinecone_query_exception",
                        error=str(e),
                        error_type=type(e).__name__,
                        query=query,
                        filter_dict=filter_dict,
                    )
                    return {
                        "results": [],
                        "count": 0,
                        "query": query,
                        "filters_applied": filter_dict,
                        "error": f"Pinecone query failed: {str(e)}"
                    }
        except Exception as e:
            logger.error(
                "pinecone_query_threadpool_exception",
                error=str(e),
                error_type=type(e).__name__,
                query=query,
                filter_dict=filter_dict,
            )
            return {
                "results": [],
                "count": 0,
                "query": query,
                "filters_applied": filter_dict,
                "error": f"Pinecone query threadpool failed: {str(e)}"
            }
        
        logger.info(
            "search_vacancies_tool_completed",
            query=query,
            filter_dict=filter_dict,
            results_count=len(search_results),
            top_scores=[r.get("score", 0.0) for r in search_results[:3]] if search_results else [],
            all_scores=[r.get("score", 0.0) for r in search_results] if search_results else [],
            min_score=min([r.get("score", 0.0) for r in search_results]) if search_results else 0.0,
            max_score=max([r.get("score", 0.0) for r in search_results]) if search_results else 0.0,
            results_sample=[
                {
                    "id": r.get("id"),
                    "score": r.get("score"),
                    "title": r.get("metadata", {}).get("title"),
                    "company": r.get("metadata", {}).get("company_name"),
                    "location": r.get("metadata", {}).get("location"),
                    "industry": r.get("metadata", {}).get("industry"),
                    "company_stage": r.get("metadata", {}).get("company_stage"),
                }
                for r in search_results[:5]
            ] if search_results else [],
        )
        
        return {
            "results": search_results,
            "count": len(search_results),
            "query": query,
            "filters_applied": filter_dict,
        }
        
    except Exception as e:
        logger.error(
            "search_vacancies_tool_error",
            query=query,
            filter_dict=filter_dict,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "results": [],
            "count": 0,
            "query": query,
            "filters_applied": filter_dict,
            "error": f"Search failed: {str(e)}"
        }
