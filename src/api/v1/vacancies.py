"""
Vacancy Search Service API endpoints.
Supports both mock data and Firecrawl-based real search.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import httpx
import structlog
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schemas.vacancy import VacancyFilter, Vacancy, CompanyStage
from shared.pinecone_client import VectorStore
from src.services.firecrawl_service import FirecrawlService
from src.services.exceptions import (
    FirecrawlAuthError,
    FirecrawlAPIError,
    FirecrawlRateLimitError,
    FirecrawlConnectionError,
)
from apps.orchestrator.chat_search import ChatSearchAgent
from apps.orchestrator.agents.matchmaker import MatchmakerAgent
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Configure structlog for structured JSON logging
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

# Create router
router = APIRouter(prefix="/api/v1/vacancies", tags=["vacancies"])


class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    message: str = Field(..., description="Natural language search query")
    history: Optional[List[dict]] = Field(default=[], description="Previous conversation messages")
    persona: Optional[dict] = Field(default=None, description="User persona/CV information for personalized search")


def get_mock_vacancies() -> list[Vacancy]:
    """
    Generate mock vacancies for testing UI flow.
    Returns realistic vacancies from logistics/AI startups.
    """
    return [
        Vacancy(
            title="Senior Backend Engineer",
            company_name="LogiTech AI",
            company_stage=CompanyStage.SERIES_A,
            location="San Francisco, CA",
            industry="Logistics",
            salary_range="$150k-$200k",
            description_url="https://logitech-ai.com/careers/backend-engineer",
            required_skills=["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
            remote_option=True,
        ),
        Vacancy(
            title="ML Engineer - Supply Chain Optimization",
            company_name="RouteOptima",
            company_stage=CompanyStage.SEED,
            location="New York, NY",
            industry="Logistics",
            salary_range="$130k-$170k",
            description_url="https://routeoptima.com/jobs/ml-engineer",
            required_skills=["Python", "TensorFlow", "PyTorch", "Kubernetes", "GCP"],
            remote_option=False,
        ),
        Vacancy(
            title="Full Stack Developer",
            company_name="AI Freight Solutions",
            company_stage=CompanyStage.GROWTH,
            location="Remote",
            industry="Logistics",
            salary_range="$140k-$180k",
            description_url="https://aifreight.com/careers/fullstack",
            required_skills=["TypeScript", "React", "Node.js", "MongoDB", "GraphQL"],
            remote_option=True,
        ),
    ]


def filter_vacancies(vacancies: list[Vacancy], filter_params: VacancyFilter) -> list[Vacancy]:
    """
    Filter vacancies based on filter parameters.
    Uses case-insensitive substring matching for industry and location.
    Supports location synonyms and flexible skill matching for persona-based searches.

    Args:
        vacancies: List of vacancies to filter
        filter_params: Filter criteria

    Returns:
        Filtered list of vacancies
    """
    filtered = vacancies

    # Location synonym mapping for common variations
    LOCATION_SYNONYMS = {
        "us": ["united states", "usa", "america", "u.s.", "u.s.a."],
        "uk": ["united kingdom", "london", "england", "britain", "great britain"],
        "united states": ["us", "usa", "america", "u.s.", "u.s.a."],
        "united kingdom": ["uk", "london", "england", "britain", "great britain"],
        "usa": ["us", "united states", "america", "u.s.", "u.s.a."],
    }

    if filter_params.role:
        # Smart Keyword Matcher for role filtering
        role_lower = filter_params.role.lower().strip()
        
        # Word stem mapping for common variations (engineering -> engineer, etc.)
        word_stem_map = {
            "engineering": "engineer",
            "engineer": "engineer",
            "developing": "developer",
            "developer": "developer",
            "development": "developer",
            "architecting": "architect",
            "architect": "architect",
            "scientist": "scientist",
            "science": "scientist",
            "analyst": "analyst",
            "analysis": "analyst",
            "manager": "manager",
            "management": "manager",
            "director": "director",
            "directing": "director",
            "specialist": "specialist",
            "consultant": "consultant",
            "designer": "designer",
            "designing": "designer",
            "programmer": "programmer",
            "programming": "programmer"
        }
        
        # Role modifiers that MUST be present (frontend, backend, mobile, ai, etc.)
        role_modifiers = ["frontend", "front-end", "backend", "back-end", "fullstack", "full-stack", 
                         "full stack", "mobile", "web", "ai", "ml", "machine learning", "data", 
                         "devops", "sre", "qa", "test", "testing", "security", "cloud", "embedded",
                         "ios", "android", "react", "vue", "angular", "node", "python", "java", "go"]
        
        # Extract role words and normalize stems
        role_words = [w for w in role_lower.split() if len(w) > 2]
        normalized_words = []
        modifier_keywords = []
        
        for word in role_words:
            # Check if it's a modifier
            if any(mod in word for mod in role_modifiers):
                modifier_keywords.append(word)
            # Normalize word stems
            normalized = word_stem_map.get(word, word)
            normalized_words.append(normalized)
        
        # Build match criteria
        def matches_role(vacancy):
            title_lower = vacancy.title.lower()
            skills_lower = " ".join([s.lower() for s in vacancy.required_skills])
            combined_text = f"{title_lower} {skills_lower}"
            
            # Check if any normalized word matches in title (prioritize title matches)
            # This handles "engineering" -> "engineer" matching
            title_match = any(norm_word in title_lower for norm_word in normalized_words)
            
            # If modifiers are present (e.g., "frontend", "backend", "mobile"), they MUST be in title or skills
            modifier_match = True
            if modifier_keywords:
                modifier_match = any(mod in combined_text for mod in modifier_keywords)
            
            # Prioritize title matches - if role words match in title, it's a strong match
            # This ensures "Backend Engineer" matches "Backend Software Engineer" in title
            return title_match and modifier_match
        
        # Filter with smart matching - prioritize title matches
        # Results are sorted by vector search, so title matches will naturally rank higher
        filtered = [v for v in filtered if matches_role(v)]

    if filter_params.skills:
        # Soft skill filtering: require at least ONE skill match instead of ALL skills
        # This prevents "0 results" issue and lets vector search + AI Matcher determine final relevance
        # We rely on Pinecone vector search for semantic matching and AI Matcher for final scoring
        skill_set = {skill.lower() for skill in filter_params.skills}
        
        # Check if vacancy has at least one matching skill (case-insensitive)
        # This is a soft filter - the vector search ranking and AI Matcher will determine true relevance
        filtered = [
            v for v in filtered 
            if any(skill.lower() in skill_set for skill in v.required_skills)
        ]

    if filter_params.location:
        location_lower = filter_params.location.lower().strip()
        # Check for location synonyms
        location_variants = [location_lower]
        for key, synonyms in LOCATION_SYNONYMS.items():
            if key in location_lower:
                location_variants.extend(synonyms)
            elif any(syn in location_lower for syn in synonyms):
                location_variants.append(key)
                location_variants.extend(synonyms)
        
        # Remove duplicates while preserving order
        location_variants = list(dict.fromkeys(location_variants))
        
        filtered = [
            v
            for v in filtered
            if any(variant in v.location.lower() for variant in location_variants)
            or (v.remote_option and "remote" in location_lower)
        ]

    if filter_params.is_remote is not None:
        filtered = [v for v in filtered if v.remote_option == filter_params.is_remote]

    if filter_params.company_stages:
        # Use robust enum comparison to handle both Enum objects and strings
        filter_vals = [CompanyStage.get_stage_value(s) for s in filter_params.company_stages]
        filtered = [
            v for v in filtered if CompanyStage.get_stage_value(v.company_stage) in filter_vals
        ]

    if filter_params.industry:
        # Case-insensitive substring matching
        industry_lower = filter_params.industry.lower()
        filtered = [v for v in filtered if industry_lower in v.industry.lower()]

    if filter_params.min_salary:
        # Simple heuristic: extract min from salary_range if available
        # For mock data, we'll just return all if min_salary is set
        # In production, this would parse salary_range properly
        pass

    return filtered


def get_firecrawl_service() -> FirecrawlService:
    """
    Get or create Firecrawl service instance.

    Returns:
        FirecrawlService instance

    Raises:
        ImportError: If firecrawl-py package is not installed
        FirecrawlAuthError: If API key is missing or invalid
        FirecrawlConnectionError: If connection fails
        Exception: Other initialization errors
    """
    # Let exceptions propagate up - don't catch them here
    # The caller (search_vacancies or health_check) will handle them appropriately
    return FirecrawlService()


async def expand_query_with_keywords(role: str, chat_agent) -> str:
    """
    Expand a role query with 3-4 related technical keywords to improve vector similarity.
    Uses LLM to generate relevant keywords for the role.
    
    Args:
        role: Job role/title to expand
        chat_agent: ChatSearchAgent instance for LLM calls
        
    Returns:
        Expanded query string with role and related keywords
    """
    try:
        from langchain_core.messages import HumanMessage
        
        expansion_prompt = f"""Given the job role "{role}", provide 3-4 related technical keywords or skills that are commonly associated with this role.

Return ONLY a comma-separated list of keywords (no explanations, no JSON, just keywords).
Example: "Python, API development, microservices, cloud infrastructure"

Role: {role}
Keywords:"""
        
        messages = [HumanMessage(content=expansion_prompt)]
        response = await chat_agent.invoke(messages, max_tokens=100)
        keywords = response.content.strip()
        
        # Clean up the response (remove any extra text, keep only keywords)
        keywords = keywords.split('\n')[0].strip()  # Take first line only
        keywords = keywords.replace('Keywords:', '').replace('keywords:', '').strip()
        
        # Combine role with keywords
        expanded_query = f"Role: {role}. Related keywords: {keywords}"
        logger.info("query_expanded_with_keywords", role=role, keywords=keywords)
        return expanded_query
        
    except Exception as e:
        logger.warning("query_expansion_failed", role=role, error=str(e))
        # Fallback to just the role
        return f"Role: {role}"


def build_search_query(filter_params: VacancyFilter, persona: Optional[dict] = None, search_mode: str = "explicit") -> str:
    """
    Build search query text from filter parameters for embedding generation.
    Prioritizes user-requested roles over CV persona skills to ensure relevance.
    
    Query Format: "Role: [Explicit Role] Skills: [Extracted Skills]"
    - If user explicitly types a role (e.g., "Backend Engineer"), it MUST be the primary component
    - Role is used as the main text for embedding to prevent CV skills from outranking explicit roles
    
    Merge Strategy:
    - If search_mode == "explicit": Use ONLY parameters explicitly extracted from user's message.
      DO NOT fill empty 'skills' or 'role' fields from CV Persona. This allows users to search
      for jobs outside their current stack.
    - If search_mode == "persona": Use CV Persona as the primary source for 'role' and 'skills'
      if the user's query is generic (e.g., "find jobs"). Extract only top 5 key skills, not entire CV text.
    
    Args:
        filter_params: VacancyFilter with search criteria
        persona: Optional persona dictionary with CV/profile information
        search_mode: Either "explicit" (use only user input) or "persona" (use persona as base)
        
    Returns:
        Search query text string optimized for Pinecone vector search with role as primary component
    """
    role_part = None
    skills_part = None
    
    # EXPLICIT MODE: Use ONLY what user explicitly specified
    if search_mode == "explicit":
        # Role MUST be primary component if explicitly provided
        if filter_params.role:
            role_part = filter_params.role
        
        # Skills if explicitly provided
        if filter_params.skills:
            skills_str = ", ".join(filter_params.skills)
            skills_part = skills_str
        
        # DO NOT fill from persona in explicit mode - allows searching outside current stack
    
    # PERSONA MODE: Use persona as primary source, fill missing fields
    elif search_mode == "persona" and persona:
        # Prioritize explicit role from user - if provided, it's the primary component
        if filter_params.role:
            role_part = filter_params.role
        elif persona.get("career_goals"):
            goals = persona.get("career_goals", [])
            if isinstance(goals, list) and goals:
                role_part = goals[0]
            elif isinstance(goals, str):
                role_part = goals
        
        # Prioritize explicit skills from user, but fall back to persona if missing
        if filter_params.skills:
            skills_str = ", ".join(filter_params.skills)
            skills_part = skills_str
        elif persona.get("technical_skills"):
            skills = persona.get("technical_skills", [])
            if isinstance(skills, list) and skills:
                # Extract only top 5 key skills - do NOT use entire CV text
                skills_str = ", ".join(skills[:5])
                skills_part = skills_str
            elif isinstance(skills, str):
                skills_part = skills
    
    # Build weighted query string: Role is primary, Skills are secondary
    query_parts = []
    if role_part:
        # Role is the primary component - this ensures role matching takes precedence
        query_parts.append(f"Role: {role_part}")
    if skills_part:
        query_parts.append(f"Skills: {skills_part}")
    
    # Fallback: If no query parts in any mode, use generic query
    if not query_parts:
        query_parts.append("job vacancy")
    
    # Return weighted query with role as primary component
    # This format ensures the embedding prioritizes role matching over skill matching
    return ". ".join(query_parts) + "."


def build_pinecone_filter(filter_params: VacancyFilter) -> Optional[Dict[str, Any]]:
    """
    Build Pinecone filter dictionary from filter parameters.
    
    Args:
        filter_params: VacancyFilter with search criteria
        
    Returns:
        Pinecone filter dictionary or None
    """
    filter_dict = {}
    
    # Pinecone supports exact matches and contains for strings
    # For industry, we can use exact match if provided
    if filter_params.industry:
        # Use $in for partial matching (Pinecone doesn't support case-insensitive contains directly)
        # We'll do case-insensitive filtering in Python instead
        pass
    
    # For location, we can try exact match, but will also filter in Python
    if filter_params.location:
        # Similar to industry, we'll filter in Python for better matching
        pass
    
    # For remote_option, we can use exact boolean match
    if filter_params.is_remote is not None:
        filter_dict["remote_option"] = {"$eq": filter_params.is_remote}
    
    # For company_stage, we can use $in for multiple values
    # Normalize the strings to ensure they match enum values (e.g., 'SeriesA' -> 'Series A')
    if filter_params.company_stages:
        normalized_stages = [CompanyStage.get_stage_value(s) for s in filter_params.company_stages]
        filter_dict["company_stage"] = {"$in": normalized_stages}
    
    return filter_dict if filter_dict else None


async def get_query_embedding(query_text: str, embedding_service_url: str) -> List[float]:
    """
    Get embedding for search query from embedding-service.
    
    Args:
        query_text: Search query text
        embedding_service_url: URL of the embedding service
        
    Returns:
        Embedding vector as list of floats
        
    Raises:
        HTTPException: If embedding service is unavailable
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{embedding_service_url}/embed",
                json={"texts": [query_text]}
            )
            response.raise_for_status()
            result = response.json()
            
            if "embeddings" not in result or not result["embeddings"]:
                raise ValueError("Invalid response from embedding service")
            
            return result["embeddings"][0]
    except httpx.TimeoutException as e:
        logger.error("embedding_service_timeout", error=str(e))
        raise HTTPException(
            status_code=503, detail="Embedding service timeout"
        ) from e
    except httpx.HTTPStatusError as e:
        logger.error("embedding_service_http_error", status_code=e.response.status_code, error=str(e))
        raise HTTPException(
            status_code=503, detail=f"Embedding service error: {str(e)}"
        ) from e
    except httpx.RequestError as e:
        logger.error("embedding_service_unreachable", error=str(e))
        raise HTTPException(
            status_code=503, detail="Embedding service unavailable"
        ) from e


def metadata_to_vacancy(metadata: Dict[str, Any]) -> Vacancy:
    """
    Convert Pinecone metadata dictionary to Vacancy object.
    
    Args:
        metadata: Metadata dictionary from Pinecone
        
    Returns:
        Vacancy object
    """
    # Handle company_stage - it might be a string that needs normalization
    company_stage_str = metadata.get("company_stage", "Growth (Series B or later)")
    try:
        # Try to match to enum
        company_stage = CompanyStage(company_stage_str)
    except ValueError:
        # Use get_stage_value to normalize, then try again
        normalized_stage = CompanyStage.get_stage_value(company_stage_str)
        try:
            company_stage = CompanyStage(normalized_stage)
        except ValueError:
            # Default to GROWTH if no match
            company_stage = CompanyStage.GROWTH
            logger.warning("unknown_company_stage", stage=company_stage_str, defaulted_to="GROWTH")
    
    return Vacancy(
        title=metadata.get("title", "Unknown"),
        company_name=metadata.get("company_name", "Unknown"),
        company_stage=company_stage,
        location=metadata.get("location", "Not specified"),
        industry=metadata.get("industry", "Technology"),
        salary_range=metadata.get("salary_range"),
        description_url=metadata.get("description_url", ""),
        required_skills=metadata.get("required_skills", []),
        remote_option=metadata.get("remote_option", False),
        source_url=metadata.get("source_url"),
    )


@router.post("/search", response_model=list[Vacancy])
async def search_vacancies(
    filter_params: VacancyFilter,
    use_firecrawl: bool = Query(
        False, description="Use Firecrawl for real search instead of Pinecone"
    ),
    use_mock: bool = Query(
        False, description="Use mock data instead of Pinecone"
    ),
) -> list[Vacancy]:
    """
    Search for vacancies based on filter criteria using Pinecone vector search.

    Supports three modes:
    - Pinecone mode (default): Fast vector search in pre-indexed Pinecone database
    - Mock mode: Returns realistic mock vacancies for testing
    - Firecrawl mode: Fetches real vacancies from a16z jobs page using Firecrawl

    Args:
        filter_params: VacancyFilter with search criteria
        use_firecrawl: If True, use Firecrawl for real search (requires FIRECRAWL_API_KEY)
        use_mock: If True, use mock data instead of Pinecone

    Returns:
        List of Vacancy objects matching the filter criteria

    Raises:
        HTTPException: If search fails
    """
    try:
        # Normalize company_stages early using CompanyStage.get_stage_value
        # This ensures 'SeriesA' becomes 'Series A', etc.
        normalized_company_stages = None
        if filter_params.company_stages:
            normalized_company_stages = [
                CompanyStage.get_stage_value(stage) for stage in filter_params.company_stages
            ]
            # Update filter_params with normalized stages for consistent use throughout
            filter_params.company_stages = normalized_company_stages
        
        # Log search request (never log API keys or sensitive data)
        logger.info(
            "vacancy_search_requested",
            role=filter_params.role,
            skills=filter_params.skills,
            location=filter_params.location,
            is_remote=filter_params.is_remote,
            company_stages=normalized_company_stages,
            industry=filter_params.industry,
            min_salary=filter_params.min_salary,
            use_firecrawl=use_firecrawl,
            use_mock=use_mock,
        )

        if use_mock:
            # Use mock data
            all_vacancies = get_mock_vacancies()
            filtered_vacancies = filter_vacancies(all_vacancies, filter_params)

            logger.info(
                "vacancy_search_completed", total_results=len(filtered_vacancies), source="mock"
            )

            return filtered_vacancies

        if use_firecrawl:
            # Use Firecrawl for real search - DO NOT fall back to mock on failure
            try:
                firecrawl_service = get_firecrawl_service()
                vacancies = firecrawl_service.fetch_vacancies(filter_params, max_results=100)

                logger.info(
                    "vacancy_search_completed", total_results=len(vacancies), source="firecrawl"
                )

                return vacancies
            except ImportError as e:
                # Firecrawl package not installed - raise error, don't fall back to mock
                error_msg = "Firecrawl package (firecrawl-py) is not installed. Please install it in requirements/api.txt."
                logger.error("firecrawl_import_error", error=str(e), error_type=type(e).__name__)
                raise HTTPException(status_code=503, detail=error_msg) from e
            except FirecrawlAuthError as e:
                error_msg = f"Firecrawl authentication failed: {str(e)}"
                logger.error("firecrawl_auth_error", error=str(e))
                raise HTTPException(status_code=401, detail=error_msg) from e
            except FirecrawlRateLimitError as e:
                error_msg = f"Firecrawl rate limit exceeded: {str(e)}"
                logger.error("firecrawl_rate_limit", error=str(e))
                raise HTTPException(status_code=429, detail=error_msg) from e
            except FirecrawlConnectionError as e:
                error_msg = f"Firecrawl service unavailable: {str(e)}"
                logger.error("firecrawl_connection_error", error=str(e))
                raise HTTPException(status_code=503, detail=error_msg) from e
            except FirecrawlAPIError as e:
                error_msg = f"Firecrawl API error: {str(e)}"
                logger.error("firecrawl_api_error", error=str(e))
                raise HTTPException(status_code=500, detail=error_msg) from e
            except Exception as e:
                # Catch any other unexpected errors - don't fall back to mock
                error_msg = f"Unexpected Firecrawl error: {str(e)}"
                logger.error(
                    "firecrawl_unexpected_error", error=str(e), error_type=type(e).__name__
                )
                raise HTTPException(status_code=500, detail=error_msg) from e

        # Default: Use Pinecone for fast vector search
        try:
            # Get embedding service URL
            embedding_service_url = os.getenv(
                "EMBEDDING_SERVICE_URL",
                "http://embedding-service:8001"
            )
            
            # Build search query from filter parameters
            search_query = build_search_query(filter_params)
            logger.info("search_query_built", query=search_query)
            
            # Get embedding for search query
            query_embedding = await get_query_embedding(search_query, embedding_service_url)
            logger.info("query_embedding_generated", dim=len(query_embedding))
            
            # Build Pinecone filter
            pinecone_filter = build_pinecone_filter(filter_params)
            logger.info("pinecone_filter_built", filter=pinecone_filter)
            
            # Initialize Pinecone client
            vector_store = VectorStore()
            
            # Query Pinecone
            top_k = 50  # Get more results, then filter in Python
            results = vector_store.query(
                query_vector=query_embedding,
                top_k=top_k,
                filter_dict=pinecone_filter,
                namespace="vacancies",
                include_values=False
            )
            
            logger.info("pinecone_query_completed", results_count=len(results))
            
            # Convert metadata to Vacancy objects
            vacancies = []
            for result in results:
                try:
                    vacancy = metadata_to_vacancy(result["metadata"])
                    vacancies.append(vacancy)
                except Exception as e:
                    logger.warning("vacancy_conversion_failed", error=str(e), metadata=result.get("metadata", {}))
                    continue
            
            # Apply additional filters in Python (for industry, location with case-insensitive matching)
            filtered_vacancies = filter_vacancies(vacancies, filter_params)
            
            logger.info(
                "vacancy_search_completed",
                total_results=len(filtered_vacancies),
                source="pinecone",
                initial_results=len(vacancies)
            )
            
            return filtered_vacancies
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error("pinecone_search_error", error=str(e), error_type=type(e).__name__)
            raise HTTPException(
                status_code=500, detail=f"Error performing Pinecone search: {str(e)}"
            ) from e

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("vacancy_search_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=500, detail=f"Error performing vacancy search: {str(e)}"
        ) from e


@router.post("/chat")
async def chat_search(request: ChatRequest) -> Dict[str, Any]:
    """
    Conversational vacancy search endpoint.
    
    Accepts natural language messages and converts them to search queries,
    then returns matching vacancies with an AI-generated summary.
    
    Args:
        request: ChatRequest with user's natural language message
        
    Returns:
        Dictionary with:
        - vacancies: List of Vacancy objects
        - summary: AI-generated summary explaining the results
        
    Raises:
        HTTPException: If search fails
    """
    try:
        logger.info("chat_search_requested", message_length=len(request.message))
        
        # Initialize chat search agent
        chat_agent = ChatSearchAgent()
        
        # Interpret user message to extract search parameters
        # Pass history and persona for context-aware search
        extracted_params = await chat_agent.interpret_message(
            request.message,
            history=request.history or [],
            persona=request.persona
        )
        
        # Extract search_mode from interpreted params
        # The Job Scout agent has already applied the merge strategy:
        # - "explicit": Only user-provided fields (no persona filling)
        # - "persona": Persona used as default base for missing role/skills
        search_mode = extracted_params.get("search_mode", "explicit")
        
        logger.info(
            "chat_params_extracted",
            role=extracted_params.get("role"),
            skills=extracted_params.get("skills"),
            industry=extracted_params.get("industry"),
            location=extracted_params.get("location"),
            company_stage=extracted_params.get("company_stage"),
            search_mode=search_mode,
        )
        
        # Convert extracted parameters to VacancyFilter
        # Note: The merge strategy has already been applied by the Job Scout agent
        # based on search_mode, so filter_params contains the final merged values
        filter_params = VacancyFilter(
            role=extracted_params.get("role"),
            skills=extracted_params.get("skills") or [],
            industry=extracted_params.get("industry"),
            location=extracted_params.get("location"),
            company_stages=[extracted_params.get("company_stage")] if extracted_params.get("company_stage") else None,
        )
        
        # Normalize company_stages if present
        if filter_params.company_stages:
            normalized_company_stages = [
                CompanyStage.get_stage_value(stage) for stage in filter_params.company_stages
            ]
            filter_params.company_stages = normalized_company_stages
        
        # Get embedding service URL
        embedding_service_url = os.getenv(
            "EMBEDDING_SERVICE_URL",
            "http://embedding-service:8001"
        )
        
        # Build search query from filter parameters, passing persona and search_mode
        # build_search_query respects search_mode to determine merge strategy:
        # - "explicit": Uses only filter_params (no persona filling)
        # - "persona": Uses persona to fill missing role/skills in filter_params
        # The query is optimized for Pinecone vector search using role and skills
        base_query = build_search_query(filter_params, persona=request.persona, search_mode=search_mode)
        
        # For explicit mode with a role, expand query with related technical keywords
        if search_mode == "explicit" and filter_params.role:
            try:
                search_query = await expand_query_with_keywords(filter_params.role, chat_agent)
                logger.info(
                    "chat_search_query_expanded", 
                    base_query=base_query,
                    expanded_query=search_query,
                    search_mode=search_mode
                )
            except Exception as e:
                logger.warning("query_expansion_failed_fallback", error=str(e))
                search_query = base_query
        else:
            search_query = base_query
        
        logger.info(
            "chat_search_query_built", 
            query=search_query, 
            search_mode=search_mode,
            has_persona=request.persona is not None
        )
        
        # Get embedding for search query
        query_embedding = await get_query_embedding(search_query, embedding_service_url)
        logger.info("chat_query_embedding_generated", dim=len(query_embedding))
        
        # Build Pinecone filter
        pinecone_filter = build_pinecone_filter(filter_params)
        logger.info("chat_pinecone_filter_built", filter=pinecone_filter)
        
        # Initialize Pinecone client
        vector_store = VectorStore()
        
        # Query Pinecone
        top_k = 50  # Get more results, then filter in Python
        results = vector_store.query(
            query_vector=query_embedding,
            top_k=top_k,
            filter_dict=pinecone_filter,
            namespace="vacancies",
            include_values=False
        )
        
        logger.info("chat_pinecone_query_completed", results_count=len(results))
        
        # Convert metadata to Vacancy objects and preserve Pinecone similarity scores
        # Store (vacancy, score) tuples to maintain the relationship
        vacancy_score_pairs = []
        for result in results:
            try:
                vacancy = metadata_to_vacancy(result["metadata"])
                # Preserve Pinecone similarity score (0.0 to 1.0)
                pinecone_score = result.get("score", 0.0)
                vacancy_score_pairs.append((vacancy, pinecone_score))
            except Exception as e:
                logger.warning("chat_vacancy_conversion_failed", error=str(e), metadata=result.get("metadata", {}))
                continue
        
        # Extract just vacancies for filtering
        vacancies = [pair[0] for pair in vacancy_score_pairs]
        
        # Apply soft filters in Python (case-insensitive matching, at least one skill match)
        # Note: Skill filtering requires at least ONE match (not all), and role uses partial match
        # This prevents "0 results" - vector search ranking and AI Matcher determine final relevance
        filtered_vacancies = filter_vacancies(vacancies, filter_params)

        logger.info(
            "chat_vacancy_search_completed",
            total_results=len(filtered_vacancies),
            initial_results=len(vacancies)
        )

        # Convert all vacancies to dicts for response and preserve Pinecone scores
        # Create a mapping from vacancy to its Pinecone score using title+company as key
        vacancy_to_score = {}
        for vacancy, score in vacancy_score_pairs:
            key = f"{vacancy.title}_{vacancy.company_name}"
            vacancy_to_score[key] = score
        
        vacancies_response = []
        for vacancy in filtered_vacancies:
            vacancy_dict = vacancy.model_dump() if hasattr(vacancy, 'model_dump') else vacancy.dict()
            # Add Pinecone similarity score if available
            key = f"{vacancy.title}_{vacancy.company_name}"
            if key in vacancy_to_score:
                vacancy_dict['pinecone_score'] = vacancy_to_score[key]
            else:
                vacancy_dict['pinecone_score'] = None
            vacancies_response.append(vacancy_dict)

        # Use MatchmakerAgent to analyze top vacancies if persona is provided
        # Analyze more candidates (up to 20) from the filtered results, then sort by AI score
        # This ensures we get the best matches based on AI analysis, not just vector similarity
        top_vacancies_for_matching = vacancies_response[:20] if len(vacancies_response) > 0 else []
        
        if request.persona and top_vacancies_for_matching:
            logger.info("matchmaker_analysis_started", vacancy_count=len(top_vacancies_for_matching))
            matchmaker = MatchmakerAgent()
            
            # Analyze each vacancy with matchmaker (with fail-safe error handling)
            successful_analyses = 0
            for vacancy_dict in top_vacancies_for_matching:
                try:
                    # Get vacancy text from the vacancy dict
                    vacancy_text = f"""
Title: {vacancy_dict.get('title', 'Unknown')}
Company: {vacancy_dict.get('company_name', 'Unknown')}
Location: {vacancy_dict.get('location', 'Not specified')}
Industry: {vacancy_dict.get('industry', 'Not specified')}
Company Stage: {vacancy_dict.get('company_stage', 'Not specified')}
Required Skills: {', '.join(vacancy_dict.get('required_skills', [])) if vacancy_dict.get('required_skills') else 'Not specified'}
Remote Option: {vacancy_dict.get('remote_option', False)}
Salary Range: {vacancy_dict.get('salary_range') or 'Not specified'}
Description URL: {vacancy_dict.get('description_url', 'N/A')}
"""
                    
                    # Get Pinecone score for this vacancy (if available)
                    pinecone_score = vacancy_dict.get('pinecone_score')
                    
                    # Analyze match with matchmaker agent (with timeout protection)
                    # The analyze_match method now returns a dict with 'score' and 'reasoning'
                    match_result = await matchmaker.analyze_match(
                        vacancy_text=vacancy_text,
                        candidate_persona=request.persona,
                        similarity_score=pinecone_score  # Pass Pinecone score for context
                    )
                    
                    # Handle new return format: dict with 'score' and 'reasoning'
                    if isinstance(match_result, dict):
                        ai_score = match_result.get('score')
                        match_reasoning = match_result.get('reasoning', '')
                        
                        # Add AI match score and reasoning to vacancy dict
                        if match_reasoning and len(match_reasoning) > 0:
                            vacancy_dict['match_reasoning'] = match_reasoning
                            vacancy_dict['ai_match_score'] = ai_score  # 0-10 scale
                            successful_analyses += 1
                        else:
                            logger.warning(
                                "matchmaker_analysis_empty_response",
                                vacancy_title=vacancy_dict.get('title', 'Unknown')
                            )
                            vacancy_dict['ai_match_score'] = None
                    else:
                        # Fallback for old format (shouldn't happen, but handle gracefully)
                        logger.warning("matchmaker_unexpected_format", vacancy_title=vacancy_dict.get('title', 'Unknown'))
                        if match_result and len(str(match_result)) > 0:
                            vacancy_dict['match_reasoning'] = str(match_result)
                            vacancy_dict['ai_match_score'] = None
                    
                except Exception as e:
                    logger.warning(
                        "matchmaker_analysis_failed",
                        vacancy_title=vacancy_dict.get('title', 'Unknown'),
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    # Set fallback values instead of leaving them empty
                    vacancy_dict['match_reasoning'] = "Match analysis temporarily unavailable."
                    vacancy_dict['ai_match_score'] = None
                    # Continue with other vacancies even if one fails
                    continue
            
            logger.info("matchmaker_analysis_completed", successful=successful_analyses, total=len(top_vacancies_for_matching))
            
            # Result Thresholding: Filter out weak matches (AI score < 5)
            # Do not show roles that the AI Matcher explicitly identifies as weak matches (score 2-4)
            before_threshold = len(vacancies_response)
            vacancies_response = [
                v for v in vacancies_response 
                if v.get('ai_match_score') is None or v.get('ai_match_score') >= 5
            ]
            after_threshold = len(vacancies_response)
            logger.info(
                "vacancies_filtered_by_threshold",
                before_count=before_threshold,
                after_count=after_threshold,
                filtered_out=before_threshold - after_threshold
            )
            
            # Sort vacancies by AI match score (descending) to prioritize best matches
            # Vacancies with AI scores are ranked higher than those without
            def sort_key(v):
                ai_score = v.get('ai_match_score')
                if ai_score is not None:
                    return (1, -ai_score)  # Has AI score, sort by score descending
                else:
                    return (0, 0)  # No AI score, sort to end
            
            vacancies_response.sort(key=sort_key, reverse=True)
            
            # Keep only top 5 after sorting by AI score
            vacancies_response = vacancies_response[:5]
            logger.info("vacancies_sorted_by_ai_score", top_5_count=len(vacancies_response))
        
        # Generate AI summary of results using the top vacancies
        # Convert dicts back to Vacancy objects for the summary function
        top_vacancies_for_summary = []
        for v_dict in vacancies_response[:5]:
            try:
                # Reconstruct Vacancy object from dict
                vacancy_obj = Vacancy(
                    title=v_dict.get('title', 'Unknown'),
                    company_name=v_dict.get('company_name', 'Unknown'),
                    company_stage=CompanyStage.get_stage_value(v_dict.get('company_stage', 'Growth (Series B or later)')),
                    location=v_dict.get('location', 'Not specified'),
                    industry=v_dict.get('industry', 'Technology'),
                    salary_range=v_dict.get('salary_range'),
                    description_url=v_dict.get('description_url', ''),
                    required_skills=v_dict.get('required_skills', []),
                    remote_option=v_dict.get('remote_option', False),
                )
                top_vacancies_for_summary.append(vacancy_obj)
            except Exception as e:
                logger.warning("vacancy_reconstruction_failed", error=str(e), vacancy_dict=v_dict)
                continue
        
        summary = await chat_agent.format_results_summary(top_vacancies_for_summary, request.message)

        logger.info("chat_search_completed", summary_length=len(summary))

        return {
            "vacancies": vacancies_response,
            "summary": summary
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("chat_search_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=500, detail=f"Error performing chat search: {str(e)}"
        ) from e


@router.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint for the vacancy search service.

    Returns:
        Dictionary with status information including Firecrawl configuration
    """
    logger.info("vacancy_search_health_check_requested")

    # Log environment variables (masked) for deployment verification
    import os

    env_vars_to_check = [
        "FIRECRAWL_API_KEY",
        "PINECONE_API_KEY",
        "DEEPSEEK_API_KEY",
        "EMBEDDING_SERVICE_URL",
        "CV_PROCESSOR_URL",
    ]

    env_status = {}
    for var_name in env_vars_to_check:
        value = os.getenv(var_name)
        # Check if value exists and is not empty string
        if value and value.strip():
            # Mask sensitive keys
            if "API_KEY" in var_name or "SECRET" in var_name:
                masked = f"{value[:4]}****" if len(value) > 4 else "****"
                env_status[var_name] = f"configured ({masked})"
            else:
                env_status[var_name] = f"configured ({value})"
        else:
            env_status[var_name] = "not set or empty"

    logger.info("environment_variables_status", **env_status)

    # Check if Firecrawl service can be initialized
    # Also check if FIRECRAWL_API_KEY is not empty string
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    firecrawl_configured = False
    firecrawl_error = None

    if not firecrawl_api_key or not firecrawl_api_key.strip():
        firecrawl_configured = False
        firecrawl_error = "FIRECRAWL_API_KEY not set or empty"
        logger.warning("firecrawl_api_key_empty")
    else:
        try:
            service = get_firecrawl_service()
            firecrawl_configured = True
            logger.info("firecrawl_service_available")
        except FirecrawlAuthError as e:
            firecrawl_configured = False
            firecrawl_error = "API key missing or invalid"
            logger.warning("firecrawl_auth_error", error=str(e))
        except ImportError as e:
            firecrawl_configured = False
            firecrawl_error = "firecrawl-py package not installed"
            logger.warning("firecrawl_import_error", error=str(e))
        except Exception as e:
            firecrawl_configured = False
            firecrawl_error = f"Initialization failed: {str(e)}"
            logger.error("firecrawl_init_error", error=str(e), error_type=type(e).__name__)

    response = {
        "status": "ok",
        "service": "vacancy-search",
        "version": "1.0.0",
        "firecrawl_configured": firecrawl_configured,
    }

    if firecrawl_error:
        response["firecrawl_error"] = firecrawl_error

    return response
