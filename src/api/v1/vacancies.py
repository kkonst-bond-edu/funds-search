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


class VacancySearchResponse(BaseModel):
    """Response schema for vacancy search with statistics."""
    vacancies: List[Vacancy] = Field(..., description="List of matching vacancies")
    total_in_db: Optional[int] = Field(None, description="Total vacancies in database (if available)")
    initial_vector_matches: int = Field(..., description="Number of results from initial vector search")
    total_after_filters: int = Field(..., description="Number of results after applying all filters")


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


def apply_soft_title_filter(vacancies: list[Vacancy], role_query: Optional[str]) -> list[Vacancy]:
    """
    Apply soft title filter to exclude vacancies with conflicting terms.
    
    If user explicitly asked for a role type (e.g., "Backend"), filter out
    vacancies with conflicting terms in the title (e.g., "Mobile", "Frontend")
    unless those terms are also specified in the query.
    
    Args:
        vacancies: List of vacancies to filter
        role_query: The expanded role query from Job Scout (may contain multiple keywords)
        
    Returns:
        Filtered list of vacancies
    """
    if not role_query:
        return vacancies
    
    role_lower = role_query.lower()
    
    # Define conflicting role categories
    role_conflicts = {
        "backend": ["frontend", "front-end", "mobile", "ios", "android", "react native"],
        "frontend": ["backend", "back-end", "mobile", "ios", "android", "react native"],
        "mobile": ["backend", "back-end", "frontend", "front-end", "web"],
        "ios": ["android", "backend", "back-end", "frontend", "front-end"],
        "android": ["ios", "backend", "back-end", "frontend", "front-end"],
    }
    
    # Detect which role category the user is searching for
    detected_categories = []
    for category, conflicts in role_conflicts.items():
        if category in role_lower:
            detected_categories.append(category)
    
    # If no specific category detected, don't filter
    if not detected_categories:
        return vacancies
    
    # Collect all conflicting terms for detected categories
    all_conflicts = set()
    for category in detected_categories:
        all_conflicts.update(role_conflicts[category])
    
    # Remove conflicts that are also mentioned in the query (user wants both)
    for conflict in list(all_conflicts):
        if conflict in role_lower:
            all_conflicts.remove(conflict)
    
    # Filter out vacancies with conflicting terms in title
    filtered = []
    for vacancy in vacancies:
        title_lower = vacancy.title.lower()
        has_conflict = any(conflict in title_lower for conflict in all_conflicts)
        
        if not has_conflict:
            filtered.append(vacancy)
        else:
            logger.debug(
                "soft_title_filter_excluded",
                vacancy_title=vacancy.title,
                conflict_terms=list(all_conflicts),
                role_query=role_query
            )
    
    return filtered


def apply_hard_keyword_filter(vacancies: list[Vacancy], required_keywords: Optional[List[str]]) -> list[Vacancy]:
    """
    Apply hard keyword filter using required_keywords.
    
    Vacancies MUST contain at least one of the required keywords in their
    title or required_skills to pass this filter.
    
    Args:
        vacancies: List of vacancies to filter
        required_keywords: List of critical keywords that must be present
        
    Returns:
        Filtered list of vacancies
    """
    if not required_keywords or len(required_keywords) == 0:
        return vacancies
    
    # Normalize keywords to lowercase for matching
    keywords_lower = [kw.lower() for kw in required_keywords if kw and len(kw.strip()) > 0]
    
    if not keywords_lower:
        return vacancies
    
    filtered = []
    for vacancy in vacancies:
        title_lower = vacancy.title.lower()
        skills_lower = [s.lower() for s in vacancy.required_skills]
        combined_text = f"{title_lower} {' '.join(skills_lower)}"
        
        # Check if at least one required keyword is present
        has_keyword = any(kw in combined_text for kw in keywords_lower)
        
        if has_keyword:
            filtered.append(vacancy)
        else:
            logger.debug(
                "hard_keyword_filter_excluded",
                vacancy_title=vacancy.title,
                required_keywords=required_keywords
            )
    
    return filtered


def prioritize_by_persona_preferences(
    vacancies: list[Vacancy], 
    persona: Optional[dict],
    search_mode: str
) -> list[Vacancy]:
    """
    Prioritize vacancies based on persona preferences when in persona mode.
    
    In persona mode, prioritize results that match:
    - preferred_startup_stage
    - industry_preferences
    
    Args:
        vacancies: List of vacancies to prioritize
        persona: User persona dictionary
        search_mode: Either "persona" or "explicit"
        
    Returns:
        Reordered list of vacancies with persona matches first
    """
    if search_mode != "persona" or not persona:
        return vacancies
    
    preferred_stage = persona.get("preferred_startup_stage")
    industry_preferences = persona.get("industry_preferences", [])
    
    if not preferred_stage and not industry_preferences:
        return vacancies
    
    # Separate vacancies into prioritized and non-prioritized
    prioritized = []
    others = []
    
    for vacancy in vacancies:
        is_prioritized = False
        
        # Check company stage match
        if preferred_stage:
            vacancy_stage = CompanyStage.get_stage_value(vacancy.company_stage)
            preferred_stage_normalized = CompanyStage.get_stage_value(preferred_stage)
            if vacancy_stage == preferred_stage_normalized:
                is_prioritized = True
        
        # Check industry match
        if industry_preferences:
            vacancy_industry_lower = vacancy.industry.lower()
            for pref in industry_preferences:
                if isinstance(pref, str) and pref.lower() in vacancy_industry_lower:
                    is_prioritized = True
                    break
        
        if is_prioritized:
            prioritized.append(vacancy)
        else:
            others.append(vacancy)
    
    # Return prioritized first, then others
    return prioritized + others


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
    Uses metadata filters for industry, location, and company_stage.
    
    Args:
        filter_params: VacancyFilter with search criteria
        
    Returns:
        Pinecone filter dictionary or None
    """
    filter_dict = {}
    
    # Metadata filtering: Use Pinecone metadata filters for industry
    # Pinecone supports exact matches - we'll use $eq for single value
    # Note: Pinecone metadata filters are case-sensitive, so we'll also do
    # case-insensitive filtering in Python, but this reduces the initial result set
    if filter_params.industry:
        # Use exact match - case-insensitive filtering happens in Python
        filter_dict["industry"] = {"$eq": filter_params.industry}
    
    # Metadata filtering: Use Pinecone metadata filters for location
    # Similar to industry, use exact match for initial filtering
    if filter_params.location:
        # Use exact match - case-insensitive filtering happens in Python
        filter_dict["location"] = {"$eq": filter_params.location}
    
    # For remote_option, we can use exact boolean match
    if filter_params.is_remote is not None:
        filter_dict["remote_option"] = {"$eq": filter_params.is_remote}
    
    # Metadata filtering: Use Pinecone metadata filters for company_stage
    # Use $in for multiple values
    # Normalize the strings to ensure they match enum values (e.g., 'SeriesA' -> 'Series A')
    if filter_params.company_stages:
        normalized_stages = [CompanyStage.get_stage_value(s) for s in filter_params.company_stages]
        filter_dict["company_stage"] = {"$in": normalized_stages}
    
    return filter_dict if filter_dict else None


def apply_polarity_filter(vacancies: list[Vacancy], role_query: Optional[str]) -> list[Vacancy]:
    """
    Apply strict polarity filter based on role domain keywords.
    
    If the user query contains domain-specific terms (e.g., "Frontend", "Backend", "Mobile"),
    vacancies that don't match this domain in the title MUST be excluded.
    
    Args:
        vacancies: List of vacancies to filter
        role_query: The role query from Job Scout (may contain domain keywords)
        
    Returns:
        Filtered list of vacancies
    """
    if not role_query:
        return vacancies
    
    role_lower = role_query.lower()
    
    # Define domain keywords and their conflicting domains
    domain_keywords = {
        "frontend": ["frontend", "front-end", "react", "vue", "angular", "ui", "user interface"],
        "backend": ["backend", "back-end", "api", "server", "microservices"],
        "mobile": ["mobile", "ios", "android", "react native", "swift", "kotlin"],
        "fullstack": ["fullstack", "full-stack", "full stack"],
        "devops": ["devops", "sre", "infrastructure", "kubernetes", "docker"],
        "data": ["data", "data science", "data scientist", "analytics", "ml", "machine learning"],
    }
    
    # Detect which domain keyword is present in the query
    detected_domains = []
    for domain, keywords in domain_keywords.items():
        if any(keyword in role_lower for keyword in keywords):
            detected_domains.append(domain)
    
    # If no specific domain detected, don't filter
    if not detected_domains:
        return vacancies
    
    # Collect all domain keywords for detected domains
    all_domain_keywords = set()
    for domain in detected_domains:
        all_domain_keywords.update(domain_keywords[domain])
    
    # Filter: Only keep vacancies that contain at least one domain keyword in the title
    filtered = []
    for vacancy in vacancies:
        title_lower = vacancy.title.lower()
        has_domain_match = any(keyword in title_lower for keyword in all_domain_keywords)
        
        if has_domain_match:
            filtered.append(vacancy)
        else:
            logger.debug(
                "polarity_filter_excluded",
                vacancy_title=vacancy.title,
                detected_domains=detected_domains,
                role_query=role_query
            )
    
    return filtered


def apply_keyword_match_filter(vacancies: list[Vacancy], required_keywords: Optional[List[str]]) -> list[Vacancy]:
    """
    Apply strict keyword match filter using required_keywords.
    
    Vacancies MUST contain at least one of the required keywords in their
    title, required_skills, or combined text to pass this filter.
    
    Args:
        vacancies: List of vacancies to filter
        required_keywords: List of critical keywords that must be present
        
    Returns:
        Filtered list of vacancies
    """
    if not required_keywords or len(required_keywords) == 0:
        return vacancies
    
    # Normalize keywords to lowercase for matching
    keywords_lower = [kw.lower() for kw in required_keywords if kw and len(kw.strip()) > 0]
    
    if not keywords_lower:
        return vacancies
    
    filtered = []
    for vacancy in vacancies:
        title_lower = vacancy.title.lower()
        skills_lower = [s.lower() for s in vacancy.required_skills]
        combined_text = f"{title_lower} {' '.join(skills_lower)}"
        
        # Check if at least one required keyword is present
        has_keyword = any(kw in combined_text for kw in keywords_lower)
        
        if has_keyword:
            filtered.append(vacancy)
        else:
            logger.debug(
                "keyword_match_filter_excluded",
                vacancy_title=vacancy.title,
                required_keywords=required_keywords
            )
    
    return filtered


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


def parse_min_salary_from_range(salary_range: Optional[str]) -> Optional[int]:
    """
    Parse minimum salary from salary range string.
    
    Handles formats like:
    - "$120k-$180k" -> 120000
    - "$150,000 - $200,000" -> 150000
    - "120k-180k" -> 120000
    
    Args:
        salary_range: Salary range string or None
        
    Returns:
        Minimum salary as integer (in dollars) or None if cannot parse
    """
    if not salary_range:
        return None
    
    try:
        # Remove currency symbols and spaces
        cleaned = salary_range.replace("$", "").replace(",", "").strip()
        
        # Try to extract first number (minimum)
        import re
        # Match patterns like "120k", "120000", "120-180k", etc.
        numbers = re.findall(r'(\d+)(?:k|K)?', cleaned)
        if numbers:
            min_salary_str = numbers[0]
            min_salary = int(min_salary_str)
            # If the number is less than 1000, assume it's in thousands (e.g., "120" in "120k")
            if min_salary < 1000 and ('k' in cleaned.lower() or 'K' in cleaned):
                min_salary *= 1000
            return min_salary
    except (ValueError, AttributeError):
        pass
    
    return None


def metadata_to_vacancy(metadata: Dict[str, Any]) -> Vacancy:
    """
    Convert Pinecone metadata dictionary to Vacancy object.
    Ensures all fields including industry and min_salary are properly extracted.
    
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
    
    # Extract industry - ensure it's properly extracted from Pinecone metadata
    # This is a critical field for filtering and must be present
    industry = metadata.get("industry")
    if not industry or industry == "" or not isinstance(industry, str):
        industry = "Technology"  # Default fallback
        logger.debug("industry_missing_in_metadata", defaulted_to="Technology")
    
    # Extract salary_range from Pinecone metadata
    salary_range = metadata.get("salary_range")
    
    # Parse min_salary from salary_range if available
    # This is extracted for frontend display (Minimum Salary field)
    min_salary = None
    if salary_range:
        min_salary = parse_min_salary_from_range(salary_range)
    # Also check if min_salary is directly in metadata (some sources may provide it directly)
    if not min_salary and "min_salary" in metadata:
        try:
            min_salary_value = metadata.get("min_salary")
            if min_salary_value is not None:
                min_salary = int(min_salary_value)
        except (ValueError, TypeError) as e:
            logger.debug("min_salary_parse_failed", error=str(e), metadata_value=metadata.get("min_salary"))
            min_salary = None
    
    vacancy = Vacancy(
        title=metadata.get("title", "Unknown"),
        company_name=metadata.get("company_name", "Unknown"),
        company_stage=company_stage,
        location=metadata.get("location", "Not specified"),
        industry=industry,
        salary_range=salary_range,
        description_url=metadata.get("description_url", ""),
        required_skills=metadata.get("required_skills", []),
        remote_option=metadata.get("remote_option", False),
        source_url=metadata.get("source_url"),
    )

    # Store min_salary as an additional attribute (not in schema, but for frontend)
    if min_salary:
        # Add min_salary to the vacancy object as a dynamic attribute
        vacancy.min_salary = min_salary

    return vacancy


@router.post("/search", response_model=VacancySearchResponse)
async def search_vacancies(
    filter_params: VacancyFilter,
    required_keywords: Optional[List[str]] = Query(
        None, description="Required keywords that must be present in vacancies"
    ),
    use_firecrawl: bool = Query(
        False, description="Use Firecrawl for real search instead of Pinecone"
    ),
    use_mock: bool = Query(
        False, description="Use mock data instead of Pinecone"
    ),
) -> VacancySearchResponse:
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
            initial_count = len(all_vacancies)
            
            # Apply all filters in sequence
            filtered_vacancies = filter_vacancies(all_vacancies, filter_params)
            
            # Apply polarity filter (strict title domain matching)
            if filter_params.role:
                filtered_vacancies = apply_polarity_filter(filtered_vacancies, filter_params.role)
            
            # Apply keyword match filter
            if required_keywords:
                filtered_vacancies = apply_keyword_match_filter(filtered_vacancies, required_keywords)

            logger.info(
                "vacancy_search_completed",
                total_results=len(filtered_vacancies),
                source="mock",
                initial_count=initial_count
            )

            return VacancySearchResponse(
                vacancies=filtered_vacancies,
                total_in_db=initial_count,
                initial_vector_matches=initial_count,
                total_after_filters=len(filtered_vacancies)
            )

        if use_firecrawl:
            # Use Firecrawl for real search - DO NOT fall back to mock on failure
            try:
                firecrawl_service = get_firecrawl_service()
                vacancies = firecrawl_service.fetch_vacancies(filter_params, max_results=100)
                initial_count = len(vacancies)
                
                # Apply all filters in sequence
                filtered_vacancies = filter_vacancies(vacancies, filter_params)
                
                # Apply polarity filter (strict title domain matching)
                if filter_params.role:
                    filtered_vacancies = apply_polarity_filter(filtered_vacancies, filter_params.role)
                
                # Apply keyword match filter
                if required_keywords:
                    filtered_vacancies = apply_keyword_match_filter(filtered_vacancies, required_keywords)

                logger.info(
                    "vacancy_search_completed",
                    total_results=len(filtered_vacancies),
                    source="firecrawl",
                    initial_count=initial_count
                )

                return VacancySearchResponse(
                    vacancies=filtered_vacancies,
                    total_in_db=None,  # Firecrawl doesn't provide total DB count
                    initial_vector_matches=initial_count,
                    total_after_filters=len(filtered_vacancies)
                )
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
            
            # Track initial vector search results count
            initial_vector_matches = len(results)
            
            # Convert metadata to Vacancy objects
            vacancies = []
            for result in results:
                try:
                    vacancy = metadata_to_vacancy(result["metadata"])
                    vacancies.append(vacancy)
                except Exception as e:
                    logger.warning("vacancy_conversion_failed", error=str(e), metadata=result.get("metadata", {}))
                    continue
            
            # Apply filters in sequence for strict hybrid filtering
            
            # Step 1: Apply standard filters (skills, location, industry, etc.)
            # Note: Industry and location are already filtered by Pinecone metadata filters,
            # but we apply additional case-insensitive matching in Python
            filtered_vacancies = filter_vacancies(vacancies, filter_params)
            logger.info(
                "standard_filters_applied",
                before=len(vacancies),
                after=len(filtered_vacancies)
            )
            
            # Step 2: Apply polarity filter (strict title domain matching)
            if filter_params.role:
                before_polarity = len(filtered_vacancies)
                filtered_vacancies = apply_polarity_filter(filtered_vacancies, filter_params.role)
                logger.info(
                    "polarity_filter_applied",
                    before=before_polarity,
                    after=len(filtered_vacancies),
                    role_query=filter_params.role
                )
            
            # Step 3: Apply keyword match filter (required_keywords)
            if required_keywords:
                before_keywords = len(filtered_vacancies)
                filtered_vacancies = apply_keyword_match_filter(filtered_vacancies, required_keywords)
                logger.info(
                    "keyword_match_filter_applied",
                    before=before_keywords,
                    after=len(filtered_vacancies),
                    required_keywords=required_keywords
                )
            
            # Final count after all filters
            total_after_filters = len(filtered_vacancies)
            
            logger.info(
                "vacancy_search_completed",
                total_results=total_after_filters,
                source="pinecone",
                initial_vector_matches=initial_vector_matches,
                total_after_filters=total_after_filters
            )
            
            return VacancySearchResponse(
                vacancies=filtered_vacancies,
                total_in_db=None,  # Pinecone doesn't provide total DB count easily
                initial_vector_matches=initial_vector_matches,
                total_after_filters=total_after_filters
            )
            
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
        
        # Inject persona skills into search parameters for persona mode
        # This forces the DB to look for embeddings matching the candidate's top skills
        if search_mode == "persona" and request.persona:
            persona_skills = request.persona.get("technical_skills")
            if persona_skills:
                # Extract top 5 technical skills from persona
                if isinstance(persona_skills, list):
                    top_5_skills = persona_skills[:5]
                elif isinstance(persona_skills, str):
                    # If it's a string, try to split by comma or use as single skill
                    top_5_skills = [s.strip() for s in persona_skills.split(",")][:5]
                else:
                    top_5_skills = []
                
                if top_5_skills:
                    # Merge with existing skills (avoid duplicates)
                    existing_skills = set(filter_params.skills or [])
                    persona_skills_set = set([s.lower() for s in top_5_skills])
                    
                    # Add persona skills that aren't already in the list
                    merged_skills = list(existing_skills)
                    for skill in top_5_skills:
                        if skill.lower() not in [s.lower() for s in merged_skills]:
                            merged_skills.append(skill)
                    
                    filter_params.skills = merged_skills
                    logger.info(
                        "persona_skills_injected",
                        top_5_skills=top_5_skills,
                        merged_skills=merged_skills,
                        search_mode=search_mode
                    )
        
        # Get embedding service URL
        embedding_service_url = os.getenv(
            "EMBEDDING_SERVICE_URL",
            "http://embedding-service:8001"
        )
        
        # Extract required_keywords for hard filtering
        required_keywords = extracted_params.get("required_keywords")
        if required_keywords and isinstance(required_keywords, list):
            required_keywords = [kw for kw in required_keywords if kw and len(str(kw).strip()) > 0]
        else:
            required_keywords = None
        
        # Use the expanded role query directly from Job Scout (it's already optimized)
        # The role field now contains an expanded search query, not just a job title
        if filter_params.role:
            # The role is already an expanded query from Job Scout, use it directly
            search_query = filter_params.role
            logger.info(
                "chat_search_using_expanded_role_query",
                role_query=search_query,
                search_mode=search_mode
            )
        else:
            # Fallback: build query if role is missing
            search_query = build_search_query(filter_params, persona=request.persona, search_mode=search_mode)
            logger.info(
                "chat_search_query_built_fallback",
                query=search_query,
                search_mode=search_mode
            )
        
        logger.info(
            "chat_search_query_built", 
            query=search_query, 
            search_mode=search_mode,
            has_persona=request.persona is not None,
            required_keywords=required_keywords
        )
        
        # Get embedding for search query
        query_embedding = await get_query_embedding(search_query, embedding_service_url)
        logger.info("chat_query_embedding_generated", dim=len(query_embedding))
        
        # Build Pinecone filter
        pinecone_filter = build_pinecone_filter(filter_params)
        logger.info("chat_pinecone_filter_built", filter=pinecone_filter)
        
        # Initialize Pinecone client
        vector_store = VectorStore()
        
        # Query Pinecone with top_k=40 for hybrid filtering
        top_k = 40
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
        
        # ========================================================================
        # STRICT HARD FILTERING: Apply hard filters BEFORE ranking/prioritization
        # ========================================================================
        # These filters remove vacancies that don't match, regardless of semantic score.
        # Vacancies that fail these filters are removed BEFORE any ranking occurs.
        # This ensures only relevant vacancies are ranked by semantic similarity.
        # 
        # Hard filters (applied in order):
        # 1. Industry (if specified)
        # 2. Location (if specified)
        # 3. Company Stage (if specified)
        # 4. Required Keywords (if specified)
        # ========================================================================
        
        before_hard_metadata = len(vacancies)
        filtered_vacancies = vacancies
        
        # HARD FILTER: Industry (strict match) - removes vacancies that don't match
        if filter_params.industry:
            industry_lower = filter_params.industry.lower()
            filtered_vacancies = [
                v for v in filtered_vacancies 
                if industry_lower in v.industry.lower()
            ]
            logger.info(
                "hard_filter_industry_applied",
                before=before_hard_metadata,
                after=len(filtered_vacancies),
                industry=filter_params.industry
            )
        
        # HARD FILTER: Location (strict match with synonyms) - removes vacancies that don't match
        if filter_params.location:
            location_lower = filter_params.location.lower().strip()
            LOCATION_SYNONYMS = {
                "us": ["united states", "usa", "america", "u.s.", "u.s.a."],
                "uk": ["united kingdom", "london", "england", "britain", "great britain"],
                "united states": ["us", "usa", "america", "u.s.", "u.s.a."],
                "united kingdom": ["uk", "london", "england", "britain", "great britain"],
                "usa": ["us", "united states", "america", "u.s.", "u.s.a."],
            }
            location_variants = [location_lower]
            for key, synonyms in LOCATION_SYNONYMS.items():
                if key in location_lower:
                    location_variants.extend(synonyms)
                elif any(syn in location_lower for syn in synonyms):
                    location_variants.append(key)
                    location_variants.extend(synonyms)
            location_variants = list(dict.fromkeys(location_variants))
            
            before_location = len(filtered_vacancies)
            filtered_vacancies = [
                v for v in filtered_vacancies
                if any(variant in v.location.lower() for variant in location_variants)
                or (v.remote_option and "remote" in location_lower)
            ]
            logger.info(
                "hard_filter_location_applied",
                before=before_location,
                after=len(filtered_vacancies),
                location=filter_params.location
            )
        
        # HARD FILTER: Company Stage (strict match) - removes vacancies that don't match
        if filter_params.company_stages:
            filter_vals = [CompanyStage.get_stage_value(s) for s in filter_params.company_stages]
            before_stage = len(filtered_vacancies)
            filtered_vacancies = [
                v for v in filtered_vacancies 
                if CompanyStage.get_stage_value(v.company_stage) in filter_vals
            ]
            logger.info(
                "hard_filter_company_stage_applied",
                before=before_stage,
                after=len(filtered_vacancies),
                stages=filter_params.company_stages
            )
        
        # Step 4: Apply HARD keyword filter (required_keywords) - STRICT FILTER
        # This is a strict requirement: vacancies MUST contain at least one required keyword
        # Vacancies that don't match are removed, regardless of semantic score
        # This filter is applied AFTER metadata filters but BEFORE ranking
        if required_keywords:
            before_keywords = len(filtered_vacancies)
            filtered_vacancies = apply_hard_keyword_filter(filtered_vacancies, required_keywords)
            logger.info(
                "hard_keyword_filter_applied",
                required_keywords=required_keywords,
                before=before_keywords,
                after=len(filtered_vacancies),
                message="Hard filter: vacancies without required keywords removed before ranking"
            )
            # NO FALLBACK - if hard filter returns 0 results, that's the result
        
        # ========================================================================
        # SOFT FILTERING: Apply soft filters AFTER hard filters
        # ========================================================================
        # These filters are less strict and allow some flexibility.
        # Applied after hard filters but before ranking.
        # ========================================================================
        
        # Step 5: Apply Soft Title Filter (exclude conflicting terms)
        if filter_params.role:
            before_soft = len(filtered_vacancies)
            filtered_vacancies = apply_soft_title_filter(filtered_vacancies, filter_params.role)
            logger.info(
                "soft_title_filter_applied",
                before=before_soft,
                after=len(filtered_vacancies)
            )
        
        # Step 6: Apply remaining soft filters (skills - at least one match)
        # Note: Skills are soft filters (at least one match), not hard filters
        before_soft_filters = len(filtered_vacancies)
        filtered_vacancies = filter_vacancies(filtered_vacancies, filter_params)
        logger.info(
            "soft_filters_applied",
            before=before_soft_filters,
            after=len(filtered_vacancies)
        )
        
        # ========================================================================
        # RANKING: Update scores and prioritize AFTER all filtering
        # ========================================================================
        # Only vacancies that passed all hard filters are ranked.
        # ========================================================================
        
        # Update vacancy_score_pairs to match filtered vacancies (preserve scores for filtered results)
        # This ensures Pinecone scores are preserved for the filtered vacancies
        filtered_vacancy_keys = {f"{v.title}_{v.company_name}" for v in filtered_vacancies}
        vacancy_score_pairs = [
            (v, s) for v, s in vacancy_score_pairs 
            if f"{v.title}_{v.company_name}" in filtered_vacancy_keys
        ]
        
        # Prioritize by persona preferences if in persona mode (AFTER filtering)
        if search_mode == "persona":
            filtered_vacancies = prioritize_by_persona_preferences(
                filtered_vacancies,
                request.persona,
                search_mode
            )
            logger.info(
                "persona_prioritization_applied",
                results_after_prioritization=len(filtered_vacancies),
                message="Prioritization applied AFTER hard filters"
            )

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
            
            # Ensure min_salary is included if available (from parsed salary_range)
            # This is extracted from Pinecone metadata in metadata_to_vacancy()
            if hasattr(vacancy, 'min_salary') and vacancy.min_salary is not None:
                vacancy_dict['min_salary'] = vacancy.min_salary
            elif vacancy_dict.get('salary_range'):
                # Parse min_salary from salary_range if not already set
                min_salary = parse_min_salary_from_range(vacancy_dict.get('salary_range'))
                if min_salary:
                    vacancy_dict['min_salary'] = min_salary
            else:
                # Explicitly set to None if not available
                vacancy_dict['min_salary'] = None
            
            # Ensure industry is included (extracted from Pinecone metadata)
            # This should already be there from metadata_to_vacancy(), but ensure it's present
            if 'industry' not in vacancy_dict or not vacancy_dict.get('industry'):
                vacancy_dict['industry'] = vacancy.industry if hasattr(vacancy, 'industry') else "Technology"
            
            # Ensure all required fields are present for frontend
            # These are extracted from Pinecone metadata in metadata_to_vacancy()
            if 'title' not in vacancy_dict:
                vacancy_dict['title'] = vacancy.title if hasattr(vacancy, 'title') else "Unknown"
            if 'company_name' not in vacancy_dict:
                vacancy_dict['company_name'] = vacancy.company_name if hasattr(vacancy, 'company_name') else "Unknown"
            if 'location' not in vacancy_dict:
                vacancy_dict['location'] = vacancy.location if hasattr(vacancy, 'location') else "Not specified"
            if 'required_skills' not in vacancy_dict:
                vacancy_dict['required_skills'] = vacancy.required_skills if hasattr(vacancy, 'required_skills') else []
            
            vacancies_response.append(vacancy_dict)

        # Use MatchmakerAgent to analyze top vacancies if persona is provided
        # Analyze top 10 candidates from the filtered results, then sort by AI score
        # This ensures we get the best matches based on AI analysis, not just vector similarity
        top_vacancies_for_matching = vacancies_response[:10] if len(vacancies_response) > 0 else []
        
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
                        
                        # Ensure ai_score is an integer (0-10 scale)
                        if ai_score is not None:
                            try:
                                ai_score = int(ai_score)
                            except (ValueError, TypeError):
                                logger.warning(
                                    "ai_score_invalid_type",
                                    vacancy_title=vacancy_dict.get('title', 'Unknown'),
                                    ai_score=ai_score
                                )
                                ai_score = None
                        
                        # Add AI match score and reasoning to vacancy dict
                        # CRITICAL: Always save the score to response object for UI to read
                        # Score is on 0-10 scale, UI will convert to percentage
                        vacancy_dict['ai_match_score'] = ai_score  # 0-10 scale, saved for frontend
                        
                        if match_reasoning and len(match_reasoning) > 0:
                            vacancy_dict['match_reasoning'] = match_reasoning
                            successful_analyses += 1
                        else:
                            logger.warning(
                                "matchmaker_analysis_empty_response",
                                vacancy_title=vacancy_dict.get('title', 'Unknown')
                            )
                            vacancy_dict['match_reasoning'] = None
                        
                        # Log that AI score was saved
                        logger.debug(
                            "ai_match_score_saved",
                            vacancy_title=vacancy_dict.get('title', 'Unknown'),
                            ai_score=ai_score
                        )
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
            
            # Re-rank: Sort vacancies by AI match score (descending) to prioritize best matches
            # Only vacancies that were analyzed by Matchmaker will have AI scores
            def sort_key(v):
                ai_score = v.get('ai_match_score')
                if ai_score is not None:
                    return ai_score  # Sort by AI score descending
                else:
                    return -1  # No AI score, sort to end
            
            # Sort all vacancies by AI score (descending)
            vacancies_response.sort(key=sort_key, reverse=True)
            
            # Filter: Only show vacancies with AI score >= 5
            # This ensures we only show high-quality matches that were analyzed by Matchmaker
            before_filter = len(vacancies_response)
            vacancies_response = [
                v for v in vacancies_response 
                if v.get('ai_match_score') is not None and v.get('ai_match_score') >= 5
            ]
            after_filter = len(vacancies_response)
            logger.info(
                "vacancies_filtered_by_ai_score",
                before_count=before_filter,
                after_count=after_filter,
                filtered_out=before_filter - after_filter,
                threshold=5
            )
            
            # AI scores are already included in each vacancy dict and will be shown in the response
            logger.info("vacancies_reranked_by_ai_score", final_count=len(vacancies_response))
        
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
        
        # Note: Hard filters are now strict (no fallback)
        # If required_keywords filter returns 0 results, that's the final result
        # No warning message needed as this is expected behavior for strict filtering

        # Extract debug_info from extracted_params
        debug_info = extracted_params.get("debug_info", {})

        logger.info("chat_search_completed", summary_length=len(summary))

        return {
            "vacancies": vacancies_response,
            "summary": summary,
            "debug_info": debug_info
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
