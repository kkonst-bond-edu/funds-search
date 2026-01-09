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
    persona: Optional[Dict[str, Any]] = Field(default=None, description="User persona/CV information for personalized search")


class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    vacancies: List[Dict[str, Any]] = Field(..., description="List of matching vacancies with match scores and AI insights")
    summary: str = Field(..., description="AI-generated summary explaining the results")
    debug_info: Dict[str, Any] = Field(default_factory=dict, description="Debug information including friendly_reasoning")
    persona_applied: bool = Field(..., description="Flag indicating if persona data was successfully used for matching")


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

    # Skip role filtering if role is None, empty, or indicates "all" search
    # This ensures that when Job Scout sets role to null for broad queries, all vacancies pass through
    if filter_params.role:
        role_lower = filter_params.role.lower().strip()
        # If role is "all", "any", "everything", or empty after strip, skip filtering entirely
        # This allows truly broad searches without title-based filtering
        if role_lower and role_lower not in ["all", "any", "everything", "null", "none"]:
            # Smart Keyword Matcher for role filtering
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
                # Also handles "designer" matching "Motion Designer", "UI Designer", etc.
                title_match = any(norm_word in title_lower for norm_word in normalized_words)
                
                # If modifiers are present (e.g., "frontend", "backend", "mobile"), they MUST be in title or skills
                # BUT: if no modifiers are present (e.g., just "Designer"), don't require modifier match
                modifier_match = True
                if modifier_keywords:
                    modifier_match = any(mod in combined_text for mod in modifier_keywords)
                
                # Prioritize title matches - if role words match in title, it's a strong match
                # This ensures "Backend Engineer" matches "Backend Software Engineer" in title
                # And "Designer" matches "Motion Designer", "UI Designer", etc.
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
        # Expand "Growth" to include Series B, Series C, and Growth
        expanded_stages = []
        for stage in filter_params.company_stages:
            normalized = CompanyStage.get_stage_value(stage)
            if normalized == "Growth":
                # Growth includes Series B, Series C, and Growth
                expanded_stages.extend(["Series B", "Series C", "Growth"])
            else:
                expanded_stages.append(normalized)
        
        # Remove duplicates
        filter_vals = list(set(expanded_stages))
        filtered = [
            v for v in filtered if CompanyStage.get_stage_value(v.company_stage) in filter_vals
        ]

    if filter_params.industry:
        # Case-insensitive substring matching
        industry_lower = filter_params.industry.lower()
        filtered = [v for v in filtered if industry_lower in v.industry.lower()]

    if filter_params.min_salary:
        # Filter by minimum salary
        # We check vacancy.min_salary (already populated in metadata_to_vacancy)
        filtered = [
            v for v in filtered
            if v.min_salary is not None and v.min_salary >= filter_params.min_salary
        ]

    if filter_params.category:
        # Filter by category (case-insensitive)
        category_lower = filter_params.category.lower()
        filtered = [v for v in filtered if v.category and category_lower in v.category.lower()]

    if filter_params.experience_level:
        # Filter by experience level (case-insensitive)
        exp_lower = filter_params.experience_level.lower()
        filtered = [v for v in filtered if v.experience_level and exp_lower in v.experience_level.lower()]

    if filter_params.employee_count:
        # Filter by employee count (exact match or substring)
        # Handle both "100-1000 employees" and "100–1000 employees" (different dashes)
        employee_count_lower = [ec.lower().replace('–', '-') for ec in filter_params.employee_count]
        filtered = [
            v for v in filtered
            if v.employee_count and any(
                ec in v.employee_count.lower().replace('–', '-') for ec in employee_count_lower
            )
        ]

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
    Apply keyword match filter using required_keywords with OR logic.
    
    Vacancies must match at least ONE of the required keywords in their
    title or required_skills to pass this filter (OR logic, not AND).
    
    Args:
        vacancies: List of vacancies to filter
        required_keywords: List of critical keywords
        
    Returns:
        Filtered list of vacancies
    """
    if not required_keywords or len(required_keywords) == 0:
        return vacancies
    
    # Normalize keywords to lowercase for matching
    keywords_lower = [kw.lower() for kw in required_keywords if kw and len(kw.strip()) > 0]
    
    if not keywords_lower:
        return vacancies
    
    # Add synonyms for common terms (e.g., "Database" -> ["database", "db", "sql", "postgresql", "mysql"])
    keyword_synonyms = {}
    for kw in keywords_lower:
        synonyms = [kw]  # Start with the keyword itself
        if "database" in kw or "db" in kw:
            synonyms.extend(["database", "db", "sql", "postgresql", "postgres", "mysql", "mongodb", "redis"])
        elif "python" in kw:
            synonyms.extend(["python", "py", "django", "flask", "fastapi"])
        elif "javascript" in kw or "js" in kw:
            synonyms.extend(["javascript", "js", "node", "nodejs", "typescript", "ts"])
        keyword_synonyms[kw] = synonyms
    
    filtered = []
    for vacancy in vacancies:
        title_lower = vacancy.title.lower()
        # Check both title and required_skills list
        skills_lower = [s.lower() for s in vacancy.required_skills]
        combined_text = f"{title_lower} {' '.join(skills_lower)}"
        
        # Check if at least ONE required keyword (or its synonym) is present
        has_keyword = False
        for kw in keywords_lower:
            # Check the keyword itself
            if kw in combined_text:
                has_keyword = True
                break
            # Check synonyms if available
            if kw in keyword_synonyms:
                for synonym in keyword_synonyms[kw]:
                    if synonym in combined_text:
                        has_keyword = True
                        break
                if has_keyword:
                    break
        
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
    
    # Build weighted query string: Role is primary, Skills are secondary, Industry adds context
    query_parts = []
    if role_part:
        # Role is the primary component - this ensures role matching takes precedence
        query_parts.append(f"Role: {role_part}")
    if skills_part:
        query_parts.append(f"Skills: {skills_part}")
    
    # Add industry for better context (if provided)
    if filter_params.industry:
        query_parts.append(f"Industry: {filter_params.industry}")
    
    # Add location for context (if provided)
    if filter_params.location:
        query_parts.append(f"Location: {filter_params.location}")
    
    # Fallback: If no query parts in any mode, use generic query
    if not query_parts:
        query_parts.append("job vacancy")
    
    # Return weighted query with role as primary component
    # This format ensures the embedding prioritizes role matching over skill matching
    return ". ".join(query_parts) + "."


def build_pinecone_filter(filter_params: VacancyFilter, remote_available: Optional[bool] = None) -> Optional[Dict[str, Any]]:
    """
    Build Pinecone filter dictionary from filter parameters.
    Uses metadata filters for industry, location, company_stage, and remote_available.
    
    Args:
        filter_params: VacancyFilter with search criteria
        remote_available: Optional boolean from Job Scout response for remote_available field
        
    Returns:
        Pinecone filter dictionary or None
    """
    filter_dict = {}
    
    # Metadata filtering: Use Pinecone metadata filters for industry
    # Use $in with case variations to handle potential case mismatches
    if filter_params.industry:
        # Generate case variations (Original, Title, Lower, Upper)
        industry_variants = list(set([
            filter_params.industry,
            filter_params.industry.title(),
            filter_params.industry.lower(),
            filter_params.industry.upper()
        ]))
        filter_dict["industry"] = {"$in": industry_variants}
    
    # Metadata filtering: Use Pinecone metadata filters for location
    # Use $in with case variations
    if filter_params.location:
        # Generate case variations
        location_variants = list(set([
            filter_params.location,
            filter_params.location.title(),
            filter_params.location.lower(),
            filter_params.location.upper()
        ]))
        filter_dict["location"] = {"$in": location_variants}
    
    # Boolean filtering: Handle remote_available from Job Scout response
    # Priority: remote_available parameter > filter_params.is_remote
    if remote_available is not None:
        # Use remote_available field from Job Scout response
        filter_dict["remote_available"] = {"$eq": remote_available}
    elif filter_params.is_remote is not None:
        # Fallback to filter_params.is_remote (for backward compatibility)
        filter_dict["remote_option"] = {"$eq": filter_params.is_remote}
    
    # Metadata filtering: Use Pinecone metadata filters for company_stage
    # Use $in for multiple values
    # Normalize the strings to ensure they match enum values (e.g., 'SeriesA' -> 'Series A')
    # Expand "Growth" to include Series B, Series C, and Growth
    if filter_params.company_stages:
        expanded_stages = []
        for stage in filter_params.company_stages:
            normalized = CompanyStage.get_stage_value(stage)
            if normalized == "Growth":
                # Growth includes Series B, Series C, and Growth
                expanded_stages.extend(["Series B", "Series C", "Growth"])
            else:
                expanded_stages.append(normalized)
        
        # Remove duplicates
        normalized_stages = list(set(expanded_stages))
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
    # Handle company_stage - it might be a string, list, or dict that needs normalization
    company_stage_raw = metadata.get("company_stage", "Growth")
    
    # Handle old format: stringified Python list like "[{'id': '...', 'label': '...'}]"
    company_stage_str = str(company_stage_raw)
    if company_stage_str.startswith("[{") and "employees" in company_stage_str:
        # Try to extract employee count from old format
        import re
        employee_match = re.search(r"'label':\s*'([^']+)'", company_stage_str)
        if employee_match:
            company_stage_str = employee_match.group(1)
        else:
            # Fallback: try to extract any text with "employees"
            employee_match = re.search(r"(\d+[-\+]?\s*employees)", company_stage_str, re.IGNORECASE)
            if employee_match:
                company_stage_str = employee_match.group(1)
            else:
                company_stage_str = "Growth"
    
    # Normalize using get_stage_value helper
    normalized_stage = CompanyStage.get_stage_value(company_stage_str)
    
    # Use normalized stage as string (since Vacancy.company_stage is now str, not Enum)
    company_stage = normalized_stage
    
    # Extract industry - ensure it's properly extracted from Pinecone metadata
    # This is a critical field for filtering and must be present
    industry = metadata.get("industry")
    if not industry or industry == "" or not isinstance(industry, str):
        industry = "Technology"  # Default fallback
        logger.debug("industry_missing_in_metadata", defaulted_to="Technology")
    
    # Extract salary_range from Pinecone metadata and clean it if it contains dicts
    salary_range = metadata.get("salary_range")
    # Clean salary_range if it contains dict representations (e.g., "{'label': 'USD', 'value': 'USD'}")
    if salary_range and isinstance(salary_range, str):
        import re
        # Check if string contains dict-like patterns
        if "'label'" in salary_range or '"label"' in salary_range:
            # Extract only numbers from the string
            numbers = re.findall(r'\d+[\d,]*', salary_range)
            if numbers:
                # Remove commas and convert to int
                clean_numbers = [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit()]
                if len(clean_numbers) == 1:
                    salary_range = f"${clean_numbers[0]:,}"
                elif len(clean_numbers) >= 2:
                    salary_range = f"${clean_numbers[0]:,} - ${clean_numbers[-1]:,}"
                else:
                    salary_range = None
            else:
                salary_range = None
    
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
    
    # Extract all additional fields from metadata
    category = metadata.get("category")
    experience_level = metadata.get("experience_level")
    employee_count = metadata.get("employee_count")
    is_hybrid = metadata.get("is_hybrid", False)
    
    vacancy = Vacancy(
        title=metadata.get("title", "Unknown"),
        company_name=metadata.get("company_name", "Unknown"),
        company_stage=company_stage,
        location=metadata.get("location", "Not specified"),
        industry=industry,
        category=category,
        experience_level=experience_level,
        salary_range=salary_range,
        description_url=metadata.get("description_url", ""),
        required_skills=metadata.get("required_skills", []),
        remote_option=metadata.get("remote_option", False),
        is_hybrid=is_hybrid,
        employee_count=employee_count,
        source_url=metadata.get("source_url"),
        full_description=metadata.get("text", "") or metadata.get("full_description", ""),
        min_salary=min_salary,
    )

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
            # Only apply if role contains domain-specific keywords (frontend, backend, etc.)
            # Skip for generic roles like "Designer", "Manager", "Engineer" without modifiers
            if filter_params.role:
                role_lower = filter_params.role.lower()
                # Check if role contains domain-specific keywords that need polarity filtering
                domain_keywords_in_role = ["frontend", "backend", "mobile", "fullstack", "devops", "data"]
                has_domain_keyword = any(keyword in role_lower for keyword in domain_keywords_in_role)
                
                if has_domain_keyword:
                    before_polarity = len(filtered_vacancies)
                    filtered_vacancies = apply_polarity_filter(filtered_vacancies, filter_params.role)
                    logger.info(
                        "polarity_filter_applied",
                        before=before_polarity,
                        after=len(filtered_vacancies),
                        role_query=filter_params.role
                    )
                else:
                    logger.info(
                        "polarity_filter_skipped",
                        reason="No domain-specific keywords in role",
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
            
            # Remove duplicates by description_url (keep first occurrence)
            seen_urls = set()
            unique_vacancies = []
            for vacancy in filtered_vacancies:
                url = vacancy.description_url
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    unique_vacancies.append(vacancy)
            
            # Update count after deduplication
            duplicates_removed = total_after_filters - len(unique_vacancies)
            if duplicates_removed > 0:
                logger.info("duplicates_removed", count=duplicates_removed, total_before=total_after_filters, total_after=len(unique_vacancies))
            
            logger.info(
                "vacancy_search_completed",
                total_results=len(unique_vacancies),
                source="pinecone",
                initial_vector_matches=initial_vector_matches,
                total_after_filters=len(unique_vacancies),
                duplicates_removed=duplicates_removed
            )
            
            return VacancySearchResponse(
                vacancies=unique_vacancies,
                total_in_db=None,  # Pinecone doesn't provide total DB count easily
                initial_vector_matches=initial_vector_matches,
                total_after_filters=len(unique_vacancies)
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


@router.post("/chat", response_model=ChatResponse)
async def chat_search(request: ChatRequest) -> ChatResponse:
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
        
        # Fallback: Log warning if persona is missing
        if not request.persona:
            logger.warning("chat_search_without_persona", detail="Chat request received without persona data. Matchmaker will return low scores.")
        else:
            logger.info("chat_search_with_persona", persona_keys=list(request.persona.keys()) if isinstance(request.persona, dict) else "not_dict")
        
        # Initialize chat search agent
        chat_agent = ChatSearchAgent()
        
        # Orchestration: Pass persona to chat_search_agent.interpret_message()
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
        
        # Extract remote_available boolean from Job Scout response
        remote_available = extracted_params.get("remote_available")
        if remote_available is not None:
            # Ensure it's a boolean
            if isinstance(remote_available, bool):
                pass  # Already boolean
            elif isinstance(remote_available, str):
                remote_available = remote_available.lower() in ("true", "1", "yes")
            else:
                remote_available = bool(remote_available)
            logger.info("remote_available_extracted", remote_available=remote_available)
        
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
        
        # Build Pinecone filter with remote_available from Job Scout response
        pinecone_filter = build_pinecone_filter(filter_params, remote_available=remote_available)
        logger.info("chat_pinecone_filter_built", filter=pinecone_filter, remote_available=remote_available)
        
        # Initialize Pinecone client
        vector_store = VectorStore()
        
        # Query Pinecone with top_k=40 to ensure we have enough candidates before filtering
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
            # Expand "Growth" to include Series B, Series C, and Growth
            expanded_stages = []
            for stage in filter_params.company_stages:
                normalized = CompanyStage.get_stage_value(stage)
                if normalized == "Growth":
                    # Growth includes Series B, Series C, and Growth
                    expanded_stages.extend(["Series B", "Series C", "Growth"])
                else:
                    expanded_stages.append(normalized)
            
            # Remove duplicates
            filter_vals = list(set(expanded_stages))
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
        
        # Step 4: Apply keyword filter (required_keywords) - OR LOGIC
        # This is an OR match requirement: vacancies must match at least ONE required keyword
        # Vacancies that don't match any keyword are removed, regardless of semantic score
        # This filter is applied AFTER metadata filters but BEFORE ranking
        if required_keywords:
            before_keywords = len(filtered_vacancies)
            filtered_vacancies = apply_hard_keyword_filter(filtered_vacancies, required_keywords)
            logger.info(
                "keyword_filter_applied",
                required_keywords=required_keywords,
                before=before_keywords,
                after=len(filtered_vacancies),
                logic="OR (at least one keyword must match)",
                message="Keyword filter: vacancies must match at least one required keyword"
            )
            # NO FALLBACK - if filter returns 0 results, that's the result
        
        # ========================================================================
        # SOFT FILTERING: Apply soft filters AFTER hard filters
        # ========================================================================
        # These filters are less strict and allow some flexibility.
        # Applied after hard filters but before ranking.
        # ========================================================================
        
        # Step 5: Apply Soft Title Filter (exclude conflicting terms)
        # Skip soft title filter if role is None, empty, or indicates "all" search
        if filter_params.role:
            role_lower = filter_params.role.lower().strip() if filter_params.role else ""
            # Only apply soft filter if role is not empty and not a special "all" keyword
            if role_lower and role_lower not in ["all", "any", "everything", "null", "none"]:
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
            
            # METADATA MAPPING: Ensure industry, salary_min, and company_stage are correctly extracted
            # Industry: extracted from Pinecone metadata in metadata_to_vacancy()
            if 'industry' not in vacancy_dict or not vacancy_dict.get('industry'):
                vacancy_dict['industry'] = vacancy.industry if hasattr(vacancy, 'industry') else "Technology"
            
            # Salary_min: already extracted in metadata_to_vacancy and included in model_dump
            # Just ensure it's present if for some reason it's missing (though it shouldn't be)
            if 'min_salary' not in vacancy_dict:
                vacancy_dict['min_salary'] = vacancy.min_salary if hasattr(vacancy, 'min_salary') else None

            
            # Company_stage: extracted from Pinecone metadata in metadata_to_vacancy()
            # This should already be present, but ensure it's correctly mapped
            if 'company_stage' not in vacancy_dict:
                vacancy_dict['company_stage'] = vacancy.company_stage.value if hasattr(vacancy.company_stage, 'value') else str(vacancy.company_stage)
            
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

        # ========================================================================
        # MATCHMAKER INTEGRATION (CRITICAL): Analyze ALL filtered vacancies
        # ========================================================================
        # After getting filtered results from Pinecone, iterate through them
        # and call matchmaker_agent.analyze_match(vacancy_text, persona)
        # The Matchmaker must return a JSON with 'score' (0-100) and 'analysis' (text)
        # ========================================================================
        
        # Check if persona data is available and valid
        has_persona = bool(request.persona) and isinstance(request.persona, dict) and len(request.persona) > 0
        persona_applied = False
        
        if vacancies_response:
            logger.info("matchmaker_analysis_started", vacancy_count=len(vacancies_response), has_persona=has_persona)
            
            if has_persona:
                matchmaker = MatchmakerAgent()
            
            # Iterate through all vacancies after filtering
            successful_analyses = 0
            for vacancy_dict in vacancies_response:
                try:
                    # If persona is missing, set default values without calling matchmaker
                    if not has_persona:
                        vacancy_dict['score'] = 0
                        vacancy_dict['match_score'] = 0
                        vacancy_dict['ai_match_score'] = 0
                        vacancy_dict['ai_insight'] = "CV missing: Upload your resume in the 'Career & Match Hub' to enable AI matching."
                        vacancy_dict['match_reason'] = "CV missing: Upload your resume in the 'Career & Match Hub' to enable AI matching."
                        vacancy_dict['persona_applied'] = False
                        continue
                    
                    # Build vacancy text from the vacancy dict for matchmaker analysis
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
                    
                    # Orchestration: Pass the SAME persona to matchmaker_agent.analyze_match()
                    # After getting search results, pass the SAME 'persona' to 'matchmaker_agent.analyze_match()'
                    # The Matchmaker must return a JSON with 'score' (0-100) and 'analysis' (text)
                    match_result = await matchmaker.analyze_match(
                        vacancy_text=vacancy_text,
                        candidate_persona=request.persona,
                        similarity_score=pinecone_score  # Pass Pinecone score for context
                    )
                    
                    # Parse Matchmaker response
                    # Expected format: {"score": 0-100, "analysis": "text"} or {"score": 0-10, "reasoning": "text"}
                    ai_score = None
                    analysis_text = None
                    
                    if isinstance(match_result, dict):
                        # Try to get score (0-100 scale) or fallback to 0-10 scale
                        ai_score = match_result.get('score')
                        # Try 'analysis' first, then fallback to 'reasoning'
                        analysis_text = match_result.get('analysis') or match_result.get('reasoning', '')
                    elif isinstance(match_result, str):
                        # If response is a string, try to parse it as JSON or extract score
                        import json
                        import re
                        try:
                            # Try to parse as JSON
                            parsed = json.loads(match_result)
                            ai_score = parsed.get('score')
                            analysis_text = parsed.get('analysis') or parsed.get('reasoning', '')
                        except (json.JSONDecodeError, ValueError):
                            # Try to extract score from string (format: "Score: X/100" or "Score: X/10")
                            score_pattern_100 = r'Score:\s*(\d+)/100'
                            score_pattern_10 = r'Score:\s*(\d+)/10'
                            score_match = re.search(score_pattern_100, match_result, re.IGNORECASE)
                            if not score_match:
                                score_match = re.search(score_pattern_10, match_result, re.IGNORECASE)
                            if score_match:
                                try:
                                    ai_score = int(score_match.group(1))
                                    # If extracted from /10 format, convert to 0-100 scale
                                    if '/10' in match_result:
                                        ai_score = ai_score * 10  # Convert 0-10 to 0-100
                                    ai_score = max(0, min(100, ai_score))  # Clamp to 0-100
                                    # Remove score line from analysis text
                                    analysis_text = re.sub(r'Score:\s*\d+/(?:100|10)\s*', '', match_result, flags=re.IGNORECASE).strip()
                                except (ValueError, TypeError):
                                    analysis_text = match_result
                            else:
                                analysis_text = match_result
                    else:
                        # Fallback: convert to string
                        analysis_text = str(match_result) if match_result else None
                    
                    # Ensure ai_score is properly converted to integer (0-100 scale)
                    if ai_score is not None:
                        try:
                            # Convert to float first to handle string/float/int inputs
                            ai_score = float(ai_score)
                            # If score is in 0-10 range, convert to 0-100
                            if 0 <= ai_score <= 10:
                                ai_score = ai_score * 10
                            # Clamp to valid range (0-100)
                            ai_score = max(0.0, min(100.0, ai_score))
                            # Always convert to int for clean UI display
                            ai_score = int(round(ai_score))
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                "ai_score_invalid_type",
                                vacancy_title=vacancy_dict.get('title', 'Unknown'),
                                ai_score=ai_score,
                                error=str(e)
                            )
                            ai_score = 0  # Default to 0 on conversion error
                    else:
                        # If score not found, use a default (0)
                        logger.warning(
                            "matchmaker_analysis_no_score",
                            vacancy_title=vacancy_dict.get('title', 'Unknown')
                        )
                        ai_score = 0  # Default score if not found
                    
                    # SCORE MAPPING: Map 'score' to both 'score' and 'match_score' fields
                    # This is the primary score field used by UI - ensure it's always an integer (0-100)
                    vacancy_dict['score'] = ai_score
                    vacancy_dict['match_score'] = ai_score  # Add match_score field for UI compatibility
                    
                    # Also keep ai_match_score for backward compatibility (0-100 scale)
                    vacancy_dict['ai_match_score'] = ai_score
                    
                    # Mark that persona was successfully applied
                    vacancy_dict['persona_applied'] = True
                    persona_applied = True
                    
                    # Log successful score mapping for debugging
                    logger.debug(
                        "matchmaker_score_mapped",
                        vacancy_title=vacancy_dict.get('title', 'Unknown'),
                        score=ai_score,
                        match_score=ai_score,
                        has_analysis=bool(analysis_text),
                        persona_applied=True
                    )
                    
                    # SCORE MAPPING: Map 'analysis' to a new field 'ai_insight'
                    vacancy_dict['ai_insight'] = analysis_text if analysis_text and len(analysis_text) > 0 else None
                    
                    # Also keep match_reason for backward compatibility
                    vacancy_dict['match_reason'] = analysis_text if analysis_text and len(analysis_text) > 0 else None
                    
                    if ai_score is not None and ai_score > 0:
                        successful_analyses += 1
                        logger.debug(
                            "matchmaker_analysis_saved",
                            vacancy_title=vacancy_dict.get('title', 'Unknown'),
                            score=ai_score,
                            has_analysis=bool(analysis_text)
                        )
                    
                except Exception as e:
                    logger.error(
                        "matchmaker_analysis_failed",
                        vacancy_title=vacancy_dict.get('title', 'Unknown'),
                        error=str(e),
                        error_type=type(e).__name__
                    )
                    # Set fallback values instead of leaving them empty
                    vacancy_dict['score'] = 0  # Default score (0-100 scale)
                    vacancy_dict['match_score'] = 0  # Add match_score field for UI compatibility
                    vacancy_dict['ai_match_score'] = 0
                    vacancy_dict['persona_applied'] = False
                    vacancy_dict['ai_insight'] = "Match analysis temporarily unavailable."
                    vacancy_dict['match_reason'] = "Match analysis temporarily unavailable."
                    # Continue with other vacancies even if one fails
                    continue
            
            logger.info("matchmaker_analysis_completed", successful=successful_analyses, total=len(vacancies_response))
            
            # ========================================================================
            # SORTING: Sort the final result list by 'score' (AI Score) descending
            # ========================================================================
            def sort_key(v):
                score = v.get('score')
                if score is not None:
                    return float(score)  # Sort by score (0-100) descending
                else:
                    return -1.0  # No score, sort to end
            
        # Remove duplicates by description_url before sorting (keep highest score)
        url_to_best = {}
        for v in vacancies_response:
            url = v.get('description_url', '')
            if url:
                if url not in url_to_best:
                    url_to_best[url] = v
                else:
                    # Keep the one with higher score
                    current_score = v.get('score', 0) or 0
                    existing_score = url_to_best[url].get('score', 0) or 0
                    if current_score > existing_score:
                        url_to_best[url] = v
        
        # Convert back to list
        vacancies_response = list(url_to_best.values())
        duplicates_removed = len(vacancies_response) - len(url_to_best)
        if duplicates_removed > 0:
            logger.info("chat_duplicates_removed", count=duplicates_removed, total_before=len(vacancies_response) + duplicates_removed, total_after=len(vacancies_response))
        
        # Sort all vacancies by score (descending)
        vacancies_response.sort(key=sort_key, reverse=True)
        logger.info("vacancies_sorted_by_score", final_count=len(vacancies_response))
        
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
        if not isinstance(debug_info, dict):
            debug_info = {}
        
        # Ensure friendly_reasoning from the Job Scout is passed into the final response object
        friendly_reasoning = extracted_params.get("friendly_reasoning")
        # Always include friendly_reasoning in debug_info (even if None) so UI can access it
        debug_info["friendly_reasoning"] = friendly_reasoning
        if friendly_reasoning:
            logger.info("friendly_reasoning_added_to_debug_info", friendly_reasoning=friendly_reasoning)
        else:
            logger.warning("friendly_reasoning_missing", detail="Job Scout did not return friendly_reasoning")

        logger.info("chat_search_completed", summary_length=len(summary), persona_applied=persona_applied)

        return ChatResponse(
            vacancies=vacancies_response,
            summary=summary,
            debug_info=debug_info,
            persona_applied=persona_applied  # Flag indicating if persona was successfully used
        )
        
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
