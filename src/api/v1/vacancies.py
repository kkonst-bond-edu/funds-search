"""
Vacancy Search Service API endpoints.
Supports both mock data and Firecrawl-based real search.
"""

import logging
import sys
from pathlib import Path

import structlog
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schemas.vacancy import VacancyFilter, Vacancy, CompanyStage
from src.services.firecrawl_service import FirecrawlService
from src.services.exceptions import (
    FirecrawlAuthError,
    FirecrawlAPIError,
    FirecrawlRateLimitError,
    FirecrawlConnectionError,
)

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

    Args:
        vacancies: List of vacancies to filter
        filter_params: Filter criteria

    Returns:
        Filtered list of vacancies
    """
    filtered = vacancies

    if filter_params.role:
        role_lower = filter_params.role.lower()
        filtered = [v for v in filtered if role_lower in v.title.lower()]

    if filter_params.skills:
        skill_set = {skill.lower() for skill in filter_params.skills}
        filtered = [
            v for v in filtered if any(skill.lower() in skill_set for skill in v.required_skills)
        ]

    if filter_params.location:
        location_lower = filter_params.location.lower()
        filtered = [
            v
            for v in filtered
            if location_lower in v.location.lower()
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


@router.post("/search", response_model=list[Vacancy])
async def search_vacancies(
    filter_params: VacancyFilter,
    use_firecrawl: bool = Query(
        False, description="Use Firecrawl for real search instead of mock data"
    ),
) -> list[Vacancy]:
    """
    Search for vacancies based on filter criteria.

    Supports two modes:
    - Mock mode (default): Returns realistic mock vacancies for testing
    - Firecrawl mode: Fetches real vacancies from a16z jobs page using Firecrawl

    Args:
        filter_params: VacancyFilter with search criteria
        use_firecrawl: If True, use Firecrawl for real search (requires FIRECRAWL_API_KEY)

    Returns:
        List of Vacancy objects matching the filter criteria

    Raises:
        HTTPException: If search fails
    """
    try:
        # Log search request (never log API keys or sensitive data)
        logger.info(
            "vacancy_search_requested",
            role=filter_params.role,
            skills=filter_params.skills,
            location=filter_params.location,
            is_remote=filter_params.is_remote,
            company_stages=[CompanyStage.get_stage_value(s) for s in filter_params.company_stages]
            if filter_params.company_stages
            else None,
            industry=filter_params.industry,
            min_salary=filter_params.min_salary,
            use_firecrawl=use_firecrawl,
        )

        if use_firecrawl:
            # Use Firecrawl for real search - DO NOT fall back to mock on failure
            try:
                firecrawl_service = get_firecrawl_service()
                vacancies = firecrawl_service.fetch_vacancies(filter_params, max_results=10)

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
        else:
            # Use mock data
            all_vacancies = get_mock_vacancies()
            filtered_vacancies = filter_vacancies(all_vacancies, filter_params)

            logger.info(
                "vacancy_search_completed", total_results=len(filtered_vacancies), source="mock"
            )

            return filtered_vacancies

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("vacancy_search_error", error=str(e), error_type=type(e).__name__)
        raise HTTPException(
            status_code=500, detail=f"Error performing vacancy search: {str(e)}"
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
