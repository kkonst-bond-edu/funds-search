"""
Firecrawl service for fetching real vacancies from job boards.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import structlog

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from firecrawl import Firecrawl
except ImportError:
    Firecrawl = None

from src.schemas.vacancy import Vacancy, VacancyFilter, CompanyStage
from src.services.exceptions import (
    FirecrawlAuthError,
    FirecrawlAPIError,
    FirecrawlRateLimitError,
    FirecrawlConnectionError,
)

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger("INFO"),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class FirecrawlService:
    """Service for fetching vacancies using Firecrawl API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Firecrawl service.

        Args:
            api_key: Firecrawl API key. If not provided, reads from FIRECRAWL_API_KEY env var.

        Raises:
            FirecrawlAuthError: If API key is missing
        """
        if Firecrawl is None:
            raise ImportError(
                "firecrawl package is not installed. Install it with: pip install firecrawl-py"
            )

        api_key = api_key or os.getenv("FIRECRAWL_API_KEY")

        # Check if API key is not set or is empty string
        if not api_key or not api_key.strip():
            raise FirecrawlAuthError(
                "FIRECRAWL_API_KEY environment variable is required and must not be empty. "
                "Please set it in your .env file."
            )

        # Security: Never log the full API key - mask as fc-****
        masked_key = f"fc-****" if len(api_key) > 4 else "****"
        logger.info("firecrawl_service_initializing", api_key_masked=masked_key)

        try:
            self.client = Firecrawl(api_key=api_key)
            logger.info(
                "firecrawl_service_initialized", api_key_masked=masked_key, api_key_received=True
            )
        except Exception as e:
            logger.error("firecrawl_init_failed", error=str(e), error_type=type(e).__name__)
            raise FirecrawlConnectionError(
                f"Failed to initialize Firecrawl client: {str(e)}"
            ) from e

    def fetch_vacancies(self, filter_params: VacancyFilter, max_results: int = 10) -> List[Vacancy]:
        """
        Fetch vacancies from a16z jobs page using Firecrawl.

        Args:
            filter_params: VacancyFilter with search criteria
            max_results: Maximum number of vacancies to return

        Returns:
            List of Vacancy objects

        Raises:
            FirecrawlAuthError: If API key is invalid
            FirecrawlAPIError: If API request fails
            FirecrawlRateLimitError: If rate limit is exceeded
            FirecrawlConnectionError: If connection fails
        """
        jobs_url = "https://jobs.a16z.com/jobs"

        logger.info(
            "firecrawl_fetch_started",
            url=jobs_url,
            role=filter_params.role,
            max_results=max_results,
        )

        try:
            # Build extraction schema matching our Vacancy model
            extraction_schema = {
                "type": "object",
                "properties": {
                    "jobs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "company_name": {"type": "string"},
                                "company_stage": {"type": "string"},
                                "location": {"type": "string"},
                                "industry": {"type": "string"},
                                "salary_range": {"type": "string"},
                                "description_url": {"type": "string"},
                                "required_skills": {"type": "array", "items": {"type": "string"}},
                                "remote_option": {"type": "boolean"},
                            },
                            "required": ["title", "company_name", "location", "description_url"],
                        },
                    }
                },
                "required": ["jobs"],
            }

            # Build search prompt based on filters
            extraction_prompt = self._build_search_prompt(filter_params)

            # Use Firecrawl SDK v1.x/v2.x API: scrape() with formats as list of config objects
            # Timeout set to 120 seconds (120000 ms) to accommodate LLM extraction (~31 seconds)
            response = self.client.scrape(
                url=jobs_url,
                formats=[
                    {
                        "type": "json",
                        "schema": extraction_schema,
                        "prompt": extraction_prompt,
                    }
                ],
                timeout=120000,  # 120 seconds in milliseconds
            )

            # Parse response and convert to Vacancy objects
            vacancies = self._parse_firecrawl_response(response, filter_params)

            # Limit results
            if len(vacancies) > max_results:
                vacancies = vacancies[:max_results]

            logger.info(
                "firecrawl_fetch_completed", total_results=len(vacancies), max_results=max_results
            )

            return vacancies

        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__

            # Handle specific error types
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                logger.error("firecrawl_rate_limit", error=error_msg)
                raise FirecrawlRateLimitError(f"Firecrawl rate limit exceeded: {error_msg}") from e
            elif "401" in error_msg or "403" in error_msg or "unauthorized" in error_msg.lower():
                logger.error("firecrawl_auth_error", error=error_msg)
                raise FirecrawlAuthError(f"Firecrawl authentication failed: {error_msg}") from e
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                logger.error("firecrawl_connection_error", error=error_msg)
                raise FirecrawlConnectionError(f"Firecrawl connection error: {error_msg}") from e
            else:
                logger.error("firecrawl_api_error", error=error_msg, error_type=error_type)
                raise FirecrawlAPIError(f"Firecrawl API error: {error_msg}") from e

    def _build_search_prompt(self, filter_params: VacancyFilter) -> str:
        """
        Build search prompt based on filter parameters.

        Args:
            filter_params: VacancyFilter with search criteria

        Returns:
            Search prompt string
        """
        prompt_parts = ["Extract job listings from the page."]
        if filter_params.role:
            prompt_parts.append(f"Focus on roles matching: {filter_params.role}")
        if filter_params.skills:
            skills_str = ", ".join(filter_params.skills)
            prompt_parts.append(f"Required skills: {skills_str}")
        if filter_params.location:
            prompt_parts.append(f"Location: {filter_params.location}")
        if filter_params.is_remote is not None:
            remote_str = "remote" if filter_params.is_remote else "on-site"
            prompt_parts.append(f"Remote option: {remote_str}")
        if filter_params.industry:
            prompt_parts.append(f"Industry: {filter_params.industry}")

        return " ".join(prompt_parts)

    def _parse_firecrawl_response(
        self, response: Any, filter_params: VacancyFilter
    ) -> List[Vacancy]:
        """
        Parse Firecrawl SDK v1.x/v2.x response and convert to Vacancy objects.

        Args:
            response: Firecrawl API response (may be Document object or dict)
            filter_params: Filter parameters for post-processing

        Returns:
            List of Vacancy objects
        """
        vacancies = []

        # Firecrawl SDK v1.x/v2.x: Extract JSON from response
        # New SDK may return a Document object with .json attribute, or a dict
        if hasattr(response, "json"):
            # The new SDK returns a Document object with a .json attribute
            extracted_json = response.json
        elif isinstance(response, dict):
            # Fallback for dictionary-style responses
            extracted_json = response.get("json") or response.get("data", {}).get("json")
        else:
            # Try to access as attribute or use response directly
            extracted_json = (
                getattr(response, "data", response) if hasattr(response, "data") else response
            )

        if not extracted_json:
            # Fallback: try old format for compatibility
            if isinstance(response, dict):
                extracted_json = response.get("extract", {}) or response.get("data", {}) or response
            else:
                extracted_json = {}

        # Extract jobs array from the JSON
        if isinstance(extracted_json, dict):
            jobs_data = extracted_json.get("jobs", [])
        else:
            # If extracted_json is already a list or other structure
            logger.warning(
                "firecrawl_unexpected_response_format", response_type=type(extracted_json).__name__
            )
            jobs_data = []

        if not jobs_data:
            logger.warning(
                "firecrawl_no_vacancies_found",
                response_keys=list(response.keys()) if isinstance(response, dict) else "non-dict",
                extracted_json_type=type(extracted_json).__name__ if extracted_json else "None",
            )
            return vacancies

        for job_data in jobs_data:
            try:
                # Map Firecrawl data to Vacancy model
                company_stage_str = job_data.get("company_stage", "Growth")
                try:
                    company_stage = CompanyStage(company_stage_str)
                except ValueError:
                    # Default to Growth if stage doesn't match enum
                    company_stage = CompanyStage.GROWTH

                # Determine remote option
                location = job_data.get("location", "")
                remote_option = job_data.get("remote_option", False)
                if not remote_option and location:
                    remote_option = "remote" in location.lower()

                vacancy = Vacancy(
                    title=job_data.get("title", "Unknown Position"),
                    company_name=job_data.get("company_name", "Unknown Company"),
                    company_stage=company_stage,
                    location=location or "Not specified",
                    industry=job_data.get("industry", "Technology"),
                    salary_range=job_data.get("salary_range"),
                    description_url=job_data.get("description_url", ""),
                    required_skills=job_data.get("required_skills", []),
                    remote_option=remote_option,
                )

                vacancies.append(vacancy)

            except Exception as e:
                logger.warning("firecrawl_vacancy_parse_error", job_data=job_data, error=str(e))
                continue

        return vacancies
