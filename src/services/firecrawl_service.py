"""
Firecrawl service for fetching real vacancies from job boards.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
from urllib.parse import quote_plus, quote
import structlog

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from firecrawl import Firecrawl
    from firecrawl.v2.types import JsonFormat
except ImportError:
    Firecrawl = None
    JsonFormat = None

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

    # Mapping from VacancyFilter fields to a16z URL parameters
    A16Z_FILTER_MAP = {
        "role": "jobTypes",
        "industry": "markets",
        "location": "locations",
        "is_remote": "remoteOnly",
        "company_stages": "stages",
    }

    # Industry name normalization mapping
    INDUSTRY_NORMALIZATION = {
        "bio + health": "Bio + Health",
        "bio+health": "Bio + Health",
        "bio health": "Bio + Health",
        "bio & health": "Bio + Health",
        "fintech": "Fintech",
        "finteck": "Fintech",
    }

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

    def build_a16z_url(self, filter_params: VacancyFilter) -> str:
        """
        Build a16z jobs URL with query parameters matching their filtering system.

        Args:
            filter_params: VacancyFilter with search criteria

        Returns:
            Complete URL with query parameters
        """
        base_url = "https://jobs.a16z.com/jobs"
        params = []

        # Handle is_remote specially (boolean -> remoteOnly=true)
        if filter_params.is_remote is True:
            params.append("remoteOnly=true")

        # Handle role -> jobTypes (using A16Z_FILTER_MAP)
        if filter_params.role:
            job_type = quote_plus(filter_params.role)
            params.append(f"{self.A16Z_FILTER_MAP['role']}={job_type}")

        # Handle location -> locations (using A16Z_FILTER_MAP)
        if filter_params.location:
            location_encoded = quote_plus(filter_params.location)
            params.append(f"{self.A16Z_FILTER_MAP['location']}={location_encoded}")

        # Handle industry -> markets (using A16Z_FILTER_MAP, with normalization)
        if filter_params.industry:
            # Normalize industry name using INDUSTRY_NORMALIZATION
            industry_val = self.INDUSTRY_NORMALIZATION.get(
                filter_params.industry.lower(), filter_params.industry
            )
            # Use quote() instead of quote_plus() to preserve + as %2B for "Bio + Health"
            industry_encoded = quote(industry_val)
            params.append(f"{self.A16Z_FILTER_MAP['industry']}={industry_encoded}")

        # Handle company_stages -> stages (using A16Z_FILTER_MAP, multiple parameters)
        if filter_params.company_stages:
            for stage in filter_params.company_stages:
                stage_value = CompanyStage.get_stage_value(stage)
                # Map our enum values to a16z URL format
                # Always use quote_plus to handle spaces and special characters correctly
                if stage_value == "Seed":
                    params.append(f"{self.A16Z_FILTER_MAP['company_stages']}=Seed")
                elif stage_value == "Series A":
                    params.append(
                        f"{self.A16Z_FILTER_MAP['company_stages']}={quote_plus('Series A')}"
                    )  # Becomes "Series+A"
                elif stage_value == "Growth (Series B or later)":
                    params.append(
                        f"{self.A16Z_FILTER_MAP['company_stages']}={quote_plus('Growth (Series B or later)')}"
                    )
                elif stage_value == "1-10 employees":
                    params.append(f"{self.A16Z_FILTER_MAP['company_stages']}=1-10+employees")
                elif stage_value == "10-100 employees":
                    params.append(f"{self.A16Z_FILTER_MAP['company_stages']}=10-100+employees")
                else:
                    # Fallback: URL encode the stage value
                    params.append(
                        f"{self.A16Z_FILTER_MAP['company_stages']}={quote_plus(stage_value)}"
                    )

        # Handle skills -> skills (multiple parameters, not in A16Z_FILTER_MAP but handled separately)
        if filter_params.skills:
            for skill in filter_params.skills:
                skill_encoded = quote_plus(skill.strip())
                params.append(f"skills={skill_encoded}")

        # Build final URL
        if params:
            url = f"{base_url}?{'&'.join(params)}"
        else:
            url = base_url

        logger.info("a16z_url_built", url=url, filters=filter_params.dict())
        return url

    def _normalize_industry_name(self, industry: str) -> str:
        """
        Normalize industry names to match a16z format using INDUSTRY_NORMALIZATION.

        Common normalizations:
        - "Finteck" or "fintech" -> "Fintech"
        - "bio+health" or "bio health" -> "Bio + Health"
        - Case-insensitive matching for common industries

        Args:
            industry: Raw industry name from user input

        Returns:
            Normalized industry name
        """
        if not industry:
            return industry

        industry_lower = industry.strip().lower()

        # Check if we have a normalization mapping in INDUSTRY_NORMALIZATION
        if industry_lower in self.INDUSTRY_NORMALIZATION:
            return self.INDUSTRY_NORMALIZATION[industry_lower]

        # Additional common industry normalizations (fallback)
        fallback_map = {
            "logistics": "Logistics",
            "healthcare": "Healthcare",
            "enterprise": "Enterprise",
            "consumer": "Consumer",
            "crypto": "Crypto",
            "cryptocurrency": "Crypto",
            "ai": "AI",
            "artificial intelligence": "AI",
            "ml": "ML",
            "machine learning": "ML",
        }

        if industry_lower in fallback_map:
            return fallback_map[industry_lower]

        # If no mapping found, capitalize first letter of each word
        # This handles cases like "fintech" -> "Fintech" generically
        words = industry.strip().split()
        normalized = " ".join(word.capitalize() for word in words)
        return normalized

    def fetch_vacancies(
        self, filter_params: VacancyFilter, max_results: int = 100
    ) -> List[Vacancy]:
        """
        Fetch vacancies from a16z jobs page using Firecrawl with targeted URL filtering.

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
        # Build targeted a16z URL with filters applied
        jobs_url = self.build_a16z_url(filter_params)

        logger.info(
            "firecrawl_fetch_started",
            url=jobs_url,
            role=filter_params.role,
            is_remote=filter_params.is_remote,
            company_stages=[CompanyStage.get_stage_value(s) for s in filter_params.company_stages]
            if filter_params.company_stages
            else None,
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

            # Build strict extraction prompt for pre-filtered a16z page
            # The URL already contains the filters, so we just need to extract visible listings
            extraction_prompt = (
                "Extract EVERY job vacancy from the page. I expect 30 results and truncation is NOT allowed. "
                "Do NOT stop after 10. Extract all of them into a single list. "
                "Collect all titles, companies, and links. "
                "The page has been scrolled multiple times to load all content. Extract ALL job cards visible, "
                "including those that appear after scrolling. "
                "For each card, extract title, company_name, location, salary_range, required_skills, "
                "company_stage, description_url, and remote_option information."
            )

            # Log the URL before making the request
            logger.info("firecrawl_request_url", url=jobs_url)

            # Build JSON format with schema and prompt
            json_format = JsonFormat(
                type="json",
                schema=extraction_schema,
                prompt=extraction_prompt,
            )

            # Build scrape parameters dictionary for SDK v4
            # Use **scrape_params to unpack dictionary as keyword arguments
            scrape_params = {
                "formats": [json_format],
                "wait_for": 3000,  # Wait 3 seconds for JS to load all cards (snake_case)
                "actions": [
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                ],
                "timeout": 120000,  # 120 seconds to accommodate LLM extraction
            }

            # Use Firecrawl SDK v4: scrape() with **scrape_params to unpack dictionary
            response = self.client.scrape(jobs_url, **scrape_params)

            # Parse response and convert to Vacancy objects
            vacancies = self._parse_firecrawl_response(response, filter_params, jobs_url)

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
        self, response: Any, filter_params: VacancyFilter, source_url: str
    ) -> List[Vacancy]:
        """
        Parse Firecrawl SDK v4 response and convert to Vacancy objects.

        Args:
            response: Firecrawl API response (may be Document object with .data attribute or dict)
            filter_params: Filter parameters for post-processing
            source_url: Source URL used to fetch the vacancies

        Returns:
            List of Vacancy objects
        """
        vacancies = []

        # Firecrawl SDK v4: Extract JSON from response
        # SDK v4 returns a Document object (Pydantic model) with .json attribute containing extracted data
        logger.debug("firecrawl_response_type", response_type=type(response).__name__)
        
        # Convert Document to dict to access fields
        if hasattr(response, "model_dump"):
            response_dict = response.model_dump()
        elif hasattr(response, "dict"):
            response_dict = response.dict()
        elif isinstance(response, dict):
            response_dict = response
        else:
            response_dict = {}
        
        # Check for warnings in the response
        if isinstance(response_dict, dict) and response_dict.get("warning"):
            logger.warning("firecrawl_response_warning", warning=response_dict.get("warning"))
        
        # The JSON extraction result is in response_dict["json"]
        # This should be a dict containing the extracted data matching our schema
        json_data = response_dict.get("json") if isinstance(response_dict, dict) else None
        
        if json_data is not None:
            # json_data is the extracted JSON object (should be a dict)
            extracted_json = json_data if isinstance(json_data, dict) else {}
            logger.debug("firecrawl_json_extracted", json_type=type(json_data).__name__, has_data=bool(extracted_json))
        else:
            # JSON extraction returned None - extraction may have failed
            logger.warning(
                "firecrawl_json_extraction_failed",
                response_keys=list(response_dict.keys()) if isinstance(response_dict, dict) else "not-dict",
            )
            extracted_json = {}

        # Extract jobs array from the JSON
        if isinstance(extracted_json, dict):
            logger.debug("firecrawl_extracted_json_keys", keys=list(extracted_json.keys()))
            jobs_data = extracted_json.get("jobs", [])
            # Also check for other possible keys
            if not jobs_data:
                jobs_data = extracted_json.get("job", [])
            if not jobs_data and isinstance(extracted_json.get("data"), list):
                jobs_data = extracted_json.get("data", [])
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
                extracted_json_keys=list(extracted_json.keys()) if isinstance(extracted_json, dict) else "not-dict",
                extracted_json_preview=str(extracted_json)[:500] if extracted_json else "None",
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
                    source_url=source_url,
                )

                vacancies.append(vacancy)

            except Exception as e:
                logger.warning("firecrawl_vacancy_parse_error", job_data=job_data, error=str(e))
                continue

        return vacancies

    def get_page_markdown(self, url: str) -> str:
        """
        Get page content as Markdown using Firecrawl.
        
        Args:
            url: URL to scrape
            
        Returns:
            Markdown content of the page
            
        Raises:
            FirecrawlAuthError: If API key is invalid
            FirecrawlAPIError: If API request fails
            FirecrawlRateLimitError: If rate limit is exceeded
            FirecrawlConnectionError: If connection fails
        """
        logger.info("firecrawl_markdown_scrape_started", url=url)

        try:
            # Build scrape parameters for markdown extraction
            # Include scrolling actions to load all content
            scrape_params = {
                "formats": ["markdown"],
                "wait_for": 3000,  # Wait 3 seconds for JS to load
                "actions": [
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                ],
                "timeout": 120000,  # 120 seconds timeout
            }

            # Use Firecrawl SDK v4: scrape() with **scrape_params to unpack dictionary
            response = self.client.scrape(url, **scrape_params)

            # Extract markdown from response
            if hasattr(response, "model_dump"):
                response_dict = response.model_dump()
            elif hasattr(response, "dict"):
                response_dict = response.dict()
            elif isinstance(response, dict):
                response_dict = response
            else:
                response_dict = {}

            markdown_content = response_dict.get("markdown") if isinstance(response_dict, dict) else None

            if not markdown_content or not markdown_content.strip():
                logger.critical(
                    "firecrawl_empty_markdown",
                    url=url,
                    response_keys=list(response_dict.keys()) if isinstance(response_dict, dict) else "not-dict",
                )
                raise FirecrawlAPIError(f"Firecrawl returned empty markdown for URL: {url}")

            logger.info("firecrawl_markdown_scrape_completed", url=url, markdown_length=len(markdown_content))
            return markdown_content

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

    def scrape_url_with_custom_prompt(
        self, url: str, extraction_prompt: str, max_results: int = 100
    ) -> List[Vacancy]:
        """
        Scrape a URL directly with a custom extraction prompt.
        Useful for data ingestion scripts that need specific extraction instructions.

        Args:
            url: URL to scrape
            extraction_prompt: Custom prompt for extraction
            max_results: Maximum number of vacancies to return

        Returns:
            List of Vacancy objects

        Raises:
            FirecrawlAuthError: If API key is invalid
            FirecrawlAPIError: If API request fails
            FirecrawlRateLimitError: If rate limit is exceeded
            FirecrawlConnectionError: If connection fails
        """
        logger.info("firecrawl_custom_scrape_started", url=url, max_results=max_results)

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

            # Build JSON format with schema and prompt
            json_format = JsonFormat(
                type="json",
                schema=extraction_schema,
                prompt=extraction_prompt,
            )

            # Build scrape parameters dictionary for SDK v4
            # Use **scrape_params to unpack dictionary as keyword arguments
            scrape_params = {
                "formats": [json_format],
                "wait_for": 3000,  # Wait 3 seconds for JS to load all cards (snake_case)
                "actions": [
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                    {"type": "scroll", "direction": "down", "amount": 1000},
                    {"type": "wait", "milliseconds": 1000},
                ],
                "timeout": 120000,  # 120 seconds to accommodate LLM extraction
            }

            # Use Firecrawl SDK v4: scrape() with **scrape_params to unpack dictionary
            response = self.client.scrape(url, **scrape_params)

            # Create a dummy filter for parsing (not used for filtering in this case)
            dummy_filter = VacancyFilter()

            # Parse response and convert to Vacancy objects
            vacancies = self._parse_firecrawl_response(response, dummy_filter, url)

            # Limit results
            if len(vacancies) > max_results:
                vacancies = vacancies[:max_results]

            logger.info(
                "firecrawl_custom_scrape_completed",
                total_results=len(vacancies),
                max_results=max_results,
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
