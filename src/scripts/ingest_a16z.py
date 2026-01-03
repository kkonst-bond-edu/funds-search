"""
Script to ingest all a16z job vacancies and save to JSON file.
Uses Markdown extraction + DeepSeek LLM for robust extraction.
"""

import json
import sys
import asyncio
import re
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import structlog
from src.services.firecrawl_service import FirecrawlService
from src.schemas.vacancy import Vacancy, CompanyStage
from apps.orchestrator.llm import LLMProviderFactory
from langchain_core.messages import HumanMessage

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


def parse_deepseek_json_response(response_text: str) -> List[Dict[str, Any]]:
    """
    Parse JSON from DeepSeek response.
    Handles cases where response may have markdown code blocks, extra text, or be truncated.
    
    Args:
        response_text: Raw response text from DeepSeek
        
    Returns:
        List of job dictionaries
    """
    # Try to extract JSON from markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*(\[.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON array directly
        json_match = re.search(r'(\[.*)', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find the start of a JSON array
            start_idx = response_text.find('[')
            if start_idx != -1:
                json_str = response_text[start_idx:].strip()
            else:
                json_str = response_text.strip()
    
    # Try to parse the JSON
    try:
        jobs_data = json.loads(json_str)
        if isinstance(jobs_data, list):
            return jobs_data
        elif isinstance(jobs_data, dict) and "jobs" in jobs_data:
            return jobs_data["jobs"]
        else:
            logger.warning("deepseek_unexpected_json_format", json_type=type(jobs_data).__name__)
            return []
    except json.JSONDecodeError as e:
        # If JSON is malformed, try to extract valid JSON objects from the string
        logger.warning("deepseek_json_parse_error_attempting_recovery", error=str(e), error_pos=getattr(e, 'pos', None))
        
        # Try to extract complete JSON objects from the array
        jobs_data = []
        
        # Find the start of the array
        start_idx = json_str.find('[')
        if start_idx == -1:
            start_idx = 0
        
        # Try to parse objects one by one by finding complete JSON objects
        # Use a stack-based approach to handle nested structures
        i = start_idx + 1  # Skip the opening bracket
        brace_count = 0
        bracket_count = 0  # For arrays in objects
        obj_start = None
        in_string = False
        escape_next = False
        
        while i < len(json_str):
            char = json_str[i]
            
            if escape_next:
                escape_next = False
                i += 1
                continue
            
            if char == '\\':
                escape_next = True
                i += 1
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                i += 1
                continue
            
            if not in_string:
                if char == '{':
                    if brace_count == 0:
                        obj_start = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and obj_start is not None:
                        # Found a complete object
                        try:
                            obj_str = json_str[obj_start:i+1]
                            obj = json.loads(obj_str)
                            if isinstance(obj, dict) and "title" in obj:
                                jobs_data.append(obj)
                        except json.JSONDecodeError:
                            pass
                        obj_start = None
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
            
            i += 1
        
        if jobs_data:
            logger.info("deepseek_recovered_jobs_from_partial_json", recovered_count=len(jobs_data))
            return jobs_data
        
        # Last resort: log error with full context
        logger.error(
            "deepseek_json_parse_error",
            error=str(e),
            error_position=getattr(e, 'pos', None),
            response_preview=response_text[:1000],
        )
        raise ValueError(f"Failed to parse JSON from DeepSeek response: {str(e)}") from e


def normalize_to_vacancies(jobs_data: List[Dict[str, Any]], source_url: str) -> List[Vacancy]:
    """
    Normalize job data to Vacancy objects using CompanyStage.get_stage_value.
    
    Args:
        jobs_data: List of job dictionaries from DeepSeek
        source_url: Source URL used to fetch the vacancies
        
    Returns:
        List of Vacancy objects
    """
    vacancies = []
    
    for job_data in jobs_data:
        try:
            # Handle company_stage using CompanyStage.get_stage_value
            company_stage_str = job_data.get("company_stage") or job_data.get("company_stage", "Growth")
            try:
                # Try to match enum value
                company_stage = CompanyStage(company_stage_str)
            except ValueError:
                # Try to find matching stage using get_stage_value
                stage_value = CompanyStage.get_stage_value(company_stage_str)
                try:
                    company_stage = CompanyStage(stage_value)
                except ValueError:
                    # Default to Growth if no match
                    company_stage = CompanyStage.GROWTH
                    logger.warning(
                        "vacancy_stage_normalization_failed",
                        original_stage=company_stage_str,
                        defaulted_to="Growth",
                    )

            # Determine remote option
            location = job_data.get("location", "") or job_data.get("location", "")
            remote_option = job_data.get("remote_option", False)
            if not remote_option and location:
                remote_option = "remote" in location.lower()

            vacancy = Vacancy(
                title=job_data.get("title", "Unknown Position"),
                company_name=job_data.get("company_name") or job_data.get("company", "Unknown Company"),
                company_stage=company_stage,
                location=location or "Not specified",
                industry=job_data.get("industry", "Technology"),
                salary_range=job_data.get("salary_range"),
                description_url=job_data.get("description_url") or job_data.get("url", ""),
                required_skills=job_data.get("required_skills", []) or [],
                remote_option=remote_option,
                source_url=source_url,
            )

            vacancies.append(vacancy)

        except Exception as e:
            logger.warning("vacancy_normalization_error", job_data=job_data, error=str(e))
            continue

    return vacancies


async def extract_jobs_with_deepseek(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Extract job vacancies from markdown using DeepSeek LLM.
    
    Args:
        markdown_content: Markdown content from Firecrawl
        
    Returns:
        List of job dictionaries
    """
    logger.info("deepseek_extraction_started", markdown_length=len(markdown_content))

    # Truncate markdown to fit within token limits (131k tokens max)
    # Rough estimate: 1 token ≈ 4 characters, so 131k tokens ≈ 524k chars
    # But we need to leave room for the prompt, so truncate to ~250k characters (~100k tokens)
    MAX_MARKDOWN_LENGTH = 250000
    if len(markdown_content) > MAX_MARKDOWN_LENGTH:
        logger.warning(
            "markdown_truncated",
            original_length=len(markdown_content),
            truncated_length=MAX_MARKDOWN_LENGTH,
        )
        markdown_content = markdown_content[:MAX_MARKDOWN_LENGTH]
        markdown_content += "\n\n[Content truncated due to length limits...]"

        # Build extraction prompt - emphasize complete JSON
        extraction_prompt = (
            "Below is a Markdown dump of a job board. Extract ALL job vacancies into a JSON array. "
            "For each job, identify: title, company_name (or company), location, industry, "
            "description_url (or url), required_skills (as array), company_stage, and remote_option (boolean). "
            "IMPORTANT: Return a COMPLETE, valid JSON array. Ensure all strings are properly escaped. "
            "RETURN ONLY RAW JSON array starting with [ and ending with ], no markdown code blocks, no explanations."
        )

    # Create message for DeepSeek
    messages = [
        HumanMessage(
            content=f"{extraction_prompt}\n\nMarkdown content:\n{markdown_content}"
        )
    ]

    try:
        # Get DeepSeek provider
        llm_provider = LLMProviderFactory.get_provider("deepseek")
        logger.info("deepseek_provider_initialized", provider=llm_provider.name)

        # Invoke DeepSeek
        response = await llm_provider.ainvoke(messages)
        response_text = response.content if hasattr(response, "content") else str(response)

        logger.info("deepseek_extraction_completed", response_length=len(response_text))

        # Parse JSON from response
        jobs_data = parse_deepseek_json_response(response_text)
        logger.info("deepseek_jobs_parsed", jobs_count=len(jobs_data))

        return jobs_data

    except Exception as e:
        logger.error("deepseek_extraction_failed", error=str(e), error_type=type(e).__name__)
        raise


async def main_async():
    """Main async ingestion function."""
    # Hardcoded URL for a16z remote jobs
    a16z_url = "https://jobs.a16z.com/jobs?remoteOnly=true"

    logger.info("a16z_ingestion_started", url=a16z_url)

    try:
        # Initialize Firecrawl service
        firecrawl_service = FirecrawlService()
        logger.info("firecrawl_service_initialized")

        # Get page markdown
        logger.info("fetching_markdown", url=a16z_url)
        markdown_content = firecrawl_service.get_page_markdown(a16z_url)
        logger.info("markdown_fetched", content_length=len(markdown_content))

        # Extract jobs using DeepSeek
        logger.info("extracting_jobs_with_deepseek")
        jobs_data = await extract_jobs_with_deepseek(markdown_content)

        # Normalize to Vacancy objects
        logger.info("normalizing_vacancies", jobs_count=len(jobs_data))
        vacancies = normalize_to_vacancies(jobs_data, a16z_url)
        logger.info("vacancies_normalized", vacancies_count=len(vacancies))

        # Convert to dict for JSON serialization
        vacancies_data = [vacancy.dict() for vacancy in vacancies]

        # Create data directory if it doesn't exist
        data_dir = project_root / "data"
        data_dir.mkdir(exist_ok=True)
        logger.info("data_directory_ready", path=str(data_dir))

        # Save to JSON file
        output_file = data_dir / "vacancies_dump.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(vacancies_data, f, indent=2, ensure_ascii=False)

        logger.info(
            "ingestion_completed",
            total_vacancies=len(vacancies),
            output_file=str(output_file),
        )

        print(f"\n✅ Successfully ingested {len(vacancies)} vacancies")
        print(f"📁 Saved to: {output_file}")

    except Exception as e:
        logger.error("ingestion_failed", error=str(e), error_type=type(e).__name__)
        print(f"\n❌ Ingestion failed: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point - runs async function."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

