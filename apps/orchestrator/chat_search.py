"""
Conversational Vacancy Search Agent.

Uses LLM to interpret natural language messages and extract search parameters,
then generates friendly summaries of search results.
"""

import json
import logging
from typing import Any, Dict, List
import structlog
from langchain_core.messages import HumanMessage, SystemMessage

from apps.orchestrator.llm import LLMProviderFactory
from src.schemas.vacancy import Vacancy

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


class ChatSearchAgent:
    """
    AI agent that converts natural language chat messages into search queries
    and generates friendly summaries of search results.
    """

    def __init__(self):
        """Initialize the chat search agent with DeepSeek LLM provider."""
        self.llm_provider = LLMProviderFactory.get_provider("deepseek")
        logger.info("chat_search_agent_initialized", provider=self.llm_provider.name)

    async def interpret_message(self, user_input: str) -> Dict[str, Any]:
        """
        Extract search parameters from natural language user input.

        Args:
            user_input: Natural language message from the user

        Returns:
            Dictionary with extracted parameters:
            - role: Optional[str] - Job role/title
            - skills: Optional[List[str]] - Required skills
            - industry: Optional[str] - Industry sector
            - location: Optional[str] - Job location
            - company_stage: Optional[str] - Company funding stage

            Missing fields will be None.
        """
        logger.info("interpreting_user_message", user_input_length=len(user_input))

        system_prompt = """You are a helpful assistant that extracts job search parameters from natural language messages.

Extract the following information from the user's message:
- role: Job title or role (e.g., "Software Engineer", "Python Developer", "Backend Engineer")
- skills: List of technical skills mentioned (e.g., ["Python", "Go", "React"])
- industry: Industry sector (e.g., "Fintech", "AI", "Healthcare")
- location: Job location (e.g., "San Francisco", "Remote", "New York")
- company_stage: Company funding stage (e.g., "Seed", "Series A", "Growth")

If a field is not mentioned in the user's message, set it to null.

Return ONLY a valid JSON object with this exact structure:
{
  "role": "string or null",
  "skills": ["string"] or null,
  "industry": "string or null",
  "location": "string or null",
  "company_stage": "string or null"
}

Do not include any explanation or additional text, only the JSON object."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]

            response = await self.llm_provider.ainvoke(messages)
            response_text = response.content.strip()

            # Try to extract JSON from the response (in case LLM adds extra text)
            # Look for JSON object in the response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1

            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
            else:
                json_str = response_text

            # Parse JSON response
            extracted_params = json.loads(json_str)

            # Normalize the response to ensure all expected fields exist
            result = {
                "role": extracted_params.get("role"),
                "skills": extracted_params.get("skills"),
                "industry": extracted_params.get("industry"),
                "location": extracted_params.get("location"),
                "company_stage": extracted_params.get("company_stage"),
            }

            # Convert empty strings to None
            for key in result:
                if result[key] == "":
                    result[key] = None
                elif key == "skills" and result[key] is not None and len(result[key]) == 0:
                    result[key] = None

            logger.info(
                "message_interpreted",
                role=result["role"],
                skills=result["skills"],
                industry=result["industry"],
                location=result["location"],
                company_stage=result["company_stage"],
            )

            return result

        except json.JSONDecodeError as e:
            logger.error("json_parse_error", error=str(e), response=response_text if 'response_text' in locals() else None)
            # Return empty result on parse error
            return {
                "role": None,
                "skills": None,
                "industry": None,
                "location": None,
                "company_stage": None,
            }
        except Exception as e:
            logger.error("interpretation_error", error=str(e), error_type=type(e).__name__)
            raise

    async def format_results_summary(
        self, vacancies: List[Vacancy], user_input: str
    ) -> str:
        """
        Generate a friendly response summarizing why these specific vacancies were found.

        Args:
            vacancies: List of Vacancy objects found in the search
            user_input: Original user message

        Returns:
            Friendly summary string explaining the search results
        """
        logger.info("formatting_results_summary", vacancy_count=len(vacancies))

        if not vacancies:
            return "I couldn't find any vacancies matching your criteria. Try adjusting your search parameters or check back later for new opportunities!"

        # Build a summary of the vacancies
        vacancy_summaries = []
        for i, vacancy in enumerate(vacancies[:5], 1):  # Limit to top 5 for summary
            skills_str = ", ".join(vacancy.required_skills[:3]) if vacancy.required_skills else "various skills"
            if len(vacancy.required_skills) > 3:
                skills_str += "..."

            summary = (
                f"{i}. {vacancy.title} at {vacancy.company_name} "
                f"({vacancy.company_stage.value if hasattr(vacancy.company_stage, 'value') else vacancy.company_stage}) "
                f"in {vacancy.location} - {vacancy.industry} industry. "
                f"Key skills: {skills_str}."
            )
            vacancy_summaries.append(summary)

        vacancies_text = "\n".join(vacancy_summaries)

        if len(vacancies) > 5:
            vacancies_text += f"\n\n...and {len(vacancies) - 5} more opportunities."

        system_prompt = """You are a helpful assistant that explains job search results in a friendly, conversational way.

Based on the user's original query and the vacancies found, explain why these specific opportunities match their search.

Be concise (2-3 sentences), friendly, and highlight the key matches (skills, industry, location, company stage).

Do not make up information that wasn't provided."""

        user_prompt = f"""User's original query: "{user_input}"

Found {len(vacancies)} matching vacancies:

{vacancies_text}

Generate a friendly 2-3 sentence summary explaining why these vacancies match the user's search."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = await self.llm_provider.ainvoke(messages)
            summary = response.content.strip()

            logger.info("results_summary_generated", summary_length=len(summary))

            return summary

        except Exception as e:
            logger.error("summary_generation_error", error=str(e), error_type=type(e).__name__)
            # Fallback to a simple summary
            return (
                f"I found {len(vacancies)} vacancy(ies) matching your search. "
                f"Here are the top opportunities that align with your criteria."
            )
