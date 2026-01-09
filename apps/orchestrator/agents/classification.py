"""
Classification Agent - Categorizes job vacancies according to a16z taxonomy.

This agent analyzes job vacancies and classifies them by category, industry,
experience level, and remote work options.
"""

import json
import logging
from typing import Dict, Any
import structlog
from langchain_core.messages import HumanMessage

from apps.orchestrator.agents.base import BaseAgent

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


class ClassificationAgent(BaseAgent):
    """
    Classification Agent - Categorizes job vacancies according to a16z taxonomy.

    This agent uses the classification_agent configuration from agents.yaml
    and the classification.txt prompt.
    """

    def __init__(self):
        """Initialize the Classification agent with configuration from agents.yaml."""
        super().__init__(agent_name="classification_agent")
        logger.info("classification_agent_initialized", agent_name="classification_agent")

    def _clean_json_response(self, text: str) -> str:
        """
        Clean JSON response by removing markdown code blocks and whitespace.
        
        LLMs often wrap JSON in triple backticks (```json ... ```).
        This helper method strips these markers and any leading/trailing whitespace.
        
        Args:
            text: Raw response text that may contain markdown code blocks
            
        Returns:
            Cleaned text ready for json.loads()
        """
        # Simple and robust cleaning: strip whitespace and remove markdown code blocks
        cleaned = text.strip().replace("```json", "").replace("```", "").strip()
        return cleaned

    async def classify(self, title: str, description: str) -> Dict[str, Any]:
        """
        Classify a job vacancy by category, industry, experience level, and remote option.

        Args:
            title: Job vacancy title
            description: Job vacancy description

        Returns:
            Dictionary with fields:
            - category: str - One of the standard function categories
            - industry: str - One of the industry sectors
            - experience_level: str - Seniority level
            - remote_option: bool - Whether remote work is available
            - required_skills: List[str] - List of skills from the taxonomy
        """
        logger.info(
            "classifying_vacancy",
            agent_name=self.agent_name,
            title=title[:100] if title else None,  # Log first 100 chars
            description_length=len(description) if description else 0,
        )

        try:
            # Load the prompt template
            prompt_template = self._load_prompt(self.prompt_file)
            
            # Format the prompt with vacancy information
            formatted_prompt = f"{prompt_template}\n\nVacancy Title: {title}\n\nVacancy Description:\n{description}"
            
            # Create the user message
            messages = [HumanMessage(content=formatted_prompt)]
            
            # Invoke the LLM (this will use the system prompt from classification.txt)
            # We pass None as system_prompt to use the default from the file
            response = await self.invoke(messages, system_prompt=None)
            
            # Extract the response content
            response_text = response.content if hasattr(response, "content") else str(response)
            
            logger.debug("classification_response_received", response_length=len(response_text))
            
            # Clean the JSON response (remove markdown code blocks if present)
            cleaned_response = self._clean_json_response(response_text)
            
            # Parse JSON
            try:
                classification_result = json.loads(cleaned_response)
                
                # Validate and extract required fields with defaults
                result = {
                    "category": classification_result.get("category", "Other"),
                    "industry": classification_result.get("industry", "Other"),
                    "experience_level": classification_result.get("experience_level", "Other"),
                    "remote_option": classification_result.get("remote_option", False),
                    "required_skills": classification_result.get("required_skills", []),
                }
                
                # Ensure remote_option is boolean
                if not isinstance(result["remote_option"], bool):
                    result["remote_option"] = str(result["remote_option"]).lower() in ("true", "1", "yes")
                
                # Ensure required_skills is a list
                if not isinstance(result["required_skills"], list):
                    if isinstance(result["required_skills"], str):
                        # Try to parse as comma-separated string
                        result["required_skills"] = [s.strip() for s in result["required_skills"].split(",") if s.strip()]
                    else:
                        result["required_skills"] = []
                
                logger.info(
                    "classification_successful",
                    category=result["category"],
                    industry=result["industry"],
                    experience_level=result["experience_level"],
                    remote_option=result["remote_option"],
                    required_skills=result["required_skills"],
                )
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(
                    "json_parse_error",
                    error=str(e),
                    response_text=response_text[:500],  # Log first 500 chars for debugging
                )
                # Return default values on JSON parsing failure
                return {
                    "category": "Other",
                    "industry": "Other",
                    "experience_level": "Other",
                    "remote_option": False,
                    "required_skills": [],
                }
                
        except Exception as e:
            logger.error(
                "classification_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return default values on any error to keep pipeline running
            return {
                "category": "Other",
                "industry": "Other",
                "experience_level": "Other",
                "remote_option": False,
                "required_skills": [],
            }
