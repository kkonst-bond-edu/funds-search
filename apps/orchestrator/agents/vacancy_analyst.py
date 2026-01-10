"""
Vacancy Analyst - Analyzes and enriches job vacancies.

This agent performs classification (category, industry, seniority) and 
deep enrichment (segmentation, entity extraction, summarization) of job vacancies.
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


class VacancyAnalyst(BaseAgent):
    """
    Vacancy Analyst - Analyzes and enriches job vacancies.

    This agent uses the vacancy_analyst configuration from agents.yaml
    and the classification.txt / enrichment.txt prompts.
    """

    def __init__(self):
        """Initialize the Vacancy Analyst with configuration from agents.yaml."""
        super().__init__(agent_name="vacancy_analyst")
        logger.info("vacancy_analyst_initialized", agent_name="vacancy_analyst")

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
        if not text:
            return ""
        
        # Simple and robust cleaning: strip whitespace and remove markdown code blocks
        cleaned = text.strip()
        
        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        if "```" in cleaned:
            # Try to extract JSON from code blocks
            import re
            # Match ```json ... ``` or ``` ... ```
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)
            else:
                # Fallback: just remove ``` markers
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        # Remove leading/trailing whitespace and newlines
        cleaned = cleaned.strip()
        
        # Try to find JSON object if there's extra text around it
        if not cleaned.startswith("{"):
            import re
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
        
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
            # Load the prompt template (default from config, usually classification.txt)
            prompt_template = self._load_prompt(self.prompt_file)
            
            # Format the prompt with vacancy information
            formatted_prompt = f"{prompt_template}\n\nVacancy Title: {title}\n\nVacancy Description:\n{description}"
            
            # Create the user message
            messages = [HumanMessage(content=formatted_prompt)]
            
            # Invoke the LLM
            # We pass None as system_prompt to use the default from the file if configured
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

    async def enrich(self, title: str, description: str, required_skills: list = None) -> Dict[str, Any]:
        """
        Enrich a job vacancy with block-based segmentation and extracted entities.
        
        This performs a single LLM call that returns the complete enriched JSON structure
        including blocks, extracted entities, evidence_map, ai_ready_views, and normalization_warnings.

        Args:
            title: Job vacancy title
            description: Job vacancy description
            required_skills: Optional list of required skills from scraper/API to merge with extracted skills

        Returns:
            Dictionary with enriched structure:
            - blocks: Block-based segmentation (META, CONTEXT, WORK, FIT, OFFER)
            - extracted: Extracted entities (role, company, offer, constraints)
            - evidence_map: Mapping of fields to source evidence quotes
            - ai_ready_views: Compacted summaries (role_profile_text, company_profile_text)
            - normalization_warnings: List of data quality warnings
        """
        logger.info(
            "enriching_vacancy",
            agent_name=self.agent_name,
            title=title[:100] if title else None,
            description_length=len(description) if description else 0,
        )

        try:
            # Load the enrichment prompt template
            enrichment_prompt = self._load_prompt("enrichment.txt")
            
            # Build the input prompt with vacancy information
            # Include required_skills if provided for merging
            skills_context = ""
            if required_skills:
                skills_context = f"\n\nExisting required_skills from scraper: {json.dumps(required_skills)}"
            
            formatted_prompt = (
                f"{enrichment_prompt}\n\n"
                f"Vacancy Title: {title}\n\n"
                f"Vacancy Description:\n{description}"
                f"{skills_context}"
            )
            
            # Create the user message
            messages = [HumanMessage(content=formatted_prompt)]
            
            # Invoke the LLM with enrichment prompt
            # Use max_tokens to ensure we get a complete response (enrichment can be large)
            response = await self.invoke(messages, system_prompt=enrichment_prompt, max_tokens=8000)
            
            # Extract the response content
            response_text = response.content if hasattr(response, "content") else str(response)
            
            logger.debug("enrichment_response_received", response_length=len(response_text))
            
            # Clean the JSON response (remove markdown code blocks if present)
            cleaned_response = self._clean_json_response(response_text)
            
            # Parse JSON
            try:
                enrichment_result = json.loads(cleaned_response)
                
                # Validate structure and ensure all required fields exist with defaults
                result = {
                    "blocks": enrichment_result.get("blocks"),
                    "extracted": enrichment_result.get("extracted"),
                    "evidence_map": enrichment_result.get("evidence_map", {}),
                    "ai_ready_views": enrichment_result.get("ai_ready_views"),
                    "normalization_warnings": enrichment_result.get("normalization_warnings", []),
                }
                
                # Ensure evidence_map is a dict
                if not isinstance(result["evidence_map"], dict):
                    result["evidence_map"] = {}
                
                # Ensure normalization_warnings is a list
                if not isinstance(result["normalization_warnings"], list):
                    result["normalization_warnings"] = []
                
                logger.info(
                    "enrichment_successful",
                    has_blocks=result["blocks"] is not None,
                    has_extracted=result["extracted"] is not None,
                    warnings_count=len(result["normalization_warnings"]),
                )
                
                return result
                
            except json.JSONDecodeError as e:
                # Log detailed error information for debugging
                error_position = getattr(e, 'pos', None)
                error_line = getattr(e, 'lineno', None)
                error_col = getattr(e, 'colno', None)
                
                logger.error(
                    "enrichment_json_parse_error",
                    error=str(e),
                    error_position=error_position,
                    error_line=error_line,
                    error_col=error_col,
                    response_length=len(response_text),
                    cleaned_length=len(cleaned_response),
                    response_preview=response_text[:500],  # First 500 chars
                    cleaned_preview=cleaned_response[:500],  # First 500 chars of cleaned
                    response_suffix=response_text[-500:] if len(response_text) > 500 else "",  # Last 500 chars
                )
                
                # Try to extract partial JSON if possible
                try:
                    # Try to find and extract just the blocks or extracted part
                    import re
                    # Look for blocks: {...}
                    blocks_match = re.search(r'"blocks"\s*:\s*(\{[^}]*\})', cleaned_response, re.DOTALL)
                    extracted_match = re.search(r'"extracted"\s*:\s*(\{[^}]*\})', cleaned_response, re.DOTALL)
                    
                    if blocks_match or extracted_match:
                        logger.warning("Found partial JSON structure, but full parse failed")
                except Exception:
                    pass
                
                # Return empty structure on JSON parsing failure
                return {
                    "blocks": None,
                    "extracted": None,
                    "evidence_map": {},
                    "ai_ready_views": None,
                    "normalization_warnings": [
                        f"Failed to parse enrichment JSON from LLM: {str(e)}"
                        + (f" (position {error_position})" if error_position else "")
                    ],
                }
                
        except Exception as e:
            logger.error(
                "enrichment_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return empty structure on any error to keep pipeline running
            return {
                "blocks": None,
                "extracted": None,
                "evidence_map": {},
                "ai_ready_views": None,
                "normalization_warnings": [f"Enrichment error: {str(e)}"],
            }
