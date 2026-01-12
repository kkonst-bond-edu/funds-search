"""
Job Scout Agent - Generates search parameters from user persona.

This agent specializes in converting user persona preferences into structured
search parameters (semantic query + metadata filters) for Pinecone vector search.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
import structlog
from langchain_core.messages import HumanMessage

from apps.orchestrator.agents.base import BaseAgent
from shared.schemas import UserPersona
from src.schemas.vacancy import VacancyFilter, CompanyStage

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


class JobScoutAgent(BaseAgent):
    """
    Job Scout Agent - Converts user persona to search parameters.

    This agent uses the job_scout configuration from agents.yaml and the
    job_scout.txt prompt to generate semantic queries and metadata filters
    from user persona preferences.
    """

    def __init__(self):
        """Initialize the Job Scout agent with configuration from agents.yaml."""
        super().__init__(agent_name="job_scout")
        logger.info("job_scout_agent_initialized", agent_name="job_scout")

    def _clean_json_response(self, text: str) -> str:
        """
        Clean JSON response by removing markdown code blocks and whitespace.
        
        Args:
            text: Raw response text that may contain markdown code blocks
            
        Returns:
            Cleaned text ready for json.loads()
        """
        if not text:
            return "{}"
        
        cleaned = text.strip()
        
        # Remove markdown code blocks
        if "```" in cleaned:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1)
            else:
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        cleaned = cleaned.strip()
        
        # Try to find JSON object if there's extra text around it
        if not cleaned.startswith("{"):
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
        
        return cleaned

    def _persona_to_dict(self, persona: Any) -> Dict[str, Any]:
        """
        Convert persona to dictionary format.
        
        Args:
            persona: UserPersona object, dict, or None
            
        Returns:
            Dictionary representation of persona
        """
        if persona is None:
            return {}
        
        if isinstance(persona, UserPersona):
            return persona.model_dump(exclude_none=True)
        
        if isinstance(persona, dict):
            return persona
        
        logger.warning("unexpected_persona_type", persona_type=type(persona).__name__)
        return {}

    def _format_persona_for_prompt(self, persona_dict: Dict[str, Any]) -> str:
        """
        Format persona dictionary into a readable text format for the prompt.
        
        Args:
            persona_dict: Persona dictionary
            
        Returns:
            Formatted persona text
        """
        parts = []
        
        if persona_dict.get("technical_skills"):
            skills = persona_dict["technical_skills"]
            if isinstance(skills, list):
                parts.append(f"Technical Skills: {', '.join(skills)}")
            else:
                parts.append(f"Technical Skills: {skills}")
        
        if persona_dict.get("career_goals"):
            goals = persona_dict["career_goals"]
            if isinstance(goals, list):
                parts.append(f"Career Goals: {', '.join(goals)}")
            else:
                parts.append(f"Career Goals: {goals}")
        
        if persona_dict.get("preferred_company_stages"):
            stages = persona_dict["preferred_company_stages"]
            if isinstance(stages, list):
                parts.append(f"Preferred Company Stages: {', '.join(stages)}")
            else:
                parts.append(f"Preferred Company Stages: {stages}")
        
        if persona_dict.get("preferred_locations"):
            locations = persona_dict["preferred_locations"]
            if isinstance(locations, list):
                parts.append(f"Preferred Locations: {', '.join(locations)}")
            else:
                parts.append(f"Preferred Locations: {locations}")
        
        if persona_dict.get("salary_min"):
            parts.append(f"Minimum Salary: ${persona_dict['salary_min']:,}")
        
        if persona_dict.get("remote_only"):
            parts.append("Remote Only: Yes")
        
        if persona_dict.get("cultural_preferences"):
            cultural = persona_dict["cultural_preferences"]
            if isinstance(cultural, list):
                parts.append(f"Cultural Preferences: {', '.join(cultural)}")
            else:
                parts.append(f"Cultural Preferences: {cultural}")
        
        return "\n".join(parts) if parts else "No persona information available."

    def _map_persona_to_filters(self, persona_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Direct mapping from UserPersona fields to VacancyFilter structure.
        
        This provides a baseline filter structure that can be enhanced by LLM.
        
        Args:
            persona_dict: Persona dictionary
            
        Returns:
            Dictionary with filter fields compatible with VacancyFilter
        """
        filters = {}
        
        # Map preferred_company_stages -> company_stages
        if persona_dict.get("preferred_company_stages"):
            stages = persona_dict["preferred_company_stages"]
            if isinstance(stages, list) and stages:
                filters["company_stages"] = stages
            elif isinstance(stages, str):
                filters["company_stages"] = [stages]
        
        # Map preferred_locations -> location (take first if multiple)
        if persona_dict.get("preferred_locations"):
            locations = persona_dict["preferred_locations"]
            if isinstance(locations, list) and locations:
                # Use first location, or join if multiple
                filters["location"] = locations[0] if len(locations) == 1 else ", ".join(locations)
            elif isinstance(locations, str):
                filters["location"] = locations
        
        # Map salary_min -> min_salary
        if persona_dict.get("salary_min"):
            filters["min_salary"] = persona_dict["salary_min"]
        
        # Map remote_only -> is_remote
        if persona_dict.get("remote_only"):
            filters["is_remote"] = True
        
        # Map technical_skills -> skills
        if persona_dict.get("technical_skills"):
            skills = persona_dict["technical_skills"]
            if isinstance(skills, list):
                filters["skills"] = skills
            elif isinstance(skills, str):
                filters["skills"] = [skills]
        
        return filters

    async def create_search_params(
        self,
        user_persona: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create search parameters from user persona.
        
        This method:
        1. Maps persona fields directly to filter structure
        2. Uses LLM to generate semantic query and enhance filters
        3. Returns both semantic_query and metadata_filters
        
        Args:
            user_persona: UserPersona dictionary or object (may be None/empty)
        
        Returns:
            Dictionary with:
            - semantic_query: str - Query text for vector embedding
            - metadata_filters: Dict[str, Any] - Pinecone metadata filters
            - filter_params: VacancyFilter-compatible dict
        """
        persona_dict = self._persona_to_dict(user_persona)
        
        logger.info(
            "creating_search_params",
            agent_name=self.agent_name,
            has_persona=bool(persona_dict),
            persona_keys=list(persona_dict.keys()) if persona_dict else [],
        )

        # Start with direct mapping from persona to filters
        base_filters = self._map_persona_to_filters(persona_dict)
        
        # If persona is empty, return minimal search params
        if not persona_dict:
            logger.info("persona_empty_returning_minimal_params")
            return {
                "semantic_query": "Software Engineer",
                "metadata_filters": None,
                "filter_params": {},
            }
        
        try:
            # Format persona for prompt
            persona_text = self._format_persona_for_prompt(persona_dict)
            
            # Create a virtual user message that represents the persona preferences
            # This allows the LLM to use the existing job_scout.txt prompt logic
            virtual_message = """Find jobs matching my profile and preferences."""
            
            # Build the prompt with persona context
            analysis_prompt = f"""User Profile (Persona):
{persona_text}

User Message: {virtual_message}

IMPORTANT: The user wants to find jobs based on their profile. Extract search parameters from the User Profile above.
- Use technical_skills to build the semantic query (role field)
- Use preferred_company_stages for company_stage filter
- Use preferred_locations for location filter
- Use salary_min for min_salary filter
- Use remote_only to set remote_available filter
- Generate an expanded semantic query that combines role keywords with top technical skills
- Set search_mode to "persona" since we're using persona data"""

            # Create the user message
            messages = [HumanMessage(content=analysis_prompt)]
            
            # Invoke the LLM
            response = await self.invoke(messages, max_tokens=1500)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            logger.debug("job_scout_response_received", response_length=len(response_text))
            
            # Clean and parse JSON response
            cleaned_response = self._clean_json_response(response_text)
            
            llm_params = {}
            try:
                llm_params = json.loads(cleaned_response)
                if not isinstance(llm_params, dict):
                    logger.warning("llm_params_not_dict", params_type=type(llm_params).__name__)
                    llm_params = {}
            except json.JSONDecodeError as e:
                logger.error(
                    "job_scout_json_parse_error",
                    error=str(e),
                    response_preview=response_text[:200],
                )
                # Fallback to base filters if parsing fails
                llm_params = {}
            
            # Merge LLM params with base filters (LLM params take precedence)
            filter_params = {**base_filters, **llm_params}
            
            # Extract semantic query from LLM response (role field)
            semantic_query = filter_params.get("role")
            if not semantic_query:
                # Fallback: build query from technical skills
                skills = persona_dict.get("technical_skills", [])
                if skills:
                    top_skills = skills[:3] if isinstance(skills, list) else [skills]
                    semantic_query = " ".join(top_skills)
                else:
                    semantic_query = "Software Engineer"
            
            # Build metadata filters for Pinecone
            metadata_filters = self._build_metadata_filters(filter_params)
            
            # Clean up filter_params (remove fields not in VacancyFilter)
            clean_filter_params = {
                "role": filter_params.get("role"),
                "skills": filter_params.get("skills") or filter_params.get("required_keywords"),
                "location": filter_params.get("location"),
                "is_remote": filter_params.get("is_remote") or filter_params.get("remote_available"),
                "company_stages": filter_params.get("company_stages"),
                "industry": filter_params.get("industry"),
                "min_salary": filter_params.get("min_salary"),
                "category": filter_params.get("category"),
                "experience_level": filter_params.get("experience_level"),
            }
            
            # Remove None values
            clean_filter_params = {k: v for k, v in clean_filter_params.items() if v is not None}
            
            logger.info(
                "search_params_created",
                semantic_query=semantic_query,
                has_metadata_filters=metadata_filters is not None,
                filter_keys=list(clean_filter_params.keys()),
            )
            
            return {
                "semantic_query": semantic_query,
                "metadata_filters": metadata_filters,
                "filter_params": clean_filter_params,
            }
            
        except Exception as e:
            logger.error(
                "create_search_params_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback: return minimal params with base filters
            semantic_query = "Software Engineer"
            if persona_dict.get("technical_skills"):
                skills = persona_dict["technical_skills"]
                if isinstance(skills, list) and skills:
                    semantic_query = " ".join(skills[:3])
            
            metadata_filters = self._build_metadata_filters(base_filters)
            
            return {
                "semantic_query": semantic_query,
                "metadata_filters": metadata_filters,
                "filter_params": base_filters,
            }

    def _build_metadata_filters(self, filter_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Pinecone metadata filter dictionary from filter parameters.
        
        This matches the structure used in build_pinecone_filter from vacancies.py.
        
        Args:
            filter_params: Dictionary with filter parameters
            
        Returns:
            Pinecone filter dictionary or None
        """
        filter_dict = {}
        
        # Industry filter (with case variations)
        if filter_params.get("industry"):
            industry = filter_params["industry"]
            industry_variants = list(set([
                industry,
                industry.title(),
                industry.lower(),
                industry.upper()
            ]))
            filter_dict["industry"] = {"$in": industry_variants}
        
        # Location filter (with case variations)
        if filter_params.get("location"):
            location = filter_params["location"]
            location_variants = list(set([
                location,
                location.title(),
                location.lower(),
                location.upper()
            ]))
            filter_dict["location"] = {"$in": location_variants}
        
        # Remote filter
        remote_value = filter_params.get("remote_available") or filter_params.get("is_remote")
        if remote_value is not None:
            filter_dict["remote_available"] = {"$eq": bool(remote_value)}
        
        # Company stage filter
        if filter_params.get("company_stages"):
            stages = filter_params["company_stages"]
            if not isinstance(stages, list):
                stages = [stages]
            
            # Normalize stages using CompanyStage enum
            normalized_stages = []
            for stage in stages:
                normalized = CompanyStage.get_stage_value(stage)
                if normalized == "Growth":
                    # Growth includes Series B, Series C, and Growth
                    normalized_stages.extend(["Series B", "Series C", "Growth"])
                else:
                    normalized_stages.append(normalized)
            
            # Remove duplicates
            normalized_stages = list(set(normalized_stages))
            filter_dict["company_stage"] = {"$in": normalized_stages}
        
        # Category filter
        if filter_params.get("category"):
            filter_dict["category"] = {"$eq": filter_params["category"]}
        
        # Experience level filter
        if filter_params.get("experience_level"):
            filter_dict["experience_level"] = {"$eq": filter_params["experience_level"]}
        
        # Employee count filter
        if filter_params.get("employee_count"):
            employee_count = filter_params["employee_count"]
            if not isinstance(employee_count, list):
                employee_count = [employee_count]
            filter_dict["employee_count"] = {"$in": employee_count}
        
        return filter_dict if filter_dict else None
