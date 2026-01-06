"""
Conversational Vacancy Search Agent.

Uses LLM to interpret natural language messages and extract search parameters,
then generates friendly summaries of search results.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import structlog
from langchain_core.messages import HumanMessage, AIMessage

from apps.orchestrator.agents.base import BaseAgent
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


class ChatSearchAgent(BaseAgent):
    """
    AI agent that converts natural language chat messages into search queries
    and generates friendly summaries of search results.

    This is the Job Scout agent implementation, using the BaseAgent infrastructure.
    """

    def __init__(self):
        """Initialize the chat search agent (Job Scout) with configuration from agents.yaml."""
        super().__init__(agent_name="job_scout")
        # Store the summary prompt file path (not content) for reloading on each request
        self.summary_prompt_file = "job_summary.txt"
        logger.info("chat_search_agent_initialized", agent_name="job_scout")

    def _load_summary_prompt(self) -> str:
        """
        Load the summary prompt from job_summary.txt (reloaded on each call).

        Returns:
            Summary prompt text as a string
        """
        # Get the path to prompts directory
        orchestrator_dir = Path(__file__).parent
        prompts_dir = orchestrator_dir / "prompts"
        prompt_path = prompts_dir / self.summary_prompt_file

        if not prompt_path.exists():
            logger.warning("summary_prompt_file_not_found", path=str(prompt_path))
            # Fallback to default prompt
            return """You are a helpful assistant that explains job search results in a friendly, conversational way.

Based on the user's original query and the vacancies found, explain why these specific opportunities match their search.

Be concise (2-3 sentences), friendly, and highlight the key matches (skills, industry, location, company stage).

Do not make up information that wasn't provided."""

        with open(prompt_path, "r") as f:
            prompt_text = f.read().strip()

        logger.info("summary_prompt_loaded", prompt_length=len(prompt_text))
        return prompt_text

    async def interpret_message(
        self, 
        user_input: str, 
        history: List[dict] = None, 
        persona: dict = None
    ) -> Dict[str, Any]:
        """
        Extract search parameters from natural language user input.

        Args:
            user_input: Natural language message from the user
            history: Optional list of previous messages in format [{"role": "user/assistant", "content": "..."}]
            persona: Optional dictionary with user persona/CV information

        Returns:
            Dictionary with extracted parameters:
            - role: Optional[str] - Job role/title
            - skills: Optional[List[str]] - Required skills
            - industry: Optional[str] - Industry sector
            - location: Optional[str] - Job location
            - company_stage: Optional[str] - Company funding stage
            - search_mode: str - Either "persona" (generic queries) or "explicit" (specific criteria)

            Missing fields will be None (except search_mode which is always present).
        """
        logger.info(
            "interpreting_user_message", 
            user_input_length=len(user_input), 
            agent_name=self.agent_name,
            has_history=history is not None and len(history) > 0,
            has_persona=persona is not None
        )

        try:
            # Define job title keywords for detection (used in multiple places)
            job_title_keywords = [
                "engineer", "developer", "architect", "scientist", "analyst", "manager",
                "director", "lead", "specialist", "consultant", "designer", "programmer",
                "backend", "frontend", "fullstack", "full stack", "mobile", "web",
                "data", "ml", "ai", "devops", "sre", "qa", "test", "security"
            ]
            
            # Handle special case: "all" or "все" means "find all vacancies matching my profile"
            user_input_lower = user_input.lower().strip()
            if user_input_lower in ["all", "все", "show all", "show me all"]:
                if persona:
                    # Return minimal filters to get all matching vacancies
                    logger.info("interpreting_all_request_with_persona")
                    return {
                        "role": None,  # Don't filter by role
                        "skills": None,  # Don't filter by skills (too restrictive)
                        "industry": None,  # Don't filter by industry
                        "location": None,  # Don't filter by location
                        "company_stage": None,  # Don't filter by company stage
                        "search_mode": "persona",  # Generic query, use persona
                    }
                else:
                    # Without persona, return all nulls to get all vacancies
                    logger.info("interpreting_all_request_without_persona")
                    return {
                        "role": None,
                        "skills": None,
                        "industry": None,
                        "location": None,
                        "company_stage": None,
                        "search_mode": "explicit",  # No persona, so explicit (though filters are null)
                    }

            # Build messages list with history and persona context
            messages = []

            # Add conversation history first (if provided)
            if history:
                for msg in history:
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "user":
                        messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        messages.append(AIMessage(content=content))

            # Check if user input contains a job title keyword (before building context)
            has_job_title = any(keyword in user_input_lower for keyword in job_title_keywords)
            
            # Add persona context BEFORE the current user message
            # This provides context for the LLM to use when the query is vague
            if persona:
                persona_text = self._format_persona(persona)
                # Check if user message is generic (needs persona defaults)
                generic_queries = ["find vacancies", "find jobs", "show vacancies", "show jobs", 
                                 "vacancies", "jobs", "search", "find", "show"]
                is_generic = (user_input_lower in generic_queries or len(user_input_lower.split()) <= 3) and not has_job_title
                
                if is_generic:
                    # For generic queries, instruct LLM to use persona as default source
                    user_input_with_context = f"""User Profile: {persona_text}

User Message: {user_input}

IMPORTANT INSTRUCTIONS:
- The user's message is generic/vague. Use the User Profile as the DEFAULT source for role and skills.
- Extract role from User Profile's career_goals or CV text if the user message doesn't specify a role.
- Extract skills from User Profile's technical_skills if the user message doesn't specify skills.
- Only extract location, industry, or company_stage if explicitly mentioned in the user's message.
- Do NOT infer specific cities or strict filters unless explicitly mentioned."""
                else:
                    # For specific queries, prevent hallucination and ensure explicit mode
                    explicit_mode_hint = ""
                    if has_job_title:
                        explicit_mode_hint = "\n\nCRITICAL: The user mentioned a job title. You MUST set search_mode to 'explicit' and extract ONLY from the current message, NOT from the User Profile or history."
                    
                    user_input_with_context = f"""User Profile: {persona_text}

User Message: {user_input}

IMPORTANT: Extract ONLY the search parameters explicitly mentioned in the CURRENT user message. 
- Do NOT infer specific cities, locations, or strict filters from the User Profile unless the user explicitly mentions them.
- Use the User Profile ONLY for context when the query is vague (e.g., "remote ones" refers to previous search).
- If the user says "all" or asks to see everything, return all fields as null.
- Only extract what the user actually said in THIS message, not what you think they might want based on their profile or previous messages.{explicit_mode_hint}"""
                messages.append(HumanMessage(content=user_input_with_context))
            else:
                # Add current user input without persona context
                messages.append(HumanMessage(content=user_input))

            # BaseAgent.invoke will automatically prepend the system prompt from the prompt file
            response = await self.invoke(messages)
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

            # Store raw extracted values for debug_info (before any enrichment)
            raw_extracted_role = extracted_params.get("role")
            raw_required_keywords = extracted_params.get("required_keywords")
            raw_identified_filters = {
                "role": extracted_params.get("role"),
                "skills": extracted_params.get("skills"),
                "industry": extracted_params.get("industry"),
                "location": extracted_params.get("location"),
                "company_stage": extracted_params.get("company_stage"),
                "required_keywords": raw_required_keywords,
            }

            # Normalize the response to ensure all expected fields exist
            result = {
                "role": extracted_params.get("role"),
                "skills": extracted_params.get("skills"),
                "industry": extracted_params.get("industry"),
                "location": extracted_params.get("location"),
                "company_stage": extracted_params.get("company_stage"),
                "required_keywords": extracted_params.get("required_keywords"),
                "search_mode": extracted_params.get("search_mode"),  # Will validate and default below
                "friendly_reasoning": extracted_params.get("friendly_reasoning"),  # Extract friendly_reasoning from Job Scout response
                "remote_available": extracted_params.get("remote_available"),  # Extract remote_available from Job Scout response
            }
            
            # Validate and normalize search_mode - default to "explicit" if missing, null, or invalid
            # Safety fallback: If search_mode is missing or null, default to "explicit" to ensure
            # the system respects the user's specific typed query rather than forcing CV data
            search_mode = result.get("search_mode")
            
            # Let the LLM decide the search_mode based on the prompt rules
            # The prompt prioritizes persona mode when "me", "my", or "for me" is mentioned
            # We trust the LLM's decision and don't override it here
            
            # Check if search_mode is missing, null, or invalid - default to "explicit" for safety
            if not search_mode or search_mode not in ["persona", "explicit"]:
                logger.warning(
                    "search_mode_missing_or_invalid",
                    provided_mode=search_mode,
                    user_input=user_input_lower,
                    message="Defaulting to 'explicit' mode to ensure system respects user's specific query"
                )
                # Default to explicit to ensure the system respects the user's specific typed query
                # This prevents accidentally forcing CV data on a specific user request
                search_mode = "explicit"
                result["search_mode"] = search_mode
            
            # Apply persona fallback logic based on search_mode
            search_mode = result.get("search_mode")
            
            if search_mode == "explicit":
                # Explicit mode: Use ONLY what the user typed in the CURRENT message.
                # Do NOT fill from Persona, history, or CV.
                # This allows searching for something totally different from their profile.
                # Even if skills are provided but role is missing, do NOT fill role from persona.
                
                # Ensure role and skills contain ONLY what was extracted from current message
                # If LLM accidentally included persona data, we've already prevented that above
                # But double-check: if role/skills exist, they should be from user input only
                
                logger.info(
                    "search_mode_explicit", 
                    role=result.get("role"),
                    skills=result.get("skills"),
                    message="Using only explicit user input from current message, ignoring persona and history"
                )
            
            elif search_mode == "persona":
                # Persona mode: Use Persona as the default base. Fill missing fields from persona.
                # CRITICAL: Only use persona data if it's actually available and valid
                # Don't "fake" a match if CV is empty - fall back to explicit mode behavior
                if persona and isinstance(persona, dict) and len(persona) > 0:
                    # Fill role from persona if missing
                    if not result.get("role") and persona.get("career_goals"):
                        career_goals = persona.get("career_goals", [])
                        if isinstance(career_goals, list) and career_goals:
                            # Use first career goal as role hint
                            result["role"] = career_goals[0] if isinstance(career_goals[0], str) else None
                        elif isinstance(career_goals, str):
                            result["role"] = career_goals
                    
                    # Fill skills from persona if missing
                    if not result.get("skills") and persona.get("technical_skills"):
                        skills = persona.get("technical_skills", [])
                        if isinstance(skills, list):
                            # Use top 5 skills from persona
                            result["skills"] = skills[:5]
                        elif isinstance(skills, str):
                            result["skills"] = [skills]
                    
                    # Hybrid: If user provides skills but no role, use persona role as complement
                    # This only applies in persona mode, not explicit mode
                    if result.get("skills") and not result.get("role") and persona.get("career_goals"):
                        career_goals = persona.get("career_goals", [])
                        if isinstance(career_goals, list) and career_goals:
                            result["role"] = career_goals[0] if isinstance(career_goals[0], str) else None
                        elif isinstance(career_goals, str):
                            result["role"] = career_goals
                        logger.info(
                            "persona_hybrid_mode_applied",
                            user_skills=result.get("skills"),
                            persona_role=result.get("role"),
                            message="Using user skills + persona role in persona mode"
                        )
                    
                    # PERSONA ENRICHMENT: Enrich role expansion with technical skills and experience
                    # This creates a high-density search query for better embedding matching
                    if result.get("role"):
                        enriched_role = self._enrich_role_with_persona(
                            result["role"],
                            persona
                        )
                        if enriched_role != result["role"]:
                            logger.info(
                                "role_enriched_with_persona",
                                original_role=result["role"],
                                enriched_role=enriched_role
                            )
                            result["role"] = enriched_role
                    
                    logger.info(
                        "search_mode_persona", 
                        role=result.get("role"), 
                        skills_count=len(result.get("skills") or []),
                        message="Using persona as base for search"
                    )
                else:
                    # Persona mode requested but no persona data available
                    # Don't fake a match - log warning and proceed without persona enrichment
                    logger.warning(
                        "persona_mode_without_persona",
                        search_mode=search_mode,
                        message="Persona mode requested but no persona data available. Proceeding without persona enrichment."
                    )

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
                search_mode=result["search_mode"],
            )

            # Add debug_info to response
            result["debug_info"] = {
                "extracted_role": raw_extracted_role,
                "identified_filters": raw_identified_filters,
                "required_keywords": raw_required_keywords,
                "search_mode": result.get("search_mode", "explicit"),  # Include search_mode for UI display
            }

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
                "search_mode": "explicit",  # Default fallback
                "required_keywords": None,
                "friendly_reasoning": None,  # Missing on parse error
                "remote_available": None,  # Missing on parse error
                "debug_info": {
                    "extracted_role": None,
                    "identified_filters": {},
                    "required_keywords": None,
                }
            }
        except Exception as e:
            logger.error("interpretation_error", error=str(e), error_type=type(e).__name__)
            raise

    def _enrich_role_with_persona(self, role: str, persona: dict) -> str:
        """
        Enrich role expansion with technical skills and experience from CV Persona.
        Creates a high-density search query for better embedding matching.
        
        Args:
            role: The role query from Job Scout (may already be expanded)
            persona: User persona dictionary with CV information
            
        Returns:
            Enriched role query string with technical skills and experience
        """
        if not role or not persona:
            return role
        
        enrichment_parts = []
        
        # Add top technical skills (up to 5) to the role query
        technical_skills = persona.get("technical_skills")
        if technical_skills:
            if isinstance(technical_skills, list):
                top_skills = technical_skills[:5]
                if top_skills:
                    # Add skills as keywords to the role query
                    skills_str = " ".join(top_skills)
                    enrichment_parts.append(skills_str)
            elif isinstance(technical_skills, str):
                enrichment_parts.append(technical_skills)
        
        # Add experience level if available
        experience_years = persona.get("experience_years")
        if experience_years:
            if isinstance(experience_years, (int, float)):
                if experience_years >= 5:
                    enrichment_parts.append("Senior")
                elif experience_years >= 3:
                    enrichment_parts.append("Mid-level")
        
        # Combine role with enrichment parts
        if enrichment_parts:
            enriched_role = f"{role} {' '.join(enrichment_parts)}"
            return enriched_role.strip()
        
        return role

    def _format_persona(self, persona: dict) -> str:
        """
        Format persona dictionary into a readable text format.

        Args:
            persona: Dictionary with user persona information (from CV/profile)

        Returns:
            Formatted persona text
        """
        parts = []
        
        if persona.get("technical_skills"):
            skills = persona["technical_skills"]
            if isinstance(skills, list):
                parts.append(f"Technical Skills: {', '.join(skills)}")
            else:
                parts.append(f"Technical Skills: {skills}")
        
        if persona.get("career_goals"):
            goals = persona["career_goals"]
            if isinstance(goals, list):
                parts.append(f"Career Goals: {', '.join(goals)}")
            else:
                parts.append(f"Career Goals: {goals}")
        
        if persona.get("experience_years"):
            parts.append(f"Years of Experience: {persona['experience_years']}")
        
        if persona.get("preferred_startup_stage"):
            parts.append(f"Preferred Company Stage: {persona['preferred_startup_stage']}")
        
        if persona.get("industry_preferences"):
            industries = persona["industry_preferences"]
            if isinstance(industries, list):
                parts.append(f"Industry Preferences: {', '.join(industries)}")
            else:
                parts.append(f"Industry Preferences: {industries}")
        
        # If persona has raw text (from CV), include it
        if persona.get("cv_text"):
            parts.append(f"\nCV Text:\n{persona['cv_text']}")  # Include full CV text (already limited to 2000 chars in CV processor)
        
        return "\n".join(parts) if parts else "No persona information available."

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

        # Use the externalized summary prompt
        user_prompt = f"""User's original query: "{user_input}"

Found {len(vacancies)} matching vacancies:

{vacancies_text}

Generate a friendly 2-3 sentence summary explaining why these vacancies match the user's search."""

        try:
            messages = [HumanMessage(content=user_prompt)]
            # Reload the summary prompt on each request (no caching)
            summary_prompt = self._load_summary_prompt()
            response = await self.invoke(messages, system_prompt=summary_prompt)
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
