"""
Job Scout Agent - Performs intelligent job search using tools.

This agent analyzes user profile and preferences, then uses search_vacancies_tool
to find matching job vacancies in Pinecone vector database.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
import structlog
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from apps.orchestrator.agents.base import BaseAgent
from apps.orchestrator.tools.search_tool import search_vacancies_tool, SearchSchema
from shared.schemas import UserPersona
from src.schemas.vacancy import VacancyFilter, CompanyStage, RoleCategory, ExperienceLevel

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
        # Bind the search tool to the LLM
        # Access llm_provider._llm for tool binding
        if hasattr(self.llm_provider, "_llm"):
            self.llm_with_tools = self.llm_provider._llm.bind_tools([search_vacancies_tool])
        else:
            # Fallback: use llm_provider directly if it's already an LLM
            self.llm_with_tools = self.llm_provider.bind_tools([search_vacancies_tool])
        logger.info("job_scout_agent_initialized", agent_name="job_scout", tools_bound=True)

    def _load_prompt(self, prompt_file: str) -> str:
        """
        Load prompt text from a file and inject enum values dynamically.
        
        This overrides the base class method to inject RoleCategory, ExperienceLevel,
        and CompanyStage enum values into the prompt at runtime, ensuring the prompt
        and database schema are always in sync.
        
        Args:
            prompt_file: Name of the prompt file (e.g., "job_scout.txt")
            
        Returns:
            Prompt text with enum values injected
        """
        # Load the base prompt from file
        prompt_text = super()._load_prompt(prompt_file)
        
        # Get enum values dynamically
        valid_categories = [cat.value for cat in RoleCategory.__members__.values()]
        valid_experience_levels = [level.value for level in ExperienceLevel.__members__.values()]
        valid_company_stages = [stage.value for stage in CompanyStage.__members__.values()]
        
        # Format as comma-separated list with quotes for clarity
        categories_str = ", ".join([f'"{cat}"' for cat in valid_categories])
        experience_levels_str = ", ".join([f'"{level}"' for level in valid_experience_levels])
        company_stages_str = ", ".join([f'"{stage}"' for stage in valid_company_stages])
        
        # Replace placeholders with actual enum values
        prompt_text = prompt_text.replace("{{VALID_CATEGORIES}}", categories_str)
        prompt_text = prompt_text.replace("{{VALID_EXPERIENCE_LEVELS}}", experience_levels_str)
        prompt_text = prompt_text.replace("{{VALID_COMPANY_STAGES}}", company_stages_str)
        
        logger.info(
            "prompt_enum_values_injected",
            categories_count=len(valid_categories),
            experience_levels_count=len(valid_experience_levels),
            company_stages_count=len(valid_company_stages),
        )
        
        return prompt_text

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
        
        if persona_dict.get("target_roles"):
            roles = persona_dict["target_roles"]
            if isinstance(roles, list):
                parts.append(f"Target Roles: {', '.join(roles)}")
            else:
                parts.append(f"Target Roles: {roles}")

        if persona_dict.get("preferred_categories"):
            categories = persona_dict["preferred_categories"]
            if isinstance(categories, list):
                parts.append(f"Preferred Categories: {', '.join(categories)}")
            else:
                parts.append(f"Preferred Categories: {categories}")

        if persona_dict.get("preferred_experience_levels"):
            levels = persona_dict["preferred_experience_levels"]
            if isinstance(levels, list):
                parts.append(f"Preferred Experience Levels: {', '.join(levels)}")
            else:
                parts.append(f"Preferred Experience Levels: {levels}")

        if persona_dict.get("preferred_industries"):
            industries = persona_dict["preferred_industries"]
            if isinstance(industries, list):
                parts.append(f"Preferred Industries: {', '.join(industries)}")
            else:
                parts.append(f"Preferred Industries: {industries}")

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

        # Map preferred_categories -> category
        if persona_dict.get("preferred_categories"):
            categories = persona_dict["preferred_categories"]
            if isinstance(categories, list) and categories:
                filters["category"] = categories[0]
            elif isinstance(categories, str):
                filters["category"] = categories

        # Map preferred_experience_levels -> experience_level
        if persona_dict.get("preferred_experience_levels"):
            levels = persona_dict["preferred_experience_levels"]
            if isinstance(levels, list) and levels:
                filters["experience_level"] = levels[0]
            elif isinstance(levels, str):
                filters["experience_level"] = levels

        # Map preferred_industries -> industry
        if persona_dict.get("preferred_industries"):
            industries = persona_dict["preferred_industries"]
            if isinstance(industries, list) and industries:
                filters["industry"] = industries[0]
            elif isinstance(industries, str):
                filters["industry"] = industries
        
        # Map technical_skills -> skills
        if persona_dict.get("technical_skills"):
            skills = persona_dict["technical_skills"]
            if isinstance(skills, list):
                filters["skills"] = skills
            elif isinstance(skills, str):
                filters["skills"] = [skills]
        
        return filters

    async def search_with_tool(
        self,
        user_profile: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform job search using the search_vacancies_tool.
        
        This method:
        1. Analyzes user profile and conversation history
        2. Invokes LLM with tool binding to generate search strategy
        3. Handles tool calls and executes search
        4. Returns search results and analysis
        
        Args:
            user_profile: UserProfile dictionary with skills, experience, preferences
            conversation_history: Optional list of conversation messages
        
        Returns:
            Dictionary with:
            - search_results: List of vacancy results from tool
            - search_params: Dictionary with query and filters used
            - analysis: Agent's analysis of results
        """
        logger.info(
            "job_scout_search_starting",
            agent_name=self.agent_name,
            has_profile=bool(user_profile),
            profile_keys=list(user_profile.keys()) if user_profile else [],
        )
        
        # Build context message for the agent
        context_parts = []
        if user_profile:
            if user_profile.get("target_role"):
                context_parts.append(f"Target role: {user_profile['target_role']}")
            if user_profile.get("category"):
                context_parts.append(f"Category: {user_profile['category']}")
            if user_profile.get("experience_level"):
                context_parts.append(f"Experience level: {user_profile['experience_level']}")
            if user_profile.get("industry"):
                context_parts.append(f"Industry: {user_profile['industry']}")
            if user_profile.get("company_stage"):
                context_parts.append(f"Company stage: {user_profile['company_stage']}")
            if user_profile.get("skills"):
                context_parts.append(f"Skills: {', '.join(user_profile['skills'][:5])}")
            if user_profile.get("years_of_experience"):
                context_parts.append(f"Experience: {user_profile['years_of_experience']} years")
            if user_profile.get("location"):
                context_parts.append(f"Location preference: {user_profile['location']}")
            if user_profile.get("remote_preference"):
                context_parts.append(f"Remote preference: {user_profile['remote_preference']}")
            if user_profile.get("salary_expectation"):
                context_parts.append(f"Salary expectation: {user_profile['salary_expectation']}")
        
        context_message = "User Profile:\n" + "\n".join(context_parts) if context_parts else "No user profile available."
        
        if conversation_history:
            context_message += "\n\nConversation History:\n" + "\n".join([
                f"{msg.get('role', 'user')}: {msg.get('content', '')[:200]}"
                for msg in conversation_history[-3:]  # Last 3 messages
            ])
        
        context_message += "\n\nPlease analyze the user's profile and preferences, then use search_vacancies_tool to find matching job vacancies."
        
        # Create messages for LLM
        messages = [HumanMessage(content=context_message)]
        
        try:
            # Invoke LLM with tool binding
            response = await self.llm_with_tools.ainvoke(messages)
            
            logger.info(
                "job_scout_llm_response",
                has_tool_calls=hasattr(response, "tool_calls") and len(response.tool_calls) > 0,
                tool_calls_count=len(response.tool_calls) if hasattr(response, "tool_calls") else 0,
            )
            
            # Check if LLM wants to call the tool
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tool calls
                tool_results = []
                for tool_call in response.tool_calls:
                    tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
                    if tool_name == "search_vacancies_tool":
                        # Extract arguments - handle both dict and object formats
                        if isinstance(tool_call, dict):
                            args = tool_call.get("args", {})
                            tool_call_id = tool_call.get("id", "")
                        else:
                            args = getattr(tool_call, "args", {})
                            tool_call_id = getattr(tool_call, "id", "")
                        
                        # Extract individual parameters (new tool signature)
                        query = args.get("query", "Software Engineer") if isinstance(args, dict) else getattr(args, "query", "Software Engineer")
                        category = args.get("category") if isinstance(args, dict) else getattr(args, "category", None)
                        experience_level = args.get("experience_level") if isinstance(args, dict) else getattr(args, "experience_level", None)
                        company_stage = args.get("company_stage") if isinstance(args, dict) else getattr(args, "company_stage", None)
                        remote_option = args.get("remote_option") if isinstance(args, dict) else getattr(args, "remote_option", None)
                        location = args.get("location") if isinstance(args, dict) else getattr(args, "location", None)
                        industry = args.get("industry") if isinstance(args, dict) else getattr(args, "industry", None)
                        salary_min = args.get("salary_min") if isinstance(args, dict) else getattr(args, "salary_min", None)
                        employee_count = args.get("employee_count") if isinstance(args, dict) else getattr(args, "employee_count", None)
                        top_k = args.get("top_k", 10) if isinstance(args, dict) else getattr(args, "top_k", 10)
                        
                        # Build params dict for validation
                        search_params = {
                            "query": query,
                            "category": category,
                            "experience_level": experience_level,
                            "company_stage": company_stage,
                            "remote_option": remote_option,
                            "location": location,
                            "industry": industry,
                            "salary_min": salary_min,
                            "employee_count": employee_count,
                        }
                        
                        # Validate parameters against SearchSchema (self-correction before DB)
                        validated_schema, validation_error = self._validate_search_params(search_params)
                        
                        if validation_error:
                            logger.error(
                                "search_params_validation_failed",
                                error=validation_error,
                                params=search_params,
                            )
                            # Return error in tool result
                            tool_results.append(ToolMessage(
                                content=json.dumps({
                                    "results": [],
                                    "count": 0,
                                    "query": query,
                                    "filters_applied": None,
                                    "error": validation_error,
                                }),
                                tool_call_id=tool_call_id,
                            ))
                            continue
                        
                        logger.info(
                            "executing_search_tool",
                            query=query,
                            category=category,
                            experience_level=experience_level,
                            company_stage=company_stage,
                            remote_option=remote_option,
                            location=location,
                            industry=industry,
                            salary_min=salary_min,
                            employee_count=employee_count,
                            top_k=top_k,
                        )
                        
                        # Call the tool with validated parameters
                        tool_result = search_vacancies_tool.invoke({
                            "query": query,
                            "category": category,
                            "experience_level": experience_level,
                            "company_stage": company_stage,
                            "remote_option": remote_option,
                            "location": location,
                            "industry": industry,
                            "salary_min": salary_min,
                            "employee_count": employee_count,
                            "top_k": top_k,
                        })
                        
                        tool_results.append(ToolMessage(
                            content=json.dumps(tool_result),
                            tool_call_id=tool_call_id,
                        ))
                
                # Add tool results to messages and get final response
                messages.append(response)
                messages.extend(tool_results)
                
                # Get final analysis from agent (best-effort)
                analysis_text = ""
                analysis_error = None
                try:
                    final_response = await self.llm_with_tools.ainvoke(messages)
                    analysis_text = (
                        final_response.content
                        if hasattr(final_response, "content")
                        else str(final_response)
                    )
                except Exception as e:
                    analysis_error = f"Post-search analysis failed: {str(e)}"
                    logger.error(
                        "job_scout_analysis_error",
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                
                # Extract search results from tool results
                search_results = []
                tool_error = None
                if tool_results:
                    try:
                        tool_result_data = json.loads(tool_results[0].content)
                        search_results = tool_result_data.get("results", [])
                        tool_error = tool_result_data.get("error")
                    except (json.JSONDecodeError, KeyError):
                        pass
                
                # Build search_params dict with all parameters used
                search_params_dict = {
                    "query": query if 'query' in locals() else "Software Engineer",
                }
                if 'category' in locals() and category is not None:
                    search_params_dict["category"] = category
                if 'experience_level' in locals() and experience_level is not None:
                    search_params_dict["experience_level"] = experience_level
                if 'company_stage' in locals() and company_stage is not None:
                    search_params_dict["company_stage"] = company_stage
                if 'remote_option' in locals() and remote_option is not None:
                    search_params_dict["remote_option"] = remote_option
                if 'salary_min' in locals() and salary_min is not None:
                    search_params_dict["salary_min"] = salary_min
                if 'employee_count' in locals() and employee_count is not None:
                    search_params_dict["employee_count"] = employee_count
                
                # Prefer tool error, but keep analysis error if present
                combined_error = tool_error or analysis_error
                return {
                    "search_results": search_results,
                    "search_params": search_params_dict,
                    "analysis": analysis_text or "Search completed. Analysis unavailable.",
                    "error": combined_error,
                }
            else:
                # LLM didn't call tool, return response as analysis
                return {
                    "search_results": [],
                    "search_params": {},
                    "analysis": response.content if hasattr(response, "content") else str(response),
                    "error": "Agent did not call search tool",
                }
                
        except Exception as e:
            logger.error(
                "job_scout_search_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            return {
                "search_results": [],
                "search_params": {},
                "analysis": f"Search failed: {str(e)}",
                "error": str(e),
            }

    def _validate_search_params(self, params: Dict[str, Any]) -> Tuple[Optional[SearchSchema], Optional[str]]:
        """
        Validate search parameters against SearchSchema.
        
        This performs self-correction before hitting the database, similar to validator_node logic.
        
        Args:
            params: Dictionary with search parameters from LLM tool call
            
        Returns:
            Tuple of (SearchSchema instance if valid, error message if invalid)
        """
        try:
            # Ensure company_stage is a list if provided
            if "company_stage" in params and params["company_stage"] is not None:
                if not isinstance(params["company_stage"], list):
                    params["company_stage"] = [params["company_stage"]]
            
            # Ensure employee_count is a list if provided
            if "employee_count" in params and params["employee_count"] is not None:
                if not isinstance(params["employee_count"], list):
                    params["employee_count"] = [params["employee_count"]]
            
            # Ensure salary_min is an integer if provided
            if "salary_min" in params and params["salary_min"] is not None:
                try:
                    params["salary_min"] = int(params["salary_min"])
                except (ValueError, TypeError):
                    logger.warning("invalid_salary_min_type", value=params["salary_min"])
                    params["salary_min"] = None
            
            # Validate remote_option value
            if "remote_option" in params and params["remote_option"] is not None:
                remote_val = params["remote_option"]
                if isinstance(remote_val, bool):
                    # Convert boolean to string format
                    params["remote_option"] = "remote" if remote_val else "office"
                elif remote_val not in ["remote", "office", "hybrid"]:
                    logger.warning("invalid_remote_option_value", value=remote_val)
                    # Try to map common variations
                    if remote_val in [True, "true", "True", "remote", "Remote"]:
                        params["remote_option"] = "remote"
                    elif remote_val in [False, "false", "False", "office", "Office"]:
                        params["remote_option"] = "office"
                    else:
                        params["remote_option"] = None
            
            # Create SearchSchema instance
            schema = SearchSchema(**params)
            return schema, None
            
        except Exception as e:
            error_msg = f"Search parameter validation failed: {str(e)}"
            logger.error("search_params_validation_error", error=str(e), params=params)
            return None, error_msg

    def _build_metadata_filters(self, filter_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Build Pinecone metadata filter dictionary from filter parameters.
        
        This method is kept for backward compatibility but is now primarily used
        for legacy code paths. New code should use SearchSchema validation.
        
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
        
        # Remote filter - UPDATED to use remote_option
        # IMPORTANT: Pinecone metadata uses "remote_option", not "remote_available" or "is_remote"
        # Support both old format (is_remote boolean) and new format (remote_option string)
        remote_option = filter_params.get("remote_option")
        if remote_option is None:
            # Fallback to old format for backward compatibility
            remote_value = filter_params.get("remote_available") or filter_params.get("is_remote")
            if remote_value is not None:
                filter_dict["remote_option"] = {"$eq": bool(remote_value)}
        else:
            # New format: remote_option is a string ("remote", "office", "hybrid")
            if remote_option == "remote":
                filter_dict["remote_option"] = {"$eq": True}
            elif remote_option == "office":
                filter_dict["remote_option"] = {"$eq": False}
            elif remote_option == "hybrid":
                # Hybrid roles have remote_option=False in Pinecone
                filter_dict["remote_option"] = {"$eq": False}
        
        # Company stage filter - UPDATED to use $in for lists
        if filter_params.get("company_stages") or filter_params.get("company_stage"):
            stages = filter_params.get("company_stages") or filter_params.get("company_stage")
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
        
        # Employee count filter - UPDATED to use $in for lists
        if filter_params.get("employee_count"):
            employee_count = filter_params["employee_count"]
            if not isinstance(employee_count, list):
                employee_count = [employee_count]
            filter_dict["employee_count"] = {"$in": employee_count}
        
        # Salary min filter - UPDATED to use $gte with integer
        if filter_params.get("min_salary") or filter_params.get("salary_min"):
            salary_min = filter_params.get("min_salary") or filter_params.get("salary_min")
            if salary_min is not None:
                try:
                    salary_min_int = int(salary_min)
                    filter_dict["min_salary"] = {"$gte": salary_min_int}
                except (ValueError, TypeError):
                    logger.warning("invalid_salary_min", value=salary_min)
        
        return filter_dict if filter_dict else None
