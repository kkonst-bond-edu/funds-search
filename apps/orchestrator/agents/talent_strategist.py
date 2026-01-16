"""
Talent Strategist Agent - Maintains and updates user persona from conversations.

This agent specializes in incrementally updating user persona profiles based on
conversation history, ensuring persona evolves over time without losing existing data.
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
import structlog
from langchain_core.messages import HumanMessage

from apps.orchestrator.agents.base import BaseAgent
from shared.schemas import UserPersona

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


class TalentStrategistAgent(BaseAgent):
    """
    Talent Strategist Agent - Updates user persona incrementally from conversations.

    This agent uses the talent_strategist configuration from agents.yaml and the
    talent_strategist.txt prompt to analyze user messages and merge new information
    into the existing persona profile.
    """

    def __init__(self):
        """Initialize the Talent Strategist agent with configuration from agents.yaml."""
        super().__init__(agent_name="talent_strategist")
        logger.info("talent_strategist_agent_initialized", agent_name="talent_strategist")

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
            return "{}"
        
        # Simple and robust cleaning: strip whitespace and remove markdown code blocks
        cleaned = text.strip()
        
        # Remove markdown code blocks (```json ... ``` or ``` ... ```)
        if "```" in cleaned:
            # Try to extract JSON from code blocks
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
        
        # Fallback: try to convert to dict
        logger.warning("unexpected_persona_type", persona_type=type(persona).__name__)
        return {}

    def _merge_persona_updates(
        self, 
        current_persona: Dict[str, Any], 
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge updates into current persona, preserving existing data.
        
        This implements the merge logic:
        - Lists are merged (additive, unless explicitly cleared)
        - Scalar values are replaced if provided
        - None values clear the field
        - Empty arrays clear the list
        - Only valid UserPersona fields are accepted
        
        Args:
            current_persona: Current persona dictionary
            updates: Dictionary with updates from LLM
            
        Returns:
            Merged persona dictionary
        """
        # Valid fields from UserPersona schema
        valid_fields = {
            "technical_skills",
            "career_goals",
            "preferred_startup_stage",
            "cultural_preferences",
            "user_id",
            "target_roles",
            "preferred_categories",
            "preferred_experience_levels",
            "preferred_industries",
            "preferred_company_stages",
            "preferred_locations",
            "salary_min",
            "remote_only",
            "chat_context",
            "skip_questions",
        }
        
        merged = current_persona.copy()
        
        for key, value in updates.items():
            # Filter out invalid fields
            if key not in valid_fields:
                logger.warning(f"Invalid persona field ignored: {key}")
                continue
                
            if value is None:
                # None means clear the field
                merged.pop(key, None)
            elif isinstance(value, list):
                # For lists, merge additively unless it's explicitly an empty list
                if len(value) == 0:
                    # Empty list means clear
                    merged[key] = []
                else:
                    # Merge lists: combine unique values
                    existing = merged.get(key, [])
                    if not isinstance(existing, list):
                        existing = []
                    # Add new values that aren't already present
                    merged_list = list(existing)
                    for item in value:
                        if item not in merged_list:
                            merged_list.append(item)
                    merged[key] = merged_list
            else:
                # Scalar values: replace
                merged[key] = value
        
        return merged

    async def update_persona(
        self,
        current_persona: Optional[Dict[str, Any]],
        user_message: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Update user persona by analyzing the latest message and merging with existing persona.
        
        This method:
        1. Takes the current persona (may be empty or partial)
        2. Analyzes the user's latest message in context of chat history
        3. Extracts new information using LLM
        4. Merges updates into existing persona (preserving existing data)
        5. Returns the updated persona
        
        Args:
            current_persona: Current persona dictionary or UserPersona object (may be None/empty)
            user_message: Latest user message to analyze
            chat_history: Optional list of previous messages for context
                          Format: [{"role": "user|assistant", "content": "..."}, ...]
        
        Returns:
            Updated persona dictionary with merged information
        """
        # Convert persona to dict
        persona_dict = self._persona_to_dict(current_persona)
        
        logger.info(
            "updating_persona",
            agent_name=self.agent_name,
            message_length=len(user_message),
            has_existing_persona=bool(persona_dict),
            persona_keys=list(persona_dict.keys()) if persona_dict else [],
            history_length=len(chat_history) if chat_history else 0,
        )

        try:
            # Format current persona for prompt
            persona_text = "No existing persona data."
            if persona_dict:
                persona_text = json.dumps(persona_dict, indent=2)
            
            # Format chat history for context
            history_section = ""
            if chat_history:
                history_lines = []
                for msg in chat_history[-5:]:  # Last 5 messages for context
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    history_lines.append(f"{role.capitalize()}: {content}")
                history_text = "\n".join(history_lines)
                history_section = f"Recent Chat History:\n{history_text}\n"
            
            # Build the analysis prompt
            analysis_prompt = f"""Current User Persona (JSON):
{persona_text}

Latest User Message:
{user_message}

{history_section}Analyze the latest user message and extract any new information about their preferences, skills, goals, or requirements. Return ONLY a JSON object with fields that should be UPDATED. Preserve existing data - only change what is explicitly mentioned or contradicted."""

            # Create the user message
            messages = [HumanMessage(content=analysis_prompt)]
            
            # Invoke the LLM with a reasonable token limit for structured output
            response = await self.invoke(messages, system_prompt=system_prompt, max_tokens=1000)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            logger.debug("persona_update_response_received", response_length=len(response_text))
            
            # Clean and parse JSON response
            cleaned_response = self._clean_json_response(response_text)
            
            updates = {}
            try:
                updates = json.loads(cleaned_response)
                if not isinstance(updates, dict):
                    logger.warning("persona_updates_not_dict", updates_type=type(updates).__name__)
                    updates = {}
            except json.JSONDecodeError as e:
                logger.error(
                    "persona_update_json_parse_error",
                    error=str(e),
                    response_preview=response_text[:200],
                )
                # Return current persona unchanged if parsing fails
                return persona_dict
            
            # Merge updates into current persona
            updated_persona = self._merge_persona_updates(persona_dict, updates)
            
            # Log what was updated
            changed_fields = [k for k in updates.keys() if updates.get(k) != persona_dict.get(k)]
            logger.info(
                "persona_update_completed",
                updated_fields=changed_fields,
                updates_count=len(updates),
                final_persona_keys=list(updated_persona.keys()),
            )
            
            return updated_persona
            
        except Exception as e:
            logger.error(
                "persona_update_error",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Return current persona unchanged on error
            return persona_dict
