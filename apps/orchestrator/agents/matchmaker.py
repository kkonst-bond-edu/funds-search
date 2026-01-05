"""
Matchmaker Agent - Analyzes vacancy-candidate matches.

This agent specializes in analyzing why a specific vacancy is a good fit
for a candidate based on their persona and the vacancy requirements.
"""

import logging
from typing import Any, Dict, List, Optional
import structlog

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


class MatchmakerAgent(BaseAgent):
    """
    Matchmaker Agent - Analyzes vacancy against candidate persona and explains the match.

    This agent uses the matchmaker configuration from agents.yaml and the matchmaker.txt prompt.
    """

    def __init__(self):
        """Initialize the Matchmaker agent with configuration from agents.yaml."""
        super().__init__(agent_name="matchmaker")
        logger.info("matchmaker_agent_initialized", agent_name="matchmaker")

    async def analyze_match(
        self,
        vacancy_text: str,
        candidate_persona: Dict[str, Any],
        similarity_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Analyze why a vacancy matches a candidate's persona.

        Args:
            vacancy_text: Full text description of the vacancy
            candidate_persona: Dictionary with candidate persona information
            similarity_score: Optional cosine similarity score from vector search

        Returns:
            Dictionary with:
            - score: int - AI match score from 0 to 10
            - reasoning: str - Text explanation of the match
        """
        logger.info(
            "analyzing_match",
            agent_name=self.agent_name,
            vacancy_length=len(vacancy_text),
            has_persona=bool(candidate_persona),
            similarity_score=similarity_score
        )

        # Format persona information
        persona_text = self._format_persona(candidate_persona)

        # Build the analysis prompt
        # Limit vacancy text to 2000 chars to avoid token limits
        limited_vacancy_text = vacancy_text[:2000] if len(vacancy_text) > 2000 else vacancy_text
        
        # Format similarity score safely
        similarity_str = f"{similarity_score:.4f}" if similarity_score is not None else "N/A"
        
        analysis_prompt = f"""Vacancy Description:
{limited_vacancy_text}

Candidate Persona:
{persona_text}

Similarity Score: {similarity_str}

Analyze why this vacancy matches the candidate. Be extremely concise: 3-4 bullet points, max 150 words."""

        try:
            from langchain_core.messages import HumanMessage
            import re
            messages = [HumanMessage(content=analysis_prompt)]
            # Limit response to 500 tokens (approximately 375 words) to prevent verbose responses
            # This ensures responses stay under 150 words as requested
            response = await self.invoke(messages, max_tokens=500)
            reasoning = response.content if hasattr(response, "content") else str(response)
            
            # Extract score from response (format: "Score: X/10" or "Score: X/10" at the beginning)
            ai_score = None
            score_pattern = r'Score:\s*(\d+)/10'
            score_match = re.search(score_pattern, reasoning, re.IGNORECASE)
            if score_match:
                try:
                    ai_score = int(score_match.group(1))
                    # Clamp score to 0-10 range
                    ai_score = max(0, min(10, ai_score))
                    # Remove score line from reasoning text
                    reasoning = re.sub(r'Score:\s*\d+/10\s*', '', reasoning, flags=re.IGNORECASE).strip()
                except (ValueError, AttributeError):
                    logger.warning("match_score_extraction_failed", reasoning_preview=reasoning[:100])
            
            # If score not found, default to None (will be handled by API)
            if ai_score is None:
                logger.warning("match_score_not_found", reasoning_preview=reasoning[:100])
            
            # Additional safety: truncate if somehow still too long (shouldn't happen with max_tokens)
            if len(reasoning) > 1000:  # ~150 words at ~6.5 chars/word
                reasoning = reasoning[:1000] + "..."
                logger.warning("match_analysis_truncated", original_length=len(response.content) if hasattr(response, "content") else 0)

            logger.info("match_analysis_completed", reasoning_length=len(reasoning), ai_score=ai_score)
            return {
                "score": ai_score,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error("match_analysis_error", error=str(e), error_type=type(e).__name__)
            # Return a fallback result instead of raising to prevent crashing the entire search
            return {
                "score": None,
                "reasoning": f"Match analysis unavailable: {str(e)[:100]}"
            }

    def _format_persona(self, persona: Dict[str, Any]) -> str:
        """
        Format persona dictionary into a readable text format.

        Args:
            persona: Dictionary with candidate persona information

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
            parts.append(f"\nCV Summary:\n{persona['cv_text'][:500]}...")  # Limit to 500 chars

        return "\n".join(parts) if parts else "No persona information available."

