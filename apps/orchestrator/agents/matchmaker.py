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

    def _extract_score_from_text(self, text: str) -> Optional[int]:
        """
        Extract score from text format (e.g., "85/100", "Score: 85/100", "8/10").
        
        Args:
            text: Text that may contain a score
            
        Returns:
            Integer score (0-100) or None if not found
        """
        import re
        
        # Try various patterns for score extraction
        patterns = [
            r'(\d+)/100',  # "85/100"
            r'(\d+)/10',   # "8/10"
            r'Score:\s*(\d+)/100',  # "Score: 85/100"
            r'Score:\s*(\d+)/10',   # "Score: 8/10"
            r'"score":\s*(\d+)',    # JSON-like: "score": 85
            r'score["\']?\s*:\s*(\d+)',  # score: 85 or score": 85
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    # If extracted from /10 format, convert to 0-100 scale
                    if '/10' in pattern or '/10' in text:
                        score = score * 10
                    # Clamp to valid range (0-100)
                    score = max(0, min(100, score))
                    return score
                except (ValueError, TypeError):
                    continue
        
        return None

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
            - score: int - AI match score from 0 to 100
            - analysis: str - Text explanation highlighting connections and gaps
            - reasoning: str - Same as analysis (for backward compatibility)
        """
        # Data Validation: Check if candidate_persona is empty
        has_persona_data = bool(candidate_persona) and (
            isinstance(candidate_persona, dict) and len(candidate_persona) > 0
        )
        
        # Fallback Score: If persona is missing, prepare fallback response
        if not has_persona_data:
            logger.warning("matchmaker_called_without_persona", detail="Matchmaker called without persona data. This will result in lower match scores.")
            # Return fallback response immediately if no persona
            return {
                "score": 0,  # Default to 0 instead of None
                "analysis": "No CV profile found. Please upload your CV in the Career & Match Hub for a detailed analysis.",
                "reasoning": "No CV profile found. Please upload your CV in the Career & Match Hub for a detailed analysis."
            }
        
        logger.info(
            "analyzing_match",
            agent_name=self.agent_name,
            vacancy_length=len(vacancy_text),
            has_persona=has_persona_data,
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
            import json
            import re
            messages = [HumanMessage(content=analysis_prompt)]
            # Limit response to 500 tokens (approximately 375 words) to prevent verbose responses
            # This ensures responses stay under 150 words as requested
            response = await self.invoke(messages, max_tokens=500)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Parse JSON response (expected format: {"score": 0-100, "analysis": "text"})
            ai_score = None
            analysis_text = None
            
            try:
                # JSON Cleaning: Use helper method to strip markdown code blocks
                cleaned_text = self._clean_json_response(response_text)
                
                # Try to parse as JSON
                parsed_json = json.loads(cleaned_text)
                ai_score = parsed_json.get("score")
                analysis_text = parsed_json.get("analysis", "")
                
                # Explicit Score Extraction: If score is a string like "85/100", extract the number
                if ai_score is not None:
                    if isinstance(ai_score, str):
                        # Try to extract score from string format
                        extracted_score = self._extract_score_from_text(ai_score)
                        if extracted_score is not None:
                            ai_score = extracted_score
                        else:
                            # Try to convert string directly to int
                            try:
                                ai_score = int(ai_score)
                            except (ValueError, TypeError):
                                logger.warning("match_score_string_conversion_failed", score=ai_score)
                                ai_score = None
                    elif isinstance(ai_score, (int, float)):
                        # Convert to int and clamp to 0-100 range
                        ai_score = int(ai_score)
                        ai_score = max(0, min(100, ai_score))
                    else:
                        logger.warning("match_score_invalid_type", score=ai_score, score_type=type(ai_score).__name__)
                        ai_score = None
                
            except json.JSONDecodeError:
                # Fallback: Try to extract score from text format (for backward compatibility)
                logger.warning("match_json_parse_failed", response_preview=response_text[:200])
                
                # Explicit Score Extraction: Try to extract score from text
                ai_score = self._extract_score_from_text(response_text)
                
                if ai_score is not None:
                    # Remove score line from analysis text
                    analysis_text = re.sub(r'Score:\s*\d+/(?:100|10)\s*', '', response_text, flags=re.IGNORECASE).strip()
                    # Also clean any markdown code blocks from analysis
                    analysis_text = self._clean_json_response(analysis_text)
                else:
                    # No score found in text, use full response as analysis
                    analysis_text = self._clean_json_response(response_text)
            
            # Fallback Score: If parsing fails or score not found, default to 0
            if ai_score is None:
                logger.warning("match_score_not_found", response_preview=response_text[:200])
                ai_score = 0  # Default to 0 instead of None
                if not analysis_text or len(analysis_text.strip()) == 0:
                    analysis_text = "Data missing for analysis. Unable to generate match score."
            
            # Additional safety: truncate analysis if somehow still too long
            if analysis_text and len(analysis_text) > 1000:  # ~150 words at ~6.5 chars/word
                analysis_text = analysis_text[:1000] + "..."
                logger.warning("match_analysis_truncated", original_length=len(response_text))

            logger.info("match_analysis_completed", analysis_length=len(analysis_text) if analysis_text else 0, ai_score=ai_score)
            return {
                "score": ai_score,
                "analysis": analysis_text,
                "reasoning": analysis_text  # Keep for backward compatibility
            }

        except Exception as e:
            logger.error("match_analysis_error", error=str(e), error_type=type(e).__name__)
            # Fallback Score: Return fallback result instead of raising to prevent crashing
            # Ensure score defaults to 0 and message is clear
            return {
                "score": 0,  # Default to 0 instead of None
                "analysis": "Data missing for analysis. Unable to generate match score due to an error.",
                "reasoning": "Data missing for analysis. Unable to generate match score due to an error."
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

