"""
LangGraph nodes for multi-agent orchestration.

This module contains the node implementations for the LangGraph workflow,
each node representing a step in the multi-agent system.
"""
import json
import logging
import os
import re
import asyncio
from typing import Dict, Any, List, Optional, Tuple
import structlog
import httpx
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage

from apps.orchestrator.graph.state import AgentState, UserProfile
from apps.orchestrator.agents.talent_strategist import TalentStrategistAgent
from apps.orchestrator.tools.search_tool import search_vacancies_tool
from apps.orchestrator.agents.matchmaker import MatchmakerAgent
from shared.pinecone_client import VectorStore
from src.schemas.vacancy import Vacancy, RoleCategory, ExperienceLevel, CompanyStage

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

NON_LOCATION_PHRASES = {
    "office only",
    "onsite",
    "on-site",
    "on site",
    "in office",
    "in-office",
    "not remote",
    "non-remote",
}


def _normalize_office_only_location(location: Optional[str]) -> Optional[str]:
    if not location:
        return None
    normalized = location.strip().lower()
    if normalized in NON_LOCATION_PHRASES:
        return None
    return location


def _escape_xml_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def get_context_prompt(state: AgentState) -> str:
    """
    Build XML-like context block for system prompts.
    """
    user_profile = state.get("user_profile")
    cv_text = state.get("cv_text") or ""
    is_cv_attached = bool(state.get("is_cv_attached", False))

    summary = user_profile.summary if user_profile else ""
    skills = user_profile.skills if user_profile and user_profile.skills else []
    experience_years = user_profile.experience_years if user_profile else 0
    location_pref = user_profile.location_pref if user_profile else ""
    salary_expectation = user_profile.salary_expectation if user_profile else 0
    industries = user_profile.industries if user_profile and user_profile.industries else []

    skills_lines = "\n".join(
        f"      <skill>{_escape_xml_value(skill)}</skill>" for skill in skills
    )
    industries_lines = "\n".join(
        f"      <industry>{_escape_xml_value(industry)}</industry>" for industry in industries
    )

    context_lines = [
        "<context>",
        "  <priority_note>Always prioritize this context over conflicting message history.</priority_note>",
        "  <user_profile>",
        f"    <summary>{_escape_xml_value(summary)}</summary>",
        f"    <experience_years>{experience_years}</experience_years>",
        f"    <location_pref>{_escape_xml_value(location_pref)}</location_pref>",
        f"    <salary_expectation>{salary_expectation}</salary_expectation>",
        "    <skills>",
        skills_lines if skills_lines else "      <skill></skill>",
        "    </skills>",
        "    <industries>",
        industries_lines if industries_lines else "      <industry></industry>",
        "    </industries>",
        "  </user_profile>",
        f"  <cv_text attached=\"{str(is_cv_attached).lower()}\">{_escape_xml_value(cv_text)}</cv_text>",
        "</context>",
    ]

    return "\n".join(context_lines)


def _clean_json_response(text: str) -> str:
    if not text:
        return "{}"
    cleaned = text.strip()
    if "```" in cleaned:
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(1)
        else:
            cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    cleaned = cleaned.strip()
    if not cleaned.startswith("{"):
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)
    return cleaned


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _parse_user_profile_response(response_text: str, fallback: UserProfile) -> UserProfile:
    cleaned = _clean_json_response(response_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(data, dict):
        return fallback

    merged = fallback.model_dump()
    for key in [
        "summary",
        "skills",
        "experience_years",
        "location_pref",
        "salary_expectation",
        "salary_min",
        "industries",
        "company_stage_pref",
        "target_seniority",
    ]:
        if key in data and data[key] is not None:
            merged[key] = data[key]

    merged["skills"] = _coerce_str_list(merged.get("skills"))
    merged["industries"] = _coerce_str_list(merged.get("industries"))
    merged["company_stage_pref"] = _coerce_str_list(merged.get("company_stage_pref"))

    try:
        merged["experience_years"] = max(0, int(merged.get("experience_years", 0)))
    except (ValueError, TypeError):
        merged["experience_years"] = fallback.experience_years

    try:
        merged["salary_expectation"] = max(0, int(merged.get("salary_expectation", 0)))
    except (ValueError, TypeError):
        merged["salary_expectation"] = fallback.salary_expectation

    try:
        merged["salary_min"] = max(0, int(merged.get("salary_min", merged.get("salary_expectation", 0))))
    except (ValueError, TypeError):
        merged["salary_min"] = fallback.salary_min

    return UserProfile(**merged)


def _build_profile_update_prompt(
    current_profile: UserProfile,
    user_message: str,
    cv_text: Optional[str],
    chat_history: List[Dict[str, str]],
) -> str:
    history_lines = []
    for msg in chat_history[-5:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role}: {content}")

    history_text = "\n".join(history_lines) if history_lines else "No recent history."
    profile_json = json.dumps(current_profile.model_dump(), indent=2)
    cv_block = cv_text.strip() if cv_text else ""

    return f"""Update the user profile using the latest interaction.

Current User Profile (JSON):
{profile_json}

Latest User Message:
{user_message or "No new user message."}

Recent Chat History:
{history_text}

CV Text (if provided):
{cv_block or "No CV text available."}

Return ONLY valid JSON matching this schema:
{{
  "summary": string,
  "skills": [string],
  "experience_years": int,
  "location_pref": string,
  "salary_expectation": int,
  "salary_min": int,
  "company_stage_pref": [string],
  "target_seniority": string,
  "industries": [string]
}}

Rules:
- Preserve existing values unless the user explicitly changes them.
- Normalize salary_expectation and experience_years to integers.
- Parse salary mentions like "$500k" into 500000.
- Keep salary_min aligned with the user's stated minimum salary intent.
- Extract company_stage_pref from explicit user intent (Seed, Series A, Series B, Series C, Growth, Public).
- If user states a target seniority (e.g., "Junior"), set target_seniority to that even if CV indicates otherwise.
- Keep skills and industries as string lists (dedupe if needed).
"""


def _format_technical_snapshot(profile: UserProfile, search_params: Dict[str, Any]) -> str:
    skills = ", ".join(profile.skills) if profile.skills else "None"
    industries = ", ".join(profile.industries) if profile.industries else "None"
    location_pref = profile.location_pref or "None"
    summary = profile.summary or "None"

    company_stage_pref = ", ".join(profile.company_stage_pref) if profile.company_stage_pref else "None"
    target_seniority = profile.target_seniority or "None"

    snapshot_lines = [
        "Technical Snapshot",
        f"- Summary: {summary}",
        f"- Skills: {skills}",
        f"- Experience Years: {profile.experience_years}",
        f"- Target Seniority: {target_seniority}",
        f"- Location Preference: {location_pref}",
        f"- Salary Expectation: {profile.salary_expectation}",
        f"- Salary Min: {profile.salary_min}",
        f"- Company Stage Preference: {company_stage_pref}",
        f"- Industries: {industries}",
    ]

    if search_params:
        snapshot_lines.append("- Search Filters:")
        query = search_params.get("query")
        if query:
            snapshot_lines.append(f"  - Query: {query}")
        filter_params = search_params.get("filter_params")
        if filter_params:
            snapshot_lines.append(f"  - Filter Params: {json.dumps(filter_params, ensure_ascii=True)}")
        metadata_filters = search_params.get("metadata_filters")
        if metadata_filters:
            snapshot_lines.append(f"  - Metadata Filters: {json.dumps(metadata_filters, ensure_ascii=True)}")

    return "\n".join(snapshot_lines)


def _format_internal_feedback(feedback: Any) -> str:
    if isinstance(feedback, str):
        return feedback
    if not isinstance(feedback, dict):
        return str(feedback)
    message = feedback.get("message") or "Job Scout reported an issue."
    suggestions = feedback.get("suggested_changes") or []
    if isinstance(suggestions, str):
        suggestions = [suggestions]
    suggestions_text = ""
    if suggestions:
        suggestions_text = " Suggested Snapshot changes: " + "; ".join(str(s) for s in suggestions if s)
    return f"{message}{suggestions_text}"


def _build_no_results_coach_message(profile: UserProfile, feedback: str) -> str:
    criteria_parts = []
    if profile.target_seniority:
        criteria_parts.append(f"{profile.target_seniority} roles")
    if profile.salary_expectation:
        criteria_parts.append(f"a salary of ${profile.salary_expectation:,}")
    if profile.salary_min:
        criteria_parts.append(f"minimum salary ${profile.salary_min:,}")
    if profile.company_stage_pref:
        if len(profile.company_stage_pref) == 1:
            criteria_parts.append(f"at {profile.company_stage_pref[0]} stage companies")
        else:
            criteria_parts.append(f"at {', '.join(profile.company_stage_pref)} stage companies")
    if profile.industries:
        if len(profile.industries) == 1:
            criteria_parts.append(f"in the {profile.industries[0]} sector")
        else:
            criteria_parts.append(f"in the {', '.join(profile.industries)} sectors")
    if profile.location_pref:
        criteria_parts.append(f"in {profile.location_pref}")

    if criteria_parts:
        criteria_text = " ".join(criteria_parts)
        opening = (
            "I tried to find roles matching your criteria, but it looks like "
            f"{criteria_text} is unrealistic for the current market/database."
        )
    else:
        opening = (
            "I tried to find roles matching your criteria, but the requirements "
            "seem unrealistic for the current market/database."
        )

    suggestions = []
    if profile.salary_expectation:
        suggestions.append("lowering the salary expectation")
    if profile.salary_min:
        suggestions.append("lowering the minimum salary")
    if profile.location_pref:
        suggestions.append("expanding the location preference")
    if profile.company_stage_pref:
        suggestions.append("broadening the company stage preference")
    if profile.industries:
        suggestions.append("broadening the industry focus")
    if profile.target_seniority:
        suggestions.append("adjusting the target seniority")

    if suggestions:
        suggestion_text = "Try " + " or ".join(suggestions) + "."
    else:
        suggestion_text = "Try relaxing one or more filters in the Technical Snapshot."

    if feedback:
        return f"{opening}\n{feedback}\n{suggestion_text}"
    return f"{opening}\n{suggestion_text}"


def _build_violations_disclaimer(violations: List[Dict[str, Any]]) -> str:
    if not violations:
        return ""
    filter_types = sorted({v.get("filter") for v in violations if v.get("filter")})
    if not filter_types:
        return "I found these matching your skills, but some criteria differ from your preferences."
    readable = []
    if "salary" in filter_types:
        readable.append("salary")
    if "company_stage" in filter_types:
        readable.append("company stage")
    if "location" in filter_types:
        readable.append("location")
    if not readable:
        readable = [f for f in filter_types if f]
    return (
        "I found these matching your skills, but please note their "
        + " and ".join(readable)
        + " differs from your preference."
    )


def _build_final_summary_prompt(job_items: List[Dict[str, str]]) -> str:
    job_lines = "\n".join(
        f"- {item.get('title', 'Untitled')} at {item.get('company', 'Unknown')}"
        for item in job_items
    )
    return f"""Summarize the matching roles for the user.

Primary context (titles + companies):
{job_lines}

Rules:
- Do NOT claim "no roles were found" if roles are listed above.
- Provide a concise 2-4 sentence summary of what's available.
"""


def _collect_job_items_for_summary(state: AgentState) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    match_results = state.get("match_results", []) or []
    candidate_pool = state.get("candidate_pool", []) or []

    for match in match_results[:10]:
        items.append({
            "title": match.get("vacancy_title") or "Untitled",
            "company": match.get("company_name") or "Unknown",
        })

    if not items:
        for vacancy in candidate_pool[:10]:
            items.append({
                "title": getattr(vacancy, "title", None) or "Untitled",
                "company": getattr(vacancy, "company_name", None) or "Unknown",
            })

    return items


def _extract_cv_text_from_messages(messages: List[BaseMessage]) -> Optional[str]:
    """
    Extract CV text from messages if present.
    
    CV might be attached as:
    - A message with CV content
    - Metadata in a message
    - A separate attachment field
    
    Args:
        messages: List of chat messages
        
    Returns:
        CV text if found, None otherwise
    """
    # Look for CV in message content or metadata
    for message in messages:
        if hasattr(message, "content"):
            content = message.content
            # Check if content contains CV indicators
            if isinstance(content, str):
                # Look for CV markers in content
                if "cv_text" in content.lower() or "resume" in content.lower():
                    # Try to extract CV text from structured content
                    try:
                        # If content is JSON-like, parse it
                        if content.strip().startswith("{"):
                            data = json.loads(content)
                            if "cv_text" in data:
                                return data["cv_text"]
                    except (json.JSONDecodeError, KeyError):
                        pass
                    # Otherwise, return the content if it looks like CV text
                    if len(content) > 500:  # CV text is typically long
                        return content
        # Check message metadata/additional_kwargs for CV
        if hasattr(message, "additional_kwargs"):
            kwargs = message.additional_kwargs
            if isinstance(kwargs, dict) and "cv_text" in kwargs:
                return kwargs["cv_text"]
    
    return None


def _convert_user_persona_to_profile(persona: Dict[str, Any]) -> UserProfile:
    """
    Convert UserPersona (from shared.schemas) to UserProfile (new schema).
    
    Args:
        persona: Dictionary with UserPersona fields
        
    Returns:
        UserProfile instance
    """
    # Map UserPersona fields to the new UserProfile schema.
    skills = persona.get("technical_skills", [])
    if not isinstance(skills, list):
        skills = []

    # Extract salary from salary_min as integer
    salary_expectation = 0
    salary_min = persona.get("salary_min")
    if salary_min is not None:
        try:
            salary_expectation = max(0, int(salary_min))
        except (ValueError, TypeError):
            salary_expectation = 0

    # Extract location from preferred_locations (take first if multiple)
    location_pref = ""
    preferred_locations = persona.get("preferred_locations", [])
    if preferred_locations and isinstance(preferred_locations, list):
        location_pref = preferred_locations[0] if preferred_locations else ""
    elif isinstance(preferred_locations, str):
        location_pref = preferred_locations

    # Extract years of experience from chat_context or explicit field
    experience_years = 0
    explicit_years = persona.get("experience_years")
    if explicit_years is not None:
        try:
            experience_years = max(0, int(explicit_years))
        except (ValueError, TypeError):
            experience_years = 0
    else:
        chat_context = persona.get("chat_context", "")
        if chat_context:
            years_match = re.search(r'(\d+)\s*years?\s*(?:of\s*)?experience', chat_context, re.IGNORECASE)
            if years_match:
                try:
                    experience_years = max(0, int(years_match.group(1)))
                except ValueError:
                    experience_years = 0

    # Extract industries list
    industries = persona.get("preferred_industries", [])
    if not isinstance(industries, list):
        industries = [industries] if industries else []

    # Build a short summary from available context
    summary = ""
    if persona.get("chat_context"):
        summary = str(persona.get("chat_context", "")).strip()
    elif persona.get("career_goals"):
        career_goals = persona.get("career_goals")
        if isinstance(career_goals, list):
            summary = ", ".join(career_goals)
        else:
            summary = str(career_goals)

    return UserProfile(
        summary=summary,
        skills=skills,
        experience_years=experience_years,
        location_pref=location_pref,
        salary_expectation=salary_expectation,
        salary_min=salary_expectation,
        company_stage_pref=_coerce_str_list(persona.get("preferred_company_stages")),
        target_seniority=str(persona.get("target_seniority", "")).strip(),
        industries=industries,
    )


def _check_profile_completeness(profile: UserProfile) -> Tuple[bool, List[str]]:
    """
    Check if user profile has enough information for search.
    
    Required fields for search:
    - At least one skill (key tech stack)
    - Experience years
    - Location preference
    
    Note: Salary expectation and industries are optional.
    
    Args:
        profile: UserProfile to check
        
    Returns:
        Tuple of (is_complete, missing_fields)
    """
    missing_fields = []
    
    if not profile.skills:
        missing_fields.append("skills")

    if profile.experience_years <= 0:
        missing_fields.append("experience_years")

    if not profile.location_pref:
        missing_fields.append("location_pref")
    
    is_complete = len(missing_fields) == 0
    return is_complete, missing_fields


def _generate_questions_for_missing_info(missing_fields: List[str]) -> List[str]:
    """
    Generate user-friendly questions for missing information.
    
    Args:
        missing_fields: List of missing field identifiers
        
    Returns:
        List of questions to ask the user
    """
    questions = []
    
    field_to_question = {
        "skills": "What are your main technical skills or programming languages? (e.g., Python, React, Java)",
        "experience_years": "How many years of professional experience do you have?",
        "location_pref": "What is your preferred work location? (e.g., San Francisco, Remote, London)",
        "industries": "Do you have preferred industries? (e.g., Bio + Health, Fintech, Enterprise) You can say 'no preference'.",
        "salary_expectation": "What is your desired annual salary range? (e.g., 120000 or $150k-$200k)",
    }
    
    for field in missing_fields:
        if field in field_to_question:
            questions.append(field_to_question[field])
    
    return questions


async def strategist_node(state: AgentState) -> AgentState:
    """
    Talent Strategist node - analyzes messages and CV, updates user profile.
    
    This node:
    1. Analyzes messages and attached CV (if present)
    2. Updates user_profile in state using TalentStrategistAgent
    3. Checks if profile has enough data for search
    4. Generates questions if data is missing and sets status to 'awaiting_info'
    5. Sets status to 'ready_for_search' if profile is complete
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with user_profile and status
    """
    logger.info("strategist_node_started", status=state.get("status"))

    internal_feedback = state.get("internal_feedback")
    feedback_message = _format_internal_feedback(internal_feedback) if internal_feedback else ""
    needs_no_results_coaching = state.get("status") == "no_results" and bool(internal_feedback)
    needs_search_failure_explainer = state.get("status") in {"no_results", "needs_research"} and bool(internal_feedback)
    
    # Initialize TalentStrategistAgent
    strategist = TalentStrategistAgent()
    messages = state.get("messages", [])
    cv_text = state.get("cv_text") or _extract_cv_text_from_messages(messages)

    current_profile = state.get("user_profile") or UserProfile()
    context_prompt = get_context_prompt({**state, "user_profile": current_profile})
    base_prompt = strategist._load_prompt(strategist.prompt_file)
    system_prompt = f"{context_prompt}\n\n{base_prompt}"

    user_message = ""
    chat_history: List[Dict[str, str]] = []
    for msg in messages:
        content = msg.content if hasattr(msg, "content") else str(msg)
        if not isinstance(content, str):
            continue
        if isinstance(msg, HumanMessage):
            user_message = content
            chat_history.append({"role": "user", "content": content})
        else:
            chat_history.append({"role": "assistant", "content": content})

    update_prompt = _build_profile_update_prompt(
        current_profile=current_profile,
        user_message=user_message,
        cv_text=cv_text,
        chat_history=chat_history,
    )

    try:
        response = await strategist.invoke(
            [HumanMessage(content=update_prompt)],
            system_prompt=system_prompt,
            max_tokens=800,
        )
        response_text = response.content if hasattr(response, "content") else str(response)
        updated_profile = _parse_user_profile_response(response_text, current_profile)
    except Exception as e:
        logger.error("profile_update_error", error=str(e), error_type=type(e).__name__)
        updated_profile = current_profile

    is_complete, missing_fields = _check_profile_completeness(updated_profile)
    missing_info = []
    missing_questions = []
    if not is_complete:
        missing_info = missing_fields
        missing_questions = _generate_questions_for_missing_info(missing_fields)
        logger.info("profile_incomplete", missing_fields=missing_fields, questions=missing_questions)

    snapshot = _format_technical_snapshot(updated_profile, state.get("search_params", {}))
    if internal_feedback:
        feedback_message = _format_internal_feedback(internal_feedback)

    response_parts = []
    if missing_questions:
        response_parts.append("Questions to refine the snapshot: " + " ".join(missing_questions))
    response_parts.append(snapshot)
    if missing_questions:
        response_parts.append("Questions to refine the snapshot: " + " ".join(missing_questions))
    response_parts.append(snapshot)
    response_text = "\n\n".join(response_parts)

    updated_state: AgentState = {
        **state,
        "user_profile": updated_profile,
        "missing_info": missing_info,
        "missing_questions": missing_questions,
        "status": "ready_for_search" if is_complete else "awaiting_info",
        "messages": [AIMessage(content=response_text)],
        "internal_feedback": None,
    }
    
    logger.info(
        "strategist_node_completed",
        status=updated_state["status"],
        profile_complete=is_complete,
        missing_info_count=len(missing_info),
    )
    
    return updated_state


def _create_metadata_schema_rag_tool() -> str:
    """
    Create RAG tool description with metadata schema information.
    
    This provides the agent with information about available metadata fields
    and their valid enum values for proper filter construction.
    
    Returns:
        String description of metadata schema for RAG tool
    """
    schema_description = f"""
METADATA SCHEMA FOR VACANCY SEARCH:

The following metadata fields are available for filtering in Pinecone:

1. CATEGORY (RoleCategory enum):
   Valid values: {', '.join([e.value for e in RoleCategory])}
   Use this field to filter by job function/category.

2. EXPERIENCE_LEVEL (ExperienceLevel enum):
   Valid values: {', '.join([e.value for e in ExperienceLevel])}
   Use this field to filter by required experience level.

3. COMPANY_STAGE (CompanyStage enum):
   Valid values: {', '.join([e.value for e in CompanyStage])}
   Use this field to filter by company funding stage.

4. INDUSTRY (string):
   Industry sector. Supported values: 'AI', 'Bio + Health', 'Consumer', 'Enterprise', 
   'Fintech', 'American Dynamism', 'Logistics', 'Marketing', 'Other'.
   Multiple industries can be comma-separated (e.g., 'AI, Enterprise').
   Case-insensitive matching is supported via $in operator with case variations.

5. LOCATION (string):
   Job location (e.g., "San Francisco", "Remote", "New York", "London").
   IMPORTANT: Location values in database may be stored in various formats:
   - Simple: "London"
   - With country: "London, UK", "London, England"
   - With full country: "London, United Kingdom"
   - Multiple formats: "London, UK, United Kingdom, London"
   The search tool automatically generates location variants to match these formats.
   Case-insensitive matching is supported via $in operator with case variations.

6. REMOTE_OPTION (boolean):
   Whether remote work is available (true/false).
   IMPORTANT: The field name in Pinecone metadata is "remote_option", NOT "remote_available".

7. MIN_SALARY (integer):
   Minimum salary in USD (e.g., 120000 for $120k).

8. REQUIRED_SKILLS (list of strings):
   List of required technical skills.

IMPORTANT:
- Always use exact enum values for category, experience_level, and company_stage.
- For company_stage, use CompanyStage.get_stage_value() helper to normalize values.
- Location and industry support case-insensitive matching.
- When building filters, use Pinecone filter syntax: {{"field": {{"$eq": value}}}} or {{"field": {{"$in": [value1, value2]}}}}.
"""
    return schema_description


async def _get_query_embedding(query_text: str, embedding_service_url: str) -> List[float]:
    """
    Get embedding for search query from embedding service.
    
    Args:
        query_text: Search query text
        embedding_service_url: URL of the embedding service
        
    Returns:
        Embedding vector as list of floats
        
    Raises:
        Exception: If embedding service is unavailable
    """
    if not query_text or not query_text.strip():
        raise ValueError("Query text is empty or None")
    
    try:
        logger.info("requesting_embedding", query_text=query_text[:100], service_url=embedding_service_url)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{embedding_service_url}/embed",
                json={"texts": [query_text]}
            )
            response.raise_for_status()
            result = response.json()
            
            if "embeddings" not in result or not result["embeddings"]:
                error_msg = f"Invalid response from embedding service: missing 'embeddings' key. Response: {result}"
                logger.error("embedding_service_invalid_response", response_keys=list(result.keys()) if isinstance(result, dict) else "not_a_dict")
                raise ValueError(error_msg)
            
            embedding = result["embeddings"][0]
            if not embedding or not isinstance(embedding, list):
                error_msg = f"Invalid embedding format: expected list, got {type(embedding)}"
                logger.error("embedding_service_invalid_format", embedding_type=type(embedding).__name__)
                raise ValueError(error_msg)
            
            logger.info("embedding_received", dim=len(embedding))
            return embedding
    except httpx.TimeoutException as e:
        error_msg = f"Embedding service timeout: {embedding_service_url}"
        logger.error("embedding_service_timeout", service_url=embedding_service_url, error=str(e))
        raise Exception(error_msg) from e
    except httpx.HTTPStatusError as e:
        error_msg = f"Embedding service HTTP error {e.response.status_code}: {e.response.text[:200]}"
        logger.error("embedding_service_http_error", status_code=e.response.status_code, response_text=e.response.text[:200])
        raise Exception(error_msg) from e
    except httpx.RequestError as e:
        error_msg = f"Embedding service connection error: {str(e)}"
        logger.error("embedding_service_connection_error", service_url=embedding_service_url, error=str(e))
        raise Exception(error_msg) from e
    except Exception as e:
        error_msg = f"Embedding service error: {str(e)}"
        logger.error("embedding_service_error", error=str(e), error_type=type(e).__name__, service_url=embedding_service_url)
        raise Exception(error_msg) from e


def _metadata_to_vacancy(metadata: Dict[str, Any]) -> Optional[Vacancy]:
    """
    Convert Pinecone metadata dictionary to Vacancy object.
    
    This is a simplified version of metadata_to_vacancy from vacancies.py.
    
    Args:
        metadata: Metadata dictionary from Pinecone
        
    Returns:
        Vacancy object or None if conversion fails
    """
    try:
        # Extract company_stage and normalize
        company_stage_raw = metadata.get("company_stage", "Growth")
        company_stage = CompanyStage.get_stage_value(company_stage_raw)
        
        # Extract industry
        industry = metadata.get("industry") or "Technology"
        
        # Extract salary_range
        salary_range = metadata.get("salary_range")
        
        # Parse min_salary from salary_range
        min_salary = None
        if salary_range and isinstance(salary_range, str):
            # Simple parsing: extract first number from salary range
            numbers = re.findall(r'\d+[\d,]*', salary_range)
            if numbers:
                try:
                    min_salary = int(numbers[0].replace(',', ''))
                except ValueError:
                    pass
        
        # Also check if min_salary is directly in metadata
        if not min_salary and "min_salary" in metadata:
            try:
                min_salary_value = metadata.get("min_salary")
                if min_salary_value is not None:
                    min_salary = int(min_salary_value)
            except (ValueError, TypeError):
                pass
        
        # Create Vacancy object
        vacancy = Vacancy(
            title=metadata.get("title", "Unknown"),
            company_name=metadata.get("company_name", "Unknown"),
            company_stage=company_stage,
            location=metadata.get("location", "Not specified"),
            industry=industry,
            category=metadata.get("category"),
            experience_level=metadata.get("experience_level"),
            salary_range=salary_range,
            description_url=metadata.get("description_url", ""),
            required_skills=metadata.get("required_skills", []),
            remote_option=metadata.get("remote_option", False),
            is_hybrid=metadata.get("is_hybrid", False),
            employee_count=metadata.get("employee_count"),
            full_description=metadata.get("text", "") or metadata.get("full_description", ""),
            min_salary=min_salary,
        )
        
        return vacancy
    except Exception as e:
        logger.warning(
            "vacancy_conversion_failed",
            error=str(e),
            error_type=type(e).__name__,
            metadata_keys=list(metadata.keys()),
            metadata_sample={
                "title": metadata.get("title", "N/A"),
                "company_name": metadata.get("company_name", "N/A"),
                "location": metadata.get("location", "N/A"),
                "company_stage": metadata.get("company_stage", "N/A"),
                "industry": metadata.get("industry", "N/A"),
                "category": metadata.get("category", "N/A"),
            },
        )
        return None


def _identify_strict_criteria(search_params: Dict[str, Any], results_count: int, min_score: float) -> List[str]:
    """
    Identify which search criteria might be too strict based on results.
    
    Args:
        search_params: Search parameters that were used
        results_count: Number of results found
        min_score: Minimum similarity score from results
        
    Returns:
        List of criteria that might be too strict
    """
    strict_criteria = []
    
    filter_params = search_params.get("filter_params", {})
    metadata_filters = search_params.get("metadata_filters", {})
    
    # Check location filter
    if filter_params.get("location") or metadata_filters.get("location"):
        strict_criteria.append("location")
    
    # Check company_stage filter
    if filter_params.get("company_stages") or metadata_filters.get("company_stage"):
        strict_criteria.append("company_stage")
    
    # Check industry filter
    if filter_params.get("industry") or metadata_filters.get("industry"):
        strict_criteria.append("industry")
    
    # Check experience_level filter
    if filter_params.get("experience_level") or metadata_filters.get("experience_level"):
        strict_criteria.append("experience_level")
    
    # Check category filter
    if filter_params.get("category") or metadata_filters.get("category"):
        strict_criteria.append("category")
    
    # Check remote filter
    # IMPORTANT: Pinecone uses "remote_option", not "remote_available"
    if filter_params.get("is_remote") is not None or metadata_filters.get("remote_option") or metadata_filters.get("remote_available"):
        strict_criteria.append("remote_preference")
    
    # Check salary filter
    if filter_params.get("min_salary") or metadata_filters.get("min_salary"):
        strict_criteria.append("salary")
    
    # Check skills filter
    if filter_params.get("skills"):
        strict_criteria.append("skills")
    
    return strict_criteria


async def job_scout_node(state: AgentState) -> AgentState:
    """
    Job Scout node - performs job search using profile-only inputs.
    
    This node:
    1. Builds query and filters strictly from state.user_profile
    2. Calls search_vacancies_tool directly (no chat context)
    3. Converts tool results to Vacancy objects
    4. Emits internal_feedback when 0 results are returned
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with candidate_pool and search_params
    """
    logger.info("job_scout_node_started", status=state.get("status"))
    state["candidate_pool"] = []
    state["match_results"] = []
    return_data = {"candidate_pool": [], "match_results": [], "internal_feedback": ""}
    state = {**state, **return_data, "missing_questions": []}
    
    # Check if user_profile exists
    user_profile = state.get("user_profile")
    if not user_profile:
        logger.warning("no_user_profile", setting_status="awaiting_info")
        internal_feedback = {
            "source": "job_scout",
            "message": "Search skipped: user_profile is missing in state.",
            "suggested_changes": ["Provide profile summary, skills, and location preference."],
        }
        updated_state: AgentState = {
            **state,
            "status": "awaiting_info",
            "missing_info": ["user_profile"],
            "missing_questions": [],
            "candidate_pool": [],
            "search_params": {},
            "internal_feedback": internal_feedback,
        }
        return updated_state
    
    is_complete, missing_fields = _check_profile_completeness(user_profile)
    if missing_fields:
        logger.info("job_scout_profile_incomplete", missing_fields=missing_fields)
        internal_feedback = {
            "source": "job_scout",
            "message": "Search skipped: user_profile missing required fields.",
            "suggested_changes": missing_fields,
        }
        updated_state: AgentState = {
            **state,
            "status": "awaiting_info",
            "missing_info": missing_fields,
            "missing_questions": [],
            "candidate_pool": [],
            "search_params": {},
            "internal_feedback": internal_feedback,
        }
        return updated_state

    query_parts = []
    if user_profile.summary:
        query_parts.append(user_profile.summary)
    if user_profile.skills:
        query_parts.append(" ".join(user_profile.skills[:8]))
    query = " ".join(query_parts).strip()

    if not query:
        internal_feedback = {
            "source": "job_scout",
            "message": "Search skipped: user_profile has no summary or skills to build a query.",
            "suggested_changes": ["Add a profile summary or skills list."],
        }
        updated_state: AgentState = {
            **state,
            "status": "awaiting_info",
            "missing_info": ["summary", "skills"],
            "candidate_pool": [],
            "search_params": {},
            "internal_feedback": internal_feedback,
        }
        return updated_state

    location = user_profile.location_pref or None
    industries = ", ".join(user_profile.industries) if user_profile.industries else None
    salary_min = user_profile.salary_min if user_profile.salary_min > 0 else None
    company_stage = _coerce_str_list(user_profile.company_stage_pref)
    if not company_stage:
        company_stage = None
    filter_params = {
        "location": location,
        "industry": industries,
        "salary_min": salary_min,
        "company_stage": company_stage,
    }
    filter_params = {k: v for k, v in filter_params.items() if v}
    current_search_params = {
        "query": query,
        "filter_params": filter_params,
    }
    state["search_params"] = current_search_params

    try:
        tool_result = search_vacancies_tool.invoke({
            "query": query,
            "location": location,
            "industry": industries,
            "salary_min": salary_min,
            "company_stage": company_stage,
            "top_k": 50,
        })

        search_results_raw = tool_result.get("results", [])
        results_count = tool_result.get("count", 0)
        search_error = tool_result.get("error")
        current_search_params = {
            **current_search_params,
            "filter_params": filter_params,
            "metadata_filters": tool_result.get("filters_applied"),
        }
        state["search_params"] = current_search_params

        if results_count == 0:
            filters = []
            if company_stage:
                filters.append(f"company_stage in {', '.join(company_stage)}")
            if salary_min:
                filters.append(f"salary > {salary_min}")
            if location:
                filters.append(f"location in {location}")
            if industries:
                filters.append(f"industry in {industries}")

            if filters:
                feedback = "No jobs found for " + " and ".join(filters) + "."
            else:
                feedback = f"No jobs found for query '{query}'."

            return {
                **state,
                "status": "no_results",
                "internal_feedback": feedback,
                "search_params": current_search_params,
            }

        logger.info(
            "tool_based_search_completed",
            results_count_before_conversion=len(search_results_raw),
            search_params=current_search_params,
            raw_results_sample=[
                {
                    "id": r.get("id", "unknown"),
                    "title": r.get("metadata", {}).get("title", "Unknown"),
                    "score": r.get("score", 0.0),
                }
                for r in search_results_raw[:5]
            ] if search_results_raw else [],
        )
        
        # Convert raw results to Vacancy objects
        vacancies = []
        conversion_failures = []
        for result in search_results_raw:
            metadata = result.get("metadata", {})
            vacancy = _metadata_to_vacancy(metadata)
            if vacancy:
                vacancies.append(vacancy)
            else:
                # Log conversion failures for debugging
                conversion_failures.append({
                    "id": result.get("id", "unknown"),
                    "title": metadata.get("title", "Unknown"),
                    "score": result.get("score", 0.0),
                    "metadata_keys": list(metadata.keys()),
                })
        
        if conversion_failures:
            logger.warning(
                "vacancy_conversion_failures",
                failures_count=len(conversion_failures),
                failures=conversion_failures[:10],  # Log first 10 failures
            )

        updated_state: AgentState = {
            **state,
            "status": "search_complete",
            "candidate_pool": vacancies,
            "search_params": current_search_params,
            "missing_info": [],
            "internal_feedback": None,
        }
        
        logger.info(
            "job_scout_node_completed",
            status=updated_state["status"],
            vacancies_count=len(vacancies),
        )
        
        return updated_state
        
    except Exception as e:
        logger.error(
            "job_scout_node_error",
            error=str(e),
            error_type=type(e).__name__,
        )
        internal_feedback = {
            "source": "job_scout",
            "message": f"Search failed: {str(e)}",
            "suggested_changes": ["Try reducing filters or retry the search later."],
        }
        updated_state: AgentState = {
            **state,
            "status": "awaiting_info",
            "missing_info": ["search_error"],
            "candidate_pool": [],
            "search_params": current_search_params,
            "internal_feedback": internal_feedback,
        }
        return updated_state


def _format_extracted_entities(extracted) -> str:
    """
    Format extracted entities from vacancy for matchmaker analysis.
    
    Args:
        extracted: ExtractedEntities object from vacancy
        
    Returns:
        Formatted string with extracted information
    """
    if not extracted:
        return "No extracted entities available."
    
    parts = []
    
    # Role information
    if hasattr(extracted, 'role') and extracted.role:
        role = extracted.role
        role_parts = []
        
        if role.must_skills:
            role_parts.append(f"Must-have skills: {', '.join(role.must_skills)}")
        if role.nice_skills:
            role_parts.append(f"Nice-to-have skills: {', '.join(role.nice_skills)}")
        if role.tech_stack:
            role_parts.append(f"Tech stack: {', '.join(role.tech_stack)}")
        if role.experience_years_min:
            role_parts.append(f"Minimum experience: {role.experience_years_min} years")
        if role.seniority_signal:
            role_parts.append(f"Seniority: {role.seniority_signal}")
        if role.responsibilities_core:
            role_parts.append(f"Core responsibilities: {len(role.responsibilities_core)} items")
        
        if role_parts:
            parts.append("ROLE REQUIREMENTS:\n" + "\n".join(f"  - {p}" for p in role_parts))
    
    # Company information
    if hasattr(extracted, 'company') and extracted.company:
        company = extracted.company
        company_parts = []
        
        if company.domain_tags:
            company_parts.append(f"Domain tags: {', '.join(company.domain_tags)}")
        if company.product_type:
            company_parts.append(f"Product type: {company.product_type}")
        if company.go_to_market:
            company_parts.append(f"Go-to-market: {company.go_to_market}")
        if company.culture_signals:
            company_parts.append(f"Culture: {', '.join(company.culture_signals)}")
        if company.scale_signals:
            company_parts.append(f"Scale indicators: {', '.join(company.scale_signals)}")
        
        if company_parts:
            parts.append("COMPANY PROFILE:\n" + "\n".join(f"  - {p}" for p in company_parts))
    
    # Constraints
    if hasattr(extracted, 'constraints') and extracted.constraints:
        constraints = extracted.constraints
        constraint_parts = []
        
        if constraints.timezone:
            constraint_parts.append(f"Timezone: {constraints.timezone}")
        if constraints.visa_or_work_auth:
            constraint_parts.append(f"Visa/Work auth: {constraints.visa_or_work_auth}")
        if constraints.travel_required:
            constraint_parts.append("Travel required: Yes")
        
        if constraint_parts:
            parts.append("CONSTRAINTS:\n" + "\n".join(f"  - {p}" for p in constraint_parts))
    
    return "\n\n".join(parts) if parts else "No extracted entities available."


def _format_blocks(blocks) -> str:
    """
    Format blocks from vacancy for matchmaker analysis.
    
    Args:
        blocks: Blocks object from vacancy
        
    Returns:
        Formatted string with block information
    """
    if not blocks:
        return "No block structure available."
    
    parts = []
    
    block_order = ["META", "CONTEXT", "WORK", "FIT", "OFFER"]
    block_names = {
        "META": "Metadata",
        "CONTEXT": "Company Context",
        "WORK": "Work Responsibilities",
        "FIT": "Requirements & Fit",
        "OFFER": "Offer & Benefits"
    }
    
    for block_key in block_order:
        block_content = getattr(blocks, block_key, None)
        if block_content:
            block_parts = []
            
            if hasattr(block_content, 'headings') and block_content.headings:
                block_parts.append("Headings: " + " | ".join(block_content.headings))
            
            if hasattr(block_content, 'units') and block_content.units:
                # Limit to first 5 units to avoid token limits
                units_to_show = block_content.units[:5]
                block_parts.append(f"Key points ({len(block_content.units)} total):")
                for unit in units_to_show:
                    # Limit unit length
                    unit_text = unit[:200] + "..." if len(unit) > 200 else unit
                    block_parts.append(f"  • {unit_text}")
            
            if block_parts:
                parts.append(f"{block_names.get(block_key, block_key)}:\n" + "\n".join(block_parts))
    
    return "\n\n".join(parts) if parts else "No block structure available."


def _get_evidence_quotes(evidence_map: Dict[str, List[str]], field: str) -> List[str]:
    """
    Get evidence quotes for a specific field from evidence_map.
    
    Args:
        evidence_map: Evidence map dictionary
        field: Field name to get evidence for
        
    Returns:
        List of evidence quotes
    """
    if not evidence_map:
        return []
    
    # Try exact match first
    if field in evidence_map:
        return evidence_map[field]
    
    # Try case-insensitive match
    field_lower = field.lower()
    for key, values in evidence_map.items():
        if key.lower() == field_lower:
            return values
    
    # Try partial match (e.g., "must_skills" matches "skills")
    for key, values in evidence_map.items():
        if field_lower in key.lower() or key.lower() in field_lower:
            return values
    
    return []


def _select_evidence_quotes(
    evidence_map: Optional[Dict[str, List[str]]],
    blocks,
    evidence_keys: List[str],
    block_keys: List[str],
    max_quotes: int = 2
) -> List[str]:
    quotes = []
    if evidence_map:
        for key in evidence_keys:
            quotes.extend(_get_evidence_quotes(evidence_map, key))
            if len(quotes) >= max_quotes:
                return quotes[:max_quotes]

    if blocks:
        for block_key in block_keys:
            block_content = getattr(blocks, block_key, None)
            if block_content and hasattr(block_content, "units") and block_content.units:
                for unit in block_content.units:
                    if unit and isinstance(unit, str):
                        quotes.append(unit.strip())
                        if len(quotes) >= max_quotes:
                            return quotes[:max_quotes]

    return quotes[:max_quotes]


async def _analyze_vacancy_match(
    matchmaker: MatchmakerAgent,
    vacancy: Vacancy,
    user_profile: UserProfile,
    context_prompt: str,
    similarity_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Analyze a single vacancy match against user profile.
    
    Args:
        matchmaker: MatchmakerAgent instance
        vacancy: Vacancy to analyze
        user_profile: User profile
        similarity_score: Optional similarity score from vector search
        
    Returns:
        Dictionary with match results including role_match and company_match summaries
    """
    # Convert UserProfile to persona dict
    # Parse salary_expectation string to numeric value if needed
    salary_min = None
    if user_profile.salary_expectation:
        try:
            # Parse salary_expectation string (e.g., "$150k" -> 150000, "150000" -> 150000)
            salary_str = user_profile.salary_expectation.replace("$", "").replace(",", "").strip()
            if salary_str.lower().endswith("k"):
                salary_min = int(float(salary_str[:-1]) * 1000)
            else:
                salary_min = int(float(salary_str))
        except (ValueError, TypeError, AttributeError):
            # If parsing fails, leave as None
            salary_min = None
    
    persona_dict = {
        "technical_skills": user_profile.skills or [],
        "salary_min": salary_min,
        "preferred_locations": [user_profile.location_pref] if user_profile.location_pref else [],
        "location": user_profile.location_pref,
        "industries": user_profile.industries or [],
        "summary": user_profile.summary,
    }

    if user_profile.experience_years:
        persona_dict["experience_years"] = user_profile.experience_years
    
    # Build vacancy context from extracted and blocks
    vacancy_parts = []
    
    # Add basic vacancy info first
    basic_info = f"Title: {vacancy.title}\nCompany: {vacancy.company_name}\nLocation: {vacancy.location}\nIndustry: {vacancy.industry}\nCompany Stage: {vacancy.company_stage}\nRequired Skills: {', '.join(vacancy.required_skills) if vacancy.required_skills else 'Not specified'}\nSalary Range: {vacancy.salary_range or 'Not specified'}"
    vacancy_parts.append(basic_info)
    
    # Add extracted entities
    if vacancy.extracted:
        extracted_text = _format_extracted_entities(vacancy.extracted)
        if extracted_text and extracted_text != "No extracted entities available.":
            vacancy_parts.append(extracted_text)
    
    # Add blocks
    if vacancy.blocks:
        blocks_text = _format_blocks(vacancy.blocks)
        if blocks_text and blocks_text != "No block structure available.":
            vacancy_parts.append(blocks_text)
    
    # Fallback to full_description if extracted/blocks not available
    if vacancy.full_description and vacancy.full_description.strip() and vacancy.full_description != "Parsing Error":
        vacancy_parts.append(f"Full Description: {vacancy.full_description[:2000]}")
    
    # If we still have no content, add a note
    if len(vacancy_parts) == 1:  # Only basic_info
        vacancy_parts.append("Note: Limited vacancy information available. Analysis based on metadata only.")
    
    vacancy_context = "\n\n".join(vacancy_parts)

    role_evidence_quotes = _select_evidence_quotes(
        vacancy.evidence_map,
        vacancy.blocks,
        evidence_keys=["must_skills", "skills", "requirements", "responsibilities"],
        block_keys=["FIT", "WORK"],
        max_quotes=2,
    )

    company_evidence_quotes = _select_evidence_quotes(
        vacancy.evidence_map,
        vacancy.blocks,
        evidence_keys=["domain", "company", "culture", "industry"],
        block_keys=["CONTEXT", "META"],
        max_quotes=2,
    )

    role_prompt = """You are scoring Candidate vs Role fit.
Return ONLY valid JSON:
{
  "score": integer (0-10),
  "analysis": "2-4 sentences summary focused on role fit and gaps."
}

Rules:
- Use candidate skills, role requirements, seniority and tech stack.
- Use Evidence Quotes when referencing requirements.
- Be concise. No bullet lists. No extra text."""

    company_prompt = """You are scoring Candidate vs Company fit.
Return ONLY valid JSON:
{
  "score": integer (0-10),
  "analysis": "2-4 sentences summary focused on company/industry/stage fit and gaps."
}

Rules:
- Use candidate industry/stage/location/remote preference when relevant.
- Use Evidence Quotes when referencing company context.
- Be concise. No bullet lists. No extra text."""

    role_details = []
    if vacancy.extracted and vacancy.extracted.role:
        role = vacancy.extracted.role
        if role.must_skills:
            role_details.append(f"Must skills: {', '.join(role.must_skills)}")
        if role.tech_stack:
            role_details.append(f"Tech stack: {', '.join(role.tech_stack[:10])}")
        if role.responsibilities_core:
            role_details.append(f"Responsibilities: {', '.join(role.responsibilities_core[:8])}")
        if role.experience_years_min:
            role_details.append(f"Experience years min: {role.experience_years_min}")
        if role.seniority_signal:
            role_details.append(f"Seniority signal: {role.seniority_signal}")

    company_details = []
    if vacancy.extracted and vacancy.extracted.company:
        company = vacancy.extracted.company
        if company.domain_tags:
            company_details.append(f"Domain tags: {', '.join(company.domain_tags[:6])}")
        if company.product_type:
            company_details.append(f"Product type: {company.product_type}")
        if company.go_to_market:
            company_details.append(f"GTM: {company.go_to_market}")
        if company.culture_signals:
            company_details.append(f"Culture signals: {', '.join(company.culture_signals[:6])}")
        if company.scale_signals:
            company_details.append(f"Scale signals: {', '.join(company.scale_signals[:6])}")

    role_context = f"""ROLE CONTEXT
Title: {vacancy.title}
Category: {vacancy.category}
Experience Level: {vacancy.experience_level}
Required Skills: {', '.join(vacancy.required_skills) if vacancy.required_skills else 'Not specified'}
Role Details:
{chr(10).join(role_details) if role_details else 'No extracted role details.'}

Evidence Quotes:
{chr(10).join(role_evidence_quotes) if role_evidence_quotes else 'None'}"""

    company_context = f"""COMPANY CONTEXT
Company: {vacancy.company_name}
Industry: {vacancy.industry}
Company Stage: {vacancy.company_stage}
Location: {vacancy.location}
Company Details:
{chr(10).join(company_details) if company_details else 'No extracted company details.'}

Evidence Quotes:
{chr(10).join(company_evidence_quotes) if company_evidence_quotes else 'None'}"""

    role_system_prompt = f"{context_prompt}\n\n{role_prompt}"
    role_match = await matchmaker.analyze_match(
        vacancy_text=role_context,
        candidate_persona=persona_dict,
        similarity_score=similarity_score,
        system_prompt=role_system_prompt,
        score_max=10,
    )

    company_system_prompt = f"{context_prompt}\n\n{company_prompt}"
    company_match = await matchmaker.analyze_match(
        vacancy_text=company_context,
        candidate_persona=persona_dict,
        similarity_score=similarity_score,
        system_prompt=company_system_prompt,
        score_max=10,
    )
    
    # Generate Role-specific summary
    role_summary = ""
    role_evidence = ""
    
    if vacancy.extracted and vacancy.extracted.role:
        role = vacancy.extracted.role
        role_requirements = []
        
        if role.must_skills:
            role_requirements.append(f"must-have skills: {', '.join(role.must_skills)}")
        if role.experience_years_min:
            role_requirements.append(f"minimum {role.experience_years_min} years experience")
        if role.tech_stack:
            role_requirements.append(f"tech stack: {', '.join(role.tech_stack[:5])}")
        
        # Get evidence quotes
        role_evidence_quotes = []
        if vacancy.evidence_map:
            # Try to get evidence for skills
            skills_evidence = _get_evidence_quotes(vacancy.evidence_map, "must_skills")
            if not skills_evidence:
                skills_evidence = _get_evidence_quotes(vacancy.evidence_map, "skills")
            if skills_evidence:
                role_evidence_quotes.extend(skills_evidence[:2])  # Take first 2
        
        # Build role summary
        user_skills_match = []
        if user_profile.skills:
            matching_skills = [s for s in user_profile.skills if any(s.lower() in req.lower() or req.lower() in s.lower() for req in (role.must_skills or []))]
            if matching_skills:
                user_skills_match = matching_skills

        role_score = role_match.get("score", 0)
        # Score already in 0-10 scale
        role_score_10 = role_score
        
        role_summary = f"Match {role_score_10}/10: "
        
        if user_skills_match:
            role_summary += f"Candidate has {', '.join(user_skills_match[:3])} skills"
            if user_profile.experience_years and role.experience_years_min:
                if user_profile.experience_years >= role.experience_years_min:
                    role_summary += f", {user_profile.experience_years} years experience (meets {role.experience_years_min}+ requirement)"
        elif user_profile.skills:
            role_summary += f"Candidate has {', '.join(user_profile.skills[:3])} skills"
        else:
            role_summary += "Candidate profile available"
        
        # Add evidence quote to requirement if available
        evidence_quote_for_req = None
        if role_evidence_quotes:
            evidence_quote_for_req = role_evidence_quotes[0]
            if len(evidence_quote_for_req) > 80:
                evidence_quote_for_req = evidence_quote_for_req[:80] + "..."
        
        if role_requirements:
            # Format requirement as [X] where X is the requirement text or evidence quote
            if evidence_quote_for_req:
                # Use evidence quote as the requirement reference
                req_text = evidence_quote_for_req
            else:
                req_text = role_requirements[0] if role_requirements else ""
            role_summary += f", matching requirement [{req_text}]"
        
        # Add evidence source
        if vacancy.blocks and vacancy.blocks.FIT:
            role_evidence = " (Source: FIT block)"
        else:
            role_evidence = ""
    
    # Generate Company-specific summary
    company_summary = ""
    company_evidence = ""
    
    if vacancy.extracted and vacancy.extracted.company:
        company = vacancy.extracted.company
        company_features = []
        
        if company.domain_tags:
            company_features.append(f"domains: {', '.join(company.domain_tags[:3])}")
        if company.product_type:
            company_features.append(f"product: {company.product_type}")
        if company.go_to_market:
            company_features.append(f"GTM: {company.go_to_market}")
        
        # Get evidence quotes
        company_evidence_quotes = []
        if vacancy.evidence_map:
            domain_evidence = _get_evidence_quotes(vacancy.evidence_map, "domain")
            if domain_evidence:
                company_evidence_quotes.extend(domain_evidence[:1])
        
        company_score = company_match.get("score", 0)
        # Score already in 0-10 scale
        company_score_10 = company_score
        company_summary = f"Company match {company_score_10}/10: "
        
        if company_features:
            company_summary += f"Company operates in {', '.join(company_features[:2])}"
        else:
            company_summary += f"Company: {vacancy.company_name} ({vacancy.company_stage})"
        
        # Add evidence quote
        if company_evidence_quotes:
            evidence_quote = company_evidence_quotes[0]
            if len(evidence_quote) > 100:
                evidence_quote = evidence_quote[:100] + "..."
            company_evidence = f" (Source: CONTEXT block - \"{evidence_quote}\")"
        elif vacancy.blocks and vacancy.blocks.CONTEXT:
            company_evidence = " (Source: CONTEXT block)"
    
    # Combine summaries with evidence
    role_match_summary = role_summary + role_evidence if role_summary else role_match.get("analysis", "")
    company_match_summary = company_summary + company_evidence if company_summary else company_match.get("analysis", "")

    role_score = role_match.get("score", 0)
    company_score = company_match.get("score", 0)
    overall_score = round((role_score + company_score) / 2, 1)
    overall_analysis = f"Role: {role_match_summary} Company: {company_match_summary}".strip()
    
    return {
        "vacancy_id": getattr(vacancy, 'id', None) or vacancy.description_url,
        "vacancy_title": vacancy.title,
        "company_name": vacancy.company_name,
        "overall_score": overall_score,
        "overall_analysis": overall_analysis,
        "role_match_summary": role_match_summary,
        "company_match_summary": company_match_summary,
        "role_score": role_score,
        "company_score": company_score,
        "role_evidence": role_evidence_quotes,
        "company_evidence": company_evidence_quotes,
        "similarity_score": similarity_score,
    }


async def matchmaker_node(state: AgentState) -> AgentState:
    """
    Matchmaker node - analyzes all vacancies from candidate_pool against user profile.
    
    This node:
    1. Runs parallel scoring for each vacancy using MatchmakerAgent
    2. Uses extracted and blocks fields from vacancies
    3. Generates text summaries for Candidate vs Role and Candidate vs Company
    4. Includes evidence quotes from evidence_map in summaries
    5. Writes results to match_results
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with match_results
    """
    logger.info("matchmaker_node_started", status=state.get("status"))
    
    # Check if user_profile exists
    user_profile = state.get("user_profile")
    if not user_profile:
        logger.warning("no_user_profile_for_matching")
        updated_state: AgentState = {
            **state,
            "status": "error",
            "match_results": [],
        }
        return updated_state
    
    # Get candidate pool
    candidate_pool = state.get("candidate_pool", [])
    if not candidate_pool:
        logger.warning("no_candidate_pool")
        updated_state: AgentState = {
            **state,
            "status": "matching_complete",
            "match_results": [],
        }
        return updated_state
    
    logger.info("starting_parallel_matching", vacancies_count=len(candidate_pool))
    
    # Initialize MatchmakerAgent (single instance for all analyses)
    matchmaker = MatchmakerAgent()
    context_prompt = get_context_prompt(state)
    
    # Create tasks for parallel analysis
    async def analyze_single_vacancy(vacancy: Vacancy) -> Dict[str, Any]:
        """Analyze a single vacancy."""
        try:
            # Get similarity score from search if available
            # Note: similarity scores might not be in state, so we'll use None
            similarity_score = None
            
            result = await _analyze_vacancy_match(
                matchmaker=matchmaker,
                vacancy=vacancy,
                user_profile=user_profile,
                context_prompt=context_prompt,
                similarity_score=similarity_score
            )
            
            logger.debug("vacancy_analyzed", vacancy_title=vacancy.title, score=result.get("overall_score"))
            return result
            
        except Exception as e:
            logger.error(
                "vacancy_analysis_error",
                vacancy_title=vacancy.title if hasattr(vacancy, 'title') else 'unknown',
                error=str(e),
                error_type=type(e).__name__
            )
            # Return error result
            return {
                "vacancy_id": getattr(vacancy, 'id', None) or getattr(vacancy, 'description_url', 'unknown'),
                "vacancy_title": getattr(vacancy, 'title', 'Unknown'),
                "company_name": getattr(vacancy, 'company_name', 'Unknown'),
                "overall_score": 0,
                "overall_analysis": f"Error analyzing vacancy: {str(e)}",
                "role_match_summary": "",
                "company_match_summary": "",
                "similarity_score": None,
                "error": str(e),
            }
    
    # Run all analyses in parallel
    try:
        tasks = [analyze_single_vacancy(vacancy) for vacancy in candidate_pool]
        match_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(match_results):
            if isinstance(result, Exception):
                logger.error(
                    "vacancy_analysis_exception",
                    vacancy_index=i,
                    error=str(result),
                    error_type=type(result).__name__
                )
                processed_results.append({
                    "vacancy_id": f"error_{i}",
                    "vacancy_title": "Unknown",
                    "company_name": "Unknown",
                    "overall_score": 0,
                    "overall_analysis": f"Error: {str(result)}",
                    "role_match_summary": "",
                    "company_match_summary": "",
                    "similarity_score": None,
                    "error": str(result),
                })
            else:
                processed_results.append(result)
        
        # Sort by overall_score (descending)
        processed_results.sort(key=lambda x: x.get("overall_score", 0), reverse=True)
        
        logger.info(
            "matchmaker_node_completed",
            results_count=len(processed_results),
            avg_score=sum(r.get("overall_score", 0) for r in processed_results) / len(processed_results) if processed_results else 0,
        )
        
        updated_state: AgentState = {
            **state,
            "status": "matching_complete",
            "match_results": processed_results,
        }
        
        return updated_state
        
    except Exception as e:
        logger.error("matchmaker_node_error", error=str(e), error_type=type(e).__name__)
        updated_state: AgentState = {
            **state,
            "status": "error",
            "match_results": [],
        }
        return updated_state


async def validator_node(state: AgentState) -> AgentState:
    """
    Validator node - audits match_results against user's hard filters.
    
    This node uses TalentStrategistAgent in audit mode to check if match_results
    comply with user's hard requirements (e.g., Remote only, specific location, etc.).
    If violations are found, it adjusts search_params and sends back to job_scout.
    
    IMPORTANT: Only triggers violations if:
    1. candidate_pool is empty (no results found)
    2. Agent hasn't already tried to relax filters (_filters_relaxed flag not set)
    
    Args:
        state: Current agent state
        
    Returns:
        Updated agent state with validation results and potentially adjusted search_params
    """
    logger.info("validator_node_started", status=state.get("status"))
    
    # Get user profile and match results
    user_profile = state.get("user_profile")
    match_results = state.get("match_results", [])
    candidate_pool = state.get("candidate_pool", [])
    search_params = state.get("search_params", {})
    
    if not user_profile:
        logger.warning("no_user_profile_for_validation")
        updated_state: AgentState = {
            **state,
            "status": "validation_complete",
        }
        return updated_state
    
    # Check if candidate_pool is empty AND filters haven't been relaxed yet
    # This prevents infinite loops when filters have already been relaxed
    filters_already_relaxed = search_params.get("_filters_relaxed", False)
    candidate_pool_empty = len(candidate_pool) == 0
    
    if candidate_pool_empty and not filters_already_relaxed:
        logger.info(
            "validator_skipping_empty_pool",
            candidate_pool_empty=True,
            filters_already_relaxed=False,
            reason="Candidate pool is empty but filters haven't been relaxed yet. Let job_scout handle relaxation."
        )
        # Don't trigger violations - let job_scout_node handle the relaxation
        updated_state: AgentState = {
            **state,
            "status": "validation_complete",
        }
        return updated_state
    
    if not match_results:
        logger.warning("no_match_results_for_validation")
        updated_state: AgentState = {
            **state,
            "status": "validation_complete",
        }
        return updated_state
    
    # Identify hard filters from user profile
    hard_filters = {}
    violations = []
    
    # Check location preference
    normalized_location = _normalize_office_only_location(user_profile.location_pref)
    if normalized_location:
        hard_filters["location"] = normalized_location
        # Check if any vacancy doesn't match location
        location_lower = normalized_location.lower()
        for i, result in enumerate(match_results):
            vacancy_id = result.get("vacancy_id")
            vacancy = next((v for v in candidate_pool if (getattr(v, 'id', None) or v.description_url) == vacancy_id), None)
            if vacancy:
                vacancy_location = (vacancy.location or "").lower()
                # Check if location matches (exact or contains)
                if location_lower not in vacancy_location and vacancy_location not in location_lower:
                    # Allow "Remote" to match any location preference
                    if "remote" not in vacancy_location.lower():
                        violations.append({
                            "vacancy_id": vacancy_id,
                            "vacancy_title": result.get("vacancy_title", "Unknown"),
                            "filter": "location",
                            "issue": f"Vacancy location '{vacancy.location}' doesn't match required '{user_profile.location_pref}'"
                        })
    
    # Check salary requirement
    if user_profile.salary_expectation:
        required_salary = user_profile.salary_expectation
        hard_filters["min_salary"] = required_salary
        for i, result in enumerate(match_results):
            vacancy_id = result.get("vacancy_id")
            vacancy = next((v for v in candidate_pool if (getattr(v, 'id', None) or v.description_url) == vacancy_id), None)
            if vacancy:
                vacancy_min_salary = vacancy.min_salary
                if vacancy_min_salary and vacancy_min_salary < required_salary:
                    violations.append({
                        "vacancy_id": vacancy_id,
                        "vacancy_title": result.get("vacancy_title", "Unknown"),
                        "filter": "salary",
                        "issue": f"Vacancy min salary ${vacancy_min_salary:,} is below required ${required_salary:,}"
                    })
    
    # Use TalentStrategistAgent to analyze violations and suggest corrections
    # Only proceed if we have violations AND candidate_pool is not empty
    # If candidate_pool is empty, let job_scout handle relaxation instead of triggering re-search
    # Note: We already checked above that if candidate_pool is empty and filters haven't been relaxed,
    # we skip validation entirely. So at this point, violations are valid to process.
    # BUT: Only trigger re-search if we have results but they violate filters
    if violations and len(candidate_pool) > 0:
        # Group violations by filter type for better logging
        violations_by_filter = {}
        for v in violations:
            filter_type = v.get("filter", "unknown")
            if filter_type not in violations_by_filter:
                violations_by_filter[filter_type] = []
            violations_by_filter[filter_type].append(v)
        
        logger.info(
            "validation_violations_found",
            violations_count=len(violations),
            violations_by_filter={k: len(v) for k, v in violations_by_filter.items()},
            candidate_pool_empty=candidate_pool_empty,
            filters_already_relaxed=filters_already_relaxed,
            violation_details=[
                {
                    "vacancy_title": v.get("vacancy_title", "Unknown"),
                    "filter": v.get("filter", "unknown"),
                    "issue": v.get("issue", ""),
                }
                for v in violations[:10]  # Log first 10 violations
            ],
        )
        
        # Initialize TalentStrategistAgent for audit
        strategist = TalentStrategistAgent()
        
        # Build violation summary
        violation_summary = f"Found {len(violations)} violations of hard filters:\n"
        for v in violations[:5]:  # Limit to first 5
            violation_summary += f"- {v['vacancy_title']}: {v['issue']}\n"
        
        # Create audit prompt
        salary_str = str(user_profile.salary_expectation) if user_profile.salary_expectation else "None"
        audit_prompt = f"""AUDIT MODE: Review search results for compliance with hard user filters.

User Hard Filters:
- Location: {user_profile.location_pref or 'None'}
- Min salary: {salary_str}

Violations Found:
{violation_summary}

Current Search Parameters:
{json.dumps(search_params, indent=2)}

Analyze the violations and suggest adjustments to search_params to ensure future results comply with hard filters.
Return JSON with:
- adjusted_filters: Updated filter parameters
- reasoning: Explanation of adjustments
"""
        
        try:
            from langchain_core.messages import HumanMessage
            messages = [HumanMessage(content=audit_prompt)]
            response = await strategist.invoke(messages, max_tokens=1000)
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Try to parse JSON response
            try:
                cleaned_response = strategist._clean_json_response(response_text)
                audit_result = json.loads(cleaned_response)
                
                # Update search_params with adjusted filters
                adjusted_filters = audit_result.get("adjusted_filters", {})
                if adjusted_filters:
                    # Merge adjusted filters into search_params
                    if "filter_params" in search_params:
                        search_params["filter_params"].update(adjusted_filters)
                    else:
                        search_params["filter_params"] = adjusted_filters
                    
                    # Update metadata_filters if needed
                    if "metadata_filters" in adjusted_filters:
                        search_params["metadata_filters"] = adjusted_filters["metadata_filters"]
                    
                    logger.info("search_params_adjusted", adjustments=adjusted_filters)

                # Enforce user intent for company stage and salary after adjustments
                if user_profile.company_stage_pref:
                    search_params.setdefault("filter_params", {})
                    search_params["filter_params"]["company_stage"] = user_profile.company_stage_pref
                if user_profile.salary_min:
                    search_params.setdefault("filter_params", {})
                    search_params["filter_params"]["salary_min"] = user_profile.salary_min
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("audit_response_parse_failed", error=str(e))
                # Fallback: manually adjust filters based on violations
                if "filter_params" not in search_params:
                    search_params["filter_params"] = {}
                
                # Explicitly add location to search_params if location violations found
                if "location" in violations_by_filter and user_profile.location_pref:
                    search_params["filter_params"]["location"] = user_profile.location_pref
                    logger.info("location_filter_added_to_search_params", location=user_profile.location_pref)

                if user_profile.company_stage_pref:
                    search_params["filter_params"]["company_stage"] = user_profile.company_stage_pref

                if user_profile.salary_min:
                    search_params["filter_params"]["salary_min"] = user_profile.salary_min
                
                if "remote_only" in hard_filters:
                    search_params["filter_params"]["is_remote"] = True
                    if "metadata_filters" not in search_params:
                        search_params["metadata_filters"] = {}
                    # IMPORTANT: Use "remote_option" field name in Pinecone metadata
                    search_params["metadata_filters"]["remote_option"] = {"$eq": True}
                
        except Exception as e:
            logger.error("audit_analysis_error", error=str(e), error_type=type(e).__name__)
            # Fallback: explicitly add location if location violations found
            if "location" in violations_by_filter and user_profile.location_pref:
                if "filter_params" not in search_params:
                    search_params["filter_params"] = {}
                search_params["filter_params"]["location"] = user_profile.location_pref
                logger.info("location_filter_added_to_search_params_fallback", location=user_profile.location_pref)

            if user_profile.company_stage_pref:
                search_params.setdefault("filter_params", {})
                search_params["filter_params"]["company_stage"] = user_profile.company_stage_pref

            if user_profile.salary_min:
                search_params.setdefault("filter_params", {})
                search_params["filter_params"]["salary_min"] = user_profile.salary_min
        
        # Set status to indicate need for re-search
        updated_state: AgentState = {
            **state,
            "status": "needs_research",
            "search_params": search_params,
            "violations": violations,
            "internal_feedback": {
                "source": "validator",
                "message": f"Found {len(violations)} violations of hard filters.",
                "details": violations[:10],
            },
            "missing_info": [],
        }
        
        logger.info("validator_node_completed_with_violations", violations_count=len(violations))
        return updated_state
    
    # No violations found
    logger.info("validation_passed", results_count=len(match_results))
    updated_state: AgentState = {
        **state,
        "status": "validation_complete",
        "violations": [],
    }
    return updated_state


def final_validation_node(state: AgentState) -> AgentState:
    """
    Final node used by Studio to render a human-readable response.
    Adds a summary message to the state so it appears in the UI.
    """
    match_results = state.get("match_results", []) or []
    candidate_pool = state.get("candidate_pool", []) or []
    missing_info = state.get("missing_info", []) or []
    missing_questions = state.get("missing_questions", []) or []

    if missing_questions or missing_info:
        return {
            **state,
        }

    if not match_results:
        search_failed_reason = state.get("search_failed_reason")
        if not search_failed_reason:
            search_params = state.get("search_params", {}) or {}
            filter_params = search_params.get("filter_params", {}) or {}
            search_failed_reason = (
                "No results for filters: "
                + json.dumps(filter_params, ensure_ascii=True)
            )
        coach_message = (
            "I searched using your filters but found 0 results. "
            f"{search_failed_reason}"
        )
        return {
            **state,
            "status": "no_results",
            "search_failed_reason": search_failed_reason,
            "internal_feedback": search_failed_reason,
            "messages": [AIMessage(content=coach_message)],
        }

    violations = state.get("violations") or []
    violations_by_id: Dict[str, List[str]] = {}
    for v in violations:
        vacancy_id = v.get("vacancy_id")
        filter_name = v.get("filter")
        if not vacancy_id or not filter_name:
            continue
        violations_by_id.setdefault(vacancy_id, []).append(filter_name)

    # Build quick lookup for vacancy details (title, url, location)
    vacancy_by_id = {}
    for vacancy in candidate_pool:
        vacancy_id = getattr(vacancy, "id", None)
        description_url = getattr(vacancy, "description_url", None)
        if vacancy_id:
            vacancy_by_id[vacancy_id] = vacancy
        if description_url:
            vacancy_by_id[description_url] = vacancy

    lines = []
    if match_results:
        sorted_matches = sorted(
            match_results,
            key=lambda item: item.get("overall_score", 0),
            reverse=True,
        )

        lines.append(f"I found {len(sorted_matches)} matching vacancies:")
        for idx, match in enumerate(sorted_matches[:5], start=1):
            vacancy_id = match.get("vacancy_id")
            vacancy = vacancy_by_id.get(vacancy_id)
            title = match.get("vacancy_title") or (getattr(vacancy, "title", None) if vacancy else None) or "Untitled"
            company = match.get("company_name") or (getattr(vacancy, "company_name", None) if vacancy else None) or "Unknown company"
            location = getattr(vacancy, "location", None) if vacancy else None
            url = getattr(vacancy, "description_url", None) if vacancy else None
            overall_score = match.get("overall_score", 0)
            role_score = match.get("role_score", 0)
            company_score = match.get("company_score", 0)
            role_summary = match.get("role_match_summary") or ""
            company_summary = match.get("company_match_summary") or ""
            role_evidence = match.get("role_evidence") or []
            company_evidence = match.get("company_evidence") or []

            line_parts = [
                f"{idx}. {title} — {company} (overall: {overall_score}, role: {role_score}/10, company: {company_score}/10)"
            ]
            if location:
                line_parts.append(f"Location: {location}")
            if url:
                line_parts.append(f"URL: {url}")
            lines.append(" | ".join(line_parts))
            if role_summary:
                lines.append(f"Role summary: {role_summary}")
            if role_evidence:
                lines.append(f"Role evidence: {', '.join(role_evidence[:2])}")
            if company_summary:
                lines.append(f"Company summary: {company_summary}")
            if company_evidence:
                lines.append(f"Company evidence: {', '.join(company_evidence[:2])}")

            if vacancy_id in violations_by_id:
                filters = violations_by_id.get(vacancy_id, [])
                readable = []
                if "salary" in filters:
                    readable.append("salary")
                if "company_stage" in filters:
                    readable.append("stage")
                label = "/".join(readable) if readable else "salary/stage"
                lines.append(f"Note: This role's {label} differs from your current preference.")
            lines.append("")
    else:
        lines.append("I found 0 matching vacancies.")

    summary = "\n".join(lines)
    return {
        **state,
        "messages": [AIMessage(content=summary)],
    }
