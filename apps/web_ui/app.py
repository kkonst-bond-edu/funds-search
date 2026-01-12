"""
Streamlit web UI for funds-search system.
Provides a user-friendly interface for CV processing, vacancy processing, and matching.
"""

import logging
import os
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import streamlit as st

# Note: PYTHONPATH should be set to project root, no manual sys.path manipulation needed
from shared.schemas import MatchingReport, VacancyMatchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
# Note: cv-processor listens on port 8001 internally (8002 is the external host mapping)
# Use localhost for local development, Azure will override with actual URL
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8000")
CV_PROCESSOR_URL = os.getenv("CV_PROCESSOR_URL", "http://cv-processor:8001")
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")

# Retry configuration for cold start handling
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
TIMEOUT = 300.0  # seconds (5 minutes) - increased for CV processing and matching operations

# Page configuration
st.set_page_config(
    page_title="AI Talent Matcher - Job Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .match-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: bold;
        background-color: #1f77b4;
        color: white;
    }
    .vacancy-id {
        font-family: monospace;
        color: #666;
        font-size: 0.9rem;
    }
    .reasoning-text {
        background-color: white;
        padding: 1rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        line-height: 1.6;
    }
    /* Vacancy Card Styles */
    .vacancy-card {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 16px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: box-shadow 0.2s ease;
    }
    .vacancy-card:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border-color: #d1d5db;
    }
    .company-name {
        font-size: 14px;
        font-weight: 600;
        color: #4b5563;
        margin-bottom: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .vacancy-title {
        font-size: 22px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 12px;
        line-height: 1.3;
    }
    .salary-loc-row {
        display: flex;
        gap: 24px;
        align-items: center;
        margin-bottom: 8px;
        color: #374151;
        font-size: 15px;
        font-weight: 500;
    }
    .meta-item {
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .secondary-meta {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 16px;
    }
    .tags-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-bottom: 16px;
    }
    .tech-tag {
        background-color: white;
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 13px;
        color: #374151;
        font-weight: 500;
    }
    .signal-tag {
        background-color: #f3f4f6;
        border-radius: 4px;
        padding: 4px 10px;
        font-size: 13px;
        color: #1f2937;
        font-weight: 500;
        display: inline-block;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .summary-text {
        font-size: 14px;
        color: #4b5563;
        margin-bottom: 20px;
        line-height: 1.6;
        background-color: #f9fafb;
        padding: 12px;
        border-radius: 6px;
    }
    .apply-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: white;
        color: #0f172a;
        font-weight: 600;
        font-size: 14px;
        text-decoration: none;
        padding: 0;
    }
    .apply-btn:hover {
        text-decoration: underline;
        color: #2563eb;
    }
    .tag-category {
        margin-bottom: 12px;
    }
    .tag-category-label {
        font-size: 11px;
        color: #6b7280;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 6px;
        letter-spacing: 0.05em;
    }
    .profile-section {
        margin-bottom: 12px;
        background-color: #f9fafb;
        padding: 12px;
        border-radius: 6px;
        border-left: 3px solid #e5e7eb;
    }
    .profile-label {
        font-size: 11px;
        color: #6b7280;
        font-weight: 700;
        margin-bottom: 4px;
        text-transform: uppercase;
    }
    .profile-text {
        font-size: 14px;
        color: #374151;
        line-height: 1.5;
    }
    </style>
""",
    unsafe_allow_html=True,
)


def check_service_health(
    service_url: str, service_name: str, timeout: float = 5.0
) -> Tuple[bool, str]:
    """
    Check if a service is healthy and responding.

    Args:
        service_url: Base URL of the service
        service_name: Name of the service for logging
        timeout: Request timeout in seconds

    Returns:
        Tuple of (is_healthy: bool, message: str)
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(f"{service_url}/health")
            if response.status_code == 200:
                return True, "Online"
            else:
                return False, f"Unhealthy (HTTP {response.status_code})"
    except socket.gaierror:
        error_msg = f"API not reachable at {service_url}. Please verify {service_name.upper()}_URL configuration."
        logger.error(f"DNS resolution error for {service_name}: {error_msg}")
        return False, error_msg
    except httpx.TimeoutException:
        return False, "Timeout (may be starting up)"
    except httpx.ConnectError:
        return False, "Connection refused (may be starting up)"
    except Exception as e:
        logger.warning(f"Health check error for {service_name}: {str(e)}")
        return False, f"Error: {str(e)[:50]}"


def make_request_with_retry(
    method: str,
    url: str,
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
    **kwargs,
) -> httpx.Response:
    """
    Make an HTTP request with retry logic for handling cold starts.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        max_retries: Maximum number of retry attempts
        retry_delay: Initial delay between retries (exponential backoff)
        **kwargs: Additional arguments to pass to httpx request

    Returns:
        httpx.Response object

    Raises:
        Exception: If all retries fail
    """
    last_exception = None

    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                if method.upper() == "GET":
                    response = client.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = client.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                # If we get a response (even error status), return it
                # Only retry on connection/timeout errors
                return response

        except socket.gaierror as e:
            # DNS resolution error - don't retry, show user-friendly message
            error_msg = f"API not reachable at {url}. Please verify BACKEND_API_URL configuration."
            logger.error(f"DNS resolution error: {error_msg}")
            raise Exception(error_msg) from e
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2**attempt)  # Exponential backoff
                logger.warning(
                    f"Request to {url} failed (attempt {attempt + 1}/{max_retries}): {str(e)}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} retry attempts failed for {url}")
                raise Exception(
                    f"Service unavailable after {max_retries} attempts. "
                    f"The service may be starting up (cold start). Last error: {str(e)}"
                )
        except Exception as e:
            # For other exceptions, don't retry
            logger.error(f"Non-retryable error for {url}: {str(e)}")
            raise

    # Should not reach here, but just in case
    raise Exception(f"Request failed after {max_retries} attempts: {str(last_exception)}")


def process_cv_upload(file, user_id: str) -> dict:
    """
    Upload and process a CV file.

    Args:
        file: Uploaded file object
        user_id: User ID for the candidate

    Returns:
        Response dictionary with status and resume_id
    """
    try:
        files = {"file": (file.name, file.read(), "application/pdf")}
        params = {"user_id": user_id}
        response = make_request_with_retry(
            "POST", f"{CV_PROCESSOR_URL}/process-cv", files=files, params=params
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error processing CV: {e.response.status_code} - {e.response.text}")
        raise Exception(f"Failed to process CV: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error processing CV: {str(e)}")
        error_msg = str(e)
        if "cold start" in error_msg.lower() or "starting up" in error_msg.lower():
            raise Exception(
                f"CV Processor service is starting up. Please wait a moment and try again. "
                f"Error: {error_msg}"
            )
        raise Exception(f"Error processing CV: {error_msg}")


def process_vacancy(vacancy_text: str, vacancy_id: Optional[str] = None) -> dict:
    """
    Process a vacancy description.

    Args:
        vacancy_text: Text description of the vacancy
        vacancy_id: Optional vacancy ID (generated if not provided)

    Returns:
        Response dictionary with status and vacancy_id
    """
    if not vacancy_id:
        vacancy_id = str(uuid.uuid4())

    try:
        response = make_request_with_retry(
            "POST",
            f"{CV_PROCESSOR_URL}/process-vacancy",
            json={"vacancy_id": vacancy_id, "text": vacancy_text},
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error processing vacancy: {e.response.status_code} - {e.response.text}")
        raise Exception(f"Failed to process vacancy: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error processing vacancy: {str(e)}")
        error_msg = str(e)
        if "cold start" in error_msg.lower() or "starting up" in error_msg.lower():
            raise Exception(
                f"CV Processor service is starting up. Please wait a moment and try again. "
                f"Error: {error_msg}"
            )
        raise Exception(f"Error processing vacancy: {error_msg}")


def get_matches(candidate_id: str, top_k: int = 10) -> List[VacancyMatchResult]:
    """
    Get matches for a candidate.

    Implements improved retry logic for Azure Container Apps cold starts:
    - Retries every 10 seconds for a maximum of 4 minutes (24 retries)
    - Treats 404/500 errors as potential cold start symptoms on first attempt
    - Shows user-friendly message about Azure services initializing during retries
    - Uses 300s timeout to allow backend orchestrator time for internal retries
    - Only shows final error if all retries fail

    Args:
        candidate_id: Candidate/user ID
        top_k: Number of top matches to return

    Returns:
        List of VacancyMatchResult objects
    """
    max_retries = 24  # 4 minutes / 10 seconds = 24 retries
    retry_delay = 10  # seconds
    max_wait_time = 240  # 4 minutes in seconds
    request_timeout = 300.0  # 5 minutes - allows backend orchestrator time for internal retries

    last_exception = None
    last_status_code = None
    start_time = time.time()
    info_message_shown = False

    for attempt in range(max_retries):
        try:
            # Check if we've exceeded max wait time
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_wait_time:
                break

            # Show info message during retries (after first attempt fails)
            if attempt > 0 and not info_message_shown:
                st.info(
                    "Searching for candidate data... Azure services might still be initializing."
                )
                info_message_shown = True

            # Make the request with extended timeout to allow backend orchestrator retries
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(
                    f"{BACKEND_API_URL}/match", json={"candidate_id": candidate_id, "top_k": top_k}
                )
                response.raise_for_status()
                results = response.json()
                # Parse results into VacancyMatchResult objects
                return [VacancyMatchResult(**result) for result in results]

        except socket.gaierror as e:
            # DNS resolution error - show user-friendly message
            error_msg = f"API not reachable at {BACKEND_API_URL}. Please verify BACKEND_API_URL configuration."
            st.error(f"❌ {error_msg}")
            logger.error(f"DNS resolution error in match_candidate: {error_msg}")
            raise Exception(error_msg) from e
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            last_exception = e
            elapsed_time = time.time() - start_time

            # Check if we should continue retrying
            if elapsed_time >= max_wait_time or attempt >= max_retries - 1:
                break

            # Show info message if not already shown
            if not info_message_shown:
                st.info(
                    "Searching for candidate data... Azure services might still be initializing."
                )
                info_message_shown = True

            # Wait before retrying
            time.sleep(retry_delay)

        except httpx.HTTPStatusError as e:
            last_exception = e
            last_status_code = e.response.status_code
            elapsed_time = time.time() - start_time

            # Treat 404/500 as potential cold start symptoms on first attempt
            if e.response.status_code in (404, 500):
                error_detail = e.response.text if hasattr(e.response, "text") else str(e)
                is_cold_start_symptom = (
                    attempt == 0  # First attempt
                    or "Candidate not found" in error_detail
                    or "not found" in error_detail.lower()
                )

                if (
                    is_cold_start_symptom
                    and elapsed_time < max_wait_time
                    and attempt < max_retries - 1
                ):
                    # Show info message if not already shown
                    if not info_message_shown:
                        st.info(
                            "Searching for candidate data... Azure services might still be initializing."
                        )
                        info_message_shown = True

                    logger.info(
                        f"HTTP {e.response.status_code} error on attempt {attempt + 1}/{max_retries}: "
                        f"{error_detail[:100]}. Treating as potential cold start, retrying..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    # Not a cold start symptom or retries exhausted
                    if e.response.status_code == 404:
                        raise ValueError(f"Candidate not found: {candidate_id}")
                    else:
                        logger.error(
                            f"HTTP error getting matches: {e.response.status_code} - {error_detail}"
                        )
                        raise Exception(
                            f"Failed to get matches: {e.response.status_code} - {error_detail}"
                        )
            else:
                # Other HTTP errors - don't retry
                logger.error(
                    f"HTTP error getting matches: {e.response.status_code} - {e.response.text}"
                )
                raise Exception(
                    f"Failed to get matches: {e.response.status_code} - {e.response.text}"
                )

        except Exception as e:
            last_exception = e
            elapsed_time = time.time() - start_time

            # For other exceptions, retry if it might be a cold start issue (first attempt only)
            if attempt == 0 and elapsed_time < max_wait_time and attempt < max_retries - 1:
                if not info_message_shown:
                    st.info(
                        "Searching for candidate data... Azure services might still be initializing."
                    )
                    info_message_shown = True
                logger.warning(
                    f"Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}. Retrying..."
                )
                time.sleep(retry_delay)
                continue
            else:
                # Don't retry other exceptions after first attempt
                logger.error(f"Error getting matches: {str(e)}")
                raise

    # If we get here, all retries failed
    if last_status_code in (404, 500):
        if last_status_code == 404:
            error_msg = (
                f"Candidate not found: {candidate_id} (after {max_wait_time} seconds of retries)"
            )
        else:
            error_msg = f"Service error after {max_wait_time} seconds. Azure services may still be starting up."
    else:
        error_msg = f"Service unavailable after {max_wait_time} seconds. Azure nodes may still be starting up."

    if last_exception:
        error_msg += f" Last error: {str(last_exception)}"

    logger.error(f"All retry attempts failed for get_matches: {error_msg}")
    raise Exception(error_msg)


def display_match_result(match: VacancyMatchResult, index: int):
    """
    Display a single match result in a styled card.

    Args:
        match: VacancyMatchResult object
        index: Index of the match (for ranking)
    """
    # Format score as percentage
    score_percent = match.score * 100

    # Color code based on score
    if score_percent >= 80:
        score_color = "#28a745"  # Green
    elif score_percent >= 60:
        score_color = "#ffc107"  # Yellow
    else:
        score_color = "#dc3545"  # Red

    st.markdown(
        f"""
        <div class="match-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h3 style="margin: 0;">Match #{index + 1}</h3>
                <span class="score-badge" style="background-color: {score_color};">
                    {score_percent:.1f}% Match
                </span>
            </div>
            <div class="vacancy-id">Vacancy ID: {match.vacancy_id}</div>
            <div class="reasoning-text">
                <strong>AI Reasoning:</strong><br>
                {match.reasoning}
            </div>
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer; color: #1f77b4;">View Vacancy Text</summary>
                <div style="margin-top: 0.5rem; padding: 0.5rem; background-color: #f8f9fa; border-radius: 0.25rem; max-height: 200px; overflow-y: auto;">
                    {match.vacancy_text[:500]}{"..." if len(match.vacancy_text) > 500 else ""}
                </div>
            </details>
        </div>
    """,
        unsafe_allow_html=True,
    )


def render_score_badges(pinecone_score: Optional[float], score: Optional[float] = None, ai_match_score: Optional[int] = None) -> None:
    """
    Render score badges (Semantic Score and AI Match Score) as HTML using st.markdown.
    
    Args:
        pinecone_score: Pinecone similarity score (0.0 to 1.0) or None
        score: AI match score (0-100 scale) or None - primary score field
        ai_match_score: AI match score (0-10 scale) or None - for backward compatibility
    """
    if pinecone_score is None and score is None and ai_match_score is None:
        return
    
    score_badges = []
    
    if pinecone_score is not None:
        # Format Pinecone score as percentage - renamed to "Semantic Score"
        pinecone_percent = pinecone_score * 100
        pinecone_badge = (
            f'<div style="background-color: #f0f9ff; padding: 0.5rem 0.75rem; border-radius: 0.5rem; '
            f'border: 1px solid #1f77b4;">'
            f'<strong style="color: #1f77b4; font-size: 0.875rem;">Semantic Score:</strong> '
            f'<span style="font-weight: bold; font-size: 1.1rem; color: #1f77b4; margin-left: 0.5rem;">'
            f'{pinecone_percent:.1f}%</span></div>'
        )
        score_badges.append(pinecone_badge)
    
    # Use primary 'score' field (0-100) if available, otherwise fallback to ai_match_score (0-10)
    ai_score_to_display = score if score is not None else (ai_match_score * 10 if ai_match_score is not None else None)
    
    if ai_score_to_display is not None:
        # Determine color based on AI match score (0-100 scale)
        if ai_score_to_display >= 80:
            score_color = "#22c55e"  # Green
        elif ai_score_to_display >= 50:
            score_color = "#eab308"  # Yellow
        else:
            score_color = "#ef4444"  # Red
        
        # Build AI match score badge
        ai_badge = (
            f'<div style="background-color: #f0f9ff; padding: 0.5rem 0.75rem; border-radius: 0.5rem; '
            f'border: 1px solid {score_color};">'
            f'<strong style="color: {score_color}; font-size: 0.875rem;">AI Match Score:</strong> '
            f'<span style="font-weight: bold; font-size: 1.1rem; color: {score_color}; margin-left: 0.5rem;">'
            f'{int(ai_score_to_display)}%</span></div>'
        )
        score_badges.append(ai_badge)
    
    # Build container with all score badges
    scores_container = (
        f'<div style="margin-top: 1rem; margin-bottom: 0.75rem; padding: 0.75rem; '
        f'background-color: #e7f3ff; border-radius: 0.5rem; border-left: 4px solid #1f77b4; '
        f'display: flex; gap: 1rem; flex-wrap: wrap; align-items: center;">'
        f'{"".join(score_badges)}</div>'
    )
    
    # Render with st.markdown and unsafe_allow_html=True to ensure HTML is rendered, not displayed as text
    st.markdown(scores_container, unsafe_allow_html=True)


def display_vacancy_card(vacancy: dict, index: int):
    """
    Display a vacancy card in the chat interface with a design similar to Shield AI / Compact Layout.
    """
    # Handle both dict and Pydantic model
    if hasattr(vacancy, 'dict'):
        vacancy = vacancy.dict()
    elif hasattr(vacancy, 'model_dump'):
        vacancy = vacancy.model_dump()

    # Helper for safe nested access
    def get_nested(d, path, default=None):
        keys = path.split('.')
        current = d
        for k in keys:
            if isinstance(current, dict):
                current = current.get(k)
            elif hasattr(current, k):
                current = getattr(current, k)
            else:
                return default
            if current is None:
                return default
        return current

    # Extract fields
    title = vacancy.get("title", "Unknown")
    company_name = vacancy.get("company_name", "Unknown")
    company_stage = vacancy.get("company_stage", "Unknown")
    if hasattr(company_stage, 'value'):
        company_stage = company_stage.value
    
    # Normalize stage string if needed (e.g. "Growth (Series B...)" -> "Growth")
    if company_stage and "growth" in company_stage.lower():
        company_stage = "Growth"
        
    location = vacancy.get("location", "Not specified")
    category = vacancy.get("category")
    industry = vacancy.get("industry")
    employee_count = vacancy.get("employee_count")
    
    # Experience logic
    experience_years = get_nested(vacancy, "extracted.role.experience_years_min")
    experience_level = vacancy.get("experience_level")
    experience_display = f"{experience_years}+ years" if experience_years is not None else experience_level

    # Salary logic
    salary_range = vacancy.get("salary_range")
    normalization_warnings = vacancy.get("normalization_warnings", [])
    if normalization_warnings:
        salary_range = None
    
    # Clean salary display
    if salary_range:
        import re
        if isinstance(salary_range, str) and ("'label'" in salary_range or '"label"' in salary_range):
            numbers = re.findall(r'\d+[\d,]*', salary_range)
            if numbers:
                clean_numbers = [int(n.replace(',', '')) for n in numbers if n.replace(',', '').isdigit()]
                if len(clean_numbers) == 1:
                    salary_range = f"${clean_numbers[0]:,}"
                elif len(clean_numbers) >= 2:
                    salary_range = f"${clean_numbers[0]:,} - ${clean_numbers[-1]:,}"
                else:
                    salary_range = None

    # Profiles
    role_profile = get_nested(vacancy, "ai_ready_views.role_profile_text")
    company_profile = get_nested(vacancy, "ai_ready_views.company_profile_text")
    
    # Tag Groups
    # 1. Tech Stack
    tech_stack = get_nested(vacancy, "extracted.role.tech_stack", [])
    if tech_stack:
        tech_stack = [t for t in tech_stack if t.lower() not in ["maintenance", "other", "n/a"]]
        
    # 2. Must Skills
    must_skills = get_nested(vacancy, "extracted.role.must_skills", [])
    
    # 3. Required Skills
    required_skills = vacancy.get("required_skills", [])
    
    # 4. Domain Tags
    domain_tags = get_nested(vacancy, "extracted.company.domain_tags", [])
    
    # Signals (excluding culture_signals as requested)
    has_equity = get_nested(vacancy, "extracted.offer.equity", False)
    is_customer_facing = get_nested(vacancy, "extracted.role.customer_facing", False)
    product_type = get_nested(vacancy, "extracted.company.product_type")
    
    description_url = vacancy.get("description_url", "")
    
    # Scores
    score = vacancy.get("score")
    ai_match_score = vacancy.get("ai_match_score")
    current_score = score if score is not None else (ai_match_score * 10 if ai_match_score is not None else None)
    ai_insight = vacancy.get("ai_insight") or vacancy.get("match_reasoning") or vacancy.get("match_reason")

    # --- HTML Construction ---
    
    # Score Badge
    score_html = ""
    has_persona = st.session_state.get("persona") is not None
    persona_applied = vacancy.get("persona_applied", False)
    is_manual_search = "persona_applied" not in vacancy or (not persona_applied and score is None and ai_match_score is None)

    if not is_manual_search and (not has_persona or not persona_applied):
         score_html = '<div style="text-align: right; font-size: 12px; color: #666;">Upload CV for match</div>'
    elif current_score is not None:
        color = "#22c55e" if current_score >= 80 else "#eab308" if current_score >= 50 else "#ef4444"
        score_html = f'<div style="text-align: right;"><div style="font-size: 24px; font-weight: 700; color: {color};">{int(current_score)}%</div><div style="font-size: 11px; color: #666;">Match</div></div>'
    
    # Signals HTML
    signals_list = []
    if has_equity:
        signals_list.append('<span class="signal-tag">💎 Equity</span>')
    if is_customer_facing:
        signals_list.append('<span class="signal-tag">👤 Customer Facing</span>')
    if product_type:
        signals_list.append(f'<span class="signal-tag">🚀 {product_type}</span>')
    
    # Meta Items
    meta_html = ""
    
    # Salary Line
    if salary_range:
        meta_html += f'<div class="meta-item"><span style="font-size:16px">💰</span> {salary_range} / year</div>'
    
    # Location
    if location and location != "Not specified":
        loc_parts = location.split(',')
        clean_loc = ", ".join([p.strip() for p in loc_parts[:2]])
        meta_html += f'<div class="meta-item"><span style="font-size:16px">📍</span> {clean_loc}</div>'
    
    # Row 2 Meta (Industry, Stage, Employees)
    row2_items = []
    if category:
        row2_items.append(category)
    if industry:
        row2_items.append(industry)
    if company_stage and company_stage != "Unknown":
        row2_items.append(company_stage)
    if employee_count:
        row2_items.append(employee_count)
    if experience_display:
        row2_items.append(experience_display)
        
    row2_html = " • ".join(row2_items)

    # Build conditional HTML sections
    
    # Build specific sections
    role_profile_html = ""
    if role_profile:
        role_profile_html = f'''<div class="profile-section">
    <div class="profile-label">Role Profile</div>
    <div class="profile-text">{role_profile}</div>
</div>'''

    company_profile_html = ""
    if company_profile:
        company_profile_html = f'''<div class="profile-section">
    <div class="profile-label">Company Profile</div>
    <div class="profile-text">{company_profile}</div>
</div>'''
    
    # Helper to build tag block
    def build_tag_block(title, tags, css_class="tech-tag"):
        if not tags: return ""
        # Limit tags to reasonable amount
        display_tags = tags[:10]
        tags_str = "".join([f'<span class="{css_class}">{t}</span>' for t in display_tags])
        return f'''<div class="tag-category">
    <div class="tag-category-label">{title}</div>
    <div class="tags-row">{tags_str}</div>
</div>'''

    # Gather existing skills to prevent duplicates in General Required Skills
    existing_skills_lower = set()
    if tech_stack:
        existing_skills_lower.update(t.lower() for t in tech_stack)
    if must_skills:
        existing_skills_lower.update(t.lower() for t in must_skills)

    # Key Signals
    signals_html = ""
    if signals_list:
        signals_str = "".join(signals_list)
        signals_html = f'''<div class="tag-category">
    <div class="tag-category-label">Key Signals</div>
    <div class="tags-row">{signals_str}</div>
</div>'''

    # Block 1: Skills (after Role Profile)
    block1_html = ""
    if tech_stack:
        block1_html += build_tag_block("Tech Stack", tech_stack)
    if must_skills:
        block1_html += build_tag_block("Must Have Skills", must_skills)
    
    # Filter required skills
    if required_skills:
        filtered_required = [s for s in required_skills if s.lower() not in existing_skills_lower]
        block1_html += build_tag_block("Required Skills (General)", filtered_required)

    # Block 2: Domain (after Company Profile)
    block2_html = ""
    if domain_tags:
        block2_html += build_tag_block("Domain & Industry", domain_tags)

    # Conditional wrappers for meta sections
    meta_section = f'<div class="salary-loc-row">{meta_html}</div>' if meta_html else ""
    secondary_meta_section = f'<div class="secondary-meta">{row2_html}</div>' if row2_html else ""

    # Build card HTML
    html_parts = []
    html_parts.append('<div class="vacancy-card">')
    
    # Header
    html_parts.append('<div style="display: flex; justify-content: space-between; align-items: flex-start;">')
    html_parts.append('<div style="flex-grow: 1;">')
    html_parts.append(f'<div class="company-name">{company_name}</div>')
    html_parts.append(f'<div class="vacancy-title">{title}</div>')
    html_parts.append(meta_section)
    html_parts.append(secondary_meta_section)
    html_parts.append('</div>') # Close Left Column
    html_parts.append(score_html)
    html_parts.append('</div>') # Close Header Flex Row
    
    # Body in new order:
    # 1. Key Signals
    html_parts.append(signals_html)
    # 2. Role Profile
    html_parts.append(role_profile_html)
    # 3. Block 1 (Skills)
    html_parts.append(block1_html)
    # 4. Company Profile
    html_parts.append(company_profile_html)
    # 5. Block 2 (Domain)
    html_parts.append(block2_html)
    
    # Footer (Apply Button)
    html_parts.append('<div style="margin-top: 12px; display: flex; justify-content: flex-end;">')
    html_parts.append(f'<a href="{description_url}" target="_blank" class="apply-btn"><span>→</span> Apply</a>')
    html_parts.append('</div>')
    
    html_parts.append('</div>') # Close vacancy-card
    
    card_html = "".join(html_parts)
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    # AI Insight (outside the HTML card as standard Streamlit component for expandability)
    if ai_insight:
        with st.expander("🤖 Why this match?", expanded=False):
             st.info(ai_insight)


def display_matching_report(report: MatchingReport, index: int):
    """
    Display a MatchingReport in an expandable card with strengths/weaknesses.

    Args:
        report: MatchingReport object
        index: Index of the report (for ranking)
    """
    # Color code based on match score
    if report.match_score >= 80:
        score_color = "#28a745"  # Green
    elif report.match_score >= 60:
        score_color = "#ffc107"  # Yellow
    else:
        score_color = "#dc3545"  # Red

    # Build strengths list HTML
    strengths_html = ""
    if report.strengths:
        strengths_html = "<ul>" + "".join([f"<li>{s}</li>" for s in report.strengths]) + "</ul>"
    else:
        strengths_html = "<em>No strengths identified</em>"

    # Build weaknesses list HTML
    weaknesses_html = ""
    if report.weaknesses:
        weaknesses_html = "<ul>" + "".join([f"<li>{w}</li>" for w in report.weaknesses]) + "</ul>"
    else:
        weaknesses_html = "<em>No weaknesses identified</em>"

    st.markdown(
        f"""
        <div class="match-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <h3 style="margin: 0;">Match Report #{index + 1}</h3>
                <span class="score-badge" style="background-color: {score_color};">
                    {report.match_score}/100
                </span>
            </div>
            <div style="margin-bottom: 1rem;">
                <strong>Value Proposition:</strong><br>
                <div class="reasoning-text">{report.value_proposition}</div>
            </div>
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer; color: #1f77b4; font-weight: bold;">✅ Strengths</summary>
                <div style="margin-top: 0.5rem; padding: 0.5rem; background-color: #d4edda; border-radius: 0.25rem;">
                    {strengths_html}
                </div>
            </details>
            <details style="margin-top: 0.5rem;">
                <summary style="cursor: pointer; color: #dc3545; font-weight: bold;">⚠️ Weaknesses</summary>
                <div style="margin-top: 0.5rem; padding: 0.5rem; background-color: #f8d7da; border-radius: 0.25rem;">
                    {weaknesses_html}
                </div>
            </details>
            <div style="margin-top: 1rem; padding: 0.75rem; background-color: #e7f3ff; border-radius: 0.25rem; border-left: 4px solid #1f77b4;">
                <strong>💡 Suggested Action:</strong><br>
                {report.suggested_action}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


# Initialize session state
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

# Initialize chat history for AI Recruiter tab
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your AI recruiter. Describe in your own words: what role are you looking for, your main tech stack, and any industry preferences (e.g., AI, Crypto, or BioTech)?"
        }
    ]

# Initialize persona if not exists
if "persona" not in st.session_state:
    st.session_state.persona = None

# Main UI
st.title("AI Talent Matcher")
st.info("✅ Search Mode: Database (Verified)")

# Sidebar - Clean: Only Configuration and Service Health
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info(f"**API URL:** {BACKEND_API_URL}\n\n**CV Processor URL:** {CV_PROCESSOR_URL}")

    st.markdown("---")
    st.subheader("🏥 Service Health")

    # Health check for API service
    api_healthy, api_message = check_service_health(BACKEND_API_URL, "API")
    if api_healthy:
        st.success(f"✅ **API Service:** {api_message}")
    else:
        st.warning(f"⚠️ **API Service:** {api_message}")
        if "starting up" in api_message.lower():
            st.caption("💡 Service may be waking from cold start. Wait a moment and refresh.")

    # Health check for CV Processor service
    cv_healthy, cv_message = check_service_health(CV_PROCESSOR_URL, "CV Processor")
    if cv_healthy:
        st.success(f"✅ **CV Processor:** {cv_message}")
    else:
        st.warning(f"⚠️ **CV Processor:** {cv_message}")
        if "starting up" in cv_message.lower():
            st.caption("💡 Service may be waking from cold start. Wait a moment and refresh.")

    # Health check for Embedding Service
    embedding_healthy, embedding_message = check_service_health(EMBEDDING_SERVICE_URL, "Embedding Service")
    if embedding_healthy:
        st.success(f"✅ **Embedding Service:** {embedding_message}")
    else:
        st.warning(f"⚠️ **Embedding Service:** {embedding_message}")
        if "starting up" in embedding_message.lower():
            st.caption("💡 Service may be waking from cold start. Wait a moment and refresh.")

    # Refresh button
    if st.button("🔄 Refresh Health Status", use_container_width=True):
        st.rerun()

# Main content tabs
tab_chat, tab_search, tab_cv, tab_diagnostics, tab_admin = st.tabs([
    "💬 AI Recruiter",
    "🔍 Manual Search",
    "📊 Career & Match Hub",
    "🛠️ Diagnostics",
    "⚙️ Admin"
])

# ============================================================================
# TAB 5: ADMIN (Scraper Control)
# ============================================================================
with tab_admin:
    st.header("⚙️ Admin & Scraper Control")
    st.markdown("Manage data ingestion and scraper settings.")
    
    st.subheader("🚀 Scraper Settings")
    
    with st.form("scraper_config"):
        col1, col2 = st.columns(2)
        
        with col1:
            remote_only = st.checkbox("Remote Only", value=True, help="Download only remote vacancies")
            max_days_old = st.number_input("Max Days Old", min_value=0, value=0, help="0 = download all history")
        
        with col2:
            locations = st.text_input("Include Locations (comma separated)", placeholder="London, Berlin, Europe")
            keywords = st.text_input("Include Title Keywords (comma separated)", placeholder="Python, Backend, Senior")
        
        st.markdown("---")
        st.caption("Advanced Filters")
        col3, col4 = st.columns(2)
        
        with col3:
            exclude_locations = st.text_input("Exclude Locations", placeholder="India, China")
            exclude_keywords = st.text_input("Exclude Title Keywords", placeholder="Staff, Principal")
            
        with col4:
            exclude_companies = st.text_input("Exclude Companies", placeholder="Company A, Company B")
            include_categories = st.text_input("Include Categories", placeholder="Engineering, Product")

        submit_scraper = st.form_submit_button("🚀 Start Scraper Job", type="primary", use_container_width=True)
        
        if submit_scraper:
            # Prepare filter config
            filters = {
                "remote_only": remote_only,
                "max_days_old": max_days_old if max_days_old > 0 else None,
                "include_locations": [x.strip() for x in locations.split(",") if x.strip()],
                "include_title_keywords": [x.strip() for x in keywords.split(",") if x.strip()],
                "exclude_locations": [x.strip() for x in exclude_locations.split(",") if x.strip()],
                "exclude_title_keywords": [x.strip() for x in exclude_keywords.split(",") if x.strip()],
                "exclude_companies": [x.strip() for x in exclude_companies.split(",") if x.strip()],
                "include_categories": [x.strip() for x in include_categories.split(",") if x.strip()],
            }
            
            with st.spinner("Starting scraper job..."):
                try:
                    response = httpx.post(
                        f"{BACKEND_API_URL}/api/v1/admin/scrape",
                        json={"filters": filters},
                        timeout=10.0
                    )
                    if response.status_code == 200:
                        st.success("✅ Scraper started successfully in background!")
                        st.json(filters)
                    else:
                        st.error(f"❌ Failed to start scraper: {response.text}")
                except Exception as e:
                    st.error(f"❌ Connection error: {str(e)}")

# ============================================================================
# TAB 1: AI RECRUITER (Chat Interface)
# ============================================================================
with tab_chat:
    st.header("💬 AI Recruiter")
    st.markdown("Chat with the AI recruiter to find your ideal role at startup companies.")

    # Show persona notification if active
    if st.session_state.get("persona"):
        st.info("🎯 AI is using your CV profile for personalized search.")

    # Display chat history
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # If this is an assistant message with vacancies, display them
            if message["role"] == "assistant" and "vacancies" in message:
                vacancies = message["vacancies"]
                debug_info = message.get("debug_info", {})
                search_stats = message.get("search_stats", {})
                persona_applied = message.get("persona_applied", False)
            
                # Check if persona is missing and show warning
                has_persona = st.session_state.get("persona") is not None
                if not has_persona:
                    st.warning("⚠️ **AI Matching is disabled.** Upload your CV to see how well you fit these roles.")
            
                # Display search mode badge at the top (prominent visual indicator)
                # Check persona_applied flag from API response instead of search_mode
                persona_actually_applied = message.get("persona_applied", False)
            
                if persona_actually_applied:
                    st.success("👤 **Persona Mode**: Matching based on your CV profile.")
                else:
                    st.info("🌐 **Broad Search Mode**: General search without personalized matching.")
            
                # Display friendly_reasoning at the top as Strategy Box
                friendly_reasoning = debug_info.get("friendly_reasoning") if debug_info else None
                if friendly_reasoning:
                    st.info(f"🛠️ **AI Search Strategy:** {friendly_reasoning}")
            
                # Display search statistics header
                if search_stats:
                    total_matches = search_stats.get("total_after_filters", len(vacancies))
                    initial_matches = search_stats.get("initial_vector_matches", len(vacancies))
                    db_size = search_stats.get("total_in_db")
                
                    stats_text = f"**Found {total_matches} match{'es' if total_matches != 1 else ''}**"
                    if initial_matches and initial_matches != total_matches:
                        stats_text += f" (Filtered from {initial_matches} initial candidates)"
                    if db_size:
                        stats_text += f". Database size: {db_size}"
                
                    st.info(f"📊 {stats_text}")
            
                # Display raw JSON in expander for debugging
                if debug_info:
                    with st.expander("🛠️ Technical Logs", expanded=False):
                        st.json(debug_info)
            
                # Show top 5 vacancies but indicate if more exist
                top_5_vacancies = vacancies[:5]
                total_vacancies = len(vacancies)
            
                for idx, vacancy in enumerate(top_5_vacancies):
                    display_vacancy_card(vacancy, idx)
            
                # Indicate if more vacancies exist
                if total_vacancies > 5:
                    st.info(f"📋 Showing top 5 results. {total_vacancies - 5} more vacancy/vacancies available.")

    # Chat input
    if user_input := st.chat_input("Describe what role you're looking for..."):
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": user_input})

        # Initialize variables for assistant response
        summary = ""
        vacancies = []

        # Show status while processing
        with st.status("AI is analyzing your request...", expanded=True) as status:
            try:
                # Call chat API
                chat_endpoint = f"{BACKEND_API_URL}/api/v1/vacancies/chat"
                logger.info(f"Calling chat endpoint: {chat_endpoint}")

                # Persona Check: Check if persona exists in session state
                persona = st.session_state.get("persona")
                # Only include persona if it's a valid non-empty dict
                if persona and isinstance(persona, dict) and len(persona) > 0:
                    persona_keys = list(persona.keys())
                    logger.info(f"persona_found_in_session: has_persona=True, persona_keys={persona_keys}")
                else:
                    persona = None  # Ensure persona is None if invalid
                    logger.warning("persona_not_found_in_session: No persona data available. CV may not have been analyzed yet.")

                # Prepare request payload with history and persona
                # Convert chat_messages to history format (exclude the current message we just added)
                history = []
                for msg in st.session_state.chat_messages[:-1]:  # Exclude the last message (current user input)
                    if msg.get("role") in ["user", "assistant"]:
                        history.append({
                            "role": msg["role"],
                            "content": msg.get("content", "")
                        })

                # API Request: Ensure persona is included in the request (or None if missing)
                chat_data = {
                    "message": user_input,
                    "persona": persona,  # Will be None if CV not uploaded
                    "history": history
                }
            
                # Debugging: Log persona data (hidden in expander for debugging)
                with st.expander("🔍 Debug: Request Payload", expanded=False):
                    debug_payload = {
                        "message": user_input,
                        "has_persona": persona is not None,
                        "persona_keys": list(persona.keys()) if isinstance(persona, dict) else None,
                        "persona_preview": {k: str(v)[:50] + "..." if len(str(v)) > 50 else v for k, v in list(persona.items())[:3]} if isinstance(persona, dict) else None,
                        "history_length": len(history)
                    }
                    st.json(debug_payload)
                    if persona:
                        st.success("✅ Persona data is being sent to API")
                    else:
                        st.warning("⚠️ No persona data - CV may not have been analyzed. Upload your CV in the Career & Match Hub tab first.")

                request_payload = chat_data

                with httpx.Client(timeout=120.0) as client:
                    response = client.post(
                        chat_endpoint,
                        json=request_payload
                    )
                    response.raise_for_status()
                    result = response.json()

                # Extract results
                vacancies = result.get("vacancies", [])
                summary = result.get("summary", "Found matching vacancies.")
                debug_info = result.get("debug_info", {})
                persona_applied = result.get("persona_applied", False)
                
                # ========================================================================
                # Save updated persona to session state (MEMORY PERSISTENCE)
                # ========================================================================
                # This ensures persona evolves over the conversation, remembering
                # user preferences, skills, and requirements from previous messages
                updated_persona = result.get("updated_persona")
                if updated_persona:
                    st.session_state.persona = updated_persona
                    persona_keys = list(updated_persona.keys()) if isinstance(updated_persona, dict) else []
                    logger.info(
                        f"persona_updated_in_session: persona_keys={persona_keys}, "
                        f"has_skills={bool(updated_persona.get('technical_skills'))}, "
                        f"remote_only={updated_persona.get('remote_only')}, "
                        f"preferred_company_stages={updated_persona.get('preferred_company_stages')}, "
                        f"preferred_locations={updated_persona.get('preferred_locations')}"
                    )
                elif not st.session_state.get("persona"):
                    # Initialize empty persona if not present
                    st.session_state.persona = {}
            
                # Extract search statistics if available (from VacancySearchResponse structure)
                # Note: The chat endpoint may not return these directly, but we can construct from available data
                search_stats = {}
                # Check if statistics are in the result (from search endpoint) or construct from vacancies
                if "total_after_filters" in result:
                    search_stats["total_after_filters"] = result.get("total_after_filters", len(vacancies))
                    search_stats["initial_vector_matches"] = result.get("initial_vector_matches", len(vacancies))
                    search_stats["total_in_db"] = result.get("total_in_db")
                else:
                    # Construct basic stats from available data
                    search_stats["total_after_filters"] = len(vacancies)
                    search_stats["initial_vector_matches"] = len(vacancies)  # Default to same if unknown

                status.update(label="✅ Search completed!", state="complete")

            except httpx.HTTPStatusError as e:
                status.update(label="❌ API Error", state="error")
                error_detail = e.response.text[:200] if e.response.text else "Unknown error"
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("detail", error_detail)
                except Exception:
                    pass

                # Add error message to chat
                error_message = "Sorry, an error occurred while searching for vacancies. Please try again or refine your query."
                if e.response.status_code == 422:
                    error_message = "Could not understand your query. Please try describing the role, skills, or industry in more detail."
                elif e.response.status_code == 500:
                    error_message = "Temporary server error. Please try again in a few seconds."

                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                st.rerun()

            except httpx.RequestError:
                status.update(label="❌ Connection Error", state="error")
                error_message = f"Could not connect to server. Please check that the API is available at {BACKEND_API_URL}."
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                st.rerun()

            except Exception as e:
                status.update(label="❌ Unexpected Error", state="error")
                logger.error(f"Chat search error: {str(e)}")
                error_message = f"An unexpected error occurred: {str(e)[:100]}"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_message
                })
                st.rerun()

        # Add assistant response with summary and vacancies (only if we have them)
        if summary or vacancies:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": summary if summary else "Found matching vacancies.",
                "vacancies": vacancies,
                "debug_info": debug_info if 'debug_info' in locals() else {},
                "search_stats": search_stats if 'search_stats' in locals() else {},
                "persona_applied": persona_applied if 'persona_applied' in locals() else False
            })

        st.rerun()

    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI recruiter. Describe in your own words: what role are you looking for, your main tech stack, and any industry preferences (e.g., AI, Crypto, or BioTech)?"
            }
        ]
        st.rerun()

    # ============================================================================
    # TAB 2: MANUAL SEARCH (Filters + Search)
    # ============================================================================
with tab_search:
    st.header("🔍 Manual Search")
    st.markdown("Search for vacancies using filters from the pre-indexed database.")

    # Display search mode status
    st.info("✅ Search Mode: Database (Verified)")

    # Initialize session state for search results
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False

    # Search form - All filters inside this tab
    col1, col2 = st.columns(2)

    with col1:
        role = st.text_input("Role / Title", placeholder="e.g., MLOps Engineer, ML Director", key="search_role")
        industry = st.text_input("Industry", placeholder="e.g., Logistics, FreightTech", key="search_industry")
        skills_input = st.text_input(
            "Required Skills (comma-separated)", placeholder="e.g., Python, Kubernetes, Docker", key="search_skills"
        )
        location = st.text_input("Location", placeholder="e.g., San Francisco, Remote", key="search_location")
    
        # Additional filters from VacancyFilter schema
        category = st.selectbox(
            "Category",
            options=["", "Engineering", "Product", "Design", "Data & Analytics", "Sales & Business Development", 
                    "Marketing", "Operations", "Finance", "Legal", "People & HR", "Other"],
            index=0,
            key="search_category"
        )
        experience_level = st.selectbox(
            "Experience Level",
            options=["", "Junior", "Mid", "Senior", "Lead", "Executive"],
            index=0,
            key="search_experience"
        )

    with col2:
        min_salary = st.number_input("Minimum Salary", min_value=0, value=0, step=10000, key="search_min_salary")
        is_remote = st.checkbox("Remote Work Available", key="search_remote")
        is_hybrid = st.checkbox("Hybrid Work Available", key="search_hybrid")

        # Company stages - must match CompanyStage enum values exactly
        st.subheader("Company Stages")
        seed = st.checkbox("Seed", value=False, key="search_seed")
        series_a = st.checkbox("Series A", value=False, key="search_series_a")
        growth = st.checkbox("Growth", value=False, key="search_growth", help="Includes Series B, Series C, and Growth stage companies")
        public = st.checkbox("Public", value=False, key="search_public")
    
        # Employee count filters
        st.subheader("Employee Count")
        employees_1_10 = st.checkbox("1-10 employees", value=False, key="search_emp_1_10")
        employees_10_100 = st.checkbox("10-100 employees", value=False, key="search_emp_10_100")
        employees_100_1000 = st.checkbox("100-1000 employees", value=False, key="search_emp_100_1000")
        employees_1000_plus = st.checkbox("1000+ employees", value=False, key="search_emp_1000_plus")

        # Build company stages list with exact enum values
        # Growth includes Series B, Series C, and Growth
        company_stages = []
        if seed:
            company_stages.append("Seed")
        if series_a:
            company_stages.append("Series A")
        if growth:
            # Growth includes Series B, Series C, and Growth - send as single "Growth" value
            # API will expand it automatically
            company_stages.append("Growth")
        if public:
            company_stages.append("Public")

    # Search button - use form to prevent page reload
    with st.form("search_form", clear_on_submit=False):
        search_clicked = st.form_submit_button("🔍 Search Database", type="primary", use_container_width=True)
    
        if search_clicked:
            # Build filter parameters
            filter_params = {}

            if role:
                filter_params["role"] = role
            if industry:
                filter_params["industry"] = industry
            if skills_input:
                filter_params["skills"] = [s.strip() for s in skills_input.split(",") if s.strip()]
            if location:
                filter_params["location"] = location
            if min_salary > 0:
                filter_params["min_salary"] = min_salary
            if is_remote:
                filter_params["is_remote"] = True
            if company_stages:
                filter_params["company_stages"] = company_stages
            if category:
                filter_params["category"] = category
            if experience_level:
                filter_params["experience_level"] = experience_level
        
            # Build employee count list
            employee_counts = []
            if employees_1_10:
                employee_counts.append("1-10 employees")
            if employees_10_100:
                employee_counts.append("10-100 employees")
            if employees_100_1000:
                employee_counts.append("100-1000 employees")
            if employees_1000_plus:
                employee_counts.append("1000+ employees")
            if employee_counts:
                filter_params["employee_count"] = employee_counts

            # Make API request to search endpoint
            try:
                search_endpoint = f"{BACKEND_API_URL}/api/v1/vacancies/search"
                logger.info(f"Searching vacancies at: {search_endpoint}")

                with st.spinner("🔍 Searching for vacancies..."):
                    with httpx.Client(timeout=120.0) as client:
                        # Always use Pinecone database (use_firecrawl=False by default)
                        params = {"use_firecrawl": "false"}
                        response = client.post(
                            search_endpoint,
                            json=filter_params,
                            params=params,
                        )
                        response.raise_for_status()
                        result = response.json()
                        vacancies = result.get("vacancies", [])
                    
                        # Remove duplicates by description_url
                        seen_urls = set()
                        unique_vacancies = []
                        for v in vacancies:
                            url = v.get("description_url", "")
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                unique_vacancies.append(v)
                    
                        # Store results in session state
                        st.session_state.search_results = unique_vacancies
                        st.session_state.search_performed = True

            except httpx.HTTPStatusError as e:
                error_detail = e.response.text[:200] if e.response.text else "Unknown error"
                try:
                    error_json = e.response.json()
                    error_detail = error_json.get("detail", error_detail)
                except Exception:
                    pass

                st.error(f"❌ API Error ({e.response.status_code}): {error_detail}")

                if e.response.status_code == 503 and "firecrawl" in error_detail.lower():
                    st.info("💡 This error indicates Firecrawl is not properly configured. Check the health status in the sidebar.")
                elif e.response.status_code == 401:
                    st.info("💡 Authentication failed. Please verify FIRECRAWL_API_KEY in your .env file.")
            except httpx.RequestError as e:
                st.error(f"❌ Connection Error: {str(e)}")
                st.info(f"💡 Cannot reach backend at {BACKEND_API_URL}. Check if the API service is running.")
            except Exception as e:
                st.error(f"❌ Unexpected Error: {str(e)}")

    # Display results (outside form to prevent reload)
    if st.session_state.search_performed and st.session_state.search_results:
        vacancies = st.session_state.search_results
        st.success(f"✅ Found {len(vacancies)} unique vacancy/vacancies")
        st.markdown("---")

        # Display vacancies in improved cards
        for idx, vacancy in enumerate(vacancies):
            display_vacancy_card(vacancy, idx)
    elif st.session_state.search_performed and not st.session_state.search_results:
        st.info("🔍 No vacancies found matching your criteria. Try adjusting your filters.")

    # ============================================================================
    # TAB 3: CV ANALYSIS (Upload, Process, Match)
    # ============================================================================
with tab_cv:
    st.header("📊 Career & Match Hub")

    # Radio button navigation for CV Analysis tools
    cv_mode = st.radio(
        "Select Tool",
        ["📤 Upload CV", "💼 Process Vacancy", "🎯 Find Matches"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("---")

    # CV Upload
    if cv_mode == "📤 Upload CV":
        st.subheader("Upload Candidate Resume")
        st.markdown("Upload a PDF resume to process and index it for matching.")

        user_id = st.text_input(
            "Candidate ID (User ID)",
            value="",
            help="Enter a unique identifier for the candidate (e.g., email, username, or UUID)",
        )

        uploaded_file = st.file_uploader(
            "Choose a PDF file", type=["pdf"], help="Upload a PDF resume file"
        )

        if st.button("Process CV", type="primary", use_container_width=True):
            if not user_id:
                st.error("Please enter a Candidate ID")
            elif not uploaded_file:
                st.error("Please upload a PDF file")
            else:
                try:
                    with st.status("Processing your CV...", expanded=True) as status:
                        status.update(label="📤 Uploading CV file to server...", state="running")
                        # Note: The backend handles model loading, text extraction, and embedding generation
                        # Models are pre-cached in the Docker image, so this should be fast
                        result = process_cv_upload(uploaded_file, user_id)
                        status.update(label="✅ CV processed successfully!", state="complete")

                    st.success("✅ CV processed successfully!")
                    st.json(result)
                    st.info(
                        f"**Resume ID:** {result.get('resume_id')}\n\n**Chunks Processed:** {result.get('chunks_processed')}"
                    )
                
                    # Save persona to session state for personalized search
                    persona = result.get("persona")
                    if persona:
                        st.session_state["persona"] = persona
                        st.success("🎯 CV profile saved! AI will use it for personalized search.")
                    else:
                        # Fallback: create basic persona from available data
                        st.session_state["persona"] = {
                            "cv_text": "",  # Will be populated if available
                            "user_id": user_id,
                            "resume_id": result.get('resume_id')
                        }
                        st.warning("⚠️ Persona not found in response, using basic structure.")
                
                    # Debug: Show persona in state
                    st.write("Debug: Persona in state:", st.session_state.get("persona"))

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"CV processing error: {str(e)}")

    # Process Vacancy
    elif cv_mode == "💼 Process Vacancy":
        st.subheader("Process Vacancy Description")
        st.markdown("Paste a vacancy description to process and index it for matching.")

        vacancy_id_input = st.text_input(
            "Vacancy ID (Optional)",
            value="",
            help="Enter a unique identifier for the vacancy. Leave empty to auto-generate.",
        )

        vacancy_text = st.text_area(
            "Vacancy Description",
            height=300,
            placeholder="Paste the full vacancy description here...",
            help="Enter the complete text description of the vacancy",
        )

        if st.button("Process Vacancy", type="primary", use_container_width=True):
            if not vacancy_text.strip():
                st.error("Please enter a vacancy description")
            else:
                # Progress bar for vacancy processing
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    vacancy_id = vacancy_id_input.strip() if vacancy_id_input.strip() else None

                    status_text.text("📝 Processing vacancy text...")
                    progress_bar.progress(20)

                    status_text.text("🧠 Generating embeddings...")
                    progress_bar.progress(50)

                    status_text.text("💾 Saving to database...")
                    progress_bar.progress(80)

                    result = process_vacancy(vacancy_text, vacancy_id)

                    progress_bar.progress(100)
                    status_text.text("✅ Processing complete!")

                    st.success("✅ Vacancy processed successfully!")
                    st.json(result)
                    st.info(
                        f"**Vacancy ID:** {result.get('vacancy_id')}\n\n**Chunks Processed:** {result.get('chunks_processed')}"
                    )

                    # Clear progress after a moment
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Vacancy processing error: {str(e)}")

    # Find Matches
    elif cv_mode == "🎯 Find Matches":
        st.subheader("Find Candidate Matches")
        st.markdown("Enter a candidate ID to find matching vacancies.")

        col1, col2 = st.columns([3, 1])

        with col1:
            candidate_id = st.text_input(
                "Candidate ID",
                value="",
                help="Enter the candidate ID (user_id) that was used when uploading the CV",
            )

        with col2:
            top_k = st.number_input(
                "Number of Matches",
                min_value=1,
                max_value=50,
                value=10,
                help="Number of top matches to return",
            )

        if st.button("Find Matches", type="primary", use_container_width=True):
            if not candidate_id:
                st.error("Please enter a Candidate ID")
            else:
                # Progress bar for matching
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    status_text.text("🔍 Fetching candidate profile...")
                    progress_bar.progress(20)

                    status_text.text("📊 Searching for matching vacancies...")
                    progress_bar.progress(50)

                    status_text.text("🤖 Analyzing matches with AI...")
                    progress_bar.progress(75)

                    matches = get_matches(candidate_id, top_k)

                    progress_bar.progress(100)
                    status_text.text("✅ Matching complete!")

                    # Clear progress after a moment
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()

                    if not matches:
                        st.warning("No matches found for this candidate.")
                    else:
                        st.success(f"✅ Found {len(matches)} matches!")
                        st.markdown("---")

                        # Display matches
                        for idx, match in enumerate(matches):
                            display_match_result(match, idx)

                        # Summary statistics
                        st.markdown("---")
                        st.subheader("📊 Summary Statistics")
                        avg_score = sum(m.score for m in matches) / len(matches) * 100
                        max_score = max(m.score for m in matches) * 100
                        min_score = min(m.score for m in matches) * 100

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Average Match", f"{avg_score:.1f}%")
                        with col2:
                            st.metric("Best Match", f"{max_score:.1f}%")
                        with col3:
                            st.metric("Lowest Match", f"{min_score:.1f}%")

                except ValueError as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ {str(e)}")
                    st.info("💡 Make sure you've uploaded a CV for this candidate ID first.")
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"Match finding error: {str(e)}")

    # ============================================================================
    # TAB 4: DIAGNOSTICS (System Diagnostics)
    # ============================================================================
with tab_diagnostics:
    st.header("🛠️ System Diagnostics")
    st.markdown("Run a comprehensive health check on all backend services and dependencies.")

    # Initialize session state for diagnostics
    if "diagnostics_result" not in st.session_state:
        st.session_state.diagnostics_result = None
    if "diagnostics_running" not in st.session_state:
        st.session_state.diagnostics_running = False

    col1, col2 = st.columns([3, 1])

    with col1:
        st.info(
            "💡 This diagnostic check will test connectivity to all services including CV Processor, Embedding Service, Pinecone Vector Store, and LLM Provider. Services may take a few seconds to wake up from cold start."
        )

    with col2:
        run_diagnostics = st.button(
            "🚀 Run Full System Check", type="primary", use_container_width=True
        )

    if run_diagnostics or st.session_state.diagnostics_running:
        st.session_state.diagnostics_running = True

        # Show loading state
        with st.spinner("🔍 Running system diagnostics... Waking up services..."):
            try:
                # Try primary endpoint first, then fallback with retry mechanism
                diagnostics_urls = [
                    f"{BACKEND_API_URL}/api/v1/system/diagnostics",
                    f"{BACKEND_API_URL}/system/diagnostics",
                ]

                diagnostics_data = None
                last_error = None
                max_retries = 2
                retry_delay = 2.0  # seconds

                for url in diagnostics_urls:
                    for attempt in range(max_retries):
                        try:
                            logger.info(
                                f"Attempting diagnostics at: {url} (attempt {attempt + 1}/{max_retries})"
                            )
                            with httpx.Client(timeout=60.0) as client:
                                response = client.get(url)
                                response.raise_for_status()
                                diagnostics_data = response.json()
                                logger.info(f"Successfully retrieved diagnostics from: {url}")
                                break
                        except socket.gaierror as e:
                            error_msg = f"API not reachable at {url}. DNS resolution failed. Please verify BACKEND_API_URL configuration."
                            last_error = error_msg
                            logger.error(f"DNS resolution error in diagnostics: {error_msg}")
                            raise Exception(error_msg) from e
                        except httpx.HTTPStatusError as e:
                            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                            logger.warning(f"Failed to get diagnostics from {url}: {last_error}")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                            if e.response.status_code != 404:
                                raise
                        except httpx.RequestError as e:
                            last_error = f"Connection error: {str(e)}"
                            logger.warning(f"Connection error calling {url}: {last_error}")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                            raise
                        except Exception as e:
                            last_error = str(e)
                            logger.warning(f"Error calling {url}: {last_error}")
                            if attempt < max_retries - 1:
                                time.sleep(retry_delay)
                                continue
                            if "404" not in str(e).lower():
                                raise

                    if diagnostics_data:
                        break

                if diagnostics_data:
                    st.session_state.diagnostics_result = diagnostics_data
                    st.session_state.diagnostics_running = False
                else:
                    raise Exception(f"All diagnostics endpoints failed. Last error: {last_error}")

            except socket.gaierror:
                st.session_state.diagnostics_running = False
                error_msg = f"API not reachable at {BACKEND_API_URL}. DNS resolution failed."
                st.error(f"❌ {error_msg}")
                logger.error(f"DNS resolution error in diagnostics: {error_msg}")
                st.session_state.diagnostics_result = None
            except httpx.HTTPStatusError as e:
                st.session_state.diagnostics_running = False
                error_detail = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                st.error(f"❌ Failed to get diagnostics: {error_detail}")
                logger.error(f"Diagnostics failed: {error_detail}")
                st.session_state.diagnostics_result = None
            except httpx.RequestError as e:
                st.session_state.diagnostics_running = False
                error_msg = f"Connection error: {str(e)}"
                st.error(f"❌ {error_msg}")
                logger.error(f"Diagnostics connection error: {error_msg}")
                st.session_state.diagnostics_result = None
            except Exception as e:
                st.session_state.diagnostics_running = False
                error_msg = str(e)
                st.error(f"❌ {error_msg}")
                logger.error(f"Diagnostics error: {error_msg}")
                st.session_state.diagnostics_result = None

    # Display results
    if st.session_state.diagnostics_result:
        result = st.session_state.diagnostics_result

        # Overall status
        overall_status = result.get("status", "unknown")
        if overall_status == "ok":
            st.success("✅ **Overall System Status: All Services Healthy**")
        elif overall_status == "partial":
            st.warning("⚠️ **Overall System Status: Partial - Some Services Unavailable**")
        else:
            st.error("❌ **Overall System Status: Error - Services Unavailable**")

        if result.get("timestamp"):
            st.caption(f"Last checked: {result.get('timestamp')}")

        # Display Agent Status prominently
        services = result.get("services", {})
        agent_info = services.get("agent", {})
        if agent_info:
            agent_name = agent_info.get("name", "Unknown")
            agent_status = agent_info.get("status", "unknown")
            agent_model = agent_info.get("model")

            if agent_status == "online":
                st.info(
                    f"🤖 **Agent Status:** {agent_name} ({agent_status.capitalize()})"
                    + (f" - Model: {agent_model}" if agent_model else "")
                )
            else:
                st.warning(f"🤖 **Agent Status:** {agent_name} ({agent_status.capitalize()})")

        st.markdown("---")

        # Services table
        st.subheader("Service Details")

        # Create a table with service status
        for service_name, service_data in services.items():
            # Skip agent info (already displayed above)
            if service_name == "agent":
                continue

            # Format service name for display
            display_name = service_name.replace("_", " ").title()

            status = service_data.get("status", "unknown")
            latency = service_data.get("latency")
            error = service_data.get("error")
            error_type = service_data.get("error_type")

            # Status icon and color
            if status == "ok":
                status_icon = "✅"
                status_color = "#28a745"
            else:
                status_icon = "❌"
                status_color = "#dc3545"

            # Create expandable section for each service
            with st.expander(
                f"{status_icon} **{display_name}** - {status.upper()}", expanded=status != "ok"
            ):
                col1, col2 = st.columns([2, 1])

                with col1:
                    if latency is not None:
                        st.metric("Latency", f"{latency} ms")
                    else:
                        st.metric("Latency", "N/A")

                with col2:
                    if status == "ok":
                        st.success("Healthy")
                    else:
                        st.error("Unhealthy")

                if error:
                    st.error(f"**Error:** {error}")

                    if error_type == "404":
                        st.warning("⚠️ **Routing/Configuration Error:** The service endpoint was not found.")
                    if error_type == "timeout":
                        st.warning("⏱️ **Timeout:** The service did not respond in time. It may be starting up (cold start).")

        st.markdown("---")

        # Summary statistics
        st.subheader("📊 Summary")
        total_services = len(services)
        healthy_services = sum(1 for s in services.values() if s.get("status") == "ok")
        unhealthy_services = total_services - healthy_services

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Services", total_services)
        with col2:
            st.metric("Healthy", healthy_services, delta=f"{healthy_services}/{total_services}")
        with col3:
            st.metric("Unhealthy", unhealthy_services, delta=f"-{unhealthy_services}")

        # Refresh button
        if st.button("🔄 Refresh Diagnostics", use_container_width=True):
            st.session_state.diagnostics_result = None
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Autonomous Vacancy Hunter - Multi-Agent RAG Matching System"
    "</div>",
    unsafe_allow_html=True,
    )
