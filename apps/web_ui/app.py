"""
Streamlit web UI for funds-search system.
Provides a user-friendly interface for CV processing, vacancy processing, and matching.
"""
import sys
import os
from pathlib import Path

# Add parent directories to path for shared imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import logging
import uuid
import time
import httpx
import streamlit as st
from typing import Optional, List, Tuple
from shared.schemas import VacancyMatchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
# Note: cv-processor listens on port 8001 internally (8002 is the external host mapping)
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://api:8000")
CV_PROCESSOR_URL = os.getenv("CV_PROCESSOR_URL", "http://cv-processor:8001")

# Retry configuration for cold start handling
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
TIMEOUT = 300.0  # seconds (5 minutes) - increased for CV processing and matching operations

# Page configuration
st.set_page_config(
    page_title="Funds Search - Candidate Matching",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
    </style>
""", unsafe_allow_html=True)


def check_service_health(service_url: str, service_name: str, timeout: float = 5.0) -> Tuple[bool, str]:
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
    **kwargs
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
                
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
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
            "POST",
            f"{CV_PROCESSOR_URL}/process-cv",
            files=files,
            params=params
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
            json={
                "vacancy_id": vacancy_id,
                "text": vacancy_text
            }
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
                st.info("Searching for candidate data... Azure services might still be initializing.")
                info_message_shown = True
            
            # Make the request with extended timeout to allow backend orchestrator retries
            with httpx.Client(timeout=request_timeout) as client:
                response = client.post(
                    f"{BACKEND_API_URL}/match",
                    json={
                        "candidate_id": candidate_id,
                        "top_k": top_k
                    }
                )
                response.raise_for_status()
                results = response.json()
                # Parse results into VacancyMatchResult objects
                return [VacancyMatchResult(**result) for result in results]
                
        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            last_exception = e
            elapsed_time = time.time() - start_time
            
            # Check if we should continue retrying
            if elapsed_time >= max_wait_time or attempt >= max_retries - 1:
                break
            
            # Show info message if not already shown
            if not info_message_shown:
                st.info("Searching for candidate data... Azure services might still be initializing.")
                info_message_shown = True
            
            # Wait before retrying
            time.sleep(retry_delay)
            
        except httpx.HTTPStatusError as e:
            last_exception = e
            last_status_code = e.response.status_code
            elapsed_time = time.time() - start_time
            
            # Treat 404/500 as potential cold start symptoms on first attempt
            if e.response.status_code in (404, 500):
                error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
                is_cold_start_symptom = (
                    attempt == 0 or  # First attempt
                    "Candidate not found" in error_detail or
                    "not found" in error_detail.lower()
                )
                
                if is_cold_start_symptom and elapsed_time < max_wait_time and attempt < max_retries - 1:
                    # Show info message if not already shown
                    if not info_message_shown:
                        st.info("Searching for candidate data... Azure services might still be initializing.")
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
                        logger.error(f"HTTP error getting matches: {e.response.status_code} - {error_detail}")
                        raise Exception(f"Failed to get matches: {e.response.status_code} - {error_detail}")
            else:
                # Other HTTP errors - don't retry
                logger.error(f"HTTP error getting matches: {e.response.status_code} - {e.response.text}")
                raise Exception(f"Failed to get matches: {e.response.status_code} - {e.response.text}")
        
        except Exception as e:
            last_exception = e
            elapsed_time = time.time() - start_time
            
            # For other exceptions, retry if it might be a cold start issue (first attempt only)
            if attempt == 0 and elapsed_time < max_wait_time and attempt < max_retries - 1:
                if not info_message_shown:
                    st.info("Searching for candidate data... Azure services might still be initializing.")
                    info_message_shown = True
                logger.warning(f"Unexpected error on attempt {attempt + 1}/{max_retries}: {str(e)}. Retrying...")
                time.sleep(retry_delay)
                continue
            else:
                # Don't retry other exceptions after first attempt
                logger.error(f"Error getting matches: {str(e)}")
                raise
    
    # If we get here, all retries failed
    if last_status_code in (404, 500):
        if last_status_code == 404:
            error_msg = f"Candidate not found: {candidate_id} (after {max_wait_time} seconds of retries)"
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
    
    st.markdown(f"""
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
                    {match.vacancy_text[:500]}{'...' if len(match.vacancy_text) > 500 else ''}
                </div>
            </details>
        </div>
    """, unsafe_allow_html=True)


# Main UI
st.title("🔍 Funds Search - Candidate Matching Dashboard")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    st.info(f"**API URL:** {BACKEND_API_URL}\n\n**CV Processor URL:** {CV_PROCESSOR_URL}")
    
    st.markdown("---")
    st.subheader("Service Health")
    
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
    
    # Refresh button
    if st.button("🔄 Refresh Health Status", use_container_width=True):
        st.rerun()

# Main content tabs
tab1, tab2, tab3 = st.tabs(["📄 Upload CV", "💼 Process Vacancy", "🎯 Find Matches"])

# Tab 1: Upload CV
with tab1:
    st.header("Upload Candidate Resume")
    st.markdown("Upload a PDF resume to process and index it for matching.")
    
    user_id = st.text_input(
        "Candidate ID (User ID)",
        value="",
        help="Enter a unique identifier for the candidate (e.g., email, username, or UUID)"
    )
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF resume file"
    )
    
    if st.button("Process CV", type="primary", use_container_width=True):
        if not user_id:
            st.error("Please enter a Candidate ID")
        elif not uploaded_file:
            st.error("Please upload a PDF file")
        else:
            # Progress bar for CV processing
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📤 Uploading CV file...")
                progress_bar.progress(10)
                
                status_text.text("🔄 Converting PDF to text...")
                progress_bar.progress(30)
                
                status_text.text("🧠 Generating embeddings...")
                progress_bar.progress(50)
                
                status_text.text("💾 Saving to database...")
                progress_bar.progress(70)
                
                result = process_cv_upload(uploaded_file, user_id)
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.success(f"✅ CV processed successfully!")
                st.json(result)
                st.info(f"**Resume ID:** {result.get('resume_id')}\n\n**Chunks Processed:** {result.get('chunks_processed')}")
                
                # Clear progress after a moment
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"CV processing error: {str(e)}")

# Tab 2: Process Vacancy
with tab2:
    st.header("Process Vacancy Description")
    st.markdown("Paste a vacancy description to process and index it for matching.")
    
    vacancy_id_input = st.text_input(
        "Vacancy ID (Optional)",
        value="",
        help="Enter a unique identifier for the vacancy. Leave empty to auto-generate."
    )
    
    vacancy_text = st.text_area(
        "Vacancy Description",
        height=300,
        placeholder="Paste the full vacancy description here...",
        help="Enter the complete text description of the vacancy"
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
                
                st.success(f"✅ Vacancy processed successfully!")
                st.json(result)
                st.info(f"**Vacancy ID:** {result.get('vacancy_id')}\n\n**Chunks Processed:** {result.get('chunks_processed')}")
                
                # Clear progress after a moment
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Error: {str(e)}")
                logger.error(f"Vacancy processing error: {str(e)}")

# Tab 3: Find Matches
with tab3:
    st.header("Find Candidate Matches")
    st.markdown("Enter a candidate ID to find matching vacancies.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        candidate_id = st.text_input(
            "Candidate ID",
            value="",
            help="Enter the candidate ID (user_id) that was used when uploading the CV"
        )
    
    with col2:
        top_k = st.number_input(
            "Number of Matches",
            min_value=1,
            max_value=50,
            value=10,
            help="Number of top matches to return"
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

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Funds Search - Multi-Agent RAG Matching System"
    "</div>",
    unsafe_allow_html=True
)

