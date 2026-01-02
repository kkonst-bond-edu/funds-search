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
from shared.schemas import VacancyMatchResult, MatchingReport, UserPersona

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
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")

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
    
    st.markdown(f"""
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
    """, unsafe_allow_html=True)


# Initialize session state
if "interview_complete" not in st.session_state:
    st.session_state.interview_complete = False

# Main UI
st.title("🔍 Autonomous Job Hunter - Candidate Matching Dashboard")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📄 Upload CV", "💼 Process Vacancy", "🎯 Find Matches", "🤖 AI Talent Strategist", "🔧 System Diagnostics"])

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

# Tab 4: AI Talent Strategist
with tab4:
    st.header("🤖 AI Talent Strategist")
    st.markdown("Have a conversation with our AI Talent Strategist to build your personalized job search profile.")
    
    # Interview status indicator
    if st.session_state.interview_complete:
        st.success("✅ Interview Complete - Your persona has been created!")
    else:
        st.info("💬 Start a conversation to build your personalized profile.")
    
    # Chat interface placeholder
    st.markdown("---")
    st.subheader("Conversation")
    
    # Initialize chat history
    if "talent_strategist_messages" not in st.session_state:
        st.session_state.talent_strategist_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI Talent Strategist. I'll help you discover the perfect job opportunities. Let's start by understanding your technical skills, career goals, and preferences. What are your main technical skills?"
            }
        ]
    
    # Display chat messages
    for message in st.session_state.talent_strategist_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message
        st.session_state.talent_strategist_messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Placeholder for AI response (will be implemented with actual LLM integration)
        with st.chat_message("assistant"):
            st.write("Thank you for that information! [Placeholder: AI Talent Strategist response will be implemented here]")
            st.session_state.talent_strategist_messages.append({
                "role": "assistant",
                "content": "Thank you for that information! [Placeholder: AI Talent Strategist response will be implemented here]"
            })
        
        st.rerun()
    
    # Reset interview button
    if st.button("🔄 Reset Interview", use_container_width=True):
        st.session_state.talent_strategist_messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI Talent Strategist. I'll help you discover the perfect job opportunities. Let's start by understanding your technical skills, career goals, and preferences. What are your main technical skills?"
            }
        ]
        st.session_state.interview_complete = False
        st.rerun()
    
    # Complete interview button (placeholder)
    if st.button("✅ Complete Interview", type="primary", use_container_width=True):
        st.session_state.interview_complete = True
        st.success("Interview completed! Your persona will be used for job matching.")
        st.rerun()

# Tab 5: System Diagnostics
with tab5:
    st.header("🔧 System Diagnostics")
    st.markdown("Run a comprehensive health check on all backend services and dependencies.")
    
    # Initialize session state for diagnostics
    if "diagnostics_result" not in st.session_state:
        st.session_state.diagnostics_result = None
    if "diagnostics_running" not in st.session_state:
        st.session_state.diagnostics_running = False
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info("💡 This diagnostic check will test connectivity to all services including CV Processor, Embedding Service, Pinecone Vector Store, and LLM Provider. Services may take a few seconds to wake up from cold start.")
    
    with col2:
        run_diagnostics = st.button("🚀 Run Full System Check", type="primary", use_container_width=True)
    
    if run_diagnostics or st.session_state.diagnostics_running:
        st.session_state.diagnostics_running = True
        
        # Show loading state
        with st.spinner("🔍 Running system diagnostics... Waking up services..."):
            try:
                # Make request to diagnostics endpoint
                with httpx.Client(timeout=60.0) as client:
                    response = client.get(f"{BACKEND_API_URL}/api/v1/system/diagnostics")
                    response.raise_for_status()
                    diagnostics_data = response.json()
                    st.session_state.diagnostics_result = diagnostics_data
                    st.session_state.diagnostics_running = False
            except httpx.HTTPStatusError as e:
                st.session_state.diagnostics_running = False
                st.error(f"❌ Failed to get diagnostics: HTTP {e.response.status_code} - {e.response.text}")
                st.session_state.diagnostics_result = None
            except Exception as e:
                st.session_state.diagnostics_running = False
                st.error(f"❌ Error running diagnostics: {str(e)}")
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
        
        st.markdown("---")
        
        # Services table
        st.subheader("Service Details")
        
        services = result.get("services", {})
        
        # Create a table with service status
        for service_name, service_data in services.items():
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
            with st.expander(f"{status_icon} **{display_name}** - {status.upper()}", expanded=status != "ok"):
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
                    
                    # Special handling for 404 errors
                    if error_type == "404":
                        st.warning("⚠️ **Routing/Configuration Error:** The service endpoint was not found. Check service configuration and routing.")
                    
                    # Special handling for timeouts
                    if error_type == "timeout":
                        st.warning("⏱️ **Timeout:** The service did not respond in time. It may be starting up (cold start).")
                        
                        # Force wake up button
                        if st.button(f"🔨 Force Wake Up {display_name}", key=f"wakeup_{service_name}"):
                            with st.spinner(f"Waking up {display_name}..."):
                                try:
                                    # Send multiple light requests to wake up the service
                                    wakeup_url = None
                                    if service_name == "cv_processor":
                                        wakeup_url = f"{CV_PROCESSOR_URL}/health"
                                    elif service_name == "embedding_service":
                                        wakeup_url = f"{EMBEDDING_SERVICE_URL}/health"
                                    
                                    if wakeup_url:
                                        for i in range(3):
                                            try:
                                                with httpx.Client(timeout=5.0) as client:
                                                    client.get(wakeup_url)
                                            except:
                                                pass
                                            time.sleep(1)
                                        st.success(f"Sent wake-up requests to {display_name}")
                                        st.info("💡 Run diagnostics again to check if the service is now available.")
                                except Exception as e:
                                    st.error(f"Failed to wake up service: {str(e)}")
        
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
    "Autonomous Job Hunter - Multi-Agent RAG Matching System"
    "</div>",
    unsafe_allow_html=True
)

