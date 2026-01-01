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
import httpx
import streamlit as st
from typing import Optional, List
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
        with httpx.Client(timeout=120.0) as client:
            files = {"file": (file.name, file.read(), "application/pdf")}
            params = {"user_id": user_id}
            response = client.post(
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
        raise Exception(f"Error processing CV: {str(e)}")


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
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
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
        raise Exception(f"Error processing vacancy: {str(e)}")


def get_matches(candidate_id: str, top_k: int = 10) -> List[VacancyMatchResult]:
    """
    Get matches for a candidate.
    
    Args:
        candidate_id: Candidate/user ID
        top_k: Number of top matches to return
        
    Returns:
        List of VacancyMatchResult objects
    """
    try:
        with httpx.Client(timeout=120.0) as client:
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
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error getting matches: {e.response.status_code} - {e.response.text}")
        if e.response.status_code == 404:
            raise ValueError(f"Candidate not found: {candidate_id}")
        raise Exception(f"Failed to get matches: {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"Error getting matches: {str(e)}")
        raise Exception(f"Error getting matches: {str(e)}")


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
    
    # Health check
    try:
        with httpx.Client(timeout=5.0) as client:
            api_health = client.get(f"{BACKEND_API_URL}/health")
            if api_health.status_code == 200:
                st.success("✅ API Service: Online")
            else:
                st.error("❌ API Service: Offline")
    except Exception as e:
        st.error(f"❌ API Service: {str(e)}")

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
            with st.spinner("Processing CV... This may take a minute."):
                try:
                    result = process_cv_upload(uploaded_file, user_id)
                    st.success(f"✅ CV processed successfully!")
                    st.json(result)
                    st.info(f"**Resume ID:** {result.get('resume_id')}\n\n**Chunks Processed:** {result.get('chunks_processed')}")
                except Exception as e:
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
            with st.spinner("Processing vacancy... This may take a minute."):
                try:
                    vacancy_id = vacancy_id_input.strip() if vacancy_id_input.strip() else None
                    result = process_vacancy(vacancy_text, vacancy_id)
                    st.success(f"✅ Vacancy processed successfully!")
                    st.json(result)
                    st.info(f"**Vacancy ID:** {result.get('vacancy_id')}\n\n**Chunks Processed:** {result.get('chunks_processed')}")
                except Exception as e:
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
            with st.spinner("Finding matches... This may take a minute."):
                try:
                    matches = get_matches(candidate_id, top_k)
                    
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
                    st.error(f"❌ {str(e)}")
                    st.info("💡 Make sure you've uploaded a CV for this candidate ID first.")
                except Exception as e:
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

