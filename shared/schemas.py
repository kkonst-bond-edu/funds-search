"""
Shared Pydantic schemas for the funds-search system.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class DocumentChunk(BaseModel):
    """Document chunk schema for semantic chunks."""
    text: str = Field(..., description="Chunk text content")
    metadata: Dict = Field(default_factory=dict, description="Chunk metadata")
    embedding: List[float] = Field(..., description="Embedding vector for the chunk")


class Job(BaseModel):
    """Job opening schema."""
    id: str = Field(..., description="Unique identifier for the job")
    company: str = Field(..., description="Company name")
    title: Optional[str] = Field(None, description="Job title")
    raw_text: str = Field(..., description="Full text content of the job posting")
    vector: Optional[List[float]] = Field(None, description="Embedding vector for similarity search")
    url: Optional[str] = Field(None, description="URL of the job posting")
    source_url: Optional[str] = Field(None, description="Original source URL where the job was discovered")
    location: Optional[str] = Field(None, description="Job location")
    remote: Optional[bool] = Field(None, description="Whether the position is remote")
    vc_fund: Optional[str] = Field(None, description="VC fund or investor associated with the company")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")


class Resume(BaseModel):
    """Resume/CV schema."""
    id: str = Field(..., description="Unique identifier for the resume")
    user_id: str = Field(..., description="Unique identifier for the user")
    raw_text: str = Field(..., description="Full text content of the resume")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of document chunks")
    processed_at: Optional[datetime] = Field(default_factory=datetime.now, description="Processing timestamp")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")


class Vacancy(BaseModel):
    """Vacancy/Job posting schema."""
    id: str = Field(..., description="Unique identifier for the vacancy")
    raw_text: str = Field(..., description="Full text content of the vacancy description")
    chunks: List[DocumentChunk] = Field(default_factory=list, description="List of document chunks")
    processed_at: Optional[datetime] = Field(default_factory=datetime.now, description="Processing timestamp")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")


class MatchResult(BaseModel):
    """Match result schema."""
    score: float = Field(..., description="Similarity score (cosine similarity)")
    reasoning: str = Field(..., description="AI-generated reasoning for the match")
    job: Job = Field(..., description="Matched job posting")
    resume: Optional[Resume] = Field(None, description="Matched resume (if applicable)")


class VacancyMatchResult(BaseModel):
    """Vacancy match result schema for candidate-vacancy matching."""
    score: float = Field(..., description="Similarity score (cosine similarity)")
    reasoning: str = Field(..., description="AI-generated reasoning explaining why the vacancy fits the candidate")
    vacancy_id: str = Field(..., description="ID of the matched vacancy")
    vacancy_text: str = Field(..., description="Text content of the vacancy")
    candidate_id: str = Field(..., description="ID of the candidate")


class MatchRequest(BaseModel):
    """Match request schema for candidate-vacancy matching."""
    candidate_id: str = Field(..., description="Unique identifier for the candidate (user_id)")
    top_k: Optional[int] = Field(10, description="Number of top matches to return")


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., description="Search query string")
    location: Optional[str] = Field(None, description="Optional location filter")
    role: Optional[str] = Field(None, description="Optional role/job title filter")
    remote: Optional[bool] = Field(None, description="Optional boolean for remote positions")
    user_id: Optional[str] = Field(None, description="Optional user ID for personalized search")


class UserPersona(BaseModel):
    """User persona schema for candidate profiling."""
    technical_skills: List[str] = Field(default_factory=list, description="List of technical skills")
    career_goals: List[str] = Field(default_factory=list, description="Career goals and aspirations")
    preferred_startup_stage: Optional[str] = Field(None, description="Preferred startup stage: Seed, Series A, Series B, Series C, Series D, Series E, IPO, or Public")
    cultural_preferences: List[str] = Field(default_factory=list, description="Cultural preferences and values")
    user_id: Optional[str] = Field(None, description="User ID associated with this persona")
    
    # Structured preferences that map to database filters
    target_roles: List[str] = Field(
        default_factory=list,
        description="Preferred role titles (e.g., ['Visual Designer', 'Backend Engineer'])"
    )
    preferred_categories: List[str] = Field(
        default_factory=list,
        description="Preferred job categories (e.g., ['Design', 'Engineering'])"
    )
    preferred_experience_levels: List[str] = Field(
        default_factory=list,
        description="Preferred experience levels (e.g., ['Junior', 'Mid'])"
    )
    preferred_industries: List[str] = Field(
        default_factory=list,
        description="Preferred industries (e.g., ['Bio + Health', 'Fintech'])"
    )
    preferred_company_stages: List[str] = Field(
        default_factory=list, 
        description="Preferred company funding stages (e.g., ['Seed', 'Series A', 'Growth', 'Public'])"
    )
    preferred_locations: List[str] = Field(
        default_factory=list, 
        description="Preferred job locations (e.g., ['San Francisco', 'Remote', 'London'])"
    )
    salary_min: Optional[int] = Field(
        None, 
        ge=0, 
        description="Minimum salary requirement in USD"
    )
    remote_only: bool = Field(
        False, 
        description="Whether the candidate only wants remote positions"
    )
    
    # Unstructured context for storing insights from conversation
    chat_context: Optional[str] = Field(
        None, 
        description="Unstructured insights and context extracted from chat conversations"
    )


class MatchingReport(BaseModel):
    """Matching report schema replacing simple reasoning."""
    match_score: int = Field(..., description="Match score (0-100)")
    strengths: List[str] = Field(default_factory=list, description="List of strengths/positive matches")
    weaknesses: List[str] = Field(default_factory=list, description="List of weaknesses/gaps")
    value_proposition: str = Field(..., description="Value proposition explaining why this match is valuable")
    suggested_action: str = Field(..., description="Suggested action for the candidate")
    job_id: Optional[str] = Field(None, description="Associated job ID")
    vacancy_id: Optional[str] = Field(None, description="Associated vacancy ID")
    candidate_id: Optional[str] = Field(None, description="Associated candidate ID")


class ServiceDiagnostic(BaseModel):
    """Service diagnostic result schema."""
    status: str = Field(..., description="Service status: 'ok', 'error', or 'timeout'")
    latency: Optional[float] = Field(None, description="Response latency in milliseconds")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")
    error_type: Optional[str] = Field(None, description="Error type: '404', 'timeout', 'connection', etc.")


class SystemDiagnosticsResponse(BaseModel):
    """System diagnostics response schema."""
    status: str = Field(..., description="Overall system status: 'ok', 'error', or 'partial'")
    services: Dict[str, ServiceDiagnostic] = Field(..., description="Diagnostic results for each service")
    timestamp: Optional[str] = Field(None, description="Timestamp of the diagnostic check")

