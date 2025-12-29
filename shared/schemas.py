"""
Shared Pydantic schemas for the funds-search system.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class Job(BaseModel):
    """Job opening schema."""
    url: str = Field(..., description="URL of the job posting")
    company: str = Field(..., description="Company name")
    text: str = Field(..., description="Full text content of the job posting")
    vector: Optional[List[float]] = Field(None, description="Embedding vector for similarity search")
    title: Optional[str] = Field(None, description="Job title")
    location: Optional[str] = Field(None, description="Job location")
    remote: Optional[bool] = Field(None, description="Whether the position is remote")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")


class Resume(BaseModel):
    """Resume/CV schema."""
    user_id: str = Field(..., description="Unique identifier for the user")
    text: str = Field(..., description="Full text content of the resume")
    vector: Optional[List[float]] = Field(None, description="Embedding vector for similarity search")
    created_at: Optional[datetime] = Field(default_factory=datetime.now, description="Creation timestamp")


class MatchResult(BaseModel):
    """Match result schema."""
    score: float = Field(..., description="Similarity score (cosine similarity)")
    reasoning: str = Field(..., description="AI-generated reasoning for the match")
    job: Job = Field(..., description="Matched job posting")
    resume: Optional[Resume] = Field(None, description="Matched resume (if applicable)")


class SearchRequest(BaseModel):
    """Search request schema."""
    query: str = Field(..., description="Search query string")
    location: Optional[str] = Field(None, description="Optional location filter")
    role: Optional[str] = Field(None, description="Optional role/job title filter")
    remote: Optional[bool] = Field(None, description="Optional boolean for remote positions")
    user_id: Optional[str] = Field(None, description="Optional user ID for personalized search")

