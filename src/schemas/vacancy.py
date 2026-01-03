"""
Vacancy Search Service schemas using Pydantic V2.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class CompanyStage(str, Enum):
    """Company funding stage enumeration matching a16z categories."""

    SEED = "Seed"
    SERIES_A = "Series A"
    GROWTH = "Growth (Series B or later)"
    EMPLOYEES_1_10 = "1-10 employees"
    EMPLOYEES_10_100 = "10-100 employees"

    @classmethod
    def get_stage_value(cls, stage) -> str:
        """
        Robust helper to get stage value from Enum or string.
        Prevents "'str' object has no attribute 'value'" errors.
        Handles a16z-specific stage strings.

        Args:
            stage: CompanyStage enum, string, or any value

        Returns:
            String value of the stage (normalized to enum value if possible)
        """
        if isinstance(stage, cls):
            return stage.value
        elif isinstance(stage, str):
            # Normalize a16z-specific strings to enum values
            stage_lower = stage.lower()
            
            # Map "Growth (Series B or later)" -> GROWTH
            if "growth" in stage_lower or "series b" in stage_lower or "series c" in stage_lower:
                return cls.GROWTH.value
            
            # Map "1000+ employees" -> GROWTH (large company = growth stage)
            if "1000+" in stage or "1000 +" in stage_lower:
                return cls.GROWTH.value
            
            # Map "Series A, 10–100 employees" -> SERIES_A (early stage with Series A)
            if "series a" in stage_lower:
                return cls.SERIES_A.value
            
            # Map employee count strings
            if "1-10 employees" in stage_lower or "1–10 employees" in stage_lower:
                return cls.EMPLOYEES_1_10.value
            if "10-100 employees" in stage_lower or "10–100 employees" in stage_lower:
                return cls.EMPLOYEES_10_100.value
            
            # If it matches an enum value exactly, return it
            for enum_member in cls:
                if stage == enum_member.value:
                    return enum_member.value
            
            # Otherwise return the string as-is
            return stage
        elif hasattr(stage, "value"):
            return stage.value
        else:
            return str(stage)


class VacancyFilter(BaseModel):
    """Filter schema for vacancy search requests."""

    role: Optional[str] = Field(None, description="Vacancy role or title filter")
    skills: Optional[List[str]] = Field(default_factory=list, description="Required skills list")
    location: Optional[str] = Field(None, description="Vacancy location filter")
    is_remote: Optional[bool] = Field(None, description="Remote work option filter")
    company_stages: Optional[List[str]] = Field(
        default_factory=list, description="Company funding stages filter"
    )
    industry: Optional[str] = Field(None, description="Industry filter")
    min_salary: Optional[int] = Field(None, ge=0, description="Minimum salary requirement")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "role": "Software Engineer",
                "skills": ["Python", "FastAPI", "PostgreSQL"],
                "location": "San Francisco",
                "is_remote": True,
                "company_stages": ["Seed", "Series A"],
                "industry": "Logistics",
                "min_salary": 120000,
            }
        }


class Vacancy(BaseModel):
    """Vacancy posting schema."""

    title: str = Field(..., description="Vacancy title")
    company_name: str = Field(..., description="Company name")
    company_stage: CompanyStage = Field(..., description="Company funding stage")
    location: str = Field(..., description="Job location")
    industry: str = Field(..., description="Industry sector")
    salary_range: Optional[str] = Field(None, description="Salary range (e.g., '$120k-$180k')")
    description_url: str = Field(..., description="URL to full job description")
    required_skills: List[str] = Field(default_factory=list, description="Required skills list")
    remote_option: bool = Field(False, description="Whether remote work is available")
    source_url: Optional[str] = Field(None, description="Source URL used to fetch this vacancy")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "title": "Senior Backend Engineer",
                "company_name": "LogiTech AI",
                "company_stage": "Series A",
                "location": "San Francisco, CA",
                "industry": "Logistics",
                "salary_range": "$150k-$200k",
                "description_url": "https://example.com/jobs/backend-engineer",
                "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
                "remote_option": True,
            }
        }
