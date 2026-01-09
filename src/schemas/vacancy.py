"""
Vacancy Search Service schemas using Pydantic V2.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class RoleCategory(str, Enum):
    """Role category enumeration for job classifications."""

    ENGINEERING = "Engineering"
    PRODUCT = "Product"
    DESIGN = "Design"
    DATA_ANALYTICS = "Data & Analytics"
    SALES_BUSINESS_DEVELOPMENT = "Sales & Business Development"
    MARKETING = "Marketing"
    OPERATIONS = "Operations"
    FINANCE = "Finance"
    LEGAL = "Legal"
    PEOPLE_HR = "People & HR"
    OTHER = "Other"


class ExperienceLevel(str, Enum):
    """Experience level enumeration for job positions."""

    JUNIOR = "Junior"
    MID = "Mid"
    SENIOR = "Senior"
    LEAD = "Lead"
    EXECUTIVE = "Executive"
    UNKNOWN = "Unknown"


class CompanyStage(str, Enum):
    """Company funding stage enumeration."""

    SEED = "Seed"
    SERIES_A = "Series A"
    SERIES_B = "Series B"
    SERIES_C = "Series C"
    GROWTH = "Growth"
    PUBLIC = "Public"

    @classmethod
    def get_stage_value(cls, stage) -> str:
        """
        Robust helper to get stage value from Enum or string.
        Prevents "'str' object has no attribute 'value'" errors.
        Handles legacy a16z-specific stage strings and maps them to new enum values.

        Args:
            stage: CompanyStage enum, string, or any value

        Returns:
            String value of the stage (normalized to enum value if possible)
        """
        if isinstance(stage, cls):
            return stage.value
        elif isinstance(stage, str):
            # Normalize legacy and a16z-specific strings to enum values
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
            
            # Map legacy employee count strings to appropriate stages
            if "1-10 employees" in stage_lower or "1–10 employees" in stage_lower:
                return cls.SEED.value  # Small companies typically at seed stage
            if "10-100 employees" in stage_lower or "10–100 employees" in stage_lower:
                return cls.SERIES_A.value  # Medium companies typically at Series A
            
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
    category: Optional[str] = Field(None, description="Job category filter (e.g., 'Engineering', 'Product')")
    experience_level: Optional[str] = Field(None, description="Experience level filter (e.g., 'Junior', 'Senior')")
    employee_count: Optional[List[str]] = Field(default_factory=list, description="Employee count filter (e.g., '100-1000 employees')")

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
    company_stage: str = Field(..., description="Company funding stage (e.g., 'Series A', 'Growth', '1000+ employees')")
    location: str = Field(..., description="Job location")
    industry: Optional[str] = Field(None, description="Industry sector (maps to 'Sector' in UI)")
    category: Optional[str] = Field(None, description="Job function category (maps to 'Job Function' in UI)")
    experience_level: Optional[str] = Field(None, description="Required experience level")
    remote_option: bool = Field(False, description="Whether remote work is available")
    is_hybrid: bool = Field(False, description="Whether the role is hybrid")
    salary_range: Optional[str] = Field(None, description="Salary range (e.g., '$120k-$180k')")
    min_salary: Optional[int] = Field(None, description="Minimum salary extracted from salary_range or metadata")
    description_url: str = Field(..., description="URL to full job description")
    required_skills: List[str] = Field(default_factory=list, description="Required skills list")
    employee_count: Optional[str] = Field(None, description="Company employee count")
    full_description: str = Field(..., description="Full job description for vector search")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True
        json_schema_extra = {
            "example": {
                "title": "Senior Backend Engineer",
                "company_name": "LogiTech AI",
                "company_stage": "Series A",
                "location": "San Francisco, CA",
                "industry": "AI",
                "category": "Engineering",
                "experience_level": "Senior",
                "remote_option": True,
                "salary_range": "$150k-$200k",
                "description_url": "https://example.com/jobs/backend-engineer",
                "required_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
            }
        }
