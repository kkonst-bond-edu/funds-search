"""
Vacancy Search Service schemas using Pydantic V2.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
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


class BlockContent(BaseModel):
    """Content structure for a block (META, CONTEXT, WORK, FIT, OFFER)."""
    
    units: List[str] = Field(default_factory=list, description="List of atomic strings/bullets")
    headings: List[str] = Field(default_factory=list, description="List of heading strings")


class Blocks(BaseModel):
    """Block-based segmentation of job description."""
    
    META: Optional[BlockContent] = Field(None, description="Metadata block")
    CONTEXT: Optional[BlockContent] = Field(None, description="Context block")
    WORK: Optional[BlockContent] = Field(None, description="Work responsibilities block")
    FIT: Optional[BlockContent] = Field(None, description="Fit/requirements block")
    OFFER: Optional[BlockContent] = Field(None, description="Offer/benefits block")


class RoleExtracted(BaseModel):
    """Extracted role information."""
    
    responsibilities_core: List[str] = Field(default_factory=list, description="3-12 core responsibilities")
    responsibilities_all: List[str] = Field(default_factory=list, description="All responsibilities")
    tech_stack: List[str] = Field(default_factory=list, description="Explicitly mentioned technologies")
    must_skills: List[str] = Field(default_factory=list, description="Required skills")
    nice_skills: List[str] = Field(default_factory=list, description="Nice-to-have skills")
    experience_years_min: Optional[int] = Field(None, description="Minimum years of experience")
    seniority_signal: Optional[str] = Field(None, description="Seniority indicator")
    customer_facing: bool = Field(False, description="Whether role involves direct client interaction")


class CompanyExtracted(BaseModel):
    """Extracted company information."""
    
    domain_tags: List[str] = Field(default_factory=list, description="Domain/industry tags")
    product_type: Optional[str] = Field(None, description="Product type")
    culture_signals: List[str] = Field(default_factory=list, description="Culture indicators")
    go_to_market: Optional[str] = Field(None, description="B2B or B2C")
    scale_signals: List[str] = Field(default_factory=list, description="Scale indicators (e.g., 'ARR $1B')")


class OfferExtracted(BaseModel):
    """Extracted offer information."""
    
    benefits: List[str] = Field(default_factory=list, description="List of benefits")
    equity: bool = Field(False, description="Whether equity is offered")
    hiring_process: List[str] = Field(default_factory=list, description="Hiring process steps")


class ConstraintsExtracted(BaseModel):
    """Extracted constraints information."""
    
    timezone: Optional[str] = Field(None, description="Timezone requirement")
    visa_or_work_auth: Optional[str] = Field(None, description="Visa/work authorization requirement")
    travel_required: bool = Field(False, description="Whether travel is required")


class ExtractedEntities(BaseModel):
    """Extracted entities from job description."""
    
    role: RoleExtracted = Field(default_factory=RoleExtracted, description="Role information")
    company: CompanyExtracted = Field(default_factory=CompanyExtracted, description="Company information")
    offer: OfferExtracted = Field(default_factory=OfferExtracted, description="Offer information")
    constraints: ConstraintsExtracted = Field(default_factory=ConstraintsExtracted, description="Constraints information")


class AIReadyViews(BaseModel):
    """AI-ready compacted summaries."""
    
    role_profile_text: Optional[str] = Field(None, description="Compacted role profile summary")
    company_profile_text: Optional[str] = Field(None, description="Compacted company profile summary")


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
    raw_html_url: Optional[str] = Field(None, description="URL to raw HTML file in Azure Blob Storage")
    
    # New block-based and extracted entity structure
    blocks: Optional[Blocks] = Field(None, description="Block-based segmentation of job description")
    extracted: Optional[ExtractedEntities] = Field(None, description="Extracted entities from job description")
    evidence_map: Dict[str, List[str]] = Field(default_factory=dict, description="Mapping of fields to source evidence quotes")
    ai_ready_views: Optional[AIReadyViews] = Field(None, description="AI-ready compacted summaries")
    normalization_warnings: List[str] = Field(default_factory=list, description="Warnings about data issues (e.g., invalid salary)")

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
