import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class FilterConfig(BaseModel):
    """Configuration for flexible vacancy download control"""
    
    # 1. Main switches
    remote_only: bool = True
    
    # 2. Geography
    include_locations: List[str] = Field(default_factory=list) # e.g. ["London", "Europe"]
    exclude_locations: List[str] = Field(default_factory=list) # e.g. ["US Only", "San Francisco"]
    
    # 3. Keywords in title
    include_title_keywords: List[str] = Field(default_factory=list) # e.g. ["Python", "Backend"]
    exclude_title_keywords: List[str] = Field(default_factory=list) # e.g. ["Senior", "Staff", "Lead"]
    
    # 4. Companies and Industries
    exclude_companies: List[str] = Field(default_factory=list)
    include_categories: List[str] = Field(default_factory=list) # e.g. ["Engineering", "AI"]
    
    # 5. Data freshness
    max_days_old: Optional[int] = None # If set, ignore old vacancies

class VacancyFilterService:
    def __init__(self, config: FilterConfig):
        self.config = config

    def should_process(self, metadata: Dict[str, Any]) -> bool:
        """
        Decides whether to download a vacancy based on list metadata.
        Returns True if the vacancy passes all filters.
        """
        title = str(metadata.get("title", "")).lower()
        company = str(metadata.get("company_name", "")).lower()
        
        # --- 1. Remote Filter ---
        if self.config.remote_only:
            is_remote_flag = metadata.get("is_remote", False)
            location_str = str(metadata.get("location", "")).lower()
            
            # Pass if flag is set OR "remote" word is in location
            if not (is_remote_flag or "remote" in location_str):
                logger.debug(f"Filter [Remote]: Skipped '{title}' at '{company}' (Location: {metadata.get('location')})")
                return False

        # --- 2. Location Filter (Include/Exclude) ---
        location_raw = str(metadata.get("location", "")).lower()
        
        # Exclude (exclude explicit mismatch first)
        for loc in self.config.exclude_locations:
            if loc.lower() in location_raw:
                logger.debug(f"Filter [Location Exclude]: Skipped '{title}' due to prohibited location '{loc}'")
                return False
                
        # Include (if list is not empty, location MUST contain one of the values)
        if self.config.include_locations:
            match_found = any(loc.lower() in location_raw for loc in self.config.include_locations)
            if not match_found:
                logger.debug(f"Filter [Location Include]: Skipped '{title}' (Location: {metadata.get('location')})")
                return False

        # --- 3. Title Keywords Filter ---
        # Exclude
        for kw in self.config.exclude_title_keywords:
            if kw.lower() in title:
                logger.debug(f"Filter [Title Exclude]: Skipped '{title}' due to keyword '{kw}'")
                return False
        
        # Include
        if self.config.include_title_keywords:
            match_found = any(kw.lower() in title for kw in self.config.include_title_keywords)
            if not match_found:
                logger.debug(f"Filter [Title Include]: Skipped '{title}' (missing keywords)")
                return False

        # --- 4. Company Filter ---
        if self.config.exclude_companies:
            if any(c.lower() in company for c in self.config.exclude_companies):
                logger.debug(f"Filter [Company]: Skipped '{company}'")
                return False

        # --- 5. Date Filter (Freshness) ---
        if self.config.max_days_old:
            published_at = metadata.get("published_at")
            if published_at:
                try:
                    # Handle timestamp or ISO string
                    if isinstance(published_at, (int, float)):
                         # Usually timestamp in seconds, but sometimes ms
                        ts = published_at if published_at < 10000000000 else published_at / 1000
                        pub_date = datetime.fromtimestamp(ts)
                    elif isinstance(published_at, str):
                        # Simple attempt to parse ISO string if possible, or skip
                        try:
                            pub_date = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
                        except ValueError:
                            # If format is complex, skip date check for now
                            return True
                    else:
                        pub_date = datetime.now() # Fallback

                    delta = datetime.now() - pub_date
                    if delta.days > self.config.max_days_old:
                        logger.debug(f"Filter [Date]: Skipped '{title}' (posted {delta.days} days ago)")
                        return False
                except Exception:
                    # If failed to parse date, skip filter
                    pass

        return True
