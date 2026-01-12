import json
import logging
import os
import re
import sys
import asyncio
from typing import Any, Optional, List, Dict
from pathlib import Path
import traceback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("backfill_enrichment")

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from apps.orchestrator.agents.vacancy_analyst import VacancyAnalyst
except ImportError:
    # Try adding one more level up if running from src/scripts
    project_root = Path(__file__).parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    from apps.orchestrator.agents.vacancy_analyst import VacancyAnalyst

def clean_scraper_noise(text: str) -> str:
    """Remove technical scraper noise from job descriptions."""
    if not text:
        return text
    
    noise_patterns = [
        r"You need to enable JavaScript to run this app\.",
        r"Privacy Policy",
        r"Security",
        r"Vulnerability Disclosure",
        r"Terms of Service",
        r"Cookie Policy",
        r"© \d{4}.*?All rights reserved",
        r"Powered by.*",
    ]
    
    cleaned = text
    for pattern in noise_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    
    return cleaned.strip()

def _is_scraper_noise(text: str) -> bool:
    """Check if a text unit is scraper noise."""
    if not text or len(text.strip()) == 0:
        return True
    
    text_lower = text.lower().strip()
    noise_patterns = [
        "you need to enable javascript",
        "privacy policy",
        "security",
        "vulnerability disclosure",
        "terms of service",
        "cookie policy",
        "all rights reserved",
        "powered by",
    ]
    
    for pattern in noise_patterns:
        if pattern in text_lower:
            return True
            
    if len(text.strip()) < 3:
        return True
    
    return False

def _normalize_product_type(product_type: Any) -> Optional[str]:
    """Normalize product_type from LLM response."""
    if product_type is None:
        return None
    
    if isinstance(product_type, str):
        return product_type.strip() if product_type.strip() else None
    
    if isinstance(product_type, list):
        if len(product_type) == 0:
            return None
        first_item = product_type[0]
        if isinstance(first_item, str):
            return first_item.strip() if first_item.strip() else None
        return ", ".join(str(item) for item in product_type if item).strip() or None
    
    return str(product_type).strip() if str(product_type).strip() else None

def _safe_int(value: Any) -> Optional[int]:
    """Safely convert value to int."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        if isinstance(value, str):
            match = re.search(r'\d+', value)
            if match:
                return int(match.group())
        return int(value)
    except (ValueError, TypeError):
        return None

async def process_vacancy(analyst: VacancyAnalyst, vacancy: Dict[str, Any]) -> bool:
    """Process a single vacancy to backfill enrichment data."""
    title = vacancy.get("title")
    full_description = vacancy.get("full_description")
    required_skills = vacancy.get("required_skills")
    
    if not title or not full_description or full_description == "Parsing Error":
        logger.warning(f"Skipping {title}: Invalid description or title")
        return False
        
    logger.info(f"Enriching vacancy: {title}")
    
    try:
        # Clean description
        description_for_enrichment = clean_scraper_noise(full_description)
        
        # Call enrichment
        enrichment_result = await analyst.enrich(
            title,
            description_for_enrichment,
            required_skills=required_skills
        )
        
        if not enrichment_result or not enrichment_result.get("extracted"):
            logger.warning(f"Enrichment returned empty result for {title}")
            return False
            
        # Process blocks
        blocks_data = enrichment_result.get("blocks")
        if blocks_data:
            blocks_dict = {}
            for block_key in ["META", "CONTEXT", "WORK", "FIT", "OFFER"]:
                block_data = blocks_data.get(block_key)
                if block_data:
                    units = block_data.get("units", [])
                    filtered_units = [u for u in units if u and not _is_scraper_noise(u)]
                    
                    headings = block_data.get("headings", [])
                    filtered_headings = [h for h in headings if h and not _is_scraper_noise(h)]
                    
                    blocks_dict[block_key] = {
                        "units": filtered_units,
                        "headings": filtered_headings
                    }
            vacancy["blocks"] = blocks_dict
            
        # Process extracted entities
        extracted_data = enrichment_result.get("extracted")
        if extracted_data:
            role_data = extracted_data.get("role", {})
            company_data = extracted_data.get("company", {})
            offer_data = extracted_data.get("offer", {})
            constraints_data = extracted_data.get("constraints", {})
            
            vacancy["extracted"] = {
                "role": {
                    "responsibilities_core": role_data.get("responsibilities_core", []),
                    "responsibilities_all": role_data.get("responsibilities_all", []),
                    "tech_stack": role_data.get("tech_stack", []),
                    "must_skills": role_data.get("must_skills", []),
                    "nice_skills": role_data.get("nice_skills", []),
                    "experience_years_min": _safe_int(role_data.get("experience_years_min")),
                    "seniority_signal": role_data.get("seniority_signal"),
                    "customer_facing": role_data.get("customer_facing") or False
                },
                "company": {
                    "domain_tags": company_data.get("domain_tags", []),
                    "product_type": _normalize_product_type(company_data.get("product_type")),
                    "culture_signals": company_data.get("culture_signals", []),
                    "go_to_market": company_data.get("go_to_market"),
                    "scale_signals": company_data.get("scale_signals", [])
                },
                "offer": {
                    "benefits": offer_data.get("benefits", []),
                    "equity": offer_data.get("equity") or False,
                    "hiring_process": offer_data.get("hiring_process", [])
                },
                "constraints": {
                    "timezone": constraints_data.get("timezone"),
                    "visa_or_work_auth": constraints_data.get("visa_or_work_auth"),
                    "travel_required": constraints_data.get("travel_required") or False
                }
            }
            
        # Evidence map
        vacancy["evidence_map"] = enrichment_result.get("evidence_map", {})
        
        # AI Ready Views
        ai_views_data = enrichment_result.get("ai_ready_views")
        if ai_views_data:
            vacancy["ai_ready_views"] = {
                "role_profile_text": ai_views_data.get("role_profile_text"),
                "company_profile_text": ai_views_data.get("company_profile_text")
            }
            
        # Normalization warnings
        vacancy["normalization_warnings"] = enrichment_result.get("normalization_warnings", [])
        
        logger.info(f"Successfully enriched {title}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to enrich {title}: {e}")
        traceback.print_exc()
        return False

async def main():
    dump_path = Path("vacancies_dump.json")
    if not dump_path.exists():
        logger.error("vacancies_dump.json not found")
        return
        
    logger.info("Loading vacancies...")
    try:
        with open(dump_path, "r") as f:
            vacancies = json.load(f)
    except json.JSONDecodeError:
        logger.error("Failed to parse vacancies_dump.json")
        return
        
    logger.info(f"Loaded {len(vacancies)} vacancies")
    
    # Filter candidates
    candidates = []
    for i, v in enumerate(vacancies):
        # Check if ai_ready_views is missing OR extracted is missing
        # AND description is valid
        is_missing_data = v.get("ai_ready_views") is None or v.get("extracted") is None
        is_valid_desc = v.get("full_description") and v.get("full_description") != "Parsing Error"
        
        if is_missing_data and is_valid_desc:
            candidates.append((i, v))
            
    logger.info(f"Found {len(candidates)} vacancies to backfill")
    
    if not candidates:
        logger.info("No vacancies to backfill.")
        return
        
    # Initialize analyst
    try:
        analyst = VacancyAnalyst()
    except Exception as e:
        logger.error(f"Failed to initialize VacancyAnalyst: {e}")
        return
        
    # Process candidates
    processed_count = 0
    success_count = 0
    
    for idx, vacancy in candidates:
        logger.info(f"Processing {processed_count + 1}/{len(candidates)}: {vacancy.get('title')}")
        
        success = await process_vacancy(analyst, vacancy)
        
        if success:
            success_count += 1
            # Update the record in the main list
            vacancies[idx] = vacancy
            
            # Save incrementally every 1 record
            if success_count % 1 == 0:
                logger.info(f"Saving progress... ({success_count} updated)")
                with open(dump_path, "w") as f:
                    json.dump(vacancies, f, indent=2, ensure_ascii=False)
        
        processed_count += 1
        
    # Final save
    if success_count > 0:
        logger.info("Saving final updates...")
        with open(dump_path, "w") as f:
            json.dump(vacancies, f, indent=2, ensure_ascii=False)
            
    logger.info(f"Backfill complete. Updated {success_count} vacancies.")

if __name__ == "__main__":
    asyncio.run(main())
