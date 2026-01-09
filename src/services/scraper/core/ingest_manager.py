"""
Ingest manager for processing vacancies and uploading to Pinecone.
"""

import os
import re
import logging
import hashlib
import unicodedata
import traceback
from typing import List, Optional, Any
from urllib.parse import unquote
import httpx

from src.schemas.vacancy import Vacancy, CompanyStage
from shared.pinecone_client import VectorStore

# Configure basic logger for console output (needed for import fallback)
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Import ClassificationAgent - use absolute import path
# File location: apps/orchestrator/agents/classification.py
# Current file: src/services/scraper/core/ingest_manager.py
# Absolute import: from apps.orchestrator.agents.classification import ClassificationAgent
# This should work if project root is in PYTHONPATH (standard practice)
try:
    from apps.orchestrator.agents.classification import ClassificationAgent
    logger.debug("Successfully imported ClassificationAgent from apps.orchestrator.agents.classification")
except ImportError as e:
    # Fallback: try adding project root to path and importing again
    import sys
    from pathlib import Path
    
    # Get project root (this file is at src/services/scraper/core/ingest_manager.py)
    # Go up 5 levels: core -> scraper -> services -> src -> root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    project_root_str = str(project_root)
    
    # Check if project root is already in sys.path
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
        logger.debug(f"Added project root to sys.path: {project_root_str}")
        try:
            from apps.orchestrator.agents.classification import ClassificationAgent
            logger.info(f"Successfully imported ClassificationAgent after adding {project_root_str} to sys.path")
        except ImportError as e2:
            ClassificationAgent = None
            expected_path = project_root / "apps" / "orchestrator" / "agents" / "classification.py"
            logger.error(
                f"Failed to import ClassificationAgent even after adding project root to path. "
                f"Original error: {e}. Second error: {e2}. "
                f"Project root: {project_root_str}. "
                f"Expected file: {expected_path}. "
                f"File exists: {expected_path.exists()}"
            )
    else:
        # Project root is already in path, but import still failed
        ClassificationAgent = None
        expected_path = project_root / "apps" / "orchestrator" / "agents" / "classification.py"
        logger.error(
            f"Failed to import ClassificationAgent. Error: {e}. "
            f"Project root ({project_root_str}) is already in sys.path. "
            f"Expected file: {expected_path}. "
            f"File exists: {expected_path.exists()}. "
            f"Please check that apps/orchestrator/agents/classification.py exists and is accessible."
        )


def get_api_text(data: Any) -> str:
    """
    Extract text from API data that can be in various formats (string, dict, list).
    
    This helper function fixes issues with dict-to-string conversion that can cause
    "Tags: 0" problems by properly extracting text from nested data structures.
    
    Args:
        data: API data in any format (string, dict, list, etc.)
        
    Returns:
        Extracted text string, or empty string if data is empty/None
    """
    if not data:
        return ""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        # Check common keys for text content
        for key in ["name", "label", "text", "display_name"]:
            if data.get(key):
                return str(data[key]).strip()
        # If no standard key found, try to convert the whole dict
        return str(data).strip()
    if isinstance(data, list):
        # If it's a list, process first non-empty item
        for item in data:
            if item:
                text = get_api_text(item)
                if text:
                    return text
        return ""
    return str(data).strip()


def slugify(text: str) -> str:
    """
    Convert text to a URL-safe slug, handling non-Latin characters and URL-encoded characters.
    
    Handles:
    1. URL-encoded characters (like %20) by decoding them first
    2. Cyrillic and other non-Latin characters by transliteration
    3. Normalizing Unicode characters
    4. Stripping remaining non-ASCII characters
    
    Args:
        text: Text to slugify (can contain non-Latin characters and URL-encoded chars)
        
    Returns:
        URL-safe slug string with only ASCII characters
    """
    if not text:
        return ""
    
    # Decode URL-encoded characters (e.g., %20 -> space, %2D -> hyphen)
    # Replace encoded characters with hyphens before processing
    try:
        # First, replace common URL-encoded characters with hyphens
        text = text.replace('%20', '-')
        text = text.replace('%2D', '-')
        text = text.replace('%2F', '-')
        text = text.replace('%5F', '-')
        # Then decode any remaining URL-encoded characters
        text = unquote(text)
    except Exception as e:
        # If decoding fails, continue with original text
        logger.debug(f"Error decoding URL-encoded text: {e}")
    
    # Normalize Unicode (NFD = Canonical Decomposition)
    # This separates base characters from combining marks
    text = unicodedata.normalize('NFD', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Try to transliterate common non-ASCII characters to ASCII equivalents
    # This handles some Cyrillic and other characters
    transliteration_map = {
        # Cyrillic to Latin approximations
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
        # Uppercase Cyrillic
        'А': 'a', 'Б': 'b', 'В': 'v', 'Г': 'g', 'Д': 'd', 'Е': 'e', 'Ё': 'yo',
        'Ж': 'zh', 'З': 'z', 'И': 'i', 'Й': 'y', 'К': 'k', 'Л': 'l', 'М': 'm',
        'Н': 'n', 'О': 'o', 'П': 'p', 'Р': 'r', 'С': 's', 'Т': 't', 'У': 'u',
        'Ф': 'f', 'Х': 'h', 'Ц': 'ts', 'Ч': 'ch', 'Ш': 'sh', 'Щ': 'sch',
        'Ъ': '', 'Ы': 'y', 'Ь': '', 'Э': 'e', 'Ю': 'yu', 'Я': 'ya',
    }
    
    # Apply transliteration
    transliterated = ''.join(transliteration_map.get(char, char) for char in text)
    
    # Remove all non-ASCII characters that weren't transliterated
    # Keep only alphanumeric, spaces, and hyphens
    text = re.sub(r'[^\x00-\x7F\w\s-]', '', transliterated)
    
    # Replace spaces and special characters with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    
    # Remove leading/trailing hyphens
    text = text.strip('-')
    
    return text


def generate_vacancy_id(company_name: str, title: str, description_url: str) -> str:
    """
    Generate an absolutely unique slugified ID for a vacancy using URL hash.
    
    Format: {slugify(company_name)}-{slugify(title)}-{hash_prefix}
    
    The MD5 hash of the description_url ensures absolute uniqueness:
    - Different companies with the same job titles get unique IDs
    - Same URL re-scraped will generate the same ID (allows updates without duplicates)
    - Prevents collisions when multiple companies have identical job titles
    
    Args:
        company_name: Company name
        title: Job title
        description_url: URL of the job description (used for hash to ensure uniqueness)
        
    Returns:
        Slugified ID like "company-title-a1b2c3" where the hash ensures uniqueness
    """
    company_slug = slugify(company_name)
    title_slug = slugify(title)
    
    # Calculate MD5 hash of the description_url and take first 6 characters
    # This ensures absolute uniqueness and allows re-scraping the same URL
    # to update data without creating duplicates
    hash_prefix = hashlib.md5(description_url.encode('utf-8')).hexdigest()[:6]
    
    return f"{company_slug}-{title_slug}-{hash_prefix}"


def generate_search_text(vacancy: Vacancy) -> str:
    """
    Generate combined search text for embedding from vacancy data.
    Creates rich context by combining Title, Company, Category, Industry, Location, Skills,
    and the first 1000 characters of full_description.
    
    Args:
        vacancy: Vacancy object
        
    Returns:
        Combined search text string
    """
    title = vacancy.title
    company_name = vacancy.company_name
    # Use get_api_text to ensure location is properly extracted (handles dict/list formats)
    location = get_api_text(vacancy.location) if vacancy.location else ""
    required_skills = vacancy.required_skills
    category = vacancy.category or ""
    # Use get_api_text to ensure industry is properly extracted (handles dict/list formats)
    industry = get_api_text(vacancy.industry) if vacancy.industry else ""
    
    skills_str = ", ".join(required_skills) if required_skills else ""
    
    # Get first 1000 characters of full_description
    full_description = vacancy.full_description if vacancy.full_description else ""
    description_preview = full_description[:1000] if len(full_description) > 1000 else full_description
    
    search_text = (
        f"Title: {title}. "
        f"Company: {company_name}. "
        f"Category: {category}. "
        f"Industry: {industry}. "
        f"Location: {location}. "
        f"Skills: {skills_str}. "
        f"Description: {description_preview}"
    )
    
    return search_text


async def get_embedding(text: str, embedding_service_url: str) -> List[float]:
    """
    Call embedding-service to generate embedding for a single text.
    
    Args:
        text: Text string to embed
        embedding_service_url: URL of the embedding service
        
    Returns:
        Embedding vector as list of floats
        
    Raises:
        httpx.HTTPError: If the embedding service is unreachable
        ValueError: If the response format is invalid
    """
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes timeout
        try:
            logger.info(f"Calling embedding service: {embedding_service_url}")
            response = await client.post(
                f"{embedding_service_url}/embed",
                json={"texts": [text]}
            )
            response.raise_for_status()
            result = response.json()
            
            if "embeddings" not in result:
                raise ValueError("Invalid response format from embedding service: missing 'embeddings' key")
            
            embeddings = result["embeddings"]
            if not embeddings or len(embeddings) == 0:
                raise ValueError("Empty embeddings list returned from embedding service")
            
            embedding = embeddings[0]
            logger.info(f"Embedding generated: dim={len(embedding)}")
            return embedding
            
        except httpx.TimeoutException as e:
            logger.error(f"Embedding service timeout: {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Embedding service HTTP error: status_code={e.response.status_code}, error={str(e)}"
            )
            raise
        except httpx.RequestError as e:
            logger.error(f"Embedding service unreachable: {str(e)}")
            raise httpx.HTTPError(f"Embedding service unreachable: {str(e)}") from e


class IngestManager:
    """Manages ingestion of vacancies into Pinecone vector database."""

    def __init__(
        self,
        embedding_service_url: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the ingest manager.
        
        Args:
            embedding_service_url: URL of the embedding service.
                                 Defaults to EMBEDDING_SERVICE_URL env var or
                                 "http://embedding-service:8001"
            pinecone_index_name: Pinecone index name.
                                Defaults to PINECONE_INDEX_NAME env var or
                                "funds-search"
        """
        self.embedding_service_url = embedding_service_url or os.getenv(
            "EMBEDDING_SERVICE_URL", "http://embedding-service:8001"
        )
        
        # Get pinecone_index_name from parameter or environment variable
        self.pinecone_index_name = pinecone_index_name or os.getenv(
            "PINECONE_INDEX_NAME", None
        )
        
        # Initialize Pinecone client
        try:
            self.vector_store = VectorStore()
            if self.pinecone_index_name:
                self.vector_store.index_name = self.pinecone_index_name
            logger.info(f"Pinecone client initialized: index={self.vector_store.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise
        
        # Initialize ClassificationAgent for AI-powered vacancy classification
        if ClassificationAgent is None:
            logger.warning("ClassificationAgent class not available (import failed), skipping initialization")
            self.classifier = None
        else:
            try:
                self.classifier = ClassificationAgent()
                logger.info("ClassificationAgent initialized successfully")
            except Exception as e:
                # Log full traceback for debugging
                error_traceback = traceback.format_exc()
                logger.error(
                    f"Failed to initialize ClassificationAgent: {str(e)}\n"
                    f"Traceback:\n{error_traceback}"
                )
                # Don't raise - allow processing to continue without classification
                self.classifier = None

    def _validate_vacancy(self, vacancy: Vacancy) -> bool:
        """
        Validate vacancy data before processing.
        
        Args:
            vacancy: Vacancy object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not vacancy.title or vacancy.title.strip() == "":
            logger.warning("Vacancy validation failed: empty title")
            return False
        
        if not vacancy.company_name or vacancy.company_name.strip() == "":
            logger.warning("Vacancy validation failed: empty company_name")
            return False
        
        if not vacancy.description_url or vacancy.description_url.strip() == "":
            logger.warning("Vacancy validation failed: empty description_url")
            return False
        
        # Check for 'Unknown' values - skip if both title and company are 'Unknown'
        if "Unknown" in vacancy.title and "Unknown" in vacancy.company_name:
            logger.warning(
                f"Vacancy validation failed: both title and company_name contain 'Unknown'. "
                f"Title: '{vacancy.title}', Company: '{vacancy.company_name}'. Skipping upload to maintain data quality."
            )
            return False
        
        return True

    async def process_new_vacancy(self, vacancy: Vacancy, namespace: str = "vacancies") -> bool:
        """
        Process a new vacancy: validate, generate embedding, and upsert to Pinecone.
        
        Args:
            vacancy: Vacancy object to process
            namespace: Pinecone namespace to use (default: "vacancies")
            
        Returns:
            True if successfully processed, False otherwise
        """
        # Check classifier status at the beginning
        if self.classifier is None:
            logger.info(f"Processing vacancy '{vacancy.title}': ClassificationAgent is None, skipping AI classification")
        else:
            logger.info(f"Processing vacancy '{vacancy.title}': ClassificationAgent is active, will perform AI classification")
        
        # Validate the vacancy
        if not self._validate_vacancy(vacancy):
            logger.error(f"Vacancy validation failed: {vacancy.title} at {vacancy.company_name}")
            return False
        
        try:
            # AI Classification: Classify vacancy before generating embedding
            if self.classifier:
                try:
                    # If description is "Parsing Error", use only title for classification
                    description_for_classification = (
                        vacancy.title 
                        if vacancy.full_description == "Parsing Error" 
                        else vacancy.full_description
                    )
                    
                    # Build AI instructions based on existing data and title keywords
                    title_lower = (vacancy.title or "").lower()
                    hints = []
                    
                    # Industry protection hint: If industry is already specific, tell AI to preserve it
                    if vacancy.industry and vacancy.industry.strip():
                        current_industry = vacancy.industry.strip()
                        current_industry_lower = current_industry.lower()
                        # Check if it's a specific industry (not generic "Technology")
                        specific_industries = [
                            "enterprise", "american dynamism", "fintech", "consumer", 
                            "bio + health", "crypto", "crypto/web3", "web3", "games"
                        ]
                        if any(specific in current_industry_lower for specific in specific_industries):
                            hints.append(f"PRIORITY: The industry is already known as '{current_industry}'. Do not change it unless it is fundamentally wrong.")
                            logger.debug(f"Added industry protection hint: {current_industry}")
                    
                    # Category hint: If title contains Finance/ERP/Budgeting, prioritize Finance or Operations
                    if any(keyword in title_lower for keyword in ["finance", "erp", "budgeting"]):
                        hints.append("PRIORITY: If the title contains 'Finance', 'ERP', or 'Budgeting', prioritize the 'Finance' or 'Operations' category over 'Engineering'.")
                        logger.debug(f"Added category hint for Finance/ERP/Budgeting role: {vacancy.title}")
                    
                    # Internal systems/operations roles hint
                    if any(keyword in title_lower for keyword in ["systems", "erp", "program lead"]):
                        hints.append("Note: This is an internal systems/operations role.")
                        logger.debug(f"Added classification hint for internal systems role: {vacancy.title}")
                    
                    # Append all hints to description
                    if hints:
                        hint_text = "\n\n" + "\n".join(hints)
                        description_for_classification = description_for_classification + hint_text
                        logger.debug(f"Added {len(hints)} AI instruction hints for: {vacancy.title}")
                    
                    classification_result = await self.classifier.classify(
                        vacancy.title, 
                        description_for_classification
                    )
                    
                    # Update vacancy fields with AI classification results
                    # Priority: API-provided data takes precedence over AI classification
                    
                    # Track which fields were updated vs protected for debug logging
                    updated_fields = []
                    protected_fields = []
                    
                    # Check location and description for "Hybrid" or "Remote"
                    # Hybrid = False (not 100% remote), Remote = True (100% remote)
                    location_lower = (vacancy.location or "").lower()
                    description_lower = (vacancy.full_description or "").lower() if vacancy.full_description != "Parsing Error" else ""
                    location_contains_hybrid = "hybrid" in location_lower
                    description_contains_hybrid = "hybrid" in description_lower
                    location_contains_remote = "remote" in location_lower and "hybrid" not in location_lower
                    description_contains_remote = "remote" in description_lower and "hybrid" not in description_lower
                    
                    # Category: Apply special rules before AI classification
                    # Rule: If title contains "Manager" and industry is "Finance" or "ERP", prefer "Operations" or "Finance"
                    title_lower = (vacancy.title or "").lower()
                    current_industry_lower = (vacancy.industry or "").lower()
                    category_override = None
                    
                    if "manager" in title_lower:
                        if "finance" in current_industry_lower or "erp" in current_industry_lower:
                            # Prefer Operations or Finance for Finance/ERP Manager roles
                            if "finance" in current_industry_lower:
                                category_override = "Finance"
                            else:
                                category_override = "Operations"
                            logger.debug(f"Category override for Manager role: {category_override} (industry: {vacancy.industry})")
                    
                    # Category: Only update if current value is None or "Unknown"
                    classified_category = classification_result.get("category")
                    if category_override:
                        # Use override instead of AI classification
                        vacancy.category = category_override
                        updated_fields.append(f"category={vacancy.category} (override for Manager role)")
                        logger.debug(f"Set category to {category_override} based on Manager + Finance/ERP rule")
                    elif classified_category and classified_category.strip():
                        current_category = (vacancy.category or "").strip()
                        if not current_category or current_category.lower() == "unknown":
                            vacancy.category = classified_category.strip()
                            updated_fields.append(f"category={vacancy.category}")
                            logger.debug(f"Updated category from AI: {vacancy.category}")
                        else:
                            protected_fields.append(f"category={current_category} (protected, AI suggested: {classified_category})")
                    
                    # Industry: Smart merge - add AI industry to existing if they don't match and both are specific
                    # DO NOT let AI change specific values to generic "AI" or "Technology"
                    # Use get_api_text to ensure industry is properly extracted (handles dict/list formats)
                    classified_industry = classification_result.get("industry")
                    if classified_industry and classified_industry.strip():
                        # Normalize current industry using get_api_text (handles dict/list formats)
                        current_industry_raw = vacancy.industry or ""
                        current_industry = get_api_text(current_industry_raw).strip() if current_industry_raw else ""
                        current_industry_lower = current_industry.lower()
                        classified_industry_clean = classified_industry.strip()
                        classified_industry_lower = classified_industry_clean.lower()
                        
                        # List of specific industry values that should be protected
                        specific_industries = [
                            "enterprise", "american dynamism", "fintech", "consumer", 
                            "bio + health", "crypto", "crypto/web3", "web3", "games"
                        ]
                        
                        # Check if current industry contains any specific values
                        current_has_specific = any(
                            specific in current_industry_lower 
                            for specific in specific_industries
                        )
                        
                        # Check if AI industry is specific (not generic)
                        ai_is_specific = any(
                            specific in classified_industry_lower 
                            for specific in specific_industries
                        )
                        ai_is_generic = classified_industry_lower in ["technology", "ai", "other"]
                        
                        # Update logic:
                        # 1. If current is empty or "Technology", update with AI
                        # 2. If current has specific values and AI is also specific but different, merge them
                        # 3. If AI is generic and current is specific, reject AI
                        # 4. If both are specific and same, no change
                        if not current_industry or current_industry_lower == "technology":
                            # Always update if current is empty or generic "Technology"
                            vacancy.industry = classified_industry_clean
                            updated_fields.append(f"industry={vacancy.industry}")
                            logger.debug(f"Updated industry from AI: {vacancy.industry} (was: {current_industry or 'empty'})")
                        elif current_industry_lower == classified_industry_lower:
                            # Same value, no change needed
                            pass
                        elif current_has_specific and ai_is_generic:
                            # Current has specific values, AI wants generic - protect current
                            protected_fields.append(f"industry={current_industry} (protected specific value, AI suggested generic: {classified_industry_clean})")
                            logger.debug(
                                f"Protected specific industry '{current_industry}' from AI generic suggestion '{classified_industry_clean}'"
                            )
                        elif current_has_specific and ai_is_specific:
                            # Both are specific but different - merge them (add AI to current)
                            # Split current industry by comma to check if AI industry is already included
                            current_industry_parts = [part.strip().lower() for part in current_industry.split(",")]
                            if classified_industry_lower not in current_industry_parts:
                                # AI industry is not in current, add it
                                vacancy.industry = f"{current_industry}, {classified_industry_clean}"
                                updated_fields.append(f"industry={vacancy.industry} (merged: {current_industry} + {classified_industry_clean})")
                                logger.debug(f"Merged industries: '{current_industry}' + '{classified_industry_clean}' = '{vacancy.industry}'")
                            else:
                                # AI industry already in current, no change
                                pass
                        elif current_has_specific:
                            # Current has specific values, AI has different value (not specific) - protect current
                            protected_fields.append(f"industry={current_industry} (protected, AI suggested: {classified_industry_clean})")
                            logger.debug(
                                f"Preserved API industry '{current_industry}' over AI suggestion '{classified_industry_clean}'"
                            )
                        else:
                            # Current is generic or empty, AI has specific - update
                            vacancy.industry = classified_industry_clean
                            updated_fields.append(f"industry={vacancy.industry}")
                            logger.debug(f"Updated industry from AI: {vacancy.industry} (was: {current_industry})")
                    
                    # Required skills: Merge scraper tags with AI findings
                    # Use dictionary for deduplication to preserve original case (e.g., "Braze" instead of "braze")
                    scraper_skills = list(vacancy.required_skills or [])
                    ai_skills = classification_result.get("required_skills", [])
                    
                    # Normalize skills using get_api_text to handle dict/list formats
                    normalized_scraper = []
                    for skill in scraper_skills:
                        text = get_api_text(skill)
                        if text and text.strip():
                            normalized_scraper.append(text.strip())
                    
                    normalized_ai = []
                    for skill in ai_skills:
                        text = get_api_text(skill)
                        if text and text.strip():
                            normalized_ai.append(text.strip())
                    
                    # Merge using dictionary: key is lowercase skill, value is original string
                    # This preserves original case (e.g., "Braze" from scraper, not "braze")
                    seen = {}  # Dictionary: lowercase -> original string
                    merged_skills = []
                    
                    # Step 1: Add scraper skills first (preserve original case from API)
                    for skill in normalized_scraper:
                        skill_lower = skill.lower()
                        if skill_lower and skill_lower not in seen:
                            seen[skill_lower] = skill  # Store original case
                            merged_skills.append(skill)
                    
                    # Step 2: Add AI skills, avoiding case-insensitive duplicates
                    for ai_skill in normalized_ai:
                        ai_skill_lower = ai_skill.lower()
                        if ai_skill_lower and ai_skill_lower not in seen:
                            seen[ai_skill_lower] = ai_skill  # Store original case
                            merged_skills.append(ai_skill)
                    
                    # Update vacancy with merged skills
                    # STRICT: Do NOT merge AI skills into required_skills. 
                    # Use ONLY scraper skills to match the website's main page exactly.
                    # vacancy.required_skills = merged_skills
                    
                    # Ensure we stick to the scraper's normalized skills
                    vacancy.required_skills = normalized_scraper
                    
                    if normalized_ai:
                        # We still log what AI found for debugging, but we don't use it
                        logger.debug(
                            f"AI suggested skills: {normalized_ai}. "
                            f"IGNORING AI skills to match website UI. Keeping {len(normalized_scraper)} scraper skills."
                        )
                        # We don't update updated_fields since we aren't changing it
                    
                    # Location: Update if "Not specified" and AI provides location
                    ai_location = classification_result.get("location")
                    if vacancy.location == "Not specified" and ai_location and ai_location.strip():
                        location_text = get_api_text(ai_location)
                        if location_text and location_text.strip():
                            vacancy.location = location_text.strip()
                            updated_fields.append(f"location={vacancy.location} (updated from AI)")
                            logger.debug(f"Updated location from AI: {vacancy.location}")
                    
                    # Experience level: Only update if currently None or "Unknown"
                    classified_experience = classification_result.get("experience_level")
                    if classified_experience and classified_experience.strip():
                        current_experience = (vacancy.experience_level or "").strip()
                        if not current_experience or current_experience.lower() == "unknown":
                            vacancy.experience_level = classified_experience.strip()
                            updated_fields.append(f"experience_level={vacancy.experience_level}")
                            logger.debug(f"Updated experience_level from AI: {vacancy.experience_level}")
                        else:
                            protected_fields.append(f"experience_level={current_experience} (protected, AI suggested: {classified_experience})")
                            logger.debug(
                                f"Preserved existing experience_level '{current_experience}' over AI suggestion"
                            )
                    
                    # Remote option: Hybrid = False (not 100% remote), Remote = True (100% remote)
                    # If vacancy.is_hybrid is True, explicitly set remote_option = False
                    # If description or location contains "Hybrid", set remote_option = False
                    classified_remote = classification_result.get("remote_option")
                    if isinstance(classified_remote, bool):
                        # Check vacancy.is_hybrid first - Hybrid is NOT Remote
                        if hasattr(vacancy, 'is_hybrid') and vacancy.is_hybrid is True:
                            vacancy.remote_option = False
                            updated_fields.append(f"remote_option=False (is_hybrid=True)")
                            logger.debug(f"Set remote_option=False because vacancy.is_hybrid=True")
                        # Check for Hybrid in location/description
                        elif location_contains_hybrid or description_contains_hybrid:
                            vacancy.remote_option = False
                            updated_fields.append(f"remote_option=False (Hybrid detected in location/description)")
                            logger.debug(f"Set remote_option=False because Hybrid was detected")
                        elif vacancy.remote_option is True:
                            # Preserve True from API - don't let AI overwrite it to False
                            protected_fields.append(f"remote_option=True (protected, AI suggested: {classified_remote})")
                            logger.debug(f"Preserved remote_option=True over AI suggestion {classified_remote}")
                        elif location_contains_remote or description_contains_remote:
                            # Location/description indicates 100% remote (not hybrid), set to True
                            vacancy.remote_option = True
                            updated_fields.append(f"remote_option=True (100% Remote detected in location/description)")
                            logger.debug(f"Set remote_option=True based on 100% Remote (not Hybrid)")
                        else:
                            # Only update if current value is False or None
                            vacancy.remote_option = classified_remote
                            updated_fields.append(f"remote_option={vacancy.remote_option}")
                            logger.debug(f"Updated remote_option from AI: {vacancy.remote_option}")
                    
                    # Debug logging: Show exactly which fields were updated vs protected
                    if updated_fields or protected_fields:
                        log_parts = []
                        if updated_fields:
                            log_parts.append(f"Updated: {', '.join(updated_fields)}")
                        if protected_fields:
                            log_parts.append(f"Protected: {', '.join(protected_fields)}")
                        logger.info(f"AI Classification for {vacancy.title}: {' | '.join(log_parts)}")
                    else:
                        logger.info(f"AI Classification for {vacancy.title}: No changes (all fields protected or empty)")
                    
                    # Validation: Ensure category is never None - set default if still empty
                    if not vacancy.category:
                        title_lower = (vacancy.title or "").lower()
                        # Check for G&A related keywords
                        if any(keyword in title_lower for keyword in ["g&a", "general", "administrative", "admin", "operations", "ops"]):
                            if "g&a" in title_lower or "general" in title_lower or "administrative" in title_lower:
                                vacancy.category = "G&A"
                                logger.debug(f"Defaulted category to 'G&A' based on title: {vacancy.title}")
                            else:
                                vacancy.category = "Operations"
                                logger.debug(f"Defaulted category to 'Operations' based on title: {vacancy.title}")
                        else:
                            vacancy.category = "Other"  # Default fallback if no specific match
                            logger.debug(f"Defaulted category to 'Other' (no specific match in title)")
                    
                    # Validation: Ensure experience_level is never None - set default if still empty
                    if not vacancy.experience_level:
                        vacancy.experience_level = "Unknown"
                        logger.debug(f"Defaulted experience_level to 'Unknown'")
                        
                except Exception as e:
                    logger.warning(
                        f"Classification failed for {vacancy.title}: {str(e)}. "
                        f"Setting default values and continuing with processing."
                    )
                    # Set defaults even if classification fails
                    if not vacancy.category:
                        vacancy.category = "Other"
                        logger.debug(f"Set default category='Other' after classification failure")
                    if not vacancy.experience_level:
                        vacancy.experience_level = "Unknown"
                        logger.debug(f"Set default experience_level='Unknown' after classification failure")
                    # Continue processing even if classification fails
            else:
                logger.warning("ClassificationAgent not available, skipping AI classification")
                # Set defaults if classifier is not available
                if not vacancy.category:
                    vacancy.category = "Other"
                    logger.debug(f"Set default category='Other' (classifier unavailable)")
                if not vacancy.experience_level:
                    vacancy.experience_level = "Unknown"
                    logger.debug(f"Set default experience_level='Unknown' (classifier unavailable)")
            
            # CRITICAL: Ensure category and experience_level are NEVER None before proceeding
            # This guarantees the vacancy object is always in a valid state, even if embedding fails
            if not vacancy.category:
                vacancy.category = "Other"
                logger.warning(f"Category was None, set to default 'Other' for {vacancy.title}")
            if not vacancy.experience_level:
                vacancy.experience_level = "Unknown"
                logger.warning(f"Experience_level was None, set to default 'Unknown' for {vacancy.title}")
            
            # Generate search text for embedding
            search_text = generate_search_text(vacancy)
            logger.info(f"Generated search text for vacancy: {vacancy.title}")
            
            # Generate embedding - this may fail, but vacancy is already updated with AI classification
            embedding = None
            try:
                embedding = await get_embedding(search_text, self.embedding_service_url)
                logger.info(f"Generated embedding for vacancy: {vacancy.title} (dim={len(embedding)})")
            except Exception as e:
                logger.error(
                    f"Failed to generate embedding for {vacancy.title}: {str(e)}. "
                    f"Vacancy object has been updated with AI classification but will not be uploaded to Pinecone."
                )
                # Re-raise to be caught by outer exception handler
                # The vacancy object is already updated with AI classification at this point
                raise
            
            # Generate vacancy ID with URL hash to prevent collisions
            vacancy_id = generate_vacancy_id(
                vacancy.company_name, 
                vacancy.title, 
                vacancy.description_url
            )
            
            # Helper function to safely convert enum to string
            def enum_to_string(value, default_value: str = ""):
                """Convert enum to string, handling missing values with defaults."""
                try:
                    if value is None:
                        return default_value
                    if hasattr(value, 'value'):
                        return value.value
                    return str(value) if value else default_value
                except Exception as e:
                    logger.warning(f"Error converting enum to string: {e}, using default: {default_value}")
                    return default_value
            
            # Helper function to convert value to string safely
            def to_string(value, default_value: str = ""):
                """Convert value to string, handling None and empty values."""
                if value is None:
                    return default_value
                return str(value) if value else default_value
            
            # Prepare metadata with all fields, using defaults for missing values
            # All values should be strings (except remote_option which is bool) to match search filters
            # required_skills is always a List[str]
            # NOTE: These values reflect the "protected" API data that takes precedence over AI:
            # - industry: Preserves specific API values (e.g., "American Dynamism") over generic AI "Technology"
            # - remote_option: Preserves True from API, won't be overwritten by AI False
            # - required_skills: Case-insensitive deduplication, prioritizes exact API tag strings
            # - category and experience_level: Guaranteed to never be None (defaults: "Other", "Unknown")
            metadata = {
                "title": vacancy.title or "",
                "company_name": vacancy.company_name or "",
                "company_stage": to_string(vacancy.company_stage, "Growth"),
                "location": vacancy.location or "",
                "industry": to_string(vacancy.industry, ""),  # Protected: API values preserved
                "category": to_string(vacancy.category, "Other"),  # Guaranteed to never be None
                "experience_level": to_string(vacancy.experience_level, "Unknown"),  # Guaranteed to never be None
                "remote_option": vacancy.remote_option if vacancy.remote_option is not None else False,  # Protected: True from API preserved
                "is_hybrid": vacancy.is_hybrid if vacancy.is_hybrid is not None else False,
                "description_url": vacancy.description_url or "",
                "required_skills": vacancy.required_skills if vacancy.required_skills else [],  # Protected: API tags prioritized
            }
            
            # Add optional fields if present
            if vacancy.salary_range:
                metadata["salary_range"] = vacancy.salary_range
            
            if vacancy.employee_count:
                metadata["employee_count"] = vacancy.employee_count
            
            # Filter out None values as Pinecone doesn't accept null
            metadata = {k: v for k, v in metadata.items() if v is not None}
            
            # Prepare vector for Pinecone
            vector = {
                "id": vacancy_id,
                "values": embedding,
                "metadata": metadata,
            }
            
            # Upsert to Pinecone
            logger.info(f"Upserting vacancy to Pinecone: {vacancy_id} (namespace={namespace})")
            self.vector_store.upsert(vectors=[vector], namespace=namespace)
            logger.info(f"Successfully processed vacancy: {vacancy_id}")
            
            return True
            
        except httpx.HTTPError as e:
            logger.error(
                f"Failed to process vacancy {vacancy.title}: embedding service error - {str(e)}. "
                f"Note: Vacancy object has been updated with AI classification (if available)."
            )
            # Ensure defaults are set even if embedding fails
            if not vacancy.category:
                vacancy.category = "Other"
            if not vacancy.experience_level:
                vacancy.experience_level = "Unknown"
            return False
        except Exception as e:
            logger.error(
                f"Failed to process vacancy {vacancy.title}: {str(e)} (error_type={type(e).__name__}). "
                f"Note: Vacancy object has been updated with AI classification (if available).",
                exc_info=True,
            )
            # Ensure defaults are set even if processing fails
            if not vacancy.category:
                vacancy.category = "Other"
            if not vacancy.experience_level:
                vacancy.experience_level = "Unknown"
            return False

