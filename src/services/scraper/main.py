"""
Main script for running the scraper engine.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Set

from src.services.scraper.core.browser import BrowserManager
from src.services.scraper.core.engine import ScraperEngine
from src.services.scraper.core.ingest_manager import IngestManager
from src.services.scraper.providers.a16z import A16ZScraper
from src.schemas.vacancy import Vacancy

# Configure basic logger for console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def load_existing_vacancies(dump_path: Path) -> List[dict]:
    """
    Load existing vacancies from vacancies_dump.json.
    
    Args:
        dump_path: Path to vacancies_dump.json file
        
    Returns:
        List of vacancy dictionaries, or empty list if file doesn't exist
    """
    if not dump_path.exists():
        logger.info(f"Vacancies dump file not found: {dump_path}, starting with empty list")
        return []
    
    try:
        with open(dump_path, "r", encoding="utf-8") as f:
            vacancies = json.load(f)
        if not isinstance(vacancies, list):
            logger.warning(f"Invalid format in {dump_path}, expected list, got {type(vacancies).__name__}")
            return []
        logger.info(f"Loaded {len(vacancies)} existing vacancies from {dump_path}")
        return vacancies
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from {dump_path}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error loading vacancies from {dump_path}: {str(e)}")
        return []


def get_existing_urls(vacancies: List[dict]) -> Set[str]:
    """
    Extract description_urls from existing vacancies to check for duplicates.
    
    Args:
        vacancies: List of vacancy dictionaries
        
    Returns:
        Set of description URLs
    """
    urls = set()
    for vacancy in vacancies:
        url = vacancy.get("description_url")
        if url:
            urls.add(url)
    return urls


def append_vacancy_to_dump(vacancy: Vacancy, dump_path: Path) -> None:
    """
    Append a single vacancy to vacancies_dump.json.
    
    Args:
        vacancy: Vacancy object to append
        dump_path: Path to vacancies_dump.json file
    """
    # Load existing vacancies
    existing_vacancies = load_existing_vacancies(dump_path)
    
    # Convert new vacancy to dict
    vacancy_dict = vacancy.dict()
    
    # Append to list
    existing_vacancies.append(vacancy_dict)
    
    # Save back to file
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(existing_vacancies, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Appended vacancy to {dump_path}: {vacancy.title} at {vacancy.company_name}")


async def main() -> None:
    """Main async function to run the scraper."""
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    dump_path = project_root / "vacancies_dump.json"
    
    logger.info("Starting scraper main process")
    
    # Initialize BrowserManager and use as async context manager
    logger.info("Initializing BrowserManager")
    async with BrowserManager(headless=True) as browser_manager:
        # Initialize A16ZScraper
        # Create a simple config object
        class ScraperConfig:
            base_url = "https://jobs.a16z.com/jobs"
        
        config = ScraperConfig()
        logger.info("Initializing A16ZScraper")
        a16z_scraper = A16ZScraper(browser_manager, config)
        
        # Initialize ScraperEngine with the a16z scraper
        logger.info("Initializing ScraperEngine")
        engine = ScraperEngine(scrapers=[a16z_scraper])
        
        # Initialize IngestManager
        logger.info("Initializing IngestManager")
        ingest_manager = IngestManager()
        
        try:
            # Load existing vacancies to check for duplicates
            existing_vacancies = load_existing_vacancies(dump_path)
            existing_urls = get_existing_urls(existing_vacancies)
            logger.info(f"Found {len(existing_urls)} existing vacancy URLs")
            
            # Run engine to get all vacancies
            logger.info("Running scraper engine to fetch vacancies")
            all_vacancies = await engine.run_all()
            logger.info(f"Scraper engine completed: found {len(all_vacancies)} vacancies")
            
            # Process each NEW vacancy
            new_count = 0
            processed_count = 0
            skipped_count = 0
            
            for vacancy in all_vacancies:
                # Check if this vacancy is already processed
                if vacancy.description_url in existing_urls:
                    skipped_count += 1
                    logger.info(f"Skipping already processed vacancy: {vacancy.description_url}")
                    continue
                
                # This is a new vacancy
                new_count += 1
                logger.info(f"Processing new vacancy: {vacancy.title} at {vacancy.company_name}")
                
                # Save to vacancies_dump.json
                try:
                    append_vacancy_to_dump(vacancy, dump_path)
                    # Update existing_urls to prevent duplicates in the same run
                    existing_urls.add(vacancy.description_url)
                except Exception as e:
                    logger.error(
                        f"Failed to save vacancy to dump: {vacancy.title} - {str(e)}",
                        exc_info=True,
                    )
                    continue
                
                # Upload to Pinecone
                try:
                    success = await ingest_manager.process_new_vacancy(vacancy)
                    if success:
                        processed_count += 1
                        logger.info(f"Successfully processed and uploaded vacancy: {vacancy.title}")
                    else:
                        logger.warning(f"Failed to process vacancy: {vacancy.title}")
                except Exception as e:
                    logger.error(
                        f"Failed to process vacancy in Pinecone: {vacancy.title} - {str(e)}",
                        exc_info=True,
                    )
            
            logger.info(
                f"Scraping completed: {new_count} new vacancies found, "
                f"{processed_count} uploaded to Pinecone, {skipped_count} skipped"
            )
            
        except Exception as e:
            logger.error(f"Scraper main process failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    asyncio.run(main())

