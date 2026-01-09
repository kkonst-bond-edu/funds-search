"""
Main script for running the scraper engine.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Set, Dict

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


async def process_vacancy(
    link: str,
    semaphore: asyncio.Semaphore,
    a16z_scraper: A16ZScraper,
    ingest_manager: IngestManager,
    dump_path: Path,
    existing_urls: Set[str],
    stats: Dict[str, int],
    lock: asyncio.Lock,
) -> None:
    """
    Process a single vacancy: extraction, saving to dump, and uploading to Pinecone.
    
    Args:
        link: URL of the vacancy to process
        semaphore: Semaphore to limit concurrent browser pages
        a16z_scraper: A16Z scraper instance
        ingest_manager: Ingest manager instance
        dump_path: Path to vacancies_dump.json
        existing_urls: Set of already processed URLs (thread-safe access via lock)
        stats: Dictionary with counters (new_count, processed_count, skipped_count)
        lock: Lock for thread-safe access to shared resources
    """
    async with semaphore:
        try:
            # Set timeout for the entire vacancy processing (180 seconds)
            # Increased to 180s to handle slow-loading job boards and embedding service delays
            await asyncio.wait_for(
                _process_vacancy_internal(
                    link, a16z_scraper, ingest_manager, dump_path, existing_urls, stats, lock
                ),
                timeout=180.0
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout processing vacancy (180s exceeded): {link}")
        except Exception as e:
            logger.error(
                f"Failed to process vacancy {link}: {str(e)}",
                exc_info=True,
            )


async def _process_vacancy_internal(
    link: str,
    a16z_scraper: A16ZScraper,
    ingest_manager: IngestManager,
    dump_path: Path,
    existing_urls: Set[str],
    stats: Dict[str, int],
    lock: asyncio.Lock,
) -> None:
    """
    Internal function to process a vacancy (without semaphore, called within timeout).
    
    Args:
        link: URL of the vacancy to process
        a16z_scraper: A16Z scraper instance
        ingest_manager: Ingest manager instance
        dump_path: Path to vacancies_dump.json
        existing_urls: Set of already processed URLs
        stats: Dictionary with counters
        lock: Lock for thread-safe access
    """
    # Check if this vacancy is already processed (thread-safe)
    async with lock:
        if link in existing_urls:
            stats["skipped_count"] += 1
            logger.info(f"Skipping already processed vacancy: {link}")
            return
    
    # Extract vacancy details
    try:
        vacancy = await a16z_scraper.extract_details(link)
    except Exception as e:
        logger.error(
            f"Failed to extract vacancy from {link}: {str(e)}",
            exc_info=True,
        )
        return
    
    # Check for data quality: skip if both title and company contain 'Unknown'
    if "Unknown" in vacancy.title and "Unknown" in vacancy.company_name:
        logger.warning(
            f"Skipping vacancy with poor data quality - both title and company contain 'Unknown'. "
            f"Title: '{vacancy.title}', Company: '{vacancy.company_name}'. "
            f"URL: {link}"
        )
        async with lock:
            stats["skipped_count"] += 1
        return
    
    # This is a new vacancy
    async with lock:
        stats["new_count"] += 1
        logger.info(f"Processing new vacancy: {vacancy.title} at {vacancy.company_name}")
    
    # FIRST: Process vacancy through ingest_manager (AI classification + Pinecone upload)
    # This updates the vacancy object with AI-classified fields (category, industry, experience_level, etc.)
    try:
        success = await ingest_manager.process_new_vacancy(vacancy)
        if success:
            async with lock:
                stats["processed_count"] += 1
            logger.info(f"Successfully processed and uploaded vacancy: {vacancy.title}")
        else:
            logger.warning(f"Failed to process vacancy: {vacancy.title}")
    except Exception as e:
        logger.error(
            f"Failed to process vacancy in Pinecone: {vacancy.title} - {str(e)}",
            exc_info=True,
        )
    
    # AFTER AI classification: Save the updated vacancy to vacancies_dump.json
    # The vacancy object now includes AI-classified fields (category, industry, experience_level, required_skills, etc.)
    try:
        # Use lock to ensure thread-safe file writing
        async with lock:
            append_vacancy_to_dump(vacancy, dump_path)
            # Update existing_urls to prevent duplicates in the same run
            existing_urls.add(vacancy.description_url)
    except Exception as e:
        logger.error(
            f"Failed to save vacancy to dump: {vacancy.title} - {str(e)}",
            exc_info=True,
        )
    
    # Add 1-second delay after processing each vacancy to reduce CPU and memory pressure
    # This helps prevent timeouts and resource exhaustion on local machines during scraping
    await asyncio.sleep(1)


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
            
            # Get all links first
            logger.info("Fetching all job links")
            all_links = await a16z_scraper.fetch_all_links()
            logger.info(f"Found {len(all_links)} job links to process")
            
            # Limit to first 50 jobs for testing
            all_links = all_links[:]
            logger.info(f"Processing first {len(all_links)} jobs for testing")
            
            # Filter out already processed links
            links_to_process = [link for link in all_links if link not in existing_urls]
            logger.info(f"Processing {len(links_to_process)} new vacancies (skipping {len(all_links) - len(links_to_process)} already processed)")
            
            # Initialize semaphore to limit concurrent browser pages (3 concurrent)
            # Reduced to 3 to reduce load on embedding service and prevent timeouts
            semaphore = asyncio.Semaphore(3)
            
            # Initialize stats dictionary and lock for thread-safe access
            stats = {
                "new_count": 0,
                "processed_count": 0,
                "skipped_count": len(all_links) - len(links_to_process),
            }
            lock = asyncio.Lock()
            
            # Process links in batches of 10
            batch_size = 10
            for i in range(0, len(links_to_process), batch_size):
                batch = links_to_process[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(links_to_process) + batch_size - 1) // batch_size
                
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} vacancies)")
                
                # Process batch concurrently using asyncio.gather
                tasks = [
                    process_vacancy(
                        link=link,
                        semaphore=semaphore,
                        a16z_scraper=a16z_scraper,
                        ingest_manager=ingest_manager,
                        dump_path=dump_path,
                        existing_urls=existing_urls,
                        stats=stats,
                        lock=lock,
                    )
                    for link in batch
                ]
                
                # Wait for all tasks in the batch to complete
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Add 1-second delay after batch completion to reduce CPU and memory pressure
                # This gives the browser and local machine a breather between batches
                await asyncio.sleep(1)
                
                logger.info(
                    f"Batch {batch_num}/{total_batches} completed. "
                    f"Progress: {stats['new_count']} new, {stats['processed_count']} uploaded, {stats['skipped_count']} skipped"
                )
            
            logger.info(
                f"Scraping completed: {stats['new_count']} new vacancies found, "
                f"{stats['processed_count']} uploaded to Pinecone, {stats['skipped_count']} skipped"
            )
            
        except Exception as e:
            logger.error(f"Scraper main process failed: {str(e)}", exc_info=True)
            raise


if __name__ == "__main__":
    asyncio.run(main())

