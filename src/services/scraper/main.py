"""
Main script for running the scraper engine.
"""

from dotenv import load_dotenv
load_dotenv()

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Set, Dict, Optional, Any
from src.services.scraper.filter_service import VacancyFilterService, FilterConfig

from src.services.scraper.core.browser import BrowserManager
from src.services.scraper.core.engine import ScraperEngine
from src.services.scraper.core.ingest_manager import IngestManager
from src.services.scraper.providers.a16z import A16ZScraper
def load_existing_vacancies(dump_path: Path) -> List[Dict[str, Any]]:
    """Load existing vacancies from the dump file."""
    if not dump_path.exists():
        return []
    
    existing_vacancies = []
    try:
        with open(dump_path, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        existing_vacancies.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        logger.error(f"Error loading existing vacancies: {e}")
        
    return existing_vacancies

def get_existing_urls(vacancies: List[Dict[str, Any]]) -> Set[str]:
    """Extract a set of existing URLs from vacancies."""
    urls = set()
    for v in vacancies:
        if isinstance(v, str):
            # If v is a string (e.g. JSON string), try to parse it
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                continue
                
        if isinstance(v, dict):
            url = v.get("description_url")
            if url:
                urls.add(url)
    return urls

# Max vacancies to process per run
MAX_PROCESS_PER_RUN = 50

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
    """Process a single vacancy link."""
    async with semaphore:
        try:
            logger.info(f"Processing: {link}")
            
            # Fetch details
            raw_html = await a16z_scraper.fetch_job_details(link)
            if not raw_html:
                logger.warning(f"Failed to fetch HTML for {link}")
                return
                
            # Parse
            parsed_data = await a16z_scraper.parse_job_page(raw_html, link)
            if not parsed_data:
                logger.warning(f"Failed to parse data for {link}")
                return
                
            # Process and Ingest
            success = await ingest_manager.process_new_vacancy(parsed_data)
            
            if success:
                # Save to dump file safely with lock
                async with lock:
                    with open(dump_path, "a") as f:
                        f.write(json.dumps(parsed_data, default=str) + "\n")
                    
                    stats["new_count"] += 1
                    stats["processed_count"] += 1
                    existing_urls.add(link)
                    logger.info(f"Successfully processed and saved: {parsed_data.get('title', 'Unknown')}")
            else:
                 async with lock:
                    stats["skipped_count"] += 1
                    
        except Exception as e:
            logger.error(f"Error processing {link}: {str(e)}")


logger = logging.getLogger(__name__)

# ... (imports)

async def main(filter_config: Optional[Dict[str, Any]] = None) -> None:
    """
    Main async function to run the scraper.
    
    Args:
        filter_config: Optional dictionary with filter configuration
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent
    dump_path = project_root / "vacancies_dump.json"
    
    logger.info("Starting scraper main process")
    
    # Initialize Filter Service
    config = FilterConfig(**(filter_config or {}))
    filter_service = VacancyFilterService(config)
    logger.info(f"Initialized FilterService with config: {config.dict()}")
    
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
        
        # Initialize IngestManager with remote_only setting
        logger.info(f"Initializing IngestManager (remote_only={filter_service.config.remote_only})")
        ingest_manager = IngestManager(remote_only=filter_service.config.remote_only)
        
        try:
            # Load existing vacancies to check for duplicates
            existing_vacancies = load_existing_vacancies(dump_path)
            existing_urls = get_existing_urls(existing_vacancies)
            logger.info(f"Found {len(existing_urls)} existing vacancy URLs")
            
            # Get all links first
            logger.info("Fetching all job links")
            all_links = await a16z_scraper.fetch_all_links()
            logger.info(f"Found {len(all_links)} job links total")
            
            # PRE-FILTER: Filter links using FilterService based on metadata available from list page
            links_to_process = []
            skipped_by_filter = 0
            
            for link in all_links:
                # Get metadata from scraper cache
                metadata = a16z_scraper._job_metadata.get(link, {})
                if filter_service.should_process(metadata):
                    links_to_process.append(link)
                else:
                    skipped_by_filter += 1
            
            logger.info(f"Filter Service: {skipped_by_filter} links skipped, {len(links_to_process)} links passed")

            # Filter out already processed links (check vacancies_dump.json)
            # This prevents re-processing if script was interrupted and restarted
            # Only check for existing URLs if they are not in the filtered list (to avoid double counting)
            new_links_to_process = [link for link in links_to_process if link not in existing_urls]
            
            logger.info(f"Found {len(new_links_to_process)} new vacancies to process (skipping {len(all_links) - len(links_to_process)} filtered, {len(links_to_process) - len(new_links_to_process)} already processed)")
            
            links_to_process = new_links_to_process
            
            # Limit to MAX_PROCESS_PER_RUN to process in batches
            if len(links_to_process) > MAX_PROCESS_PER_RUN:
                logger.info(f"Limiting processing to {MAX_PROCESS_PER_RUN} vacancies per run (found {len(links_to_process)} total)")
                links_to_process = links_to_process[:MAX_PROCESS_PER_RUN]
            
            logger.info(f"Will process {len(links_to_process)} vacancies in this run")
            
            # Initialize semaphore to limit concurrent browser pages (3 concurrent)
            # Reduced to 3 to reduce load on embedding service and prevent timeouts
            semaphore = asyncio.Semaphore(3)
            
            # Initialize stats dictionary and lock for thread-safe access
            stats = {
                "new_count": 0,
                "processed_count": 0,
                "skipped_count": (len(all_links) - len(links_to_process)), # Total skipped
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

