"""
Scraper engine for orchestrating multiple scraper providers.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Union

from src.services.scraper.core.base import BaseScraper
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


class ScraperEngine:
    """Engine for running multiple scraper providers sequentially."""

    def __init__(self, scrapers: List[BaseScraper]) -> None:
        """
        Initialize the scraper engine with a list of scraper providers.

        Args:
            scrapers: List of BaseScraper instances to run
        """
        self.scrapers = scrapers

    async def run_all(self) -> List[Vacancy]:
        """
        Run all scraper providers sequentially and collect results.
        
        If a scraper fails, the error is logged and the engine continues
        with the next scraper instead of crashing.

        Returns:
            List of all Vacancy objects collected from all scrapers
        """
        all_vacancies: List[Vacancy] = []

        for scraper in self.scrapers:
            try:
                scraper_type = type(scraper).__name__
                logger.info(f"Scraper started: {scraper_type}")
                vacancies = await scraper.run()
                all_vacancies.extend(vacancies)
                logger.info(f"Scraper completed: {scraper_type} (found {len(vacancies)} vacancies)")
            except Exception as e:
                scraper_type = type(scraper).__name__
                logger.error(
                    f"Scraper failed: {scraper_type} - {str(e)} (error_type={type(e).__name__})",
                    exc_info=True,
                )
                # Continue with next scraper instead of crashing

        logger.info(f"All scrapers completed: total vacancies collected = {len(all_vacancies)}")
        return all_vacancies

    def save_to_dump(self, vacancies: List[Vacancy], output_path: Optional[Union[Path, str]] = None) -> None:
        """
        Save vacancies to vacancies_dump.json file.
        
        Uses the same logic as src/scripts/ingest_a16z.py:
        - Converts Vacancy objects to dicts using .dict()
        - Saves as JSON with indent=2 and ensure_ascii=False
        
        Args:
            vacancies: List of Vacancy objects to save
            output_path: Optional path to output file. If None, uses project root / vacancies_dump.json
        """
        if output_path is None:
            # Default to project root / vacancies_dump.json
            project_root = Path(__file__).parent.parent.parent.parent
            output_path = project_root / "vacancies_dump.json"
        else:
            output_path = Path(output_path)

        # Convert to dict for JSON serialization
        vacancies_data = [vacancy.dict() for vacancy in vacancies]

        # Save to JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vacancies_data, f, indent=2, ensure_ascii=False)

