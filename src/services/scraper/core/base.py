"""
Base scraper abstract class for web scraping operations.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Any

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


class BaseScraper(ABC):
    """Abstract base class for web scrapers."""

    def __init__(self, browser_manager: Any, config: Any) -> None:
        """
        Initialize the scraper with browser manager and configuration.

        Args:
            browser_manager: Browser manager instance for web automation
            config: Configuration object with scraper settings
        """
        self.browser_manager = browser_manager
        self.config = config

    @abstractmethod
    async def fetch_all_links(self) -> List[str]:
        """
        Fetch all vacancy links from the source.

        Returns:
            List of URLs to vacancy detail pages
        """
        pass

    @abstractmethod
    async def extract_details(self, url: str) -> Vacancy:
        """
        Extract vacancy details from a single URL.

        Args:
            url: URL of the vacancy detail page

        Returns:
            Vacancy object with extracted details
        """
        pass

    async def run(self) -> List[Vacancy]:
        """
        Execute the full scraping workflow.

        This method:
        1. Fetches all vacancy links
        2. Extracts details from each link
        
        If extraction fails for a single vacancy, the error is logged
        and the scraper continues with the next vacancy.

        Returns:
            List of Vacancy objects
        """
        links = await self.fetch_all_links()
        vacancies = []
        
        for link in links:
            try:
                vacancy = await self.extract_details(link)
                vacancies.append(vacancy)
            except Exception as e:
                logger.error(
                    f"Vacancy extraction failed: {link} - {str(e)} (error_type={type(e).__name__})",
                    exc_info=True,
                )
                # Continue with next vacancy instead of crashing
        
        return vacancies

