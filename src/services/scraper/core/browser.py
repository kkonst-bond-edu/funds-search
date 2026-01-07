"""
Browser manager for Playwright-based web scraping.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

# Configure basic logger for console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class BrowserManager:
    """Manages Playwright browser lifecycle and provides page context."""

    def __init__(self, headless: bool = True) -> None:
        """
        Initialize the browser manager.

        Args:
            headless: Whether to run browser in headless mode
        """
        self.headless = headless
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "BrowserManager":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit - cleanup resources."""
        await self.close()

    async def start(self) -> None:
        """Start the Playwright browser instance."""
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            
            # Create context with standard Chrome on Windows User-Agent
            self._context = await self._browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            )

    async def close(self) -> None:
        """Close the browser and cleanup resources."""
        if self._context:
            await self._context.close()
            self._context = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator[Page, None]:
        """
        Get a new page from the browser context.
        
        Yields:
            Playwright Page object
            
        Example:
            async with browser_manager.get_page() as page:
                await browser_manager.goto_with_retry(page, "https://example.com")
        """
        if self._context is None:
            await self.start()
        
        page = await self._context.new_page()
        try:
            yield page
        finally:
            await page.close()

    async def goto_with_retry(
        self, page: Page, url: str, max_retries: int = 3, retry_delay: float = 1.0, **kwargs
    ) -> None:
        """
        Navigate to a URL with retry logic.
        
        If page.goto fails, it will retry up to max_retries times with a delay
        between attempts.
        
        Args:
            page: Playwright Page object
            url: URL to navigate to
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 1.0)
            **kwargs: Additional arguments to pass to page.goto()
        """
        last_exception = None
        
        for attempt in range(1, max_retries + 1):
            try:
                await page.goto(url, **kwargs)
                logger.info(f"Page navigation successful: {url} (attempt {attempt})")
                return
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Page navigation failed: {url} (attempt {attempt}/{max_retries}) - {str(e)}"
                )
                
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"Page navigation exhausted all retries: {url} (max_retries={max_retries}) - {str(e)}"
                    )
        
        # If all retries failed, raise the last exception
        if last_exception:
            raise last_exception

    async def smart_scroll(self, page: Page, wait_seconds: float = 1.5) -> None:
        """
        Scroll to the bottom of the page repeatedly until page height stops increasing.
        
        This method scrolls down incrementally and waits for content to load,
        continuing until the page height stabilizes.
        
        Args:
            page: Playwright Page object to scroll
            wait_seconds: Seconds to wait between scrolls (default: 1.5)
        """
        previous_height = 0
        current_height = await page.evaluate("document.body.scrollHeight")
        scroll_attempts = 0
        max_attempts = 100  # Safety limit to prevent infinite loops
        
        while current_height != previous_height and scroll_attempts < max_attempts:
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            # Wait for content to load
            await asyncio.sleep(wait_seconds)
            
            # Get new height
            previous_height = current_height
            current_height = await page.evaluate("document.body.scrollHeight")
            scroll_attempts += 1

