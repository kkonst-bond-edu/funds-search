"""
Browser manager for Playwright-based web scraping.
"""

import asyncio
import logging
import random
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

    async def wait_for_job_content(self, page: Page, timeout: int = 20000) -> bool:
        """
        Wait for job-specific content selectors to appear on the page.
        
        Uses a combined CSS selector to wait for ANY of the common job board selectors.
        This is faster than sequential waiting and ensures dynamic JavaScript content
        (like Ashby or Workday) has loaded.
        
        Args:
            page: Playwright Page object
            timeout: Maximum time to wait in milliseconds (default: 10000 = 10 seconds)
            
        Returns:
            True if any selector was found, False otherwise (never raises exception)
        """
        # Combined CSS selector - matches ANY of these selectors
        # Platform-specific selectors for better coverage
        combined_selector = (
            ".posting-headline h2, "  # Lever - job title
            ".app-title, "  # Greenhouse - job title
            ".job-description, "  # Ashby - primary
            ".job-body, "  # Ashby - alternative
            "#content, "  # Greenhouse - main content
            ".main-fields, "  # Greenhouse - job fields
            "#job-body, "  # Greenhouse - job body
            ".section.job-description, "  # Lever - job description section
            ".posting-description, "  # Lever - posting description
            "[data-automation-id='jobPostingDescription'], "  # Workday - automation ID
            "[data-automation-id=\"jobPostingDescription\"], "  # Workday - with double quotes
            ".description, "  # Generic description selector
            ".job-info"  # Generic job info selector
        )
        
        try:
            await page.wait_for_selector(combined_selector, timeout=timeout, state="attached")
            logger.debug(f"Found job content selector (combined match)")
            return True
        except (asyncio.TimeoutError, Exception) as e:
            # Don't throw error - just log warning and proceed with available HTML
            logger.warning(
                f"No job content selectors found within {timeout}ms timeout. "
                f"Proceeding with available HTML content. Error: {str(e)}"
            )
            return False

    async def goto_with_retry(
        self, page: Page, url: str, max_retries: int = 3, retry_delay: float = 1.0, **kwargs
    ) -> None:
        """
        Navigate to a URL with retry logic and wait for job content.
        
        If page.goto fails, it will retry up to max_retries times with a delay
        between attempts. After successful navigation, waits for job-specific
        content selectors and adds a random delay for JavaScript hydration.
        
        Args:
            page: Playwright Page object
            url: URL to navigate to
            max_retries: Maximum number of retry attempts (default: 3)
            retry_delay: Delay in seconds between retries (default: 1.0)
            **kwargs: Additional arguments to pass to page.goto()
                     Default timeout is 60000ms (60 seconds) if not specified
        """
        # Set default timeout to 60 seconds if not provided
        if "timeout" not in kwargs:
            kwargs["timeout"] = 60000
        
        last_exception = None
        
        for attempt in range(1, max_retries + 1):
            try:
                await page.goto(url, **kwargs)
                logger.info(f"Page navigation successful: {url} (attempt {attempt})")
                
                # Wait for job-specific content selectors (non-blocking - continues even if not found)
                selector_found = await self.wait_for_job_content(page, timeout=10000)
                
                # Add random delay (1-2 seconds) AFTER selector is found for JavaScript hydration
                # This ensures any remaining JavaScript has time to render dynamic text
                if selector_found:
                    hydration_delay = random.uniform(1.0, 2.0)
                    await asyncio.sleep(hydration_delay)
                    logger.debug(
                        f"Completed JavaScript hydration delay: {hydration_delay:.2f}s "
                        f"(selector found: {selector_found})"
                    )
                
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
        continuing until the page height stabilizes. Optimized to break early
        if height doesn't change for more than 2 seconds.
        
        Args:
            page: Playwright Page object to scroll
            wait_seconds: Seconds to wait between scrolls (default: 1.5)
        """
        previous_height = 0
        current_height = await page.evaluate("document.body.scrollHeight")
        scroll_attempts = 0
        max_attempts = 100  # Safety limit to prevent infinite loops
        stable_count = 0  # Count how many times height stayed the same
        max_stable_checks = 2  # Break if height unchanged for 2+ checks (2+ seconds)
        
        while current_height != previous_height and scroll_attempts < max_attempts:
            # Scroll to bottom
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            
            # Wait for content to load
            await asyncio.sleep(wait_seconds)
            
            # Get new height
            previous_height = current_height
            current_height = await page.evaluate("document.body.scrollHeight")
            scroll_attempts += 1
            
            # Check if height is stable (not changing)
            if current_height == previous_height:
                stable_count += 1
                # If height hasn't changed for 2+ checks, break early
                if stable_count >= max_stable_checks:
                    logger.debug(f"Page height stable for {stable_count} checks, breaking scroll loop")
                    break
            else:
                # Reset stable count if height changed
                stable_count = 0

