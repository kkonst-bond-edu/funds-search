"""
Browser manager for Playwright-based web scraping.
"""

import asyncio
import logging
import random
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright

logger = logging.getLogger(__name__)

class BrowserManager:
    """Manages Playwright browser lifecycle and provides page context."""

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self) -> "BrowserManager":
        """Добавлено: Вход в асинхронный контекстный менеджер."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Добавлено: Выход из асинхронного контекстного менеджера."""
        await self.close()

    async def start(self) -> None:
        if self._playwright is None:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=self.headless)
            self._context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )

    async def close(self) -> None:
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._playwright = self._browser = self._context = None

    @asynccontextmanager
    async def get_page(self) -> AsyncGenerator[Page, None]:
        if self._context is None:
            await self.start()
        page = await self._context.new_page()
        try:
            yield page
        finally:
            await page.close()

    async def wait_for_job_content(self, page: Page, timeout: int = 20000) -> bool:
        """Wait for key job content elements to appear."""
        combined_selector = (
            ".posting-headline h2, .app-title, .job-description, "
            ".job-body, #content, .posting-description, "
            "[data-automation-id='jobPostingDescription'], .description"
        )
        try:
            await page.wait_for_selector(combined_selector, timeout=timeout, state="attached")
            return True
        except Exception as e:
            logger.warning(f"Timeout waiting for selectors on {page.url}: {str(e)}")
            return False

    async def scroll_for_content(self, page: Page, wait_seconds: float = 1.5) -> None:
        """Scroll to the bottom to trigger lazy loading."""
        previous_height = 0
        current_height = await page.evaluate("document.body.scrollHeight")
        scroll_attempts = 0
        while current_height != previous_height and scroll_attempts < 10:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(wait_seconds)
            previous_height = current_height
            current_height = await page.evaluate("document.body.scrollHeight")
            scroll_attempts += 1