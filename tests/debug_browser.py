"""
Temporary test script for BrowserManager functionality.
"""

import asyncio
from pathlib import Path

from src.services.scraper.core.browser import BrowserManager


async def main():
    """Test BrowserManager with a16z jobs page."""
    url = "https://jobs.a16z.com/jobs"
    
    async with BrowserManager(headless=False) as browser_manager:
        async with browser_manager.get_page() as page:
            # Navigate to the page
            print(f"Navigating to {url}...")
            await page.goto(url, wait_until="networkidle")
            
            # Call smart_scroll method
            print("Scrolling to load all content...")
            await browser_manager.smart_scroll(page)
            print("Scrolling completed.")
            
            # Count all <a> tags on the page
            link_count = await page.evaluate("document.querySelectorAll('a').length")
            print(f"Total <a> tags found: {link_count}")
            
            # Take a screenshot of the bottom of the page
            # First, scroll to bottom to ensure we capture the bottom content
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(0.5)  # Brief wait for any final rendering
            
            # Get the viewport height to scroll to show bottom portion
            viewport_height = page.viewport_size["height"]
            page_height = await page.evaluate("document.body.scrollHeight")
            
            # Scroll to show the bottom portion of the page
            scroll_position = max(0, page_height - viewport_height)
            await page.evaluate(f"window.scrollTo(0, {scroll_position})")
            await asyncio.sleep(0.5)
            
            # Take screenshot
            screenshot_path = Path(__file__).parent / "debug_scroll.png"
            await page.screenshot(path=str(screenshot_path), full_page=False)
            print(f"Screenshot saved to: {screenshot_path}")


if __name__ == "__main__":
    asyncio.run(main())

