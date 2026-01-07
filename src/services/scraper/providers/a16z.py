"""
A16Z scraper provider implementation.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Any, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from src.schemas.vacancy import Vacancy, CompanyStage
from src.services.scraper.core.base import BaseScraper

# Configure basic logger for console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def save_raw_html(fund_name: str, url: str, html_content: str) -> Path:
    """
    Save raw HTML content to storage directory.
    
    Args:
        fund_name: Name of the fund (e.g., "a16z")
        url: URL of the scraped page
        html_content: Raw HTML content to save
        
    Returns:
        Path to the saved HTML file
    """
    # Get project root
    project_root = Path(__file__).parent.parent.parent.parent.parent
    
    # Extract job ID from URL
    # URLs are like: https://jobs.a16z.com/jobs/{job_id}
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.strip("/").split("/")
    job_id = path_parts[-1] if path_parts else "unknown"
    
    # Sanitize job_id for filename
    job_id = re.sub(r'[^\w\-_]', '_', job_id)
    
    # Create date-based directory structure
    date_str = datetime.now().strftime("%Y-%m-%d")
    storage_dir = project_root / "storage" / "raw_scrapes" / fund_name / date_str
    
    # Create directories if they don't exist
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Save HTML file
    html_file = storage_dir / f"{job_id}.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    logger.info(f"Saved raw HTML to: {html_file}")
    return html_file


class A16ZScraper(BaseScraper):
    """Scraper for a16z jobs website."""

    def __init__(self, browser_manager: Any, config: Any) -> None:
        """
        Initialize the A16Z scraper.

        Args:
            browser_manager: Browser manager instance for web automation
            config: Configuration object with scraper settings
        """
        super().__init__(browser_manager, config)
        self.base_url = getattr(config, "base_url", "https://jobs.a16z.com/jobs")

    async def fetch_all_links(self) -> List[str]:
        """
        Fetch all vacancy links from a16z jobs page.

        Returns:
            List of URLs to vacancy detail pages
        """
        async with self.browser_manager.get_page() as page:
            # Navigate to base URL with retry logic and wait for networkidle
            logger.info(f"Navigating to {self.base_url}")
            await self.browser_manager.goto_with_retry(
                page, self.base_url, wait_until="networkidle"
            )
            
            # Scroll to load all dynamic content
            logger.info("Scrolling to load all content")
            await self.browser_manager.smart_scroll(page)
            
            # Extract all href attributes from <a> tags
            logger.info("Extracting links from page")
            links = await page.evaluate("""
                () => {
                    const anchors = Array.from(document.querySelectorAll('a'));
                    return anchors.map(a => a.href).filter(href => href);
                }
            """)
            
            # Filter links: keep only those that contain /jobs/ and DO NOT end with /jobs or /jobs/
            base_domain = "https://jobs.a16z.com"
            filtered_links = []
            
            for link in links:
                if not link:
                    continue
                
                # Convert relative links to absolute URLs
                if link.startswith("/"):
                    link = urljoin(base_domain, link)
                elif not link.startswith("http"):
                    continue
                
                # Check if link contains /jobs/ and is not just /jobs or /jobs/
                if "/jobs/" in link:
                    # Remove trailing slash for comparison
                    link_normalized = link.rstrip("/")
                    if not link_normalized.endswith("/jobs"):
                        filtered_links.append(link)
            
            # Get unique links
            unique_links = list(set(filtered_links))
            logger.info(f"Found {len(unique_links)} unique job links")
            
            return unique_links

    async def extract_details(self, url: str) -> Vacancy:
        """
        Extract vacancy details from a single URL.

        Args:
            url: URL of the vacancy detail page

        Returns:
            Vacancy object with extracted details
        """
        async with self.browser_manager.get_page() as page:
            # Navigate to the job page
            logger.info(f"Extracting details from: {url}")
            await self.browser_manager.goto_with_retry(
                page, url, wait_until="networkidle"
            )
            
            # Get raw HTML content
            html_content = await page.content()
            
            # Save HTML backup
            save_raw_html("a16z", url, html_content)
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Extract title from <h1> or .job-title
            title = ""
            h1_tag = soup.find("h1")
            if h1_tag:
                title = h1_tag.get_text(strip=True)
            else:
                job_title_elem = soup.select_one(".job-title")
                if job_title_elem:
                    title = job_title_elem.get_text(strip=True)
            
            if not title:
                title = "Unknown Position"
                logger.warning(f"Could not extract title from: {url}")
            
            # Extract company name (portfolio company)
            company_name = ""
            # Try common selectors for company name
            company_selectors = [
                ".company-name",
                ".portfolio-company",
                "[data-company]",
                ".job-company",
            ]
            for selector in company_selectors:
                company_elem = soup.select_one(selector)
                if company_elem:
                    company_name = company_elem.get_text(strip=True)
                    break
            
            # Fallback: try to find company in meta tags or structured data
            if not company_name:
                meta_company = soup.find("meta", property="og:site_name")
                if meta_company and meta_company.get("content"):
                    company_name = meta_company.get("content")
                else:
                    # Try to extract from page title or other common locations
                    page_title = soup.find("title")
                    if page_title:
                        title_text = page_title.get_text(strip=True)
                        # Sometimes format is "Job Title - Company Name"
                        if " - " in title_text:
                            company_name = title_text.split(" - ")[-1]
            
            if not company_name:
                company_name = "Unknown Company"
                logger.warning(f"Could not extract company name from: {url}")
            
            # Extract description from main text container
            description = ""
            description_selectors = [
                ".job-description",
                ".job-details",
                ".description",
                "[role='main']",
                "main",
                ".content",
            ]
            for selector in description_selectors:
                desc_elem = soup.select_one(selector)
                if desc_elem:
                    # Get text while preserving newlines
                    description = desc_elem.get_text(separator="\n", strip=True)
                    if description and len(description) > 50:  # Ensure we got meaningful content
                        break
            
            if not description:
                # Fallback: get body text
                body = soup.find("body")
                if body:
                    description = body.get_text(separator="\n", strip=True)
            
            if not description:
                description = "No description available"
                logger.warning(f"Could not extract description from: {url}")
            
            # Extract location
            location = ""
            location_selectors = [
                ".location",
                ".job-location",
                "[data-location]",
            ]
            for selector in location_selectors:
                loc_elem = soup.select_one(selector)
                if loc_elem:
                    location = loc_elem.get_text(strip=True)
                    break
            
            if not location:
                location = "Not specified"
            
            # Determine if remote
            remote_option = False
            location_lower = location.lower()
            description_lower = description.lower()
            if (
                "remote" in location_lower
                or "remote" in description_lower
                or "anywhere" in location_lower
            ):
                remote_option = True
            
            # Extract industry (try to find in page)
            industry = "Technology"  # Default
            industry_selectors = [
                ".industry",
                ".market",
                "[data-industry]",
            ]
            for selector in industry_selectors:
                industry_elem = soup.select_one(selector)
                if industry_elem:
                    industry = industry_elem.get_text(strip=True)
                    break
            
            # Extract required skills (try to find in description or dedicated section)
            required_skills = []
            skills_selectors = [
                ".skills",
                ".requirements",
                ".qualifications",
            ]
            for selector in skills_selectors:
                skills_elem = soup.select_one(selector)
                if skills_elem:
                    skills_text = skills_elem.get_text()
                    # Try to extract skills from text (basic approach)
                    # Look for common tech keywords
                    common_skills = [
                        "Python",
                        "JavaScript",
                        "TypeScript",
                        "React",
                        "Node.js",
                        "AWS",
                        "Docker",
                        "Kubernetes",
                        "PostgreSQL",
                        "MongoDB",
                        "FastAPI",
                        "Django",
                        "Flask",
                    ]
                    for skill in common_skills:
                        if skill.lower() in skills_text.lower():
                            required_skills.append(skill)
                    break
            
            # Default company stage (can be improved with actual extraction)
            company_stage = CompanyStage.GROWTH
            
            # Create and return Vacancy object
            vacancy = Vacancy(
                title=title,
                company_name=company_name,
                company_stage=company_stage,
                location=location,
                industry=industry,
                salary_range=None,  # Not typically on a16z pages
                description_url=url,
                required_skills=required_skills,
                remote_option=remote_option,
                source_url=self.base_url,
            )
            
            logger.info(f"Extracted vacancy: {title} at {company_name}")
            return vacancy

