"""
A16Z scraper provider implementation.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Any, Optional, Dict
from urllib.parse import urljoin, urlparse, parse_qs, unquote

import httpx
from bs4 import BeautifulSoup

from src.schemas.vacancy import Vacancy, CompanyStage
from src.services.scraper.core.base import BaseScraper
from src.services.scraper.core.azure_storage import upload_html_to_azure_blob

# Configure basic logger for console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def extract_company_from_meta_tags(soup: BeautifulSoup) -> str:
    """
    Extract company name from meta tags in priority order.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Company name string or empty string if not found
    """
    # Priority 1: og:site_name
    meta_og = soup.find("meta", property="og:site_name")
    if meta_og and meta_og.get("content"):
        company = meta_og.get("content").strip()
        if company and company.lower() not in ["jobs", "careers", "hiring"]:
            return company
    
    # Priority 2: apple-mobile-web-app-title
    meta_apple = soup.find("meta", attrs={"name": "apple-mobile-web-app-title"})
    if meta_apple and meta_apple.get("content"):
        company = meta_apple.get("content").strip()
        if company and company.lower() not in ["jobs", "careers", "hiring"]:
            return company
    
    # Priority 3: meta name='author'
    meta_author = soup.find("meta", attrs={"name": "author"})
    if meta_author and meta_author.get("content"):
        company = meta_author.get("content").strip()
        if company and company.lower() not in ["jobs", "careers", "hiring"]:
            return company
    
    return ""


def extract_company_from_url(url: str) -> str:
    """
    Extract company name from URL for platform-specific job boards.
    
    Args:
        url: Job posting URL
        
    Returns:
        Company name string or empty string if not found
    """
    parsed = urlparse(url)
    path_parts = [p for p in parsed.path.strip("/").split("/") if p]
    
    # Lever: jobs.lever.co/COMPANY/ID
    if "lever.co" in parsed.netloc:
        if len(path_parts) >= 1:
            company = unquote(path_parts[0])  # Decode URL-encoded names like 'Flock%20Safety'
            # Clean up common suffixes
            company = company.replace("-", " ").title()
            return company
    
    # Greenhouse: boards.greenhouse.io/COMPANY/...
    if "greenhouse.io" in parsed.netloc:
        if len(path_parts) >= 1:
            company = unquote(path_parts[0])  # Decode URL-encoded names
            company = company.replace("-", " ").title()
            return company
    
    # Ashby: jobs.ashbyhq.com/COMPANY/ID
    if "ashbyhq.com" in parsed.netloc:
        if len(path_parts) >= 1:
            company = unquote(path_parts[0])  # Decode URL-encoded names
            company = company.replace("-", " ").title()
            return company
    
    return ""


def extract_company_from_domain(url: str) -> str:
    """
    Extract a fallback company name from the domain of the URL.
    
    Args:
        url: Job posting URL
        
    Returns:
        Company name string or empty string if not found
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        
        # Remove common prefixes
        domain = domain.replace("www.", "").replace("jobs.", "").replace("careers.", "")
        
        # Extract main domain name (before TLD)
        parts = domain.split(".")
        if len(parts) >= 2:
            # Get the main domain part (e.g., "company" from "company.com")
            main_part = parts[-2]
            # Capitalize and return
            return main_part.replace("-", " ").title()
    except Exception as e:
        logger.debug(f"Error extracting company from domain: {e}")
    
    return ""


def get_api_text(data: Any) -> str:
    """
    Extract text from API data that can be in various formats (string, dict, list).
    
    Args:
        data: API data in any format (string, dict, list, etc.)
        
    Returns:
        Extracted text string, or empty string if data is empty/None
    """
    if not data:
        return ""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, dict):
        for key in ["name", "label", "text", "display_name", "value"]:
            if data.get(key):
                return str(data[key]).strip()
    if isinstance(data, list):
        # Join multiple values with comma
        values = []
        for item in data:
            text = get_api_text(item)
            if text:
                values.append(text)
        return ", ".join(values)
    return str(data).strip()


def clean_job_title(title: str, company_name: str = "") -> str:
    """
    Clean job title by removing common suffixes and formatting.
    
    Args:
        title: Raw job title
        company_name: Optional company name to remove from title
        
    Returns:
        Cleaned job title
    """
    if not title:
        return ""
    cleaned = title.strip()
    if company_name:
        # Удаляем название компании только если оно в начале и отделено разделителем
        pattern = re.compile(rf"^{re.escape(company_name)}\s*[:\-\|]\s*", re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)
    # Удаляем (Remote), (Hybrid) и т.д.
    cleaned = re.sub(r'\s*[\(\[].*?[\)\]]', '', cleaned)
    return cleaned.strip()


def clean_salary_value(salary_data: Any) -> Optional[str]:
    """
    Clean and extract salary values from various data structures (list, dict, string, number).
    Extracts only text values and numbers, ignoring dictionaries.
    
    Args:
        salary_data: Salary data in any format (list, dict, string, number)
        
    Returns:
        Formatted salary string (e.g., "$120,000 - $180,000") or None if no valid data
    """
    if not salary_data:
        return None
    
    salary_parts = []
    
    # Handle list of mixed types
    if isinstance(salary_data, list):
        for item in salary_data:
            if isinstance(item, (int, float)):
                # Format numbers with commas (e.g., 120000 -> "120,000")
                formatted = f"{int(item):,}"
                salary_parts.append(formatted)
            elif isinstance(item, str):
                # Keep strings as is
                salary_parts.append(item)
            # STRICTLY ignore dictionaries (like {'label': 'USD'})
            elif isinstance(item, dict):
                continue
    
    # Handle dictionary (e.g., {'minValue': 150000, 'maxValue': 200000, 'currency': 'USD'})
    elif isinstance(salary_data, dict):
        min_salary = (
            salary_data.get("minValue") 
            or salary_data.get("min") 
            or salary_data.get("minSalary")
        )
        max_salary = (
            salary_data.get("maxValue") 
            or salary_data.get("max") 
            or salary_data.get("maxSalary")
        )
        currency = salary_data.get("currency", "USD")
        currency_symbol = "$" if currency == "USD" else currency
        
        if min_salary and max_salary:
            salary_parts.append(f"{currency_symbol}{min_salary:,}")
            salary_parts.append(f"{currency_symbol}{max_salary:,}")
        elif min_salary:
            salary_parts.append(f"{currency_symbol}{min_salary:,}")
        elif max_salary:
            salary_parts.append(f"{currency_symbol}{max_salary:,}")
    
    # Handle string
    elif isinstance(salary_data, str):
        salary_parts.append(salary_data)
    
    # Handle number
    elif isinstance(salary_data, (int, float)):
        salary_parts.append(f"{int(salary_data):,}")
    
    # If we have parts, format them
    if salary_parts:
        # Join parts with space first
        salary_range = " ".join(salary_parts)
        
        # Clean up: replace multiple spaces with single space, fix " - " formatting
        salary_range = re.sub(r'\s+', ' ', salary_range)  # Clean double spaces
        salary_range = re.sub(r'\s*-\s*', ' - ', salary_range)  # Normalize " - " spacing
        salary_range = salary_range.strip()
        
        # Prefix with "$" if missing
        if salary_range and not salary_range.startswith("$"):
            # Check if it's already a formatted range
            if " - " in salary_range:
                parts = salary_range.split(" - ")
                salary_range = " - ".join([f"${p.strip()}" if not p.strip().startswith("$") else p.strip() for p in parts])
            else:
                salary_range = f"${salary_range}"
        
        return salary_range
    
    return None


def extract_quality_description(soup: BeautifulSoup) -> str:
    """
    Extract high-quality job description, avoiding headers/footers.
    
    Uses a universal description finder with multiple fallback strategies
    to ensure high-quality data extraction.
    
    Args:
        soup: BeautifulSoup parsed HTML
        
    Returns:
        Job description text with preserved line breaks and cleaned whitespace
    """
    # Удаляем шумные элементы ПЕРЕД извлечением текста
    for noise in soup.select("script, style, button, .cookie-notice, .privacy-notice, .apply-button"):
        noise.decompose()
        
    description_selectors = [
        ".job-description", ".job-body", ".posting-description", 
        ".content", "#content", ".description", ".section-wrapper",
        "[data-qa='job-description']"
    ]
    for selector in description_selectors:
        # Use select to get all matching elements, not just the first one
        elements = soup.select(selector)
        for elem in elements:
            text = elem.get_text(separator="\n", strip=True)
            # Use the first element that has substantial content (> 100 chars)
            # This avoids selecting empty wrapper divs (common in Lever/Greenhouse)
            if len(text) > 100:
                return text
    return soup.get_text(separator="\n", strip=True)


def save_raw_html(fund_name: str, url: str, html_content: str) -> Optional[str]:
    """
    Upload raw HTML content to Azure Blob Storage.
    
    Args:
        fund_name: Name of the fund (e.g., "a16z")
        url: URL of the scraped page
        html_content: Raw HTML content to upload
        
    Returns:
        URL to the uploaded blob in Azure Blob Storage, or None if upload failed
    """
    blob_url = upload_html_to_azure_blob(fund_name, url, html_content)
    if blob_url:
        logger.info(f"Uploaded raw HTML to Azure Blob Storage: {blob_url}")
    else:
        logger.warning(f"Failed to upload HTML to Azure Blob Storage for: {url}")
    return blob_url


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
        self._job_metadata: Dict[str, Dict[str, Any]] = {}

    async def fetch_all_links(self) -> List[str]:
        """
        Fetch all vacancy links from a16z jobs API.

        Returns:
            List of URLs to vacancy detail pages
        """
        api_url = "https://jobs.a16z.com/api-boards/search-jobs"
        
        # Headers matching typical browser requests
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Content-Type": "application/json",
            "Origin": "https://jobs.a16z.com",
            "Referer": "https://jobs.a16z.com/jobs",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        
        # Initial payload
        payload: Dict[str, Any] = {
            "meta": {"size": 100},
            "board": {"id": "andreessen-horowitz", "isParent": True},
            "query": {"promoteFeatured": True},
        }
        
        all_links: List[str] = []
        sequence: Optional[str] = None
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                # Add sequence to payload if we have one from previous response
                if sequence:
                    payload["meta"]["sequence"] = sequence
                
                try:
                    logger.info(f"Sending API request (sequence: {sequence or 'initial'})")
                    response = await client.post(api_url, json=payload, headers=headers)
                    response.raise_for_status()
                    
                    data = response.json()
                    
                    # Debug logging: log response structure
                    logger.info(f"API Response keys: {list(data.keys())}")
                    
                    # Extract jobs from response - check multiple possible locations
                    jobs = data.get("jobs") or data.get("results", [])
                    if not jobs and "data" in data:
                        # Check inside data object for both jobs and results
                        data_obj = data.get("data", {})
                        jobs = data_obj.get("jobs") or data_obj.get("results", [])
                    
                    logger.info(f"Found {len(jobs)} jobs in response")
                    
                    # Process each job to extract links
                    for job in jobs:
                        # Try to get link in priority order: url, applyUrl, externalUrl
                        link = job.get("url") or job.get("applyUrl") or job.get("externalUrl")
                        
                        if not link:
                            # Build URL from jobId or slug
                            job_id = job.get("jobId") or job.get("slug")
                            if job_id:
                                link = f"https://jobs.a16z.com/jobs/{job_id}"
                        
                        if link:
                            # Collect tags from multiple possible keys
                            tags_list = []
                            # Check all possible tag keys, including keywords and skills
                            for key in ["tags", "job_tags", "jobTags", "keywords", "skills", "requiredSkills", "preferredSkills"]:
                                tags = job.get(key, [])
                                if tags:
                                    if isinstance(tags, list):
                                        for t in tags:
                                            if t:
                                                # Handle dicts in skills (e.g. {'label': 'Audit'})
                                                if isinstance(t, dict):
                                                    val = t.get("label") or t.get("name") or t.get("value")
                                                    if val:
                                                        tags_list.append(str(val))
                                                else:
                                                    tags_list.append(str(t))
                                    else:
                                        tags_list.append(str(tags))
                            
                            # Collect categories from multiple possible keys
                            categories_list = []
                            for key in ["categories", "job_categories", "jobCategories", "sectors"]:
                                cats = job.get(key, [])
                                if cats:
                                    if isinstance(cats, list):
                                        categories_list.extend([str(c) for c in cats if c])
                                    else:
                                        categories_list.append(str(cats))
                            
                            # Get company_name from API first, fallback to URL extraction
                            company_name = job.get("company_name") or job.get("companyName")
                            if not company_name:
                                # Fallback to URL extraction
                                company_name = self._get_company_from_url(link)
                            
                            # Collect departments from multiple possible keys
                            departments_list = []
                            for key in ["departments", "job_departments", "jobDepartments"]:
                                depts = job.get(key, [])
                                if depts:
                                    if isinstance(depts, list):
                                        departments_list.extend([str(d) for d in depts if d])
                                    else:
                                        departments_list.append(str(depts))
                            
                            # Collect sectors separately if available
                            sectors_list = []
                            # Check sectors and markets
                            for key in ["sectors", "markets"]:
                                items = job.get(key, [])
                                if items:
                                    if isinstance(items, list):
                                        for item in items:
                                            if item:
                                                # Handle dicts (e.g. {'label': 'Consumer'})
                                                if isinstance(item, dict):
                                                    val = item.get("label") or item.get("name") or item.get("value")
                                                    if val:
                                                        sectors_list.append(str(val))
                                                else:
                                                    sectors_list.append(str(item))
                                    else:
                                        sectors_list.append(str(items))
                            
                            # Collect location from multiple possible keys
                            location_data = job.get("location") or job.get("office") or job.get("location_string") or job.get("locations") or job.get("normalizedLocations")
                            
                            # Collect industries separately if available (different from sectors)
                            industries_list = []
                            industries = job.get("industries", [])
                            if industries:
                                if isinstance(industries, list):
                                    industries_list.extend([str(i) for i in industries if i])
                                else:
                                    industries_list.append(str(industries))
                            
                            # Store metadata for this job using the link as key
                            self._job_metadata[link] = {
                                "title": job.get("text") or job.get("title") or job.get("name"),  # Store title from API
                                "tags": tags_list,  # Already includes job_tags from the loop above
                                "job_tags": tags_list,  # Store job_tags separately for clarity
                                "categories": categories_list,
                                "departments": departments_list,
                                "sectors": sectors_list,  # Store sectors separately
                                "industries": industries_list,  # Store industries separately
                                "salary": job.get("salaryRange") or job.get("salary"),
                                "is_remote": job.get("isRemote"),
                                "location": location_data,  # Can be string, dict, or list
                                "office": job.get("office"),  # Store office separately for fallback
                                "location_string": job.get("location_string"),  # Store location_string for fallback
                                "company_name": company_name,  # Already uses unquote in _get_company_from_url if from URL
                                "stage": job.get("stage") or job.get("companyStage") or job.get("company_stage") or job.get("stages"),  # Company funding stage
                            }
                            all_links.append(link)
                    
                    # Log progress
                    logger.info(f"Fetched {len(all_links)} links so far...")
                    
                    # Check for next sequence token
                    meta = data.get("meta", {})
                    next_sequence = meta.get("sequence")
                    
                    # If no sequence or same sequence, we're done
                    if not next_sequence or next_sequence == sequence:
                        logger.info("No more pages to fetch")
                        break
                    
                    sequence = next_sequence
                    
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error {e.response.status_code}: {e.response.text}")
                    raise
                except httpx.RequestError as e:
                    logger.error(f"Request error: {e}")
                    raise
                except Exception as e:
                    logger.error(f"Unexpected error: {e}")
                    raise
        
        # Remove duplicates while preserving order
        unique_links = list(dict.fromkeys(all_links))
        logger.info(f"Found {len(unique_links)} unique job links (total fetched: {len(all_links)})")
        
        return unique_links

    def _get_company_from_url(self, url: str) -> str:
        """
        Extract company name from URL for platform-specific job boards.
        This is a helper method for the A16ZScraper class.
        Uses unquote to decode URL-encoded names like 'Flock%20Safety'.
        
        Args:
            url: Job posting URL
            
        Returns:
            Company name string or empty string if not found
        """
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        
        # Lever: jobs.lever.co/COMPANY/ID
        if "lever.co" in parsed.netloc:
            if len(path_parts) >= 1:
                company = unquote(path_parts[0])  # Decode URL-encoded names like 'Flock%20Safety'
                # Clean up common suffixes
                company = company.replace("-", " ").title()
                return company
        
        # Greenhouse: boards.greenhouse.io/COMPANY/...
        if "greenhouse.io" in parsed.netloc:
            if len(path_parts) >= 1:
                company = unquote(path_parts[0])  # Decode URL-encoded names
                company = company.replace("-", " ").title()
                return company
        
        # Ashby: jobs.ashbyhq.com/COMPANY/ID
        if "ashbyhq.com" in parsed.netloc:
            if len(path_parts) >= 1:
                company = unquote(path_parts[0])  # Decode URL-encoded names
                company = company.replace("-", " ").title()
                return company
        
        # Special handling for Google search URLs with myworkdayjobs.com
        if "google.com/search" in url and "myworkdayjobs.com" in url:
            # Extract from the q parameter or from the redirect URL
            try:
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                if "q" in query_params:
                    q_value = query_params["q"][0]
                    # Look for myworkdayjobs.com/COMPANY in the query
                    if "myworkdayjobs.com" in q_value:
                        # Extract company from URL pattern
                        match = re.search(r'myworkdayjobs\.com/([^/]+)', q_value)
                        if match:
                            company = match.group(1)
                            company = company.replace("-", " ").title()
                            return company
            except Exception as e:
                logger.debug(f"Error extracting company from Google search URL: {e}")
        
        return ""

    def _clean_company_name(self, company_name: str) -> str:
        """
        Clean company name by removing common prefixes.
        
        Args:
            company_name: Raw company name
            
        Returns:
            Cleaned company name
        """
        if not company_name:
            return ""
        
        # Remove "Careers at " prefix
        if company_name.startswith("Careers at "):
            company_name = company_name[len("Careers at "):].strip()
        
        return company_name

    async def extract_details(self, url: str) -> Vacancy:
        """
        Extract vacancy details from a single URL with improved extraction logic.

        Args:
            url: URL of the vacancy detail page

        Returns:
            Vacancy object with extracted details
        """
        # Retrieve cached metadata from fetch_all_links
        cached = self._job_metadata.get(url, {})
        
        async with self.browser_manager.get_page() as page:
            # Navigate to the job page
            logger.info(f"Extracting details from: {url}")
            await self.browser_manager.goto_with_retry(
                page, url, wait_until="domcontentloaded"
            )
            
            # Get raw HTML content
            html_content = await page.content()
            
            # Upload HTML to Azure Blob Storage
            raw_html_url = save_raw_html("a16z", url, html_content)
            
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            
            # Detect job board type from URL
            is_lever = "lever.co" in url
            is_greenhouse = "greenhouse.io" in url
            is_ashby = "ashbyhq.com" in url
            
            # ========== EXTRACT TITLE ==========
            title = ""
            
            # Priority 1: Cached API title (Most accurate for main page match)
            if cached.get("title"):
                title = cached.get("title").strip()
                logger.info(f"Using cached API title: {title}")

            # Priority 2: Try platform-specific selectors first (Lever, Greenhouse)
            if not title:
                # Lever: .posting-headline h2
                title_elem = soup.select_one(".posting-headline h2")
                if title_elem:
                    title = title_elem.get_text(strip=True)
                # Greenhouse: .app-title
                if not title:
                    title_elem = soup.select_one(".app-title")
                    if title_elem:
                        title = title_elem.get_text(strip=True)
            
            # Priority 2: Fallback to soup.title.string
            if not title:
                page_title = soup.find("title")
                if page_title and page_title.string:
                    title = page_title.string.strip()
                    # Remove common page title suffixes
                    for suffix in [" - Jobs", " | Jobs", " Careers", " - Careers"]:
                        if title.endswith(suffix):
                            title = title[:-len(suffix)].strip()
            
            # Check for generic titles: "See Job Post" or "Open Roles"
            if title and title.lower() in ["see job post", "open roles"]:
                logger.info(f"Title is generic '{title}', searching for real job title from meta tags")
                # Try og:title meta tag first
                meta_og_title = soup.find("meta", property="og:title")
                if meta_og_title and meta_og_title.get("content"):
                    candidate_title = meta_og_title.get("content").strip()
                    candidate_title = clean_job_title(candidate_title)
                    if candidate_title and candidate_title.lower() not in ["see job post", "open roles"]:
                        title = candidate_title
                        logger.info(f"Found real job title from og:title: {title}")
                
                # If still generic, try <title> tag
                if title.lower() in ["see job post", "open roles"]:
                    page_title = soup.find("title")
                    if page_title:
                        title_text = page_title.get_text(strip=True)
                        # Remove common page title suffixes
                        for suffix in [" - Jobs", " | Jobs", " Careers", " - Careers"]:
                            if title_text.endswith(suffix):
                                title_text = title_text[:-len(suffix)].strip()
                        candidate_title = clean_job_title(title_text)
                        if candidate_title and candidate_title.lower() not in ["see job post", "open roles"]:
                            title = candidate_title
                            logger.info(f"Found real job title from <title>: {title}")
                
                # If still generic and we have company name, use fallback
                if title.lower() in ["see job post", "open roles"]:
                    # We'll set this after company_name is extracted
                    pass
            
            # Smart Title: If title contains generic hiring phrases, look for real job title
            if title and any(phrase in title.lower() for phrase in ["we're hiring", "come join our team", "join our team", "we are hiring"]):
                logger.info(f"Title contains generic hiring phrase, searching for real job title: {title}")
                # Look for first h1 or h2 tag on the page (usually the real job title)
                h1_tag = soup.find("h1")
                if h1_tag:
                    candidate_title = h1_tag.get_text(strip=True)
                    candidate_title = clean_job_title(candidate_title)
                    # Only use if it's different and meaningful
                    if candidate_title and candidate_title.lower() != title.lower() and len(candidate_title) > 5:
                        title = candidate_title
                        logger.info(f"Found real job title from h1: {title}")
                else:
                    h2_tag = soup.find("h2")
                    if h2_tag:
                        candidate_title = h2_tag.get_text(strip=True)
                        candidate_title = clean_job_title(candidate_title)
                        # Only use if it's different and meaningful
                        if candidate_title and candidate_title.lower() != title.lower() and len(candidate_title) > 5:
                            title = candidate_title
                            logger.info(f"Found real job title from h2: {title}")
            
            # Priority 3: Fallback to page title
            if not title:
                logger.warning(f"Could not extract title from: {url}")
                page_title = soup.find("title")
                if page_title:
                    title_text = page_title.get_text(strip=True)
                    # Remove common page title suffixes
                    for suffix in [" - Jobs", " | Jobs", " Careers", " - Careers"]:
                        if title_text.endswith(suffix):
                            title_text = title_text[:-len(suffix)].strip()
                    title = clean_job_title(title_text)
            
            if not title:
                title = "Unknown Position"
                logger.warning(f"Title extraction failed, using fallback: {url}")
            
            # ========== EXTRACT COMPANY NAME ==========
            company_name = ""
            
            # Step 1: Primary source - og:site_name meta tag
            meta_og_site = soup.find("meta", property="og:site_name")
            if meta_og_site and meta_og_site.get("content"):
                company_name = meta_og_site.get("content").strip()
                company_name = self._clean_company_name(company_name)
                if company_name and company_name.lower() not in ["jobs", "careers", "hiring", "unknown"]:
                    logger.info(f"Extracted company from og:site_name: {company_name}")
            
            # Step 2: Check cached company_name from API
            if not company_name:
                cached_company = cached.get("company_name")
                if cached_company:
                    company_name = str(cached_company).strip()
                    # Apply unquote to decode URL-encoded names (e.g., "Flock%20Safety" -> "Flock Safety")
                    company_name = unquote(company_name)
                    company_name = self._clean_company_name(company_name)
                    if company_name:
                        logger.info(f"Extracted company from cached API data: {company_name}")
            
            # Step 3: Platform-specific URL extraction using helper method (already uses unquote)
            if not company_name:
                company_name = self._get_company_from_url(url)
                company_name = self._clean_company_name(company_name)
                if company_name:
                    logger.info(f"Extracted company from URL: {company_name}")
            
            # Step 4: Try other meta tags (fallback)
            if not company_name:
                company_name = extract_company_from_meta_tags(soup)
                company_name = self._clean_company_name(company_name)
            
            # Step 5: Try CSS selectors
            if not company_name:
                company_selectors = [
                    ".company-name",
                    ".portfolio-company",
                    "[data-company]",
                    ".job-company",
                    ".employer-name",
                ]
                for selector in company_selectors:
                    company_elem = soup.select_one(selector)
                    if company_elem:
                        company_name = company_elem.get_text(strip=True)
                        company_name = self._clean_company_name(company_name)
                        if company_name and len(company_name) > 1:
                            break
            
            # Step 6: Try page title parsing
            if not company_name:
                page_title = soup.find("title")
                if page_title:
                    title_text = page_title.get_text(strip=True)
                    # Common formats: "Job Title - Company Name" or "Company Name - Job Title"
                    for separator in [" - ", " | ", " — "]:
                        if separator in title_text:
                            parts = title_text.split(separator)
                            if len(parts) >= 2:
                                # Try last part first (most common)
                                candidate = parts[-1].strip()
                                # Remove common suffixes
                                for suffix in [" Jobs", " Careers", " Hiring"]:
                                    if candidate.endswith(suffix):
                                        candidate = candidate[:-len(suffix)].strip()
                                candidate = self._clean_company_name(candidate)
                                if candidate and len(candidate) > 1 and len(candidate) < 50:
                                    company_name = candidate
                                    break
                                # Try first part
                                candidate = parts[0].strip()
                                for suffix in [" Jobs", " Careers", " Hiring"]:
                                    if candidate.endswith(suffix):
                                        candidate = candidate[:-len(suffix)].strip()
                                candidate = self._clean_company_name(candidate)
                                if candidate and len(candidate) > 1 and len(candidate) < 50:
                                    company_name = candidate
                                    break
            
            # Step 7: Domain fallback (last resort before "Unknown")
            if not company_name:
                company_name = extract_company_from_domain(url)
                company_name = self._clean_company_name(company_name)
            
            # Step 8: Special handling for Google search URLs with myworkdayjobs.com
            if not company_name or company_name.lower() in ["unknown", "jobs", "careers", "hiring"]:
                if "google.com/search" in url and "myworkdayjobs.com" in url:
                    # Try to extract from the URL or query parameters
                    extracted = self._get_company_from_url(url)
                    if extracted:
                        company_name = self._clean_company_name(extracted)
                        logger.info(f"Extracted company from Google search URL: {company_name}")
            
            # Final validation
            if not company_name or company_name.lower() in ["unknown", "jobs", "careers", "hiring"]:
                company_name = "Unknown Company"
                logger.warning(f"Company name extraction failed for: {url}")
            else:
                logger.info(f"Final extracted company name: {company_name} from {url}")
            
            # Clean the title with company_name (if available)
            if title:
                title = clean_job_title(title, company_name=company_name if company_name != "Unknown Company" else "")
            
            # Final check: If title is still generic ("See Job Post" or "Open Roles"), use company name + " Specialist"
            if title and title.lower() in ["see job post", "open roles"]:
                if company_name and company_name != "Unknown Company":
                    title = f"{company_name} Specialist"
                    logger.info(f"Using fallback title: {title}")
                else:
                    title = "Unknown Position"
                    logger.warning(f"Title is generic and company name unavailable, using fallback: {title}")
            
            # ========== EXTRACT DESCRIPTION ==========
            # Try to extract description, but don't let failure stop enrichment of other fields
            description = extract_quality_description(soup)
            
            # Description Validation: Mark short descriptions as 'Parsing Error'
            # NOTE: This assignment does NOT stop execution - all other fields continue to be enriched
            if not description or len(description) < 100:
                logger.warning(
                    f"Could not extract quality description from: {url} "
                    f"(length: {len(description) if description else 0}). "
                    f"Will use cached metadata for other fields (location, salary, tags)."
                )
                description = "Parsing Error"
            
            # ========== EXTRACT LOCATION ==========
            # STRICT: Use ONLY cached location from API - check multiple fields and handle different data types
            # This prevents errors like "Portfolio, AI" from being extracted from description text
            location = ""
            
            # Try location, office, location_string in that order
            location_data = cached.get("location") or cached.get("office") or cached.get("location_string")
            
            if location_data:
                # Handle different data types using get_api_text
                if isinstance(location_data, list):
                    # If it's a list, process each item and join with ", "
                    location_parts = []
                    for item in location_data:
                        text = get_api_text(item)
                        if text:
                            location_parts.append(text)
                    location = ", ".join(location_parts).strip()
                else:
                    # Use get_api_text for dict, string, or other types
                    location = get_api_text(location_data)
            
            # If no cached location, use default
            if not location:
                location = "Not specified"
            
            # If location is "Not specified", try to find it in soup
            if location == "Not specified":
                # Try .location selector
                location_elem = soup.select_one(".location")
                if location_elem:
                    location = location_elem.get_text(strip=True)
                # Try .posting-categories .location selector
                if location == "Not specified":
                    location_elem = soup.select_one(".posting-categories .location")
                    if location_elem:
                        location = location_elem.get_text(strip=True)
            
            # ========== DETERMINE REMOTE OPTION ==========
            # Priority 1: Explicitly check if location contains "Hybrid" (case-insensitive)
            remote_option = None
            location_lower = (location or "").lower()
            if "hybrid" in location_lower:
                remote_option = True
                logger.debug(f"Set remote_option=True because location contains 'Hybrid': {location}")
            
            # Priority 2: Use cached is_remote from API (most reliable)
            if remote_option is None:
                cached_is_remote = cached.get("is_remote")
                if cached_is_remote is not None:
                    remote_option = bool(cached_is_remote)
            
            # Priority 3: Fallback to text-based detection if cached data not available
            if remote_option is None:
                remote_option = False
                description_lower = description.lower() if description != "Parsing Error" else ""
                if (
                    "remote" in location_lower
                    or "remote" in description_lower
                    or "anywhere" in location_lower
                    or "work from home" in description_lower
                ):
                    remote_option = True
            
            # Ensure it's always a boolean
            remote_option = bool(remote_option)
            
            # ========== EXTRACT INDUSTRY ==========
            # Collect ALL sectors/industries from multiple fields
            # Join all values with ", " to capture multiple sectors (e.g., "American Dynamism, Enterprise")
            industry = "Technology"  # Default
            
            sectors_list = []
            
            # Collect from cached sectors - use get_api_text for each element
            cached_sectors = cached.get("sectors", [])
            if cached_sectors:
                if isinstance(cached_sectors, list):
                    for s in cached_sectors:
                        text = get_api_text(s)
                        if text:
                            sectors_list.append(text)
                else:
                    text = get_api_text(cached_sectors)
                    if text:
                        sectors_list.append(text)
            
            # Collect from cached industries (separate field from sectors) - use get_api_text
            cached_industries = cached.get("industries", [])
            if cached_industries:
                if isinstance(cached_industries, list):
                    for i in cached_industries:
                        text = get_api_text(i)
                        if text:
                            sectors_list.append(text)
                else:
                    text = get_api_text(cached_industries)
                    if text:
                        sectors_list.append(text)
            
            # Also collect from cached categories (categories may contain sector info like "American Dynamism")
            cached_categories = cached.get("categories", [])
            if cached_categories:
                if isinstance(cached_categories, list):
                    for cat in cached_categories:
                        text = get_api_text(cat)
                        if text:
                            sectors_list.append(text)
                else:
                    text = get_api_text(cached_categories)
                    if text:
                        sectors_list.append(text)
            
            # Remove duplicates (case-insensitive) while preserving order
            seen = set()
            unique_sectors = []
            for sector in sectors_list:
                sector_lower = sector.lower()
                if sector_lower not in seen:
                    seen.add(sector_lower)
                    unique_sectors.append(sector)
            
            # Join all sectors into a single string (e.g., "American Dynamism, Enterprise")
            if unique_sectors:
                industry = ", ".join(unique_sectors)
            
            # Fallback to HTML selectors only if sectors is empty and industry is still default
            if industry == "Technology":
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
            
            # ========== EXTRACT REQUIRED SKILLS ==========
            # Collect ALL tags from cached API data (tags, departments, job_tags)
            # These represent the colorful tags seen on the a16z dashboard (e.g., "Braze", "Zemax", "Optics", "Salesforce")
            # Merge them all immediately to ensure we don't have "Tags: 0" in logs
            card_tags = []
            
            # Collect all tags from cached tags - use get_api_text for each element
            cached_tags = cached.get("tags", [])
            if cached_tags:
                if isinstance(cached_tags, list):
                    for tag in cached_tags:
                        text = get_api_text(tag)
                        if text:
                            card_tags.append(text)
                else:
                    text = get_api_text(cached_tags)
                    if text:
                        card_tags.append(text)
            
            # Retrieve departments for filtering (but NOT adding to skills)
            cached_departments = cached.get("departments", [])
            
            # REMOVED: Do NOT collect departments into skills/tags.
            # Departments are categories (e.g. "Sales", "Finance"), not skills.
            # This aligns with the website UI where tags are separate.
            
            # Collect from cached job_tags - use get_api_text
            cached_job_tags = cached.get("job_tags", [])
            if cached_job_tags:
                if isinstance(cached_job_tags, list):
                    for job_tag in cached_job_tags:
                        text = get_api_text(job_tag)
                        if text:
                            card_tags.append(text)
                else:
                    text = get_api_text(cached_job_tags)
                    if text:
                        card_tags.append(text)
            
            # Remove duplicates (case-insensitive) before filtering
            seen_lower = set()
            unique_card_tags = []
            for tag in card_tags:
                tag_lower = tag.lower().strip()
                if tag_lower and tag_lower not in seen_lower:
                    seen_lower.add(tag_lower)
                    unique_card_tags.append(tag)
            card_tags = unique_card_tags
            
            # Filter out generic words that don't represent actual skills
            # Exclude common job categories and employment types
            # IMPORTANT: Exclude "Engineering", "Marketing", "Full-time" as requested
            generic_words = {
                "all jobs", "software engineering", "growth", "engineering", 
                "product", "design", "data science", 
                "customer success", "g&a", "general & administrative",
                "full time", "full-time", "part time", "part-time", "contract", "internship"
            }
            card_tags = [
                skill for skill in card_tags 
                if skill.lower().strip() not in generic_words and len(skill.strip()) > 2
            ]
            
            # Initialize required_skills with card_tags immediately
            required_skills = list(card_tags)  # Start with card tags from API
            
            # Step 2.5: Remove terms from required_skills if they appear in industry or departments
            # This prevents duplicates like "Growth Division" appearing in both departments and skills
            # Create case-insensitive sets for comparison
            industry_terms = set()
            if industry and industry != "Technology":
                # Split industry by comma and normalize
                for term in industry.split(","):
                    term_clean = term.strip().lower()
                    if term_clean:
                        industry_terms.add(term_clean)
            
            department_terms = set()
            if cached_departments:
                if isinstance(cached_departments, list):
                    for dept in cached_departments:
                        if dept:
                            dept_clean = str(dept).strip().lower()
                            if dept_clean:
                                department_terms.add(dept_clean)
                elif isinstance(cached_departments, str):
                    dept_clean = cached_departments.strip().lower()
                    if dept_clean:
                        department_terms.add(dept_clean)
            
            # Remove skills that match industry or department terms (case-insensitive)
            required_skills = [
                skill for skill in required_skills
                if skill.strip().lower() not in industry_terms 
                and skill.strip().lower() not in department_terms
            ]
            
            # Step 3: Ensure uniqueness (case-insensitive) and convert to list of strings
            # required_skills is ONLY from cached data (tags, categories, departments) - no description extraction
            # Preserve exact strings from API (card_tags), but deduplicate case-insensitively
            seen_lower = set()
            unique_skills = []
            for skill in required_skills:
                skill_str = str(skill).strip()
                if skill_str:
                    skill_lower = skill_str.lower()
                    if skill_lower not in seen_lower:
                        seen_lower.add(skill_lower)
                        unique_skills.append(skill_str)
            
            required_skills = unique_skills
            
            # ========== EXTRACT SALARY RANGE ==========
            # Internal function to clean salary values
            def clean_salary(val: Any) -> Optional[str]:
                """
                Clean and extract salary values from various data structures.
                Filters out dictionaries like {'label': 'USD', 'value': 'USD'} and only keeps strings and numbers.
                Prepends "$" sign if missing but numbers are present.
                
                Args:
                    val: Salary data in any format (list, dict, string, number)
                    
                Returns:
                    Clean salary string (e.g., "$154,000 - $232,000") or None
                """
                if not val:
                    return None
                
                numbers = []
                strings = []  # Keep track of string values that might already have formatting
                
                # Handle list
                if isinstance(val, list):
                    for item in val:
                        if isinstance(item, dict):
                            # Filter out dictionaries like {'label': 'USD', 'value': 'USD'}
                            # Check if it's a label dict (label == value and no numeric value)
                            if 'label' in item and 'value' in item:
                                # Check if value is numeric
                                value = item.get('value')
                                if isinstance(value, (int, float)):
                                    numbers.append(int(value))
                                elif isinstance(value, str) and re.search(r'\d', value):
                                    # Extract number from string value
                                    matches = re.findall(r'\d+[\d,]*', value)
                                    for match in matches:
                                        num_str = match.replace(',', '')
                                        try:
                                            numbers.append(int(num_str))
                                        except ValueError:
                                            continue
                                # If label == value and no numbers, skip it (e.g., {'label': 'USD', 'value': 'USD'})
                                elif item.get('label') == item.get('value') and not re.search(r'\d', str(value)):
                                    continue
                            # Try to extract min/max from dict
                            else:
                                min_val = item.get("minValue") or item.get("min") or item.get("minSalary")
                                max_val = item.get("maxValue") or item.get("max") or item.get("maxSalary")
                                if min_val:
                                    numbers.append(int(min_val) if isinstance(min_val, (int, float)) else int(str(min_val).replace(',', '')))
                                if max_val:
                                    numbers.append(int(max_val) if isinstance(max_val, (int, float)) else int(str(max_val).replace(',', '')))
                        elif isinstance(item, (int, float)):
                            numbers.append(int(item))
                        elif isinstance(item, str):
                            # Check if string already contains "$" or formatted numbers
                            if "$" in item or re.search(r'\d', item):
                                # Extract numbers from string using regex
                                matches = re.findall(r'\d+[\d,]*', item)
                                for match in matches:
                                    # Remove commas and convert to int
                                    num_str = match.replace(',', '')
                                    try:
                                        numbers.append(int(num_str))
                                    except ValueError:
                                        continue
                            else:
                                # Pure string without numbers - might be a label, skip it
                                continue
                
                # Handle dictionary
                elif isinstance(val, dict):
                    # Filter out dictionaries that are just labels (e.g., {'label': 'USD', 'value': 'USD'})
                    if 'label' in val and 'value' in val and val.get('label') == val.get('value'):
                        # This is a label dictionary, skip it
                        return None
                    # Try to extract min/max values
                    min_val = val.get("minValue") or val.get("min") or val.get("minSalary")
                    max_val = val.get("maxValue") or val.get("max") or val.get("maxSalary")
                    if min_val:
                        numbers.append(int(min_val) if isinstance(min_val, (int, float)) else int(str(min_val).replace(',', '')))
                    if max_val:
                        numbers.append(int(max_val) if isinstance(max_val, (int, float)) else int(str(max_val).replace(',', '')))
                
                # Handle string
                elif isinstance(val, str):
                    # Check if string already has "$" sign
                    has_dollar = "$" in val
                    matches = re.findall(r'\d+[\d,]*', val)
                    for match in matches:
                        num_str = match.replace(',', '')
                        try:
                            numbers.append(int(num_str))
                        except ValueError:
                            continue
                    # If string already had "$" and we found numbers, preserve the format
                    if has_dollar and numbers:
                        strings.append(val)
                
                # Handle number
                elif isinstance(val, (int, float)):
                    numbers.append(int(val))
                
                # Format numbers
                if numbers:
                    # Remove duplicates and sort
                    numbers = sorted(list(set(numbers)))
                    if len(numbers) == 1:
                        # Single number: "$154,000" (prepend "$" if missing)
                        return f"${numbers[0]:,}"
                    elif len(numbers) >= 2:
                        # Range: "$154,000 - $232,000" (prepend "$" if missing)
                        return f"${numbers[0]:,} - ${numbers[-1]:,}"
                
                # If we have formatted strings, return the first one
                if strings:
                    return strings[0]
                
                return None
            
            # STRICTLY prioritize cached salary - handle list of mixed types (numbers and dicts)
            # This works even if description extraction failed ("Parsing Error")
            salary_range = None
            cached_salary = cached.get("salary")
            if cached_salary:
                # Use internal cleanup function
                salary_range = clean_salary(cached_salary)
            else:
                # Fallback to regex search in description (only if not "Parsing Error" and description exists)
                if description and description != "Parsing Error" and len(description) > 0:
                    salary_pattern = r"\$\d{1,3}(?:,\d{3})*(?:\s*-\s*\$\d{1,3}(?:,\d{3})*)?"
                    salary_match = re.search(salary_pattern, description)
                    if salary_match:
                        salary_range = salary_match.group(0)
            
            # ========== CHECK FOR HYBRID ==========
            # Check if location contains "(Hybrid)" or description says "hybrid role"
            is_hybrid = False
            # Check original cached location (before any cleaning) for "(Hybrid)"
            original_location = cached.get("location", "")
            if original_location:
                original_location_lower = str(original_location).lower()
                if "(hybrid)" in original_location_lower or "hybrid" in original_location_lower:
                    is_hybrid = True
                    logger.debug(f"Detected Hybrid in location: '{original_location}'")
            
            # Also check description for "hybrid role"
            if not is_hybrid and description and description != "Parsing Error":
                description_lower = description.lower()
                if "hybrid role" in description_lower or "hybrid work" in description_lower:
                    is_hybrid = True
                    logger.debug(f"Detected Hybrid in description: 'hybrid role' or 'hybrid work' mentioned")
            
            # ========== CREATE VACANCY OBJECT ==========
            # CRITICAL: Create Vacancy object even if description extraction failed ("Parsing Error")
            # All cached metadata is STRICTLY prioritized and preserved with cleaned/formatted values:
            # - required_skills: Initialized ONLY with card_tags (tags + categories + departments from API)
            #                    NO description-based skill extraction - prevents irrelevant skills
            #                    Case-insensitive deduplication, generic words filtered (e.g., "Full Time")
            #                    Works even if description is "Parsing Error" (uses only card_tags)
            # - salary_range: Robustly formatted from cached.get("salary") - handles lists of mixed types
            #                 Numbers formatted with commas, dicts STRICTLY ignored, joined with space,
            #                 cleaned up (double spaces, " - " formatting)
            # - location: STRICTLY from cached.get("location") ONLY - no HTML or regex fallbacks
            #             Prevents errors like "Portfolio, AI" from being extracted from description
            # - remote_option: Explicitly set to True if location contains "Hybrid", then cached.get("is_remote"), then text detection
            # - industry: All sectors joined with ", " (e.g., "American Dynamism, Enterprise, Marketing")
            # - company_name: Uses unquote to decode URL-encoded names (e.g., "Flock%20Safety")
            # - company_stage: Use cached.get("stage") if available, otherwise default to GROWTH
            # These fields are enriched REGARDLESS of description extraction success or failure
            
            # Determine company stage from cached data or default
            # Use get_api_text to handle complex objects (dicts, lists) and preserve exact text
            cached_stage = cached.get("stage")
            company_stage = "Growth"  # Default fallback
            employee_count = None  # New field for employee count
            
            if cached_stage:
                # Use get_api_text to extract clean string from any format (str, dict, list)
                stage_text = get_api_text(cached_stage)
                if stage_text:
                    # Check if stage_text contains employee count (e.g. "10-100 employees")
                    # and separate it from stage (e.g. "Series A")
                    parts = [p.strip() for p in stage_text.split(',')]
                    
                    # Filter parts
                    stages = []
                    counts = []
                    
                    for part in parts:
                        part_lower = part.lower()
                        if "employee" in part_lower:
                            counts.append(part)
                        else:
                            stages.append(part)
                    
                    if stages:
                        company_stage = ", ".join(stages)
                    if counts:
                        employee_count = ", ".join(counts)
                    
                    # If we only found employee counts and no stage, default stage is Growth (or empty?)
                    # For now keep default "Growth" if no specific stage found, unless it was just overwritten
                    if not stages and not counts:
                         company_stage = stage_text # Fallback to full text if parsing failed
            
            # Verify that we're using the cleaned/formatted values:
            # - salary_range: Robustly cleaned from list format (e.g., [206000, 310000] -> "$206,000 - $310,000")
            # - required_skills: ONLY from cached data (tags, categories, departments) - no description extraction
            # - industry: All sectors joined with ", " (e.g., "American Dynamism, Enterprise, Marketing")
            # - remote_option: Explicitly True if location contains "Hybrid"
            # - location: STRICTLY from cached API data only
            # - All fields enriched even if full_description is "Parsing Error"
            vacancy = Vacancy(
                title=title,
                company_name=company_name,
                company_stage=company_stage,
                location=location,  # STRICTLY from cached metadata only
                industry=industry,  # All sectors joined with ", "
                category=None,  # Will be set by AI classification
                experience_level=None,  # Will be set by AI classification
                remote_option=remote_option,  # STRICTLY prioritized from cached metadata
                salary_range=salary_range,  # Cleaned and formatted from cached metadata (list/dict -> string)
                description_url=url,
                required_skills=required_skills,  # ONLY from cached data (tags, categories, departments)
                full_description=description,  # May be "Parsing Error" but other fields still enriched
                employee_count=employee_count, # New field
                raw_html_url=raw_html_url,  # URL to HTML in Azure Blob Storage
            )
            
            # Set temporary is_hybrid attribute
            vacancy.is_hybrid = is_hybrid
            
            # Log metadata usage for debugging - show that cached data was used even if description failed
            if description == "Parsing Error":
                if cached:
                    logger.info(
                        f"Extracted vacancy with cached metadata (description failed): {title} at {company_name}. "
                        f"Tags: {len(required_skills)}, Salary: {salary_range}, Location: {location}, "
                        f"Remote: {remote_option}"
                    )
                else:
                    logger.warning(
                        f"Extracted vacancy with failed description and no cached metadata: {title} at {company_name}"
                    )
            else:
                logger.info(
                    f"Extracted vacancy: {title} at {company_name}. "
                    f"Tags: {len(required_skills)}, Salary: {salary_range}"
                )
            
            return vacancy

