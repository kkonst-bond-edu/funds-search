"""
A16Z scraper provider implementation.
"""

import logging
import re
from typing import List, Any, Optional, Dict
from bs4 import BeautifulSoup
from src.schemas.vacancy import Vacancy, CompanyStage
from src.services.scraper.core.base import BaseScraper

logger = logging.getLogger(__name__)

def extract_company_from_meta_tags(soup: BeautifulSoup) -> str:
    """Извлекает имя компании из мета-тегов."""
    meta_og = soup.find("meta", property="og:site_name")
    if meta_og and meta_og.get("content"):
        return meta_og.get("content").strip()
    return ""

def clean_noise_from_soup(soup: BeautifulSoup):
    """Удаляет кнопки и уведомления о куках."""
    noise_selectors = [
        "script", "style", "button", "iframe", "nav", "footer", 
        ".cookie-notice", ".privacy-notice", ".apply-button"
    ]
    for selector in noise_selectors:
        for element in soup.select(selector):
            element.decompose()
    return soup

def clean_job_title(title: str, company_name: str = "") -> str:
    if not title: return ""
    cleaned = title.strip()
    if company_name:
        pattern = re.compile(rf"^{re.escape(company_name)}\s*[:\-\|]\s*", re.IGNORECASE)
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r'\s*[\(\[].*?[\)\]]', '', cleaned)
    return cleaned.strip()

def extract_quality_description(soup: BeautifulSoup) -> str:
    """Извлекает очищенное описание вакансии."""
    soup = clean_noise_from_soup(soup)
    description_selectors = [".posting-description", "#content", ".job-description", ".description"]
    for selector in description_selectors:
        elem = soup.select_one(selector)
        if elem:
            return elem.get_text(separator="\n", strip=True)
    return soup.get_text(separator="\n", strip=True)

class A16ZScraper(BaseScraper):
    async def parse_vacancy(self, url: str, cached: Optional[Dict[str, Any]] = None) -> Optional[Vacancy]:
        async with self.browser_manager.get_page() as page:
            try:
                await page.goto(url, wait_until="networkidle", timeout=60000)
                await self.browser_manager.wait_for_job_content(page)
                content = await page.content()
                soup = BeautifulSoup(content, "html.parser")

                # Определение компании
                company_name = cached.get("company_name") if cached else ""
                if not company_name:
                    company_name = extract_company_from_meta_tags(soup)

                # Извлечение заголовка
                title_elem = soup.select_one(".posting-headline h2, .app-title, h1")
                if title_elem:
                    title = clean_job_title(title_elem.get_text(strip=True), company_name)
                else:
                    title = clean_job_title(soup.title.string if soup.title else "", company_name)
                
                if title.lower() == company_name.lower() and cached:
                    title = cached.get("text", title)

                # Извлечение локации
                location = cached.get("location", "Not specified") if cached else "Not specified"
                if location == "Not specified":
                    loc_elem = soup.select_one(".location, .posting-categories .location")
                    if loc_elem:
                        location = loc_elem.get_text(strip=True)

                # Описание
                description = extract_quality_description(soup)
                description = description.split("Apply for this job")[0].strip()
                
                remote_option = cached.get("remote_option", False) if cached else "remote" in location.lower()

                return Vacancy(
                    title=title,
                    company_name=company_name,
                    company_stage=CompanyStage.GROWTH,
                    location=location,
                    industry=cached.get("industry", "Not specified") if cached else "Not specified",
                    category=cached.get("category", "Not specified") if cached else "Not specified",
                    experience_level="Lead/Manager" if "Manager" in title else "Senior",
                    remote_option=remote_option,
                    salary_range=cached.get("salary_range", "Not specified") if cached else "Not specified",
                    description_url=url,
                    required_skills=cached.get("required_skills", []) if cached else [],
                    full_description=description
                )
            except Exception as e:
                logger.error(f"Error parsing {url}: {str(e)}")
                return None