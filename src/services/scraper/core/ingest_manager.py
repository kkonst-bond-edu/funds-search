"""
Ingest manager for processing vacancies and uploading to Pinecone.
"""

import os
import re
import logging
from typing import List, Optional
import httpx

from src.schemas.vacancy import Vacancy
from shared.pinecone_client import VectorStore

# Configure basic logger for console output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def slugify(text: str) -> str:
    """
    Convert text to a URL-safe slug.
    
    Args:
        text: Text to slugify
        
    Returns:
        URL-safe slug string
    """
    # Convert to lowercase
    text = text.lower()
    # Replace spaces and special characters with hyphens
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '-', text)
    # Remove leading/trailing hyphens
    text = text.strip('-')
    return text


def generate_vacancy_id(company_name: str, title: str) -> str:
    """
    Generate a unique slugified ID for a vacancy.
    
    Args:
        company_name: Company name
        title: Job title
        
    Returns:
        Slugified ID like "company-name-job-title"
    """
    company_slug = slugify(company_name)
    title_slug = slugify(title)
    return f"{company_slug}-{title_slug}"


def generate_search_text(vacancy: Vacancy) -> str:
    """
    Generate combined search text for embedding from vacancy data.
    
    Args:
        vacancy: Vacancy object
        
    Returns:
        Combined search text string
    """
    title = vacancy.title
    company_name = vacancy.company_name
    industry = vacancy.industry
    required_skills = vacancy.required_skills
    location = vacancy.location
    
    skills_str = ", ".join(required_skills) if required_skills else ""
    
    search_text = (
        f"Title: {title}. "
        f"Company: {company_name}. "
        f"Industry: {industry}. "
        f"Skills: {skills_str}. "
        f"Location: {location}."
    )
    
    return search_text


async def get_embedding(text: str, embedding_service_url: str) -> List[float]:
    """
    Call embedding-service to generate embedding for a single text.
    
    Args:
        text: Text string to embed
        embedding_service_url: URL of the embedding service
        
    Returns:
        Embedding vector as list of floats
        
    Raises:
        httpx.HTTPError: If the embedding service is unreachable
        ValueError: If the response format is invalid
    """
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes timeout
        try:
            logger.info(f"Calling embedding service: {embedding_service_url}")
            response = await client.post(
                f"{embedding_service_url}/embed",
                json={"texts": [text]}
            )
            response.raise_for_status()
            result = response.json()
            
            if "embeddings" not in result:
                raise ValueError("Invalid response format from embedding service: missing 'embeddings' key")
            
            embeddings = result["embeddings"]
            if not embeddings or len(embeddings) == 0:
                raise ValueError("Empty embeddings list returned from embedding service")
            
            embedding = embeddings[0]
            logger.info(f"Embedding generated: dim={len(embedding)}")
            return embedding
            
        except httpx.TimeoutException as e:
            logger.error(f"Embedding service timeout: {str(e)}")
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                f"Embedding service HTTP error: status_code={e.response.status_code}, error={str(e)}"
            )
            raise
        except httpx.RequestError as e:
            logger.error(f"Embedding service unreachable: {str(e)}")
            raise httpx.HTTPError(f"Embedding service unreachable: {str(e)}") from e


class IngestManager:
    """Manages ingestion of vacancies into Pinecone vector database."""

    def __init__(
        self,
        embedding_service_url: Optional[str] = None,
        pinecone_index_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the ingest manager.
        
        Args:
            embedding_service_url: URL of the embedding service.
                                 Defaults to EMBEDDING_SERVICE_URL env var or
                                 "http://embedding-service:8001"
            pinecone_index_name: Pinecone index name.
                                Defaults to PINECONE_INDEX_NAME env var or
                                "funds-search"
        """
        self.embedding_service_url = embedding_service_url or os.getenv(
            "EMBEDDING_SERVICE_URL", "http://embedding-service:8001"
        )
        
        # Get pinecone_index_name from parameter or environment variable
        self.pinecone_index_name = pinecone_index_name or os.getenv(
            "PINECONE_INDEX_NAME", None
        )
        
        # Initialize Pinecone client
        try:
            self.vector_store = VectorStore()
            if self.pinecone_index_name:
                self.vector_store.index_name = self.pinecone_index_name
            logger.info(f"Pinecone client initialized: index={self.vector_store.index_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone client: {str(e)}")
            raise

    def _validate_vacancy(self, vacancy: Vacancy) -> bool:
        """
        Validate vacancy data before processing.
        
        Args:
            vacancy: Vacancy object to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not vacancy.title or vacancy.title.strip() == "":
            logger.warning("Vacancy validation failed: empty title")
            return False
        
        if not vacancy.company_name or vacancy.company_name.strip() == "":
            logger.warning("Vacancy validation failed: empty company_name")
            return False
        
        if not vacancy.description_url or vacancy.description_url.strip() == "":
            logger.warning("Vacancy validation failed: empty description_url")
            return False
        
        # Check for 'Unknown' values
        if vacancy.title == "Unknown" or vacancy.company_name == "Unknown":
            logger.warning("Vacancy validation failed: 'Unknown' title or company_name")
            return False
        
        return True

    async def process_new_vacancy(self, vacancy: Vacancy, namespace: str = "vacancies") -> bool:
        """
        Process a new vacancy: validate, generate embedding, and upsert to Pinecone.
        
        Args:
            vacancy: Vacancy object to process
            namespace: Pinecone namespace to use (default: "vacancies")
            
        Returns:
            True if successfully processed, False otherwise
        """
        # Validate the vacancy
        if not self._validate_vacancy(vacancy):
            logger.error(f"Vacancy validation failed: {vacancy.title} at {vacancy.company_name}")
            return False
        
        try:
            # Generate search text for embedding
            search_text = generate_search_text(vacancy)
            logger.info(f"Generated search text for vacancy: {vacancy.title}")
            
            # Generate embedding
            embedding = await get_embedding(search_text, self.embedding_service_url)
            logger.info(f"Generated embedding for vacancy: {vacancy.title} (dim={len(embedding)})")
            
            # Generate vacancy ID
            vacancy_id = generate_vacancy_id(vacancy.company_name, vacancy.title)
            
            # Prepare metadata (filter out None values as Pinecone doesn't accept null)
            metadata = {
                "title": vacancy.title,
                "company_name": vacancy.company_name,
                "company_stage": vacancy.company_stage.value if hasattr(vacancy.company_stage, 'value') else str(vacancy.company_stage),
                "location": vacancy.location,
                "industry": vacancy.industry,
                "description_url": vacancy.description_url,
                "remote_option": vacancy.remote_option,
            }
            
            # Add optional fields if present
            if vacancy.salary_range:
                metadata["salary_range"] = vacancy.salary_range
            if vacancy.required_skills:
                metadata["required_skills"] = vacancy.required_skills
            if vacancy.source_url:
                metadata["source_url"] = vacancy.source_url
            
            # Prepare vector for Pinecone
            vector = {
                "id": vacancy_id,
                "values": embedding,
                "metadata": metadata,
            }
            
            # Upsert to Pinecone
            logger.info(f"Upserting vacancy to Pinecone: {vacancy_id} (namespace={namespace})")
            self.vector_store.upsert(vectors=[vector], namespace=namespace)
            logger.info(f"Successfully processed vacancy: {vacancy_id}")
            
            return True
            
        except httpx.HTTPError as e:
            logger.error(
                f"Failed to process vacancy {vacancy.title}: embedding service error - {str(e)}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to process vacancy {vacancy.title}: {str(e)} (error_type={type(e).__name__})",
                exc_info=True,
            )
            return False

