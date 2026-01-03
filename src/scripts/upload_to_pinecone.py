"""
Script to vectorize vacancies and upload them to Pinecone.
Loads vacancies from data/vacancies_dump.json, generates embeddings via embedding-service,
and uploads to Pinecone in batches.
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import httpx
import structlog
from shared.pinecone_client import VectorStore

# Configure logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger("INFO"),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


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


def generate_search_text(vacancy: Dict[str, Any]) -> str:
    """
    Generate combined search text for embedding from vacancy data.
    
    Args:
        vacancy: Vacancy dictionary
        
    Returns:
        Combined search text string
    """
    title = vacancy.get("title", "")
    company_name = vacancy.get("company_name", "")
    industry = vacancy.get("industry", "")
    required_skills = vacancy.get("required_skills", [])
    location = vacancy.get("location", "")
    
    skills_str = ", ".join(required_skills) if required_skills else ""
    
    search_text = (
        f"Title: {title}. "
        f"Company: {company_name}. "
        f"Industry: {industry}. "
        f"Skills: {skills_str}. "
        f"Location: {location}."
    )
    
    return search_text


async def get_embeddings(texts: List[str], embedding_service_url: str) -> List[List[float]]:
    """
    Call embedding-service to generate embeddings for texts.
    
    Args:
        texts: List of text strings to embed
        embedding_service_url: URL of the embedding service
        
    Returns:
        List of embedding vectors
        
    Raises:
        httpx.HTTPError: If the embedding service is unreachable
        ValueError: If the response format is invalid
    """
    async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minutes timeout
        try:
            logger.info("calling_embedding_service", url=embedding_service_url, text_count=len(texts))
            response = await client.post(
                f"{embedding_service_url}/embed",
                json={"texts": texts}
            )
            response.raise_for_status()
            result = response.json()
            
            if "embeddings" not in result:
                raise ValueError(f"Invalid response format from embedding service: missing 'embeddings' key")
            
            embeddings = result["embeddings"]
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
                )
            
            logger.info("embeddings_generated", count=len(embeddings), dim=len(embeddings[0]) if embeddings else 0)
            return embeddings
            
        except httpx.TimeoutException as e:
            logger.error("embedding_service_timeout", error=str(e))
            raise
        except httpx.HTTPStatusError as e:
            logger.error(
                "embedding_service_http_error",
                status_code=e.response.status_code,
                error=str(e)
            )
            raise
        except httpx.RequestError as e:
            logger.error("embedding_service_unreachable", error=str(e))
            raise httpx.HTTPError(f"Embedding service unreachable: {str(e)}") from e


def load_vacancies(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load vacancies from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of vacancy dictionaries
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Vacancies file not found: {file_path}")
    
    logger.info("loading_vacancies", file_path=str(file_path))
    with open(file_path, "r", encoding="utf-8") as f:
        vacancies = json.load(f)
    
    if not isinstance(vacancies, list):
        raise ValueError(f"Expected list of vacancies, got {type(vacancies).__name__}")
    
    logger.info("vacancies_loaded", count=len(vacancies))
    return vacancies


async def upload_vacancies_to_pinecone(
    vacancies: List[Dict[str, Any]],
    embedding_service_url: str,
    batch_size: int = 10
) -> Tuple[int, int]:
    """
    Upload vacancies to Pinecone in batches.
    Skips vacancies with 'Unknown' title or company_name.
    
    Args:
        vacancies: List of vacancy dictionaries
        embedding_service_url: URL of the embedding service
        batch_size: Number of vacancies to process per batch
        
    Returns:
        Tuple of (uploaded_count, skipped_count)
    """
    """
    Upload vacancies to Pinecone in batches.
    Skips vacancies with 'Unknown' title or company_name.
    
    Args:
        vacancies: List of vacancy dictionaries
        embedding_service_url: URL of the embedding service
        batch_size: Number of vacancies to process per batch
    """
    # Initialize Pinecone client
    try:
        vector_store = VectorStore()
        logger.info("pinecone_client_initialized", index_name=vector_store.index_name)
    except Exception as e:
        logger.error("pinecone_initialization_failed", error=str(e))
        raise
    
    # Clear existing data from Pinecone index
    try:
        logger.info("clearing_existing_vacancies", namespace="vacancies")
        vector_store.delete_all(namespace="vacancies")
        logger.info("existing_vacancies_cleared")
    except Exception as e:
        logger.error("failed_to_clear_vacancies", error=str(e))
        raise
    
    # Filter out 'Unknown' vacancies before processing
    valid_vacancies = []
    skipped_count = 0
    for vacancy in vacancies:
        title = vacancy.get("title", "")
        company_name = vacancy.get("company_name", "")
        
        if title == "Unknown" or company_name == "Unknown":
            skipped_count += 1
            logger.warning(
                "skipping_unknown_vacancy",
                title=title,
                company_name=company_name,
                vacancy_id=generate_vacancy_id(company_name, title)
            )
        else:
            valid_vacancies.append(vacancy)
    
    if skipped_count > 0:
        logger.info("vacancies_filtered", skipped=skipped_count, valid=len(valid_vacancies))
    
    total_vacancies = len(valid_vacancies)
    logger.info("upload_started", total_vacancies=total_vacancies, batch_size=batch_size)
    
    # Process in batches
    for batch_start in range(0, total_vacancies, batch_size):
        batch_end = min(batch_start + batch_size, total_vacancies)
        batch = valid_vacancies[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_vacancies + batch_size - 1) // batch_size
        
        logger.info(
            "processing_batch",
            batch_num=batch_num,
            total_batches=total_batches,
            batch_start=batch_start,
            batch_end=batch_end,
            batch_size=len(batch)
        )
        
        try:
            # Generate search texts for this batch
            search_texts = [generate_search_text(vacancy) for vacancy in batch]
            
            # Get embeddings from embedding-service
            embeddings = await get_embeddings(search_texts, embedding_service_url)
            
            # Prepare vectors for Pinecone
            vectors = []
            for vacancy, embedding in zip(batch, embeddings):
                vacancy_id = generate_vacancy_id(
                    vacancy.get("company_name", "unknown"),
                    vacancy.get("title", "unknown")
                )
                
                # Filter out None/null values from metadata (Pinecone doesn't accept null values)
                metadata = {k: v for k, v in vacancy.items() if v is not None}
                
                # Store full vacancy dictionary in metadata (without null values)
                vector = {
                    "id": vacancy_id,
                    "values": embedding,
                    "metadata": metadata
                }
                vectors.append(vector)
            
            # Upload to Pinecone
            logger.info("uploading_batch_to_pinecone", batch_num=batch_num, vector_count=len(vectors))
            vector_store.upsert(vectors=vectors, namespace="vacancies")
            logger.info("batch_uploaded", batch_num=batch_num, vector_count=len(vectors))
            
        except Exception as e:
            logger.error(
                "batch_upload_failed",
                batch_num=batch_num,
                error=str(e),
                error_type=type(e).__name__
            )
            # Continue with next batch instead of failing completely
            continue
    
    logger.info("upload_completed", total_vacancies=total_vacancies, skipped=skipped_count)
    return (total_vacancies, skipped_count)


async def main_async():
    """Main async function."""
    # Get configuration from environment
    embedding_service_url = os.getenv(
        "EMBEDDING_SERVICE_URL",
        "http://embedding-service:8001"
    )
    
    # Determine file path - check data/ first, then root
    data_file = project_root / "data" / "vacancies_dump.json"
    root_file = project_root / "vacancies_dump.json"
    
    if data_file.exists():
        vacancies_file = data_file
    elif root_file.exists():
        vacancies_file = root_file
        logger.warning("using_root_vacancies_file", path=str(vacancies_file))
    else:
        logger.error("vacancies_file_not_found", data_path=str(data_file), root_path=str(root_file))
        print(f"\n❌ Error: Vacancies file not found in {data_file} or {root_file}")
        sys.exit(1)
    
    logger.info("starting_upload", vacancies_file=str(vacancies_file), embedding_service_url=embedding_service_url)
    
    try:
        # Load vacancies
        vacancies = load_vacancies(vacancies_file)
        
        if not vacancies:
            logger.warning("no_vacancies_found")
            print("\n⚠️  No vacancies found in file")
            return
        
        # Upload to Pinecone
        uploaded_count, skipped_count = await upload_vacancies_to_pinecone(vacancies, embedding_service_url, batch_size=10)
        
        if skipped_count > 0:
            print(f"\n✅ Successfully uploaded {uploaded_count} vacancies to Pinecone (skipped {skipped_count} with 'Unknown' title/company)")
        else:
            print(f"\n✅ Successfully uploaded {uploaded_count} vacancies to Pinecone")
        
    except FileNotFoundError as e:
        logger.error("file_not_found", error=str(e))
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("json_decode_error", error=str(e))
        print(f"\n❌ Error: Invalid JSON in vacancies file: {str(e)}")
        sys.exit(1)
    except httpx.HTTPError as e:
        logger.error("embedding_service_error", error=str(e))
        print(f"\n❌ Error: Embedding service unavailable: {str(e)}")
        print("   Make sure the embedding-service is running and accessible.")
        sys.exit(1)
    except Exception as e:
        logger.error("upload_failed", error=str(e), error_type=type(e).__name__)
        print(f"\n❌ Upload failed: {str(e)}")
        sys.exit(1)


def main():
    """Main entry point - runs async function."""
    import asyncio
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

