"""
Script to remove duplicate vacancies from Pinecone.
Identifies duplicates by description_url and keeps only the most recent one.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from shared.pinecone_client import VectorStore
from src.schemas.vacancy import Vacancy
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

# Try to load .env from current directory or project root
load_dotenv()
# Explicitly try project root if not found
project_root = Path(__file__).parent.parent.parent
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def canonicalize_url(url: str) -> str:
    """
    Canonicalize URL by removing tracking parameters and sorting query params.
    
    Args:
        url: Input URL
        
    Returns:
        Canonicalized URL
    """
    if not url:
        return ""
    
    try:
        from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
        
        parsed = urlparse(url)
        
        # known tracking parameters to remove
        TRACKING_PARAMS = {
            'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
            'ref', 'source', 'gh_src', 'gh_ref', 's', 'fbclid', 'gclid', 
            '_hsenc', '_hsmi', 'mc_cid', 'mc_eid'
        }
        
        # Parse query parameters
        query_params = parse_qsl(parsed.query, keep_blank_values=True)
        
        # Filter and sort parameters
        filtered_params = [
            (k, v) for k, v in query_params 
            if k.lower() not in TRACKING_PARAMS
        ]
        filtered_params.sort()
        
        # Rebuild query string
        new_query = urlencode(filtered_params)
        
        # Rebuild URL
        new_parts = list(parsed)
        new_parts[4] = new_query
        
        return urlunparse(new_parts)
    except Exception as e:
        logger.warning(f"Failed to canonicalize URL {url}: {e}")
        return url

def find_duplicates(vector_store: VectorStore, namespace: str = "vacancies") -> Dict[str, List[str]]:
    """
    Find duplicate vacancies by description_url.
    
    Returns:
        Dictionary mapping description_url to list of vector IDs
    """
    logger.info("Scanning Pinecone for duplicates...")
    
    # Query all vectors (use a dummy vector with top_k=10000 to get all)
    # Note: Pinecone has limits, so we might need to paginate
    dummy_vector = [0.0] * 1024  # BGE-M3 uses 1024 dimensions
    
    # Get all vectors - Pinecone allows up to 10000 in one query
    try:
        results = vector_store.index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            namespace=namespace
        )
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        return {}
    
    total_vectors = len(results.matches)
    print(f"DEBUG: Total vectors found in Pinecone: {total_vectors}")
    logger.info(f"Total vectors found in Pinecone: {total_vectors}")

    # Group by (Title, Company, Location) tuple to find content duplicates
    # normalizing case and stripping whitespace
    content_to_ids: Dict[tuple, List[tuple]] = defaultdict(list)  # Store (id, url) tuples
    missing_content_count = 0
    
    for match in results.matches:
        metadata = match.metadata or {}
        title = str(metadata.get("title", "")).strip().lower()
        company = str(metadata.get("company_name", "")).strip().lower()
        location = str(metadata.get("location", "")).strip().lower()
        vector_id = match.id
        url = str(metadata.get("description_url", ""))
        
        if title and company:
            # Create a content key
            key = (title, company, location)
            content_to_ids[key].append((vector_id, url))
        else:
            missing_content_count += 1
            
    logger.info(f"Vectors with missing title or company: {missing_content_count}")
    
    # Filter to only duplicates (more than 1 ID per content key)
    duplicates = {}
    for key, items in content_to_ids.items():
        if len(items) > 1:
            # Store as string key with readable format
            title, company, location = key
            key_str = f"Title: '{title}' | Company: '{company}' | Location: '{location}'"
            duplicates[key_str] = items
    
    logger.info(f"Found {len(duplicates)} content groups with duplicates")
    total_duplicate_vectors = sum(len(ids) - 1 for ids in duplicates.values())
    logger.info(f"Total duplicate vectors to remove: {total_duplicate_vectors}")
    
    return duplicates


def remove_duplicates(vector_store: VectorStore, namespace: str = "vacancies", dry_run: bool = True) -> int:
    """
    Remove duplicate vacancies, keeping the most recent one (by vector ID or metadata).
    
    Args:
        vector_store: VectorStore instance
        namespace: Pinecone namespace
        dry_run: If True, only log what would be deleted without actually deleting
        
    Returns:
        Number of duplicates removed
    """
    duplicates = find_duplicates(vector_store, namespace)
    
    if not duplicates:
        logger.info("No duplicates found!")
        return 0
    
    removed_count = 0
    ids_to_delete = []
    
    for content_key, items in duplicates.items():
        # items is list of (vector_id, url) tuples
        # Keep the first one and delete the rest
        ids_to_keep, url_to_keep = items[0]
        ids_to_remove = [item[0] for item in items[1:]]
        urls_to_remove = [item[1] for item in items[1:]]
        
        logger.info(f"Content: {content_key}")
        logger.info(f"  Keeping: {ids_to_keep} (URL: {url_to_keep})")
        logger.info(f"  Removing: {ids_to_remove} (URLs: {urls_to_remove})")
        
        ids_to_delete.extend(ids_to_remove)
        removed_count += len(ids_to_remove)
    
    if dry_run:
        logger.info(f"DRY RUN: Would delete {removed_count} duplicate vectors")
        return removed_count
    
    # Delete duplicates in batches
    if ids_to_delete:
        logger.info(f"Deleting {len(ids_to_delete)} duplicate vectors...")
        try:
            # Pinecone delete accepts a list of IDs
            vector_store.index.delete(ids=ids_to_delete, namespace=namespace)
            logger.info(f"Successfully deleted {removed_count} duplicate vectors")
        except Exception as e:
            logger.error(f"Error deleting duplicates: {e}")
            return 0
    
    return removed_count


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Remove duplicate vacancies from Pinecone")
    parser.add_argument("--dry-run", action="store_true", help="Only show what would be deleted")
    parser.add_argument("--namespace", default="vacancies", help="Pinecone namespace")
    
    args = parser.parse_args()
    
    vector_store = VectorStore()
    removed = remove_duplicates(vector_store, namespace=args.namespace, dry_run=args.dry_run)
    
    if args.dry_run:
        print(f"\nDRY RUN: Would remove {removed} duplicate vectors")
        print("Run without --dry-run to actually delete them")
    else:
        print(f"\nRemoved {removed} duplicate vectors")
