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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    
    # Group by description_url
    url_to_ids: Dict[str, List[str]] = defaultdict(list)
    
    for match in results.matches:
        metadata = match.metadata or {}
        description_url = metadata.get("description_url", "")
        vector_id = match.id
        
        if description_url:
            url_to_ids[description_url].append(vector_id)
    
    # Filter to only duplicates (more than 1 ID per URL)
    duplicates = {url: ids for url, ids in url_to_ids.items() if len(ids) > 1}
    
    logger.info(f"Found {len(duplicates)} URLs with duplicates")
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
    
    for url, ids in duplicates.items():
        # Keep the first ID (or could sort by timestamp if available)
        # For now, keep the first one and delete the rest
        ids_to_keep = ids[0]
        ids_to_remove = ids[1:]
        
        logger.info(f"URL: {url}")
        logger.info(f"  Keeping: {ids_to_keep}")
        logger.info(f"  Removing: {ids_to_remove}")
        
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
