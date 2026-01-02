"""Shared modules for funds-search system."""
from shared.schemas import Job, Resume, MatchResult, SearchRequest, DocumentChunk

# Lazy import of VectorStore to avoid requiring pinecone in all contexts
try:
    from shared.pinecone_client import VectorStore
    __all__ = ["Job", "Resume", "MatchResult", "SearchRequest", "DocumentChunk", "VectorStore"]
except ImportError:
    # VectorStore not available (e.g., in web-ui without pinecone)
    __all__ = ["Job", "Resume", "MatchResult", "SearchRequest", "DocumentChunk"]

