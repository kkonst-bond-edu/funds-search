"""
Pinecone client for vector similarity search.
"""
import os
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from shared.schemas import Job, Resume


class PineconeClient:
    """Client for interacting with Pinecone vector database."""
    
    def __init__(self, api_key: Optional[str] = None, index_name: str = "funds-search"):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.index_name = index_name
        self.pc = Pinecone(api_key=self.api_key)
        self.index = None
        self._ensure_index()
    
    def _ensure_index(self):
        """Ensure the index exists, create if it doesn't."""
        if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
            # Create index with 1024 dimensions (BGE-M3 output dimension)
            self.pc.create_index(
                name=self.index_name,
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        
        self.index = self.pc.Index(self.index_name)
    
    def upsert_job(self, job: Job, vector: List[float]) -> None:
        """
        Upsert a job with its embedding vector.
        
        Args:
            job: Job object
            vector: Embedding vector
        """
        metadata = {
            "url": job.url,
            "company": job.company,
            "text": job.text[:1000],  # Limit metadata size
            "title": job.title or "",
            "location": job.location or "",
            "remote": job.remote or False,
        }
        
        self.index.upsert(
            vectors=[{
                "id": f"job_{hash(job.url)}",
                "values": vector,
                "metadata": metadata
            }]
        )
    
    def upsert_resume(self, resume: Resume, vector: List[float]) -> None:
        """
        Upsert a resume with its embedding vector.
        
        Args:
            resume: Resume object
            vector: Embedding vector
        """
        metadata = {
            "user_id": resume.user_id,
            "text": resume.text[:1000],  # Limit metadata size
        }
        
        self.index.upsert(
            vectors=[{
                "id": f"resume_{resume.user_id}",
                "values": vector,
                "metadata": metadata
            }]
        )
    
    def search_similar(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[dict] = None
    ) -> List[dict]:
        """
        Search for similar vectors using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            
        Returns:
            List of search results with metadata and scores
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
        ]

