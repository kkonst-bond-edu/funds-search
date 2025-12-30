"""
Pinecone client for vector similarity search.
"""
import os
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from shared.schemas import Resume


class VectorStore:
    """Client for interacting with Pinecone vector database."""
    
    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        """
        Initialize Pinecone client.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index (defaults to PINECONE_INDEX_NAME env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME", "funds-search")
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
    
    def upsert_resume(self, resume: Resume) -> None:
        """
        Upsert a resume with its chunks into Pinecone.
        Each chunk is stored as a separate vector.
        
        Args:
            resume: Resume object with chunks
        """
        if not resume.chunks:
            raise ValueError("Resume must have at least one chunk")
        
        vectors = []
        for idx, chunk in enumerate(resume.chunks):
            metadata = {
                "resume_id": resume.id,
                "user_id": resume.user_id,
                "chunk_index": idx,
                "text": chunk.text[:1000],  # Limit metadata size
                "type": "resume",
                **chunk.metadata  # Include any additional chunk metadata
            }
            
            vectors.append({
                "id": f"resume_{resume.id}_chunk_{idx}",
                "values": chunk.embedding,
                "metadata": metadata
            })
        
        # Batch upsert all chunks
        self.index.upsert(vectors=vectors)
    
    def search_similar_resumes(
        self,
        query_vector: List[float],
        top_k: int = 10
    ) -> List[dict]:
        """
        Search for similar resume chunks using cosine similarity.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return (default: 10)
            
        Returns:
            List of search results with metadata and scores
        """
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"type": "resume"}  # Only search resume chunks
        )
        
        return [
            {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            }
            for match in results.matches
        ]

