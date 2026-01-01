import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from shared.schemas import Resume, DocumentChunk

class VectorStore:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.index = self.pc.Index(self.index_name)

    def upsert_resume(self, resume: Resume):
        """Сохраняет все чанки резюме в Pinecone."""
        vectors = []
        for i, chunk in enumerate(resume.chunks):
            vectors.append({
                "id": f"{resume.user_id}_{resume.id}_chunk_{i}",
                "values": chunk.embedding,
                "metadata": {
                    "user_id": resume.user_id,
                    "resume_id": resume.id,
                    "text": chunk.text,
                    **chunk.metadata
                }
            })
        
        # Pinecone рекомендует загружать пачками (batching)
        self.index.upsert(vectors=vectors, namespace="resumes")

    def search_similar(self, query_vector: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None):
        """Поиск похожих документов."""
        query_result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict if filter_dict else None,
            namespace="resumes"
        )
        # Convert Pinecone query result to list of dicts with metadata and score
        results = []
        for match in query_result.matches:
            results.append({
                "metadata": match.metadata or {},
                "score": match.score or 0.0
            })
        return results