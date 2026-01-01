import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from shared.schemas import Resume, DocumentChunk, Vacancy
import numpy as np

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "funds-search")
        self.index = self.pc.Index(self.index_name)

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = "resumes"):
        """
        Generic upsert method that accepts vectors and namespace.
        
        Args:
            vectors: List of vector dictionaries with 'id', 'values', and 'metadata'
            namespace: Namespace to use (default: "resumes")
        """
        logger.info(f"Upserting {len(vectors)} vectors to namespace: {namespace}")
        self.index.upsert(vectors=vectors, namespace=namespace)
    
    def upsert_resume(self, resume: Resume, namespace: str = "cvs"):
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
        
        logger.info(f"Upserting resume {resume.id} for user {resume.user_id} to namespace: {namespace}")
        self.upsert(vectors=vectors, namespace=namespace)

    def upsert_vacancy(self, vacancy: Vacancy, namespace: str = "vacancies"):
        """Сохраняет все чанки вакансии в Pinecone с metadata type='vacancy'."""
        vectors = []
        for i, chunk in enumerate(vacancy.chunks):
            vectors.append({
                "id": f"vacancy_{vacancy.id}_chunk_{i}",
                "values": chunk.embedding,
                "metadata": {
                    "vacancy_id": vacancy.id,
                    "text": chunk.text,
                    **chunk.metadata  # This includes 'type': 'vacancy'
                }
            })
        
        logger.info(f"Upserting vacancy {vacancy.id} to namespace: {namespace}")
        self.upsert(vectors=vectors, namespace=namespace)

    def query(self, query_vector: List[float], top_k: int = 10, filter_dict: Optional[Dict[str, Any]] = None, namespace: str = "resumes", include_values: bool = False) -> List[Dict[str, Any]]:
        """
        Generic query method that accepts namespace parameter.
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filter
            namespace: Namespace to query (default: "resumes")
            include_values: Whether to include vector values in results (default: False)
            
        Returns:
            List of dictionaries with 'id', 'metadata', 'score', and optionally 'values'
        """
        logger.info(f"Querying namespace: {namespace} with top_k={top_k}, filter={filter_dict}, include_values={include_values}")
        query_result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            include_values=include_values,
            filter=filter_dict if filter_dict else None,
            namespace=namespace
        )
        
        # Convert Pinecone query result to list of dicts with metadata and score
        results = []
        for match in query_result.matches:
            result_dict = {
                "id": match.id,
                "metadata": match.metadata or {},
                "score": match.score or 0.0
            }
            if include_values and hasattr(match, 'values'):
                result_dict["values"] = match.values
            results.append(result_dict)
        logger.info(f"Query returned {len(results)} results from namespace: {namespace}")
        return results
    
    def get_candidate_embedding(self, candidate_id: str, namespace: str = "cvs") -> Optional[List[float]]:
        """
        Получает эмбеддинг кандидата из Pinecone.
        Извлекает 'values' из первого совпадения (first match).
        
        Args:
            candidate_id: user_id кандидата
            namespace: Namespace to query (default: "cvs")
            
        Returns:
            Эмбеддинг из первого совпадения или None, если кандидат не найден
        """
        # BGE-M3 uses 1024 dimensions
        EMBEDDING_DIM = 1024
        
        logger.info(f"Fetching candidate embedding for {candidate_id} from namespace: {namespace}")
        
        # Ищем все векторы с user_id = candidate_id
        # Используем dummy vector для query (фильтр сделает основную работу)
        # CRITICAL: include_values=True to get the embedding values
        query_result = self.index.query(
            vector=[0.0] * EMBEDDING_DIM,  # Dummy vector для BGE-M3 (1024 dims)
            top_k=1,  # Get only the first match
            include_metadata=True,
            include_values=True,  # CRITICAL: Include values to get embedding
            filter={"user_id": {"$eq": candidate_id}},
            namespace=namespace
        )
        
        if not query_result.matches:
            logger.warning(f"No matches found for candidate {candidate_id} in namespace: {namespace}")
            return None
        
        # Extract 'values' from the first match
        first_match = query_result.matches[0]
        if not hasattr(first_match, 'values') or first_match.values is None:
            logger.warning(f"First match for candidate {candidate_id} does not have values")
            return None
        
        # Get the embedding values from the first match
        candidate_embedding = first_match.values
        
        # Normalize for cosine similarity
        embedding_array = np.array(candidate_embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        logger.info(f"Extracted embedding for candidate {candidate_id} from first match (dim={len(embedding_array)})")
        return embedding_array.tolist()

    def search_vacancies(self, query_vector: List[float], top_k: int = 10, namespace: str = "vacancies") -> List[Dict[str, Any]]:
        """
        Ищет вакансии в Pinecone.
        
        Args:
            query_vector: Вектор запроса для поиска
            top_k: Количество топ результатов
            namespace: Namespace to query (default: "vacancies")
            
        Returns:
            Список словарей с metadata и score
        """
        logger.info(f"Searching vacancies in namespace: {namespace} with top_k={top_k}")
        return self.query(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=None,  # No filter needed if using separate namespace
            namespace=namespace
        )

    def search_similar(self, query_vector: List[float], top_k: int = 5, filter_dict: Optional[Dict[str, Any]] = None, namespace: str = "resumes"):
        """Поиск похожих документов."""
        results = self.query(
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=filter_dict,
            namespace=namespace
        )
        # Return results without 'id' field for backward compatibility
        return [{"metadata": r["metadata"], "score": r["score"]} for r in results]