import os
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from shared.schemas import Resume, DocumentChunk, Vacancy
import numpy as np

class VectorStore:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is required")
        self.pc = Pinecone(api_key=api_key)
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "funds-search")
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

    def upsert_vacancy(self, vacancy: Vacancy):
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
        
        # Pinecone рекомендует загружать пачками (batching)
        self.index.upsert(vectors=vectors, namespace="resumes")

    def get_candidate_embedding(self, candidate_id: str) -> Optional[List[float]]:
        """
        Получает эмбеддинг кандидата из Pinecone.
        Извлекает все чанки резюме кандидата и возвращает средний эмбеддинг.
        
        Args:
            candidate_id: user_id кандидата
            
        Returns:
            Средний эмбеддинг всех чанков резюме или None, если кандидат не найден
        """
        # BGE-M3 uses 1024 dimensions
        EMBEDDING_DIM = 1024
        
        # Ищем все векторы с user_id = candidate_id
        # Используем dummy vector для query (фильтр сделает основную работу)
        query_result = self.index.query(
            vector=[0.0] * EMBEDDING_DIM,  # Dummy vector для BGE-M3 (1024 dims)
            top_k=10000,  # Большое число, чтобы получить все чанки
            include_metadata=True,
            filter={"user_id": {"$eq": candidate_id}},
            namespace="resumes"
        )
        
        if not query_result.matches:
            return None
        
        # Извлекаем IDs для fetch
        vector_ids = [match.id for match in query_result.matches]
        if not vector_ids:
            return None
        
        # Fetch vectors by IDs to get actual embedding values
        fetch_result = self.index.fetch(ids=vector_ids, namespace="resumes")
        
        embeddings = []
        for vector_id, vector_data in fetch_result.vectors.items():
            if 'values' in vector_data:
                embeddings.append(vector_data['values'])
        
        if not embeddings:
            return None
        
        # Вычисляем средний эмбеддинг и нормализуем
        avg_embedding = np.mean(embeddings, axis=0)
        # Нормализуем для cosine similarity
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        return avg_embedding.tolist()

    def search_vacancies(self, query_vector: List[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Ищет вакансии в Pinecone с фильтром type='vacancy'.
        
        Args:
            query_vector: Вектор запроса для поиска
            top_k: Количество топ результатов
            
        Returns:
            Список словарей с metadata и score
        """
        query_result = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter={"type": {"$eq": "vacancy"}},
            namespace="resumes"
        )
        
        # Convert Pinecone query result to list of dicts with metadata and score
        results = []
        for match in query_result.matches:
            results.append({
                "id": match.id,
                "metadata": match.metadata or {},
                "score": match.score or 0.0
            })
        return results

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