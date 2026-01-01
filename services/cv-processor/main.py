import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import uuid
import httpx
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from docling.document_converter import DocumentConverter
from shared.schemas import Resume, DocumentChunk, Vacancy
from pydantic import BaseModel, Field
from shared.pinecone_client import VectorStore

app = FastAPI(title="CV Processor Service")
doc_converter = DocumentConverter()
vector_store = VectorStore()

EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")

async def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Вызов вашего сервиса на Azure."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{EMBEDDING_SERVICE_URL}/embed",
                json={"texts": texts}
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Embedding service unavailable: {str(e)}")

@app.post("/process-cv")
async def process_cv(user_id: str, file: UploadFile = File(...)):
    # 1. Сохраняем временный файл
    temp_path = f"temp_{uuid.uuid4()}.pdf"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    try:
        # 2. Docling: PDF -> Markdown (run in thread pool to avoid blocking event loop)
        result = await run_in_threadpool(doc_converter.convert, temp_path)
        markdown_text = result.document.export_to_markdown()

        # 3. Разбиваем на чанки (упрощенно)
        # В идеале здесь использовать LangChain RecursiveCharacterTextSplitter
        chunks_text = [markdown_text[i:i+1000] for i in range(0, len(markdown_text), 800)]
        
        # 4. Получаем эмбеддинги
        embeddings = await get_embeddings(chunks_text)

        # 5. Собираем объект Resume
        doc_chunks = [
            DocumentChunk(text=txt, embedding=emb, metadata={"source": file.filename})
            for txt, emb in zip(chunks_text, embeddings)
        ]
        
        resume = Resume(
            id=str(uuid.uuid4()),
            user_id=user_id,
            raw_text=markdown_text,
            chunks=doc_chunks
        )

        # 6. Сохраняем в Pinecone
        vector_store.upsert_resume(resume)

        return {"status": "success", "resume_id": resume.id, "chunks_processed": len(doc_chunks)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


class VacancyRequest(BaseModel):
    """Request schema for processing a vacancy."""
    text: str = Field(..., description="Text description of the vacancy")


@app.post("/process-vacancy")
async def process_vacancy(request: VacancyRequest):
    """
    Process a vacancy description: embed it and save to Pinecone with metadata {'type': 'vacancy'}.
    
    Args:
        request: VacancyRequest containing the text description
        
    Returns:
        Dictionary with status, vacancy_id, and chunks_processed
    """
    vacancy_text = request.text
    
    # 1. Разбиваем на чанки (упрощенно)
    # В идеале здесь использовать LangChain RecursiveCharacterTextSplitter
    chunks_text = [vacancy_text[i:i+1000] for i in range(0, len(vacancy_text), 800)]
    
    # 2. Получаем эмбеддинги
    embeddings = await get_embeddings(chunks_text)
    
    # 3. Собираем объект Vacancy с metadata type='vacancy'
    doc_chunks = [
        DocumentChunk(
            text=txt, 
            embedding=emb, 
            metadata={"type": "vacancy", "source": "api"}
        )
        for txt, emb in zip(chunks_text, embeddings)
    ]
    
    vacancy = Vacancy(
        id=str(uuid.uuid4()),
        raw_text=vacancy_text,
        chunks=doc_chunks
    )
    
    # 4. Сохраняем в Pinecone
    vector_store.upsert_vacancy(vacancy)
    
    return {
        "status": "success", 
        "vacancy_id": vacancy.id, 
        "chunks_processed": len(doc_chunks)
    }