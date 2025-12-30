"""
CV/Resume processor using Docling for PDF/Docx parsing.
Processes CVs, splits into chunks, generates embeddings, and stores in Pinecone.
"""
import os
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional, List
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from langchain.text_splitter import RecursiveCharacterTextSplitter
import httpx
import io

from shared.schemas import Resume, DocumentChunk
from shared.pinecone_client import VectorStore


app = FastAPI(title="CV Processor", description="Process CVs and resumes using Docling")

# Initialize text splitter for semantic chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Get embedding service URL from environment
EMBEDDING_SERVICE_URL = os.getenv("EMBEDDING_SERVICE_URL", "http://embedding-service:8001")


class ProcessResponse(BaseModel):
    """Response model for processed CV."""
    resume_id: str
    user_id: str
    chunks_count: int
    status: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


async def get_embeddings(texts: List[str], embedding_service_url: str) -> List[List[float]]:
    """
    Call the embedding service to get embeddings for texts.
    
    Args:
        texts: List of text strings to embed
        embedding_service_url: URL of the embedding service
        
    Returns:
        List of embedding vectors
        
    Raises:
        HTTPException: If embedding service is unavailable or returns an error
    """
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{embedding_service_url}/embed",
                json={"texts": texts}
            )
            response.raise_for_status()
            result = response.json()
            return result["embeddings"]
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=503,
            detail="Embedding service timeout. Please try again later."
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding service unavailable: {str(e)}"
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Embedding service error: {e.response.text}"
        )


@app.post("/process-cv", response_model=ProcessResponse)
async def process_cv(
    file: UploadFile = File(...),
    user_id: str = Form(...)
):
    """
    Process a CV/resume file (PDF or DOCX):
    1. Convert to Markdown using Docling
    2. Split into semantic chunks
    3. Generate embeddings for each chunk via embedding-service
    4. Store in Pinecone via VectorStore
    
    Args:
        file: Uploaded file (PDF or DOCX)
        user_id: User identifier
        
    Returns:
        ProcessResponse with resume_id, user_id, chunks_count, and status
    """
    try:
        # Read file content
        content = await file.read()
        
        # Determine format
        file_format = None
        if file.filename.endswith('.pdf'):
            file_format = InputFormat.PDF
        elif file.filename.endswith('.docx') or file.filename.endswith('.doc'):
            file_format = InputFormat.DOCX
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF or DOCX."
            )
        
        # Convert using Docling
        converter = DocumentConverter()
        file_obj = io.BytesIO(content)
        result = converter.convert(file_obj, target_format=file_format)
        
        # Extract markdown text
        raw_text = result.document.export_to_markdown()
        
        # Split into semantic chunks
        chunks_text = text_splitter.split_text(raw_text)
        
        if not chunks_text:
            raise HTTPException(
                status_code=400,
                detail="Could not extract text from the document."
            )
        
        # Generate embeddings for all chunks
        try:
            embeddings = await get_embeddings(chunks_text, EMBEDDING_SERVICE_URL)
        except HTTPException:
            # Re-raise HTTP exceptions from embedding service
            raise
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error generating embeddings: {str(e)}"
            )
        
        # Validate embeddings match chunks
        if len(embeddings) != len(chunks_text):
            raise HTTPException(
                status_code=500,
                detail=f"Embedding count mismatch: expected {len(chunks_text)}, got {len(embeddings)}"
            )
        
        # Create DocumentChunk objects
        document_chunks = []
        for idx, (chunk_text, embedding) in enumerate(zip(chunks_text, embeddings)):
            # Validate embedding dimension
            if len(embedding) != 1024:
                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid embedding dimension: expected 1024, got {len(embedding)}"
                )
            
            chunk = DocumentChunk(
                text=chunk_text,
                metadata={
                    "chunk_index": idx,
                    "total_chunks": len(chunks_text)
                },
                embedding=embedding
            )
            document_chunks.append(chunk)
        
        # Create Resume object
        resume_id = str(uuid.uuid4())
        resume = Resume(
            id=resume_id,
            user_id=user_id,
            raw_text=raw_text,
            chunks=document_chunks
        )
        
        # Store in Pinecone
        try:
            vector_store = VectorStore()
            vector_store.upsert_resume(resume)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error storing resume in Pinecone: {str(e)}"
            )
        
        return ProcessResponse(
            resume_id=resume_id,
            user_id=user_id,
            chunks_count=len(document_chunks),
            status="success"
        )
    
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CV: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

