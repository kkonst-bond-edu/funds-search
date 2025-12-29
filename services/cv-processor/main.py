"""
CV/Resume processor using Docling for PDF/Docx parsing.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import io


app = FastAPI(title="CV Processor", description="Process CVs and resumes using Docling")


class ProcessResponse(BaseModel):
    """Response model for processed CV."""
    text: str
    format: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/process", response_model=ProcessResponse)
async def process_cv(file: UploadFile = File(...)):
    """
    Process a CV/resume file (PDF or DOCX) and extract text as Markdown.
    
    Args:
        file: Uploaded file (PDF or DOCX)
        
    Returns:
        ProcessResponse with extracted text and format
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
        
        # Create a file-like object
        file_obj = io.BytesIO(content)
        
        # Convert document
        result = converter.convert(file_obj, target_format=file_format)
        
        # Extract markdown text
        text = result.document.export_to_markdown()
        
        return ProcessResponse(
            text=text,
            format=file_format.value
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing CV: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

