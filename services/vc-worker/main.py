"""
VC Worker service for scraping and parsing job postings from URLs.
Uses Docling for URL parsing and content extraction.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
import httpx


app = FastAPI(title="VC Worker", description="Scrape and parse VC job postings")


class ScrapeRequest(BaseModel):
    """Request model for URL scraping."""
    url: str


class ScrapeResponse(BaseModel):
    """Response model with scraped content."""
    text: str
    url: str
    format: str


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape_url(request: ScrapeRequest):
    """
    Scrape a URL and extract text content using Docling.
    
    Args:
        request: ScrapeRequest with URL
        
    Returns:
        ScrapeResponse with extracted text and metadata
    """
    try:
        # Fetch URL content
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(request.url)
            response.raise_for_status()
            content = response.content
        
        # Determine format (try HTML first, then PDF)
        converter = DocumentConverter()
        
        # Try to convert as HTML/web page
        try:
            result = converter.convert(request.url, target_format=InputFormat.HTML)
            text = result.document.export_to_markdown()
            format_type = "html"
        except Exception:
            # Fallback to PDF if HTML fails
            try:
                result = converter.convert(content, target_format=InputFormat.PDF)
                text = result.document.export_to_markdown()
                format_type = "pdf"
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unable to parse URL content: {str(e)}"
                )
        
        return ScrapeResponse(
            text=text,
            url=request.url,
            format=format_type
        )
    
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fetching URL: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error scraping URL: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

