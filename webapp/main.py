from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

app = FastAPI()


class SearchRequest(BaseModel):
    query: str
    location: Optional[str] = None
    role: Optional[str] = None
    remote: Optional[bool] = None


@app.get('/health')
def health():
    return {"status": "ok"}


@app.post('/search')
def search(request: SearchRequest):
    """
    Search for job openings at VC funds.
    
    Accepts a JSON payload with:
    - query: search query string
    - location: optional location filter
    - role: optional role/job title filter
    - remote: optional boolean for remote positions
    """
    # Stub response for now
    return {
        "items": [],
        "note": "stub - wiring only"
    }
