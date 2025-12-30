"""
Embedding service using BAAI/bge-m3 model.
Uses FastAPI with Lifespan pattern to load the model once.
"""

from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

# Global model and tokenizer
model: AutoModel | None = None
tokenizer: AutoTokenizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:
    """Lifespan context manager to load model on startup."""
    global model, tokenizer

    print("Loading BGE-M3 model...")
    model_name = "BAAI/bge-m3"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Set to evaluation mode
    model.eval()

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    print(f"Model loaded on {device}")

    yield

    # Cleanup (if needed)
    print("Shutting down embedding service...")


app = FastAPI(
    title="Embedding Service",
    description="BGE-M3 embedding service for funds-search",
    lifespan=lifespan,
)


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation."""

    texts: list[str]


class EmbeddingResponse(BaseModel):
    """Response model with embeddings."""

    embeddings: list[list[float]]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts using BGE-M3.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors
    """
    global model, tokenizer

    if model is None or tokenizer is None:
        raise RuntimeError("Model not loaded")

    device = next(model.parameters()).device

    # Tokenize inputs
    encoded_input = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=8192,  # BGE-M3 supports up to 8192 tokens
        return_tensors="pt",
    )

    # Move to device
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

    # Generate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Use dense embeddings (last_hidden_state)
        embeddings = model_output.last_hidden_state[:, 0, :].cpu().numpy()

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)

    return embeddings.tolist()


@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/embed", response_model=EmbeddingResponse)
async def embed(request: EmbeddingRequest) -> EmbeddingResponse:
    """
    Generate embeddings for the provided texts.

    Args:
        request: EmbeddingRequest with list of texts

    Returns:
        EmbeddingResponse with list of embedding vectors
    """
    if not request.texts:
        return EmbeddingResponse(embeddings=[])

    try:
        embeddings = generate_embeddings(request.texts)
        return EmbeddingResponse(embeddings=embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
