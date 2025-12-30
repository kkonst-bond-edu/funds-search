"""
Unit and integration tests for embedding-service.

All tests use mocks to avoid loading the actual BGE-M3 model,
which requires significant RAM and is not available in CI environments.
"""
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import torch

# Import TestClient - will work with updated httpx version
from fastapi.testclient import TestClient

# Import the app and functions
import sys
import importlib.util
from pathlib import Path
import os

# Determine the main.py file location
# When running from services/embedding-service/, main.py is in the parent of tests/
test_file = Path(__file__).resolve()
service_dir = test_file.parent.parent  # services/embedding-service/
main_file = service_dir / "main.py"

# If main.py not found, try workspace root approach
if not main_file.exists():
    # Calculate workspace root from test file location
    # test_file is in: services/embedding-service/tests/test_main.py
    workspace_root = test_file.parent.parent.parent.parent
    main_file = workspace_root / "services" / "embedding-service" / "main.py"
    sys.path.insert(0, str(workspace_root))
else:
    # Add service directory to path
    sys.path.insert(0, str(service_dir))

# Import using importlib to handle hyphenated module name
spec = importlib.util.spec_from_file_location("embedding_service_main", main_file)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load module from {main_file}")
embedding_main = importlib.util.module_from_spec(spec)
sys.modules["embedding_service_main"] = embedding_main
spec.loader.exec_module(embedding_main)

# Import the app and functions
app = embedding_main.app
generate_embeddings = embedding_main.generate_embeddings
EmbeddingRequest = embedding_main.EmbeddingRequest
EmbeddingResponse = embedding_main.EmbeddingResponse


@pytest.fixture
def client():
    """Create a test client for FastAPI."""
    # TestClient should work with httpx>=0.27.0 in CI
    # Local environment may have version conflicts, but CI will have correct versions
    return TestClient(app)


@pytest.fixture
def mock_model_and_tokenizer():
    """Fixture to mock model and tokenizer globally."""
    with patch.object(embedding_main, "model") as mock_model, patch.object(
        embedding_main, "tokenizer"
    ) as mock_tokenizer:
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_model_instance.eval.return_value = mock_model_instance
        mock_model_instance.to.return_value = mock_model_instance
        # parameters() should return an iterator, not a list
        mock_param = torch.tensor([1.0])
        mock_model_instance.parameters.return_value = iter([mock_param])

        # Create mock output
        batch_size = 2
        hidden_size = 1024
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(batch_size, 1, hidden_size)
        mock_model_instance.return_value = mock_output

        # Setup mock tokenizer
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        # Set the mocked instances
        embedding_main.model = mock_model_instance
        embedding_main.tokenizer = mock_tokenizer_instance

        yield mock_model_instance, mock_tokenizer_instance


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_endpoint_returns_ok(self, client):
        """Test that /health returns 200 with status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data

    def test_health_endpoint_schema(self, client):
        """Test that /health response matches expected schema."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["status"], str)
        assert isinstance(data["model_loaded"], bool)


class TestEmbedEndpoint:
    """Tests for the /embed endpoint."""

    def test_embed_endpoint_success(self, client, mock_model_and_tokenizer):
        """Test successful embedding generation."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        # Make request
        response = client.post("/embed", json={"texts": ["Hello world", "Test embedding"]})

        assert response.status_code == 200
        data = response.json()
        assert "embeddings" in data
        assert len(data["embeddings"]) == 2
        assert len(data["embeddings"][0]) == 1024
        assert len(data["embeddings"][1]) == 1024

    def test_embed_endpoint_empty_texts(self, client):
        """Test embedding endpoint with empty texts list."""
        response = client.post("/embed", json={"texts": []})
        # Should either return empty embeddings or error
        assert response.status_code in [200, 400, 422]

    def test_embed_endpoint_invalid_request(self, client):
        """Test embedding endpoint with invalid request body."""
        response = client.post("/embed", json={"invalid": "data"})
        assert response.status_code == 422  # Validation error

    def test_embed_endpoint_missing_field(self, client):
        """Test embedding endpoint with missing 'texts' field."""
        response = client.post("/embed", json={})
        assert response.status_code == 422


class TestGenerateEmbeddings:
    """Unit tests for generate_embeddings function."""

    def test_generate_embeddings_success(self, mock_model_and_tokenizer):
        """Test successful embedding generation."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        # Test function
        texts = ["Hello world", "Test embedding"]
        embeddings = generate_embeddings(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024
        assert len(embeddings[1]) == 1024
        assert all(isinstance(emb, float) for emb in embeddings[0])
        assert all(isinstance(emb, float) for emb in embeddings[1])

    def test_generate_embeddings_model_not_loaded(self):
        """Test that generate_embeddings raises error when model is not loaded."""
        original_model = embedding_main.model
        original_tokenizer = embedding_main.tokenizer

        try:
            embedding_main.model = None
            embedding_main.tokenizer = None

            with pytest.raises(RuntimeError, match="Model not loaded"):
                generate_embeddings(["test"])
        finally:
            embedding_main.model = original_model
            embedding_main.tokenizer = original_tokenizer

    def test_generate_embeddings_normalization(self, mock_model_and_tokenizer):
        """Test that embeddings are normalized."""
        mock_model, mock_tokenizer = mock_model_and_tokenizer

        # Override mock output with known values for normalization test
        batch_size = 1
        hidden_size = 3
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.tensor([[[3.0, 4.0, 0.0]]])
        mock_model.return_value = mock_output

        # Test function
        texts = ["test"]
        embeddings = generate_embeddings(texts)

        # Check normalization (should have norm ~1.0)
        embedding = np.array(embeddings[0])
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 0.01  # Allow small floating point error


class TestRequestResponseSchemas:
    """Integration tests for request/response schemas."""

    def test_embed_request_schema(self):
        """Test EmbeddingRequest schema validation."""
        # Valid request
        request = EmbeddingRequest(texts=["test1", "test2"])
        assert len(request.texts) == 2

        # Invalid request (should raise validation error)
        with pytest.raises(Exception):  # Pydantic validation error
            EmbeddingRequest(texts="not a list")

    def test_embed_response_schema(self):
        """Test EmbeddingResponse schema validation."""
        # Valid response
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        response = EmbeddingResponse(embeddings=embeddings)
        assert len(response.embeddings) == 2

        # Invalid response (should raise validation error)
        with pytest.raises(Exception):  # Pydantic validation error
            EmbeddingResponse(embeddings="not a list")
