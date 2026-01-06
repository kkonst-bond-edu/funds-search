"""
Unit and integration tests for the matching endpoint.
Tests the candidate-vacancy matching functionality.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Mock dependencies before importing
sys.modules["pinecone"] = MagicMock()
sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langgraph"] = MagicMock()

# Import after setting up mocks
from apps.api.main import app  # noqa: E402


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_candidate_embedding():
    """Mock candidate embedding vector (1024 dimensions for BGE-M3)."""
    return [0.1] * 1024


@pytest.fixture
def mock_vacancy_results():
    """Mock vacancy search results."""
    return [
        {
            "id": "vacancy_1_chunk_0",
            "metadata": {
                "vacancy_id": "vacancy_1",
                "text": "We are looking for a Python developer with 5+ years of experience.",
                "type": "vacancy"
            },
            "score": 0.85
        },
        {
            "id": "vacancy_2_chunk_0",
            "metadata": {
                "vacancy_id": "vacancy_2",
                "text": "Senior Data Scientist position requiring ML expertise.",
                "type": "vacancy"
            },
            "score": 0.78
        }
    ]


@patch('apps.orchestrator.workflow.get_pinecone_client')
@patch('apps.orchestrator.workflow.get_llm')
def test_match_endpoint_success(
    mock_get_llm,
    mock_get_pinecone_client,
    client,
    mock_candidate_embedding,
    mock_vacancy_results
):
    """Test successful matching endpoint."""
    # Setup mocks
    mock_pc_client = MagicMock()
    mock_pc_client.get_candidate_embedding.return_value = mock_candidate_embedding
    mock_pc_client.search_vacancies.return_value = mock_vacancy_results
    mock_get_pinecone_client.return_value = mock_pc_client

    # Mock LLM response (async function)
    mock_llm = AsyncMock()
    mock_response = MagicMock()
    mock_response.content = "This vacancy is a great fit because the candidate has strong Python skills and relevant experience."
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    mock_get_llm.return_value = mock_llm

    # Make request
    response = client.post(
        "/match",
        json={
            "candidate_id": "user_123",
            "top_k": 10
        }
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    # Check first result
    first_result = data[0]
    assert "score" in first_result
    assert "reasoning" in first_result
    assert "vacancy_id" in first_result
    assert "vacancy_text" in first_result
    assert "candidate_id" in first_result
    assert first_result["candidate_id"] == "user_123"
    assert first_result["score"] == 0.85


@patch('apps.orchestrator.workflow.get_pinecone_client')
def test_match_endpoint_candidate_not_found(mock_get_pinecone_client, client):
    """Test matching endpoint when candidate is not found."""
    # Setup mock to return None (candidate not found)
    mock_pc_client = MagicMock()
    mock_pc_client.get_candidate_embedding.return_value = None
    mock_get_pinecone_client.return_value = mock_pc_client

    # Make request
    response = client.post(
        "/match",
        json={
            "candidate_id": "nonexistent_user",
            "top_k": 10
        }
    )

    # Assertions
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_match_endpoint_missing_candidate_id(client):
    """Test matching endpoint with missing candidate_id."""
    response = client.post(
        "/match",
        json={
            "top_k": 10
        }
    )

    assert response.status_code == 422  # Validation error


def test_match_endpoint_default_top_k(client, mock_candidate_embedding, mock_vacancy_results):
    """Test matching endpoint with default top_k."""
    with patch('apps.orchestrator.workflow.get_pinecone_client') as mock_get_pc, \
         patch('apps.orchestrator.workflow.get_llm') as mock_get_llm:

        mock_pc_client = MagicMock()
        mock_pc_client.get_candidate_embedding.return_value = mock_candidate_embedding
        mock_pc_client.search_vacancies.return_value = mock_vacancy_results[:1]  # Return 1 result
        mock_get_pc.return_value = mock_pc_client

        mock_llm = AsyncMock()
        mock_response = MagicMock()
        mock_response.content = "Test reasoning"
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_get_llm.return_value = mock_llm

        # Make request without top_k (should default to 10)
        response = client.post(
            "/match",
            json={
                "candidate_id": "user_123"
            }
        )

        assert response.status_code == 200
        # Verify search_vacancies was called (the actual top_k used depends on implementation)
        mock_pc_client.search_vacancies.assert_called_once()


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
