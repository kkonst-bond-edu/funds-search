"""
Integration tests for the chat API endpoint.
Tests the full flow of the /api/v1/vacancies/chat endpoint.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(workspace_root))

# Mock dependencies before importing
sys.modules["pinecone"] = MagicMock()
sys.modules["langchain_google_genai"] = MagicMock()
sys.modules["langgraph"] = MagicMock()

# Import after setting up mocks
from apps.api.main import app
from src.schemas.vacancy import Vacancy, CompanyStage


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def mock_vacancy():
    """Create a mock vacancy for testing."""
    return Vacancy(
        title="Senior Python Engineer",
        company_name="TestCorp",
        company_stage=CompanyStage.SERIES_A,
        location="San Francisco, CA",
        industry="AI",
        salary_range="$150k-$200k",
        description_url="https://example.com/job1",
        required_skills=["Python", "FastAPI", "PostgreSQL"],
        remote_option=True,
    )


@pytest.fixture
def mock_pinecone_results(mock_vacancy):
    """Mock Pinecone query results."""
    return [
        {
            "id": "vacancy_1",
            "metadata": {
                "title": mock_vacancy.title,
                "company_name": mock_vacancy.company_name,
                "company_stage": mock_vacancy.company_stage.value,
                "location": mock_vacancy.location,
                "industry": mock_vacancy.industry,
                "salary_range": mock_vacancy.salary_range,
                "description_url": mock_vacancy.description_url,
                "required_skills": mock_vacancy.required_skills,
                "remote_option": mock_vacancy.remote_option,
            },
            "score": 0.85,
        }
    ]


@pytest.fixture
def mock_query_embedding():
    """Mock query embedding vector (1024 dimensions for BGE-M3)."""
    return [0.1] * 1024


@patch("src.api.v1.vacancies.get_query_embedding")
@patch("src.api.v1.vacancies.VectorStore")
@patch("apps.orchestrator.chat_search.ChatSearchAgent")
def test_chat_endpoint_success(
    mock_chat_agent_class,
    mock_vector_store_class,
    mock_get_embedding,
    client,
    mock_vacancy,
    mock_pinecone_results,
    mock_query_embedding,
):
    """Test successful chat endpoint with mocked dependencies."""
    # Mock ChatSearchAgent
    mock_chat_agent = MagicMock()
    mock_chat_agent.interpret_message = AsyncMock(
        return_value={
            "role": "Engineer",
            "skills": ["Python"],
            "industry": "AI",
            "location": None,
            "company_stage": None,
        }
    )
    mock_chat_agent.format_results_summary = AsyncMock(
        return_value="I found 1 matching vacancy for your search. This Python engineering role at TestCorp matches your criteria for AI startups."
    )
    mock_chat_agent_class.return_value = mock_chat_agent

    # Mock embedding service (async function - use AsyncMock)
    async def async_get_embedding(*args, **kwargs):
        return mock_query_embedding
    mock_get_embedding.side_effect = async_get_embedding

    # Mock Pinecone VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = mock_pinecone_results
    mock_vector_store_class.return_value = mock_vector_store

    # Make request
    response = client.post(
        "/api/v1/vacancies/chat",
        json={"message": "Looking for a Python engineer in AI"},
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "summary" in data
    assert "vacancies" in data
    assert isinstance(data["summary"], str)
    assert isinstance(data["vacancies"], list)
    assert len(data["vacancies"]) == 1

    # Verify summary content
    assert len(data["summary"]) > 0
    assert "Python" in data["summary"] or "1" in data["summary"]

    # Verify vacancy structure
    vacancy = data["vacancies"][0]
    assert vacancy["title"] == mock_vacancy.title
    assert vacancy["company_name"] == mock_vacancy.company_name
    assert vacancy["location"] == mock_vacancy.location
    assert vacancy["industry"] == mock_vacancy.industry
    assert "required_skills" in vacancy

    # Verify ChatSearchAgent methods were called
    mock_chat_agent.interpret_message.assert_called_once()
    mock_chat_agent.format_results_summary.assert_called_once()
    
    # Verify persona_applied flag is present in response
    assert "persona_applied" in data
    assert isinstance(data["persona_applied"], bool)


@patch("src.api.v1.vacancies.get_query_embedding")
@patch("src.api.v1.vacancies.VectorStore")
@patch("apps.orchestrator.chat_search.ChatSearchAgent")
def test_chat_endpoint_empty_results(
    mock_chat_agent_class,
    mock_vector_store_class,
    mock_get_embedding,
    client,
    mock_query_embedding,
):
    """Test chat endpoint with no matching vacancies."""
    # Mock ChatSearchAgent
    mock_chat_agent = MagicMock()
    mock_chat_agent.interpret_message = AsyncMock(
        return_value={
            "role": "Rare Role",
            "skills": ["Obscure Skill"],
            "industry": None,
            "location": None,
            "company_stage": None,
        }
    )
    mock_chat_agent.format_results_summary = AsyncMock(
        return_value="I couldn't find any vacancies matching your criteria. Try adjusting your search parameters or check back later for new opportunities!"
    )
    mock_chat_agent_class.return_value = mock_chat_agent

    # Mock embedding service (async function - use AsyncMock)
    async def async_get_embedding(*args, **kwargs):
        return mock_query_embedding
    mock_get_embedding.side_effect = async_get_embedding

    # Mock Pinecone VectorStore to return empty results
    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = []
    mock_vector_store_class.return_value = mock_vector_store

    # Make request
    response = client.post(
        "/api/v1/vacancies/chat",
        json={"message": "Looking for a rare role"},
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "summary" in data
    assert "vacancies" in data
    assert isinstance(data["summary"], str)
    assert isinstance(data["vacancies"], list)
    assert len(data["vacancies"]) == 0

    # Verify summary indicates no results
    assert "couldn't find" in data["summary"].lower() or "no" in data["summary"].lower()


@patch("src.api.v1.vacancies.get_query_embedding")
@patch("src.api.v1.vacancies.VectorStore")
@patch("apps.orchestrator.chat_search.ChatSearchAgent")
def test_chat_endpoint_multiple_vacancies(
    mock_chat_agent_class,
    mock_vector_store_class,
    mock_get_embedding,
    client,
    mock_query_embedding,
):
    """Test chat endpoint with multiple matching vacancies."""
    # Create multiple mock vacancies
    mock_results = [
        {
            "id": f"vacancy_{i}",
            "metadata": {
                "title": f"Engineer {i}",
                "company_name": f"Company {i}",
                "company_stage": "Series A",
                "location": "San Francisco, CA",
                "industry": "AI",
                "salary_range": "$150k-$200k",
                "description_url": f"https://example.com/job{i}",
                "required_skills": ["Python", "FastAPI"],
                "remote_option": True,
            },
            "score": 0.9 - (i * 0.1),
        }
        for i in range(3)
    ]

    # Mock ChatSearchAgent
    mock_chat_agent = MagicMock()
    mock_chat_agent.interpret_message = AsyncMock(
        return_value={
            "role": "Engineer",
            "skills": ["Python"],
            "industry": None,
            "location": None,
            "company_stage": None,
        }
    )
    mock_chat_agent.format_results_summary = AsyncMock(
        return_value="I found 3 matching vacancies for Python engineers. These roles offer great opportunities in AI startups."
    )
    mock_chat_agent_class.return_value = mock_chat_agent

    # Mock embedding service (async function - use AsyncMock)
    async def async_get_embedding(*args, **kwargs):
        return mock_query_embedding
    mock_get_embedding.side_effect = async_get_embedding

    # Mock Pinecone VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = mock_results
    mock_vector_store_class.return_value = mock_vector_store

    # Make request
    response = client.post(
        "/api/v1/vacancies/chat",
        json={"message": "Looking for Python engineers"},
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "summary" in data
    assert "vacancies" in data
    assert len(data["vacancies"]) == 3

    # Verify all vacancies have required fields
    for vacancy in data["vacancies"]:
        assert "title" in vacancy
        assert "company_name" in vacancy
        assert "location" in vacancy


def test_chat_endpoint_missing_message(client):
    """Test chat endpoint with missing message field."""
    response = client.post(
        "/api/v1/vacancies/chat",
        json={},
    )

    assert response.status_code == 422  # Validation error


def test_chat_endpoint_empty_message(client):
    """Test chat endpoint with empty message."""
    response = client.post(
        "/api/v1/vacancies/chat",
        json={"message": ""},
    )

    # Should still process (empty message is valid, just won't match anything)
    assert response.status_code in [200, 422]  # Either succeeds or validation error


@patch("src.api.v1.vacancies.get_query_embedding")
@patch("src.api.v1.vacancies.VectorStore")
@patch("apps.orchestrator.chat_search.ChatSearchAgent")
def test_chat_endpoint_data_flow(
    mock_chat_agent_class,
    mock_vector_store_class,
    mock_get_embedding,
    client,
    mock_vacancy,
    mock_pinecone_results,
    mock_query_embedding,
):
    """Test that data flows correctly through the endpoint."""
    # Mock ChatSearchAgent with specific return values
    extracted_params = {
        "role": "Backend Engineer",
        "skills": ["Go", "Kubernetes"],
        "industry": "Fintech",
        "location": "Remote",
        "company_stage": "Series A",
    }

    mock_chat_agent = MagicMock()
    mock_chat_agent.interpret_message = AsyncMock(return_value=extracted_params)
    mock_chat_agent.format_results_summary = AsyncMock(
        return_value="Found matching fintech roles for Go developers."
    )
    mock_chat_agent_class.return_value = mock_chat_agent

    # Mock embedding service (async function - use AsyncMock)
    async def async_get_embedding(*args, **kwargs):
        return mock_query_embedding
    mock_get_embedding.side_effect = async_get_embedding

    # Mock Pinecone VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = mock_pinecone_results
    mock_vector_store_class.return_value = mock_vector_store

    # Make request
    response = client.post(
        "/api/v1/vacancies/chat",
        json={"message": "Looking for a Go dev in fintech"},
    )

    # Assertions
    assert response.status_code == 200

    # Verify interpret_message was called with the correct message
    mock_chat_agent.interpret_message.assert_called_once_with("Looking for a Go dev in fintech")

    # Verify format_results_summary was called with vacancies and original message
    call_args = mock_chat_agent.format_results_summary.call_args
    assert call_args is not None
    vacancies_arg = call_args[0][0]  # First positional argument
    message_arg = call_args[0][1]  # Second positional argument
    assert isinstance(vacancies_arg, list)
    assert len(vacancies_arg) == 1
    assert message_arg == "Looking for a Go dev in fintech"

    # Verify embedding service was called
    assert mock_get_embedding.called

    # Verify Pinecone was queried
    mock_vector_store.query.assert_called_once()
    
    # Verify persona_applied flag is present
    data = response.json()
    assert "persona_applied" in data
    assert isinstance(data["persona_applied"], bool)


@patch("src.api.v1.vacancies.get_query_embedding")
@patch("src.api.v1.vacancies.VectorStore")
@patch("apps.orchestrator.chat_search.ChatSearchAgent")
@patch("apps.orchestrator.agents.matchmaker.MatchmakerAgent")
def test_chat_endpoint_with_persona_returns_persona_applied_true(
    mock_matchmaker_class,
    mock_chat_agent_class,
    mock_vector_store_class,
    mock_get_embedding,
    client,
    mock_vacancy,
    mock_pinecone_results,
    mock_query_embedding,
):
    """Test that chat endpoint with valid persona returns persona_applied: true."""
    # Mock ChatSearchAgent
    mock_chat_agent = MagicMock()
    mock_chat_agent.interpret_message = AsyncMock(
        return_value={
            "role": "Backend Engineer",
            "skills": ["Python"],
            "industry": "AI",
            "location": None,
            "company_stage": None,
            "search_mode": "persona",
            "friendly_reasoning": "Searching for backend roles matching your Python expertise."
        }
    )
    mock_chat_agent.format_results_summary = AsyncMock(
        return_value="I found 1 matching vacancy for your search."
    )
    mock_chat_agent_class.return_value = mock_chat_agent
    
    # Mock MatchmakerAgent
    mock_matchmaker = MagicMock()
    mock_matchmaker.analyze_match = AsyncMock(
        return_value={
            "score": 85,
            "analysis": "Strong match: Your Python experience aligns well with this role."
        }
    )
    mock_matchmaker_class.return_value = mock_matchmaker
    
    # Mock embedding service
    async def async_get_embedding(*args, **kwargs):
        return mock_query_embedding
    mock_get_embedding.side_effect = async_get_embedding
    
    # Mock Pinecone VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = mock_pinecone_results
    mock_vector_store_class.return_value = mock_vector_store
    
    # Make request with persona
    persona_data = {
        "technical_skills": ["Python", "FastAPI"],
        "experience_years": 5
    }
    
    response = client.post(
        "/api/v1/vacancies/chat",
        json={
            "message": "Find backend jobs for me",
            "persona": persona_data
        }
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    # Verify persona_applied flag
    assert "persona_applied" in data
    assert data["persona_applied"] is True
    
    # Verify vacancies have match scores
    assert len(data["vacancies"]) > 0
    for vacancy in data["vacancies"]:
        assert vacancy.get("score", 0) > 0
        assert "ai_insight" in vacancy
        assert "CV missing" not in vacancy.get("ai_insight", "")


@patch("src.api.v1.vacancies.get_query_embedding")
@patch("src.api.v1.vacancies.VectorStore")
@patch("apps.orchestrator.chat_search.ChatSearchAgent")
def test_chat_endpoint_without_persona_returns_persona_applied_false(
    mock_chat_agent_class,
    mock_vector_store_class,
    mock_get_embedding,
    client,
    mock_vacancy,
    mock_pinecone_results,
    mock_query_embedding,
):
    """Test that chat endpoint without persona returns persona_applied: false."""
    # Mock ChatSearchAgent
    mock_chat_agent = MagicMock()
    mock_chat_agent.interpret_message = AsyncMock(
        return_value={
            "role": "Backend Engineer",
            "skills": ["Python"],
            "industry": None,
            "location": None,
            "company_stage": None,
            "search_mode": "explicit",
            "friendly_reasoning": "Performing a general search for Backend Engineer roles."
        }
    )
    mock_chat_agent.format_results_summary = AsyncMock(
        return_value="I found 1 matching vacancy for your search."
    )
    mock_chat_agent_class.return_value = mock_chat_agent
    
    # Mock embedding service
    async def async_get_embedding(*args, **kwargs):
        return mock_query_embedding
    mock_get_embedding.side_effect = async_get_embedding
    
    # Mock Pinecone VectorStore
    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = mock_pinecone_results
    mock_vector_store_class.return_value = mock_vector_store
    
    # Make request WITHOUT persona
    response = client.post(
        "/api/v1/vacancies/chat",
        json={
            "message": "Find backend jobs",
            "persona": None
        }
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    
    # Verify persona_applied flag is False
    assert "persona_applied" in data
    assert data["persona_applied"] is False
    
    # Verify all vacancies have CV missing message and score = 0
    assert len(data["vacancies"]) > 0
    cv_missing_message = "CV missing: Upload your resume in the 'Career & Match Hub' to enable AI matching."
    for vacancy in data["vacancies"]:
        assert vacancy.get("score", None) == 0
        assert vacancy.get("ai_insight") == cv_missing_message

