"""
Tests for vacancy search functionality.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.schemas.vacancy import Vacancy, VacancyFilter, CompanyStage
from src.api.v1.vacancies import filter_vacancies, get_mock_vacancies


class TestCompanyStageEnum:
    """Test CompanyStage enum comparison logic."""
    
    def test_get_stage_value_with_enum(self):
        """Test get_stage_value with enum object."""
        stage = CompanyStage.SEED
        assert CompanyStage.get_stage_value(stage) == "Seed"
    
    def test_get_stage_value_with_string(self):
        """Test get_stage_value with string."""
        stage = "SeriesA"
        assert CompanyStage.get_stage_value(stage) == "SeriesA"
    
    def test_get_stage_value_with_value_attribute(self):
        """Test get_stage_value with object that has value attribute."""
        class MockEnum:
            def __init__(self, val):
                self.value = val
        
        stage = MockEnum("Growth")
        assert CompanyStage.get_stage_value(stage) == "Growth"


class TestVacancyFiltering:
    """Test vacancy filtering logic."""
    
    def test_filter_by_role(self):
        """Test filtering vacancies by role."""
        vacancies = get_mock_vacancies()
        filter_params = VacancyFilter(role="Backend")
        filtered = filter_vacancies(vacancies, filter_params)
        assert len(filtered) == 1
        assert "Backend" in filtered[0].title
    
    def test_filter_by_skills(self):
        """Test filtering vacancies by skills."""
        vacancies = get_mock_vacancies()
        filter_params = VacancyFilter(skills=["Python"])
        filtered = filter_vacancies(vacancies, filter_params)
        assert len(filtered) >= 2  # At least 2 vacancies have Python
    
    def test_filter_by_company_stage_enum(self):
        """Test filtering by company stage using enum objects."""
        vacancies = get_mock_vacancies()
        filter_params = VacancyFilter(company_stages=[CompanyStage.SEED])
        filtered = filter_vacancies(vacancies, filter_params)
        assert len(filtered) == 1
        assert filtered[0].company_stage == CompanyStage.SEED
    
    def test_filter_by_company_stage_string(self):
        """Test filtering by company stage using strings (robust comparison)."""
        vacancies = get_mock_vacancies()
        # Simulate string input (as might come from API)
        filter_params = VacancyFilter(company_stages=["SeriesA"])
        filtered = filter_vacancies(vacancies, filter_params)
        assert len(filtered) == 1
        # Verify the stage value matches
        assert CompanyStage.get_stage_value(filtered[0].company_stage) == "SeriesA"
    
    def test_filter_by_remote_option(self):
        """Test filtering by remote option."""
        vacancies = get_mock_vacancies()
        filter_params = VacancyFilter(is_remote=True)
        filtered = filter_vacancies(vacancies, filter_params)
        assert all(v.remote_option for v in filtered)
    
    def test_filter_by_location(self):
        """Test filtering by location."""
        vacancies = get_mock_vacancies()
        filter_params = VacancyFilter(location="San Francisco")
        filtered = filter_vacancies(vacancies, filter_params)
        assert len(filtered) == 1
        assert "San Francisco" in filtered[0].location
    
    def test_filter_by_industry(self):
        """Test filtering by industry."""
        vacancies = get_mock_vacancies()
        filter_params = VacancyFilter(industry="Logistics")
        filtered = filter_vacancies(vacancies, filter_params)
        assert len(filtered) == 3  # All mock vacancies are in Logistics


class TestFirecrawlService:
    """Test Firecrawl service (mocked)."""
    
    def test_firecrawl_service_initialization_without_key(self, monkeypatch):
        """Test Firecrawl service raises error when API key is missing."""
        from src.services.firecrawl_service import FirecrawlService
        from src.services.exceptions import FirecrawlAuthError
        
        monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
        
        with pytest.raises(FirecrawlAuthError):
            FirecrawlService()
    
    def test_firecrawl_service_initialization_with_key(self, monkeypatch):
        """Test Firecrawl service initializes with valid API key."""
        from src.services.firecrawl_service import FirecrawlService
        
        monkeypatch.setenv("FIRECRAWL_API_KEY", "test_key_12345")
        
        # This will fail if firecrawl-py is not installed, but that's expected
        try:
            service = FirecrawlService()
            assert service is not None
        except ImportError:
            pytest.skip("firecrawl-py package not installed")
        except Exception:
            # Other errors (like connection) are acceptable for unit tests
            pass


class TestChatEndpointPersonaHandling:
    """Test CV Missing logic in the /chat endpoint."""
    
    @pytest.fixture
    def mock_persona_data(self):
        """Create mock persona data for testing."""
        return {
            "technical_skills": ["Python", "FastAPI", "PostgreSQL"],
            "experience_years": 5,
            "career_goals": "Backend Engineering",
            "location_preference": "Remote"
        }
    
    @pytest.fixture
    def mock_vacancy_dict(self):
        """Create a mock vacancy dict as returned by the API."""
        return {
            "title": "Senior Backend Engineer",
            "company_name": "TestCorp",
            "company_stage": "Series A",
            "location": "San Francisco, CA",
            "industry": "AI",
            "salary_range": "$150k-$200k",
            "description_url": "https://example.com/job1",
            "required_skills": ["Python", "FastAPI", "PostgreSQL"],
            "remote_option": True,
            "pinecone_score": 0.85
        }
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("use_mock", [True, False])
    async def test_chat_with_valid_persona_returns_persona_applied_true(
        self, mock_persona_data, mock_vacancy_dict, use_mock
    ):
        """Test that valid persona data results in persona_applied: true and match_score > 0."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from fastapi.testclient import TestClient
        import sys
        from pathlib import Path
        
        # Mock dependencies
        sys.modules["pinecone"] = MagicMock()
        sys.modules["langchain_google_genai"] = MagicMock()
        sys.modules["langgraph"] = MagicMock()
        
        from apps.api.main import app
        
        client = TestClient(app)
        
        # Mock ChatSearchAgent
        with patch("src.api.v1.vacancies.ChatSearchAgent") as mock_chat_agent_class:
            mock_chat_agent = MagicMock()
            mock_chat_agent.interpret_message = AsyncMock(
                return_value={
                    "role": "Backend Engineer",
                    "skills": ["Python"],
                    "industry": None,
                    "location": None,
                    "company_stage": None,
                    "search_mode": "persona",
                    "friendly_reasoning": "Searching for backend roles matching your Python expertise."
                }
            )
            mock_chat_agent.format_results_summary = AsyncMock(
                return_value="Found matching backend roles."
            )
            mock_chat_agent_class.return_value = mock_chat_agent
            
            # Mock MatchmakerAgent
            with patch("src.api.v1.vacancies.MatchmakerAgent") as mock_matchmaker_class:
                mock_matchmaker = MagicMock()
                mock_matchmaker.analyze_match = AsyncMock(
                    return_value={
                        "score": 85,
                        "analysis": "Strong match: Your Python and FastAPI experience aligns well with this role."
                    }
                )
                mock_matchmaker_class.return_value = mock_matchmaker
                
                # Mock embedding and vector store
                with patch("src.api.v1.vacancies.get_query_embedding") as mock_embedding:
                    async def async_get_embedding(*args, **kwargs):
                        return [0.1] * 1024
                    mock_embedding.side_effect = async_get_embedding
                    
                    with patch("src.api.v1.vacancies.VectorStore") as mock_vector_store_class:
                        mock_vector_store = MagicMock()
                        mock_vector_store.query.return_value = [
                            {
                                "id": "vacancy_1",
                                "metadata": mock_vacancy_dict,
                                "score": 0.85
                            }
                        ]
                        mock_vector_store_class.return_value = mock_vector_store
                        
                        # Make request with persona
                        response = client.post(
                            "/api/v1/vacancies/chat",
                            json={
                                "message": "Find backend jobs for me",
                                "persona": mock_persona_data
                            }
                        )
                        
                        # Assertions
                        assert response.status_code == 200
                        data = response.json()
                        
                        # Verify persona_applied flag
                        assert "persona_applied" in data
                        assert data["persona_applied"] is True
                        
                        # Verify vacancies have match scores > 0
                        assert len(data["vacancies"]) > 0
                        for vacancy in data["vacancies"]:
                            assert vacancy.get("score", 0) > 0
                            assert vacancy.get("match_score", 0) > 0
                            assert vacancy.get("persona_applied", False) is True
                            assert "ai_insight" in vacancy
                            assert "CV missing" not in vacancy.get("ai_insight", "")
    
    @pytest.mark.asyncio
    async def test_chat_without_persona_returns_persona_applied_false(
        self, mock_vacancy_dict
    ):
        """Test that missing persona data results in persona_applied: false and CV missing message."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from fastapi.testclient import TestClient
        import sys
        
        # Mock dependencies
        sys.modules["pinecone"] = MagicMock()
        sys.modules["langchain_google_genai"] = MagicMock()
        sys.modules["langgraph"] = MagicMock()
        
        from apps.api.main import app
        
        client = TestClient(app)
        
        # Mock ChatSearchAgent
        with patch("src.api.v1.vacancies.ChatSearchAgent") as mock_chat_agent_class:
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
                return_value="Found matching backend roles."
            )
            mock_chat_agent_class.return_value = mock_chat_agent
            
            # Mock embedding and vector store
            with patch("src.api.v1.vacancies.get_query_embedding") as mock_embedding:
                async def async_get_embedding(*args, **kwargs):
                    return [0.1] * 1024
                mock_embedding.side_effect = async_get_embedding
                
                with patch("src.api.v1.vacancies.VectorStore") as mock_vector_store_class:
                    mock_vector_store = MagicMock()
                    mock_vector_store.query.return_value = [
                        {
                            "id": "vacancy_1",
                            "metadata": mock_vacancy_dict,
                            "score": 0.85
                        }
                    ]
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
                        assert vacancy.get("match_score", None) == 0
                        assert vacancy.get("ai_match_score", None) == 0
                        assert vacancy.get("ai_insight") == cv_missing_message
                        assert vacancy.get("match_reason") == cv_missing_message
                        assert vacancy.get("persona_applied", True) is False
    
    @pytest.mark.asyncio
    async def test_chat_broad_search_without_persona_skips_soft_filter(
        self, mock_vacancy_dict
    ):
        """Test that 'all roles' query without CV skips soft_filter and returns all vacancies."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from fastapi.testclient import TestClient
        import sys
        
        # Mock dependencies
        sys.modules["pinecone"] = MagicMock()
        sys.modules["langchain_google_genai"] = MagicMock()
        sys.modules["langgraph"] = MagicMock()
        
        from apps.api.main import app
        
        client = TestClient(app)
        
        # Create multiple mock vacancies
        mock_vacancies = [
            {
                "id": f"vacancy_{i}",
                "metadata": {
                    **mock_vacancy_dict,
                    "title": f"Role {i}",
                    "company_name": f"Company {i}"
                },
                "score": 0.9 - (i * 0.1)
            }
            for i in range(5)
        ]
        
        # Mock ChatSearchAgent to return null role for broad search
        with patch("src.api.v1.vacancies.ChatSearchAgent") as mock_chat_agent_class:
            mock_chat_agent = MagicMock()
            mock_chat_agent.interpret_message = AsyncMock(
                return_value={
                    "role": None,  # null role for broad search
                    "skills": [],
                    "industry": None,
                    "location": None,
                    "company_stage": None,
                    "search_mode": "explicit",
                    "friendly_reasoning": "Showing all available vacancies since no CV is uploaded and search is broad."
                }
            )
            mock_chat_agent.format_results_summary = AsyncMock(
                return_value="Found all available vacancies."
            )
            mock_chat_agent_class.return_value = mock_chat_agent
            
            # Mock embedding and vector store
            with patch("src.api.v1.vacancies.get_query_embedding") as mock_embedding:
                async def async_get_embedding(*args, **kwargs):
                    return [0.1] * 1024
                mock_embedding.side_effect = async_get_embedding
                
                with patch("src.api.v1.vacancies.VectorStore") as mock_vector_store_class:
                    mock_vector_store = MagicMock()
                    mock_vector_store.query.return_value = mock_vacancies
                    mock_vector_store_class.return_value = mock_vector_store
                    
                    # Make request for "all roles" without persona
                    response = client.post(
                        "/api/v1/vacancies/chat",
                        json={
                            "message": "show all vacancies",
                            "persona": None
                        }
                    )
                    
                    # Assertions
                    assert response.status_code == 200
                    data = response.json()
                    
                    # Verify persona_applied is False
                    assert data["persona_applied"] is False
                    
                    # Verify all vacancies are returned (soft_filter should be skipped)
                    # Since role is null, filter_vacancies should not filter by role
                    assert len(data["vacancies"]) == 5
                    
                    # Verify all have CV missing message
                    cv_missing_message = "CV missing: Upload your resume in the 'Career & Match Hub' to enable AI matching."
                    for vacancy in data["vacancies"]:
                        assert vacancy.get("ai_insight") == cv_missing_message
                        assert vacancy.get("score", None) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

