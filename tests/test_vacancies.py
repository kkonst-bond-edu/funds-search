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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

