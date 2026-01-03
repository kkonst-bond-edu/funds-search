"""
Tests for conversational vacancy search functionality.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.orchestrator.chat_search import ChatSearchAgent


class TestChatSearchAgent:
    """Test ChatSearchAgent parameter extraction."""
    
    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        mock_provider = MagicMock()
        mock_provider.name = "DeepSeek"
        mock_provider.model_name = "deepseek-chat"
        return mock_provider
    
    @pytest.fixture
    def chat_agent(self, mock_llm_provider):
        """Create ChatSearchAgent with mocked LLM provider."""
        with patch('apps.orchestrator.chat_search.LLMProviderFactory.get_provider', return_value=mock_llm_provider):
            agent = ChatSearchAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_interpret_message_extracts_go_and_fintech(self, chat_agent, mock_llm_provider):
        """Test that 'Looking for a Go dev in fintech' correctly extracts 'Go' and 'Fintech'."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.content = '{"role": "Developer", "skills": ["Go"], "industry": "Fintech", "location": null, "company_stage": null}'
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        # Test interpretation
        result = await chat_agent.interpret_message("Looking for a Go dev in fintech")
        
        # Verify extracted parameters
        assert result["skills"] == ["Go"], "Should extract 'Go' as a skill"
        assert result["industry"] == "Fintech", "Should extract 'Fintech' as industry"
        assert result["role"] == "Developer", "Should extract role"
        assert result["location"] is None, "Location should be None if not mentioned"
        assert result["company_stage"] is None, "Company stage should be None if not mentioned"
    
    @pytest.mark.asyncio
    async def test_interpret_message_handles_missing_fields(self, chat_agent, mock_llm_provider):
        """Test that missing fields are set to None."""
        # Mock LLM response with only some fields
        mock_response = MagicMock()
        mock_response.content = '{"role": null, "skills": null, "industry": "AI", "location": "Remote", "company_stage": null}'
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await chat_agent.interpret_message("AI jobs remotely")
        
        assert result["role"] is None
        assert result["skills"] is None
        assert result["industry"] == "AI"
        assert result["location"] == "Remote"
        assert result["company_stage"] is None
    
    @pytest.mark.asyncio
    async def test_interpret_message_handles_json_parse_error(self, chat_agent, mock_llm_provider):
        """Test that JSON parse errors are handled gracefully."""
        # Mock LLM response with invalid JSON
        mock_response = MagicMock()
        mock_response.content = "This is not valid JSON"
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await chat_agent.interpret_message("some message")
        
        # Should return all None values on parse error
        assert result["role"] is None
        assert result["skills"] is None
        assert result["industry"] is None
        assert result["location"] is None
        assert result["company_stage"] is None
    
    @pytest.mark.asyncio
    async def test_interpret_message_extracts_company_stage(self, chat_agent, mock_llm_provider):
        """Test extraction of company stage."""
        # Mock LLM response with company stage
        mock_response = MagicMock()
        mock_response.content = '{"role": "Engineer", "skills": ["Python"], "industry": null, "location": null, "company_stage": "Series A"}'
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await chat_agent.interpret_message("Python engineer in a series A startup")
        
        assert result["company_stage"] == "Series A"
        assert result["skills"] == ["Python"]
        assert result["role"] == "Engineer"
    
    @pytest.mark.asyncio
    async def test_format_results_summary_generates_summary(self, chat_agent, mock_llm_provider):
        """Test that format_results_summary generates a friendly summary."""
        from src.schemas.vacancy import Vacancy, CompanyStage
        
        # Create mock vacancies
        vacancies = [
            Vacancy(
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
        ]
        
        # Mock LLM response for summary
        mock_response = MagicMock()
        mock_response.content = "I found 1 matching vacancy for your search. This Python engineering role at TestCorp matches your criteria for AI startups."
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        summary = await chat_agent.format_results_summary(vacancies, "Looking for Python jobs in AI")
        
        assert len(summary) > 0
        assert "Python" in summary or "1" in summary  # Summary should mention the result
    
    @pytest.mark.asyncio
    async def test_format_results_summary_handles_empty_results(self, chat_agent, mock_llm_provider):
        """Test that format_results_summary handles empty results."""
        # Mock LLM response (shouldn't be called for empty results, but just in case)
        mock_llm_provider.ainvoke = AsyncMock()
        
        summary = await chat_agent.format_results_summary([], "some query")
        
        assert "couldn't find" in summary.lower() or "no vacancies" in summary.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


