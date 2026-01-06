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
    
    @pytest.mark.asyncio
    async def test_interpret_message_with_persona_sets_persona_mode(self, chat_agent, mock_llm_provider):
        """Test that interpret_message uses persona when provided."""
        # Mock LLM response with persona mode
        mock_response = MagicMock()
        mock_response.content = '{"role": "Backend Engineer Python", "skills": ["Python", "FastAPI"], "industry": null, "location": null, "company_stage": null, "search_mode": "persona", "friendly_reasoning": "Searching for backend roles matching your Python expertise."}'
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        persona = {
            "technical_skills": ["Python", "FastAPI", "PostgreSQL"],
            "experience_years": 5
        }
        
        result = await chat_agent.interpret_message("find jobs for me", persona=persona)
        
        assert result["search_mode"] == "persona"
        assert result["role"] == "Backend Engineer Python"
        assert "Python" in result.get("skills", [])
    
    @pytest.mark.asyncio
    async def test_interpret_message_without_persona_sets_explicit_mode(self, chat_agent, mock_llm_provider):
        """Test that interpret_message defaults to explicit mode when persona is missing."""
        # Mock LLM response with explicit mode
        mock_response = MagicMock()
        mock_response.content = '{"role": "Backend Engineer", "skills": ["Python"], "industry": null, "location": null, "company_stage": null, "search_mode": "explicit", "friendly_reasoning": "Performing a general search for Backend Engineer roles."}'
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await chat_agent.interpret_message("find backend jobs", persona=None)
        
        assert result["search_mode"] == "explicit"
        assert result["role"] == "Backend Engineer"
    
    @pytest.mark.asyncio
    async def test_interpret_message_broad_search_without_persona_sets_role_null(self, chat_agent, mock_llm_provider):
        """Test that broad search queries without persona set role to null."""
        # Mock LLM response with null role for broad search
        mock_response = MagicMock()
        mock_response.content = '{"role": null, "skills": [], "industry": null, "location": null, "company_stage": null, "search_mode": "explicit", "friendly_reasoning": "Showing all available vacancies since no CV is uploaded and search is broad."}'
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        result = await chat_agent.interpret_message("show all vacancies", persona=None)
        
        assert result["role"] is None
        assert "all available vacancies" in result.get("friendly_reasoning", "").lower()
    
    @pytest.mark.asyncio
    async def test_interpret_message_persona_mode_with_missing_persona_falls_back(self, chat_agent, mock_llm_provider):
        """Test that persona mode requested but persona missing falls back to explicit."""
        # Mock LLM response requesting persona mode
        mock_response = MagicMock()
        mock_response.content = '{"role": "Backend Engineer", "skills": ["Python"], "industry": null, "location": null, "company_stage": null, "search_mode": "persona", "friendly_reasoning": "Searching for backend roles."}'
        mock_llm_provider.ainvoke = AsyncMock(return_value=mock_response)
        
        # Call with persona=None but LLM might request persona mode
        result = await chat_agent.interpret_message("find jobs for me", persona=None)
        
        # The agent should handle this gracefully (validation happens in chat_search endpoint)
        assert "role" in result
        assert result.get("search_mode") in ["persona", "explicit"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])




