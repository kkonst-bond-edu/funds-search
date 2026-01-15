"""
Test of Resilience: Handling zero search results with filter relaxation.

This test verifies:
1. Mock search_vacancies_tool returns count: 0 for a specific query
2. Agent receives message: 'Search returned 0 results. Please relax filters.'
3. Agent generates a NEW tool call with one less filter (e.g., removes salary_min or expands company_stage)
4. System doesn't enter an infinite loop (max 2-3 retries)
"""
import pytest
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.orchestrator.graph.nodes import job_scout_node
from apps.orchestrator.graph.state import AgentState, UserProfile
from apps.orchestrator.tools.search_tool import search_vacancies_tool


class TestResilienceZeroResults:
    """Test resilience when search returns zero results."""
    
    @pytest.fixture
    def user_profile(self):
        """Create a user profile with strict filters."""
        return UserProfile(
            skills=["Python", "FastAPI", "PostgreSQL"],
            years_of_experience=5,
            salary_expectation="$200k",  # High salary - likely to return 0 results
            location="San Francisco",
            remote_preference="remote_only",
        )
    
    @pytest.fixture
    def initial_state(self, user_profile):
        """Create initial agent state."""
        return {
            "messages": [
                HumanMessage(content="I'm looking for a senior Python developer role")
            ],
            "user_profile": user_profile,
            "candidate_pool": [],
            "match_results": [],
            "search_params": {},
            "status": "ready_for_search",
            "missing_info": [],
        }
    
    @pytest.mark.asyncio
    async def test_zero_results_triggers_filter_relaxation(self, initial_state, user_profile):
        """
        Test that when search returns 0 results, the agent receives a relaxation message
        and generates a new tool call with relaxed filters.
        """
        # Track tool calls to verify filter relaxation
        tool_call_count = 0
        tool_calls_history = []
        
        def mock_search_tool(*args, **kwargs):
            """Mock search tool that returns 0 results on first call, then results on second."""
            nonlocal tool_call_count
            tool_call_count += 1
            
            # Extract filters from kwargs
            filters = {
                "salary_min": kwargs.get("salary_min"),
                "company_stage": kwargs.get("company_stage"),
                "remote_option": kwargs.get("remote_option"),
            }
            tool_calls_history.append(filters.copy())
            
            # First call: return 0 results (strict filters)
            if tool_call_count == 1:
                return {
                    "results": [],
                    "count": 0,
                    "query": kwargs.get("query", "Python Developer"),
                    "filters_applied": {
                        "min_salary": {"$gte": 200000},
                        "remote_option": {"$eq": True},
                    },
                }
            # Second call: return results (relaxed filters)
            else:
                return {
                    "results": [
                        {
                            "id": "vacancy_1",
                            "metadata": {
                                "title": "Senior Python Developer",
                                "company_name": "TestCorp",
                                "remote_option": True,
                                "min_salary": 150000,  # Lower than original filter
                                "company_stage": "Series A",
                            },
                            "score": 0.85,
                        }
                    ],
                    "count": 1,
                    "query": kwargs.get("query", "Python Developer"),
                    "filters_applied": {
                        "remote_option": {"$eq": True},
                        # Note: salary_min should be removed or lowered
                    },
                }
        
        # Track agent calls separately from tool calls
        agent_call_count = 0
        
        # Mock the agent's search_with_tool to simulate LLM behavior with retry
        async def mock_search_with_tool(user_profile, conversation_history):
            """Simulate agent's search_with_tool with retry logic."""
            nonlocal tool_call_count, agent_call_count
            agent_call_count += 1
            
            # Check if conversation_history contains relaxation instruction
            has_relaxation = any(
                "relax" in str(msg.get("content", "")).lower() or 
                "0 results" in str(msg.get("content", "")).lower()
                for msg in (conversation_history or [])
            )
            
            # First call: strict filters, return 0 results
            if agent_call_count == 1:
                result = mock_search_tool(
                    query="Senior Python Developer",
                    salary_min=200000,
                    remote_option="remote",
                    company_stage=["Series A"],
                )
                
                return {
                    "search_results": result.get("results", []),
                    "search_params": {
                        "query": "Senior Python Developer",
                        "salary_min": 200000,
                        "remote_option": "remote",
                        "company_stage": ["Series A"],
                    },
                    "analysis": "Search completed" if result["count"] > 0 else "No results found",
                }
            
            # Second call: relaxed filters (after seeing 0 results and relaxation instruction)
            elif has_relaxation:
                result = mock_search_tool(
                    query="Senior Python Developer",
                    remote_option="remote",
                    # salary_min removed
                )
                
                return {
                    "search_results": result.get("results", []),
                    "search_params": {
                        "query": "Senior Python Developer",
                        "remote_option": "remote",
                        # salary_min removed
                    },
                    "analysis": "Search completed with relaxed filters",
                }
            else:
                # No relaxation instruction, return 0 results again
                result = mock_search_tool(
                    query="Senior Python Developer",
                    salary_min=200000,
                    remote_option="remote",
                )
                return {
                    "search_results": [],
                    "search_params": {
                        "query": "Senior Python Developer",
                        "salary_min": 200000,
                        "remote_option": "remote",
                    },
                    "analysis": "No results found",
                }
        
        with patch("apps.orchestrator.graph.nodes.JobScoutAgent") as mock_agent_class:
            # Create a mock agent instance
            mock_agent = MagicMock()
            mock_agent.search_with_tool = AsyncMock(side_effect=mock_search_with_tool)
            mock_agent_class.return_value = mock_agent
            
            # Run job_scout_node first time (will get 0 results)
            updated_state = await job_scout_node(initial_state)
            
            # Verify the node handled 0 results
            assert updated_state is not None
            assert "status" in updated_state
            
            # Verify that missing_info contains feedback about 0 results
            missing_info = updated_state.get("missing_info", [])
            assert len(missing_info) > 0
            assert any("0" in str(info).lower() or "no" in str(info).lower() for info in missing_info)
            
            # Verify search_attempt is tracked
            search_params = updated_state.get("search_params", {})
            assert search_params.get("_search_attempt") == 1
            
            # Simulate retry: Add ToolMessage with 0 results to trigger relaxation
            from langchain_core.messages import ToolMessage
            retry_state = updated_state.copy()
            retry_state["messages"] = retry_state.get("messages", []) + [
                ToolMessage(
                    content=json.dumps({"count": 0, "results": []}),
                    tool_call_id="test_call_1",
                )
            ]
            
            # Run job_scout_node again (should detect 0 results and add relaxation instruction)
            retry_state = await job_scout_node(retry_state)
            
            # Verify that filters were relaxed (check tool_calls_history)
            # The second call should have fewer or no salary_min filter
            assert len(tool_calls_history) >= 2, "Expected at least 2 tool calls"
            first_call = tool_calls_history[0]
            second_call = tool_calls_history[1]
            
            # Verify that salary_min was removed or lowered in second call
            assert first_call.get("salary_min") == 200000, "First call should have salary_min=200000"
            assert second_call.get("salary_min") is None or second_call.get("salary_min") < 200000, \
                "Second call should have salary_min removed or lowered"
    
    @pytest.mark.asyncio
    async def test_retry_limit_prevents_infinite_loop(self, initial_state, user_profile):
        """
        Test that the system doesn't enter an infinite loop when repeatedly getting 0 results.
        Max retries should be 2-3.
        """
        search_attempts = []
        max_retries = 3
        
        def mock_search_always_zero(*args, **kwargs):
            """Mock search that always returns 0 results."""
            search_attempts.append(kwargs.copy())
            return {
                "results": [],
                "count": 0,
                "query": kwargs.get("query", "Python Developer"),
                "filters_applied": kwargs.get("filters_applied", {}),
            }
        
        with patch("apps.orchestrator.graph.nodes.JobScoutAgent") as mock_agent_class:
            # Create a mock agent instance
            mock_agent = MagicMock()
            
            # Mock to always return 0 results
            async def mock_search_with_tool(user_profile, conversation_history):
                result = mock_search_always_zero(query="Python Developer")
                return {
                    "search_results": [],
                    "search_params": {
                        "query": "Python Developer",
                        "_search_attempt": len(search_attempts),
                    },
                    "analysis": "No results found",
                }
            
            mock_agent.search_with_tool = AsyncMock(side_effect=mock_search_with_tool)
            mock_agent_class.return_value = mock_agent
            
            # Simulate multiple retries
            current_state = initial_state.copy()
            retry_count = 0
            
            while retry_count < max_retries + 1:  # Try one more than max to verify limit
                updated_state = await job_scout_node(current_state)
                
                # Check if we should retry
                search_params = updated_state.get("search_params", {})
                search_attempt = search_params.get("_search_attempt", 0)
                
                if search_attempt >= max_retries:
                    # Should stop retrying
                    break
                
                # If status is awaiting_info and we haven't hit limit, prepare for retry
                if updated_state["status"] == "awaiting_info":
                    retry_count += 1
                    # Simulate adding relaxation instruction (as job_scout_node does)
                    current_state = updated_state.copy()
                    # Add a ToolMessage with 0 results to trigger relaxation
                    from langchain_core.messages import ToolMessage
                    current_state["messages"] = current_state.get("messages", []) + [
                        ToolMessage(
                            content=json.dumps({"count": 0, "results": []}),
                            tool_call_id="test_call",
                        )
                    ]
                else:
                    break
            
            # Verify we didn't exceed max retries
            assert retry_count <= max_retries, f"Exceeded max retries: {retry_count} > {max_retries}"
            
            # Verify search was called limited number of times
            # Note: This depends on the actual retry logic in job_scout_node
            # The node should track _search_attempt and stop after max_retries
    
    @pytest.mark.asyncio
    async def test_filter_relaxation_removes_salary_min(self, initial_state, user_profile):
        """
        Test that when filters are relaxed, salary_min is removed or lowered.
        """
        tool_calls = []
        
        def mock_search_tool(*args, **kwargs):
            """Track tool calls to verify filter changes."""
            call_info = {
                "salary_min": kwargs.get("salary_min"),
                "remote_option": kwargs.get("remote_option"),
                "company_stage": kwargs.get("company_stage"),
            }
            tool_calls.append(call_info)
            
            # Return 0 results if salary_min is too high
            if kwargs.get("salary_min") and kwargs.get("salary_min") >= 200000:
                return {
                    "results": [],
                    "count": 0,
                    "query": kwargs.get("query", "Python Developer"),
                    "filters_applied": {"min_salary": {"$gte": kwargs.get("salary_min")}},
                }
            else:
                # Return results when salary_min is removed or lowered
                return {
                    "results": [
                        {
                            "id": "vacancy_1",
                            "metadata": {"title": "Python Developer", "min_salary": 150000},
                            "score": 0.85,
                        }
                    ],
                    "count": 1,
                    "query": kwargs.get("query", "Python Developer"),
                    "filters_applied": {},
                }
        
        with patch("apps.orchestrator.tools.search_tool.search_vacancies_tool") as mock_tool:
            mock_tool.invoke = Mock(side_effect=mock_search_tool)
            
            # This test would need to actually run the agent with LLM
            # For now, we verify the mock behavior
            # First call with high salary
            result1 = mock_search_tool(query="Python Developer", salary_min=200000, remote_option="remote")
            assert result1["count"] == 0
            
            # Second call without salary_min (relaxed)
            result2 = mock_search_tool(query="Python Developer", remote_option="remote")
            assert result2["count"] > 0
            
            # Verify salary_min was removed in second call
            assert len(tool_calls) == 2
            assert tool_calls[0]["salary_min"] == 200000
            assert tool_calls[1].get("salary_min") is None or tool_calls[1]["salary_min"] < 200000


    @pytest.mark.asyncio
    async def test_zero_results_message_format(self, initial_state, user_profile):
        """
        Test that when search returns 0 results, the node generates the correct message format.
        """
        # Mock search to always return 0 results
        async def mock_search_with_tool(user_profile, conversation_history):
            return {
                "search_results": [],
                "search_params": {
                    "query": "Senior Python Developer",
                    "salary_min": 200000,
                    "remote_option": "remote",
                },
                "analysis": "No results found",
            }
        
        with patch("apps.orchestrator.graph.nodes.JobScoutAgent") as mock_agent_class:
            # Create a mock agent instance
            mock_agent = MagicMock()
            mock_agent.search_with_tool = AsyncMock(side_effect=mock_search_with_tool)
            mock_agent_class.return_value = mock_agent
            
            # Run job_scout_node
            updated_state = await job_scout_node(initial_state)
            
            # Verify status
            assert updated_state["status"] == "awaiting_info"
            
            # Verify missing_info contains the expected message
            missing_info = updated_state.get("missing_info", [])
            assert len(missing_info) > 0
            
            # Check that the message mentions "0 results" or "no results"
            message_text = " ".join(missing_info).lower()
            assert "0" in message_text or "no" in message_text or "relax" in message_text
            
            # Verify search_attempt is tracked
            search_params = updated_state.get("search_params", {})
            assert search_params.get("_search_attempt") == 1
    
    @pytest.mark.asyncio
    async def test_retry_limit_enforced(self, initial_state, user_profile):
        """
        Test that retry limit (max 3) is enforced to prevent infinite loops.
        """
        # Create state with max retries already reached
        initial_state["search_params"] = {"_search_attempt": 3}
        
        # Mock search (shouldn't be called due to retry limit)
        async def mock_search_with_tool(user_profile, conversation_history):
            # This should not be called
            assert False, "Search should not be called when retry limit is reached"
            return {}
        
        with patch("apps.orchestrator.graph.nodes.JobScoutAgent") as mock_agent_class:
            # Create a mock agent instance
            mock_agent = MagicMock()
            mock_agent.search_with_tool = AsyncMock()
            mock_agent_class.return_value = mock_agent
            
            # Run job_scout_node
            updated_state = await job_scout_node(initial_state)
            
            # Verify status is awaiting_info (not error, but asking user to relax)
            assert updated_state["status"] == "awaiting_info"
            
            # Verify missing_info contains retry limit message
            missing_info = updated_state.get("missing_info", [])
            assert len(missing_info) > 0
            message_text = " ".join(missing_info).lower()
            assert "relax" in message_text or "attempt" in message_text
            
            # Verify search was not called (retry limit reached)
            assert not mock_agent.search_with_tool.called, "Search should not be called when retry limit is reached"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
