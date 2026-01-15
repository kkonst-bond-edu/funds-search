"""
Direct test for search_vacancies_tool and _build_metadata_filters.

This test verifies:
1. The search_vacancies_tool can be called with complex filters
2. _build_metadata_filters in job_scout.py correctly produces Pinecone filter dict
3. The correct field name remote_option (not is_remote) is used
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.orchestrator.tools.search_tool import search_vacancies_tool, _build_filter_dict, SearchSchema
from apps.orchestrator.agents.job_scout import JobScoutAgent


class TestSearchToolDirect:
    """Direct tests for search_vacancies_tool and filter building."""
    
    def test_build_filter_dict_with_complex_filters(self):
        """Test _build_filter_dict produces correct Pinecone filter dict."""
        # Create SearchSchema with complex filters
        schema = SearchSchema(
            query="Senior Python Developer",
            remote_option="remote",
            salary_min=150000,
            company_stage=["Seed", "Series A"]
        )
        
        # Build filter dict
        filter_dict = _build_filter_dict(schema)
        
        # Verify the filter dict structure
        assert filter_dict is not None
        assert "remote_option" in filter_dict
        assert "min_salary" in filter_dict  # Note: uses min_salary, not salary_min
        assert "company_stage" in filter_dict
        
        # Verify remote_option uses correct field name and converts to boolean
        assert filter_dict["remote_option"] == {"$eq": True}  # "remote" -> True
        
        # Verify salary_min is converted to min_salary with $gte
        assert filter_dict["min_salary"] == {"$gte": 150000}
        
        # Verify company_stage uses $in for list
        assert filter_dict["company_stage"] == {"$in": ["Seed", "Series A"]}
        
        # Verify it uses remote_option, not is_remote
        assert "is_remote" not in filter_dict
        assert "remote_option" in filter_dict
    
    def test_build_metadata_filters_with_complex_filters(self):
        """Test _build_metadata_filters in JobScoutAgent produces correct filter dict."""
        # Create a minimal mock agent instance to access the method
        # The method doesn't use self, so we can bind it to any object
        agent = MagicMock()
        agent._build_metadata_filters = JobScoutAgent._build_metadata_filters.__get__(agent, JobScoutAgent)
        
        # Test with filter_params matching the expected input format
        filter_params = {
            "remote_option": "remote",
            "salary_min": 150000,
            "company_stage": ["Seed", "Series A"]
        }
        
        # Build metadata filters
        filter_dict = agent._build_metadata_filters(filter_params)
        
        # Verify the filter dict structure
        assert filter_dict is not None
        assert "remote_option" in filter_dict
        assert "min_salary" in filter_dict  # Note: uses min_salary, not salary_min
        assert "company_stage" in filter_dict
        
        # Verify remote_option uses correct field name and converts to boolean
        # The method converts "remote" -> True, "office" -> False, "hybrid" -> False
        assert filter_dict["remote_option"] == {"$eq": True}
        
        # Verify salary_min is converted to min_salary with $gte
        assert filter_dict["min_salary"] == {"$gte": 150000}
        
        # Verify company_stage uses $in for list
        # Note: CompanyStage.get_stage_value normalizes "Seed" and "Series A"
        assert "company_stage" in filter_dict
        assert "$in" in filter_dict["company_stage"]
        # The stages should be normalized, so we check they're in the list
        stages_list = filter_dict["company_stage"]["$in"]
        assert "Seed" in stages_list or "seed" in stages_list or any("Seed" in str(s) for s in stages_list)
        assert "Series A" in stages_list or "SeriesA" in stages_list or any("Series A" in str(s) for s in stages_list)
        
        # CRITICAL: Verify it uses remote_option, not is_remote
        assert "is_remote" not in filter_dict
        assert "remote_option" in filter_dict
    
    def test_build_metadata_filters_uses_remote_option_not_is_remote(self):
        """Explicitly verify that _build_metadata_filters uses remote_option field name."""
        # Create a minimal mock agent instance to access the method
        agent = MagicMock()
        agent._build_metadata_filters = JobScoutAgent._build_metadata_filters.__get__(agent, JobScoutAgent)
        
        # Test with remote_option
        filter_params = {
            "remote_option": "remote"
        }
        
        filter_dict = agent._build_metadata_filters(filter_params)
        
        # Must use remote_option, not is_remote
        assert "remote_option" in filter_dict
        assert "is_remote" not in filter_dict
        assert filter_dict["remote_option"] == {"$eq": True}
        
        # Test backward compatibility: if is_remote is provided, it should still use remote_option
        filter_params_legacy = {
            "is_remote": True
        }
        
        filter_dict_legacy = agent._build_metadata_filters(filter_params_legacy)
        
        # Should convert is_remote to remote_option
        assert "remote_option" in filter_dict_legacy
        assert "is_remote" not in filter_dict_legacy
        assert filter_dict_legacy["remote_option"] == {"$eq": True}
    
    @pytest.mark.asyncio
    async def test_search_vacancies_tool_with_complex_filters(self):
        """Test search_vacancies_tool can be called with complex filters."""
        # Mock the embedding service
        mock_embedding = [0.1] * 384  # Typical embedding dimension
        
        # Mock VectorStore
        mock_results = [
            {
                "id": "vacancy_1",
                "metadata": {
                    "title": "Senior Python Developer",
                    "company_name": "TestCorp",
                    "remote_option": True,
                    "min_salary": 160000,
                    "company_stage": "Seed"
                },
                "score": 0.85
            }
        ]
        
        with patch("apps.orchestrator.tools.search_tool._get_query_embedding_async") as mock_embedding_func, \
             patch("apps.orchestrator.tools.search_tool.VectorStore") as mock_vector_store_class:
            
            # Mock embedding function
            async def async_embedding(*args, **kwargs):
                return mock_embedding
            mock_embedding_func.side_effect = async_embedding
            
            # Mock VectorStore
            mock_vector_store = MagicMock()
            mock_vector_store.query.return_value = mock_results
            mock_vector_store_class.return_value = mock_vector_store
            
            # Call the tool with complex filters
            result = search_vacancies_tool.invoke({
                "query": "Senior Python Developer",
                "remote_option": "remote",
                "salary_min": 150000,
                "company_stage": ["Seed", "Series A"],
                "top_k": 10
            })
            
            # Verify the tool was called
            assert result is not None
            assert "results" in result
            assert "count" in result
            assert "query" in result
            assert "filters_applied" in result
            
            # Verify filters_applied contains the expected structure
            filters_applied = result["filters_applied"]
            assert filters_applied is not None
            
            # Verify remote_option is in filters (as boolean)
            assert "remote_option" in filters_applied
            assert filters_applied["remote_option"] == {"$eq": True}
            
            # Verify min_salary is in filters (not salary_min)
            assert "min_salary" in filters_applied
            assert filters_applied["min_salary"] == {"$gte": 150000}
            
            # Verify company_stage is in filters
            assert "company_stage" in filters_applied
            assert filters_applied["company_stage"] == {"$in": ["Seed", "Series A"]}
            
            # CRITICAL: Verify it uses remote_option, not is_remote
            assert "is_remote" not in filters_applied
            assert "remote_option" in filters_applied
            
            # Verify VectorStore.query was called with correct filter_dict
            mock_vector_store.query.assert_called_once()
            call_args = mock_vector_store.query.call_args
            
            # Extract filter_dict from call
            filter_dict = call_args.kwargs.get("filter_dict")
            assert filter_dict is not None
            
            # Verify the filter_dict passed to Pinecone uses remote_option
            assert "remote_option" in filter_dict
            assert "is_remote" not in filter_dict
            assert filter_dict["remote_option"] == {"$eq": True}
            assert filter_dict["min_salary"] == {"$gte": 150000}
            assert filter_dict["company_stage"] == {"$in": ["Seed", "Series A"]}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
