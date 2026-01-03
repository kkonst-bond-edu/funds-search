"""Orchestrator module for LangGraph state machine."""
from apps.orchestrator.graph import (
    run_search,
    orchestrator,
    run_match,
    matching_orchestrator,
    get_pinecone_client,
    get_llm_provider
)
from apps.orchestrator.llm import LLMProviderFactory


# Backward compatibility: get_llm() returns the underlying LangChain model
def get_llm():
    """Backward compatibility wrapper for get_llm_provider()."""
    from apps.orchestrator.llm import get_llm as _get_llm
    return _get_llm()


__all__ = [
    "run_search",
    "orchestrator",
    "run_match",
    "matching_orchestrator",
    "get_pinecone_client",
    "get_llm",
    "get_llm_provider",
    "LLMProviderFactory"
]
