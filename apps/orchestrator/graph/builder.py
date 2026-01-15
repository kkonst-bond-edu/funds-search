"""
LangGraph builder for multi-agent orchestration workflow.

This module builds and compiles the complete LangGraph workflow connecting
all specialized agents in the multi-agent system.
"""
import logging
from langgraph.graph import StateGraph, END

from apps.orchestrator.graph.state import AgentState
from apps.orchestrator.graph.nodes import (
    strategist_node,
    job_scout_node,
    matchmaker_node,
    validator_node,
    final_validation_node,
)

logger = logging.getLogger(__name__)


def should_continue_to_job_scout(state: AgentState) -> str:
    """
    Conditional edge function: check if profile is complete to proceed to job_scout.
    
    Args:
        state: Current agent state
        
    Returns:
        "continue" if profile is ready, "await_info" if needs more information
    """
    status = state.get("status", "")
    
    if status == "ready_for_search":
        return "continue"
    elif status == "awaiting_info":
        return "await_info"
    else:
        # Default to continue if status is unclear
        logger.warning("unclear_status_in_conditional", status=status)
        return "continue"


def should_research(state: AgentState) -> str:
    """
    Conditional edge function: check if validator found violations requiring re-search.
    
    Includes protection against infinite loops by tracking research iterations.
    Synchronized with max_search_retries in job_scout_node.
    
    Args:
        state: Current agent state
        
    Returns:
        "research" if needs re-search, "complete" if validation passed
    """
    status = state.get("status", "")
    
    if status == "needs_research":
        # Check if we've already done too many research iterations
        # Synchronized with job_scout_node limits
        search_params = state.get("search_params", {})
        research_count = search_params.get("_research_iterations", 0)
        search_attempt = search_params.get("_search_attempt", 0)
        max_research_iterations = 2  # Max 2 re-search attempts
        max_total_attempts = 3  # Maximum total attempts (search + research)
        
        total_attempts = search_attempt + research_count
        
        if research_count >= max_research_iterations or total_attempts >= max_total_attempts:
            logger.warning(
                "max_research_iterations_reached",
                iterations=research_count,
                search_attempt=search_attempt,
                total_attempts=total_attempts,
                max_research_iterations=max_research_iterations,
                max_total_attempts=max_total_attempts,
            )
            return "complete"  # Stop re-searching to prevent infinite loop
        
        # Increment iteration counter
        search_params["_research_iterations"] = research_count + 1
        return "research"
    elif status == "validation_complete":
        return "complete"
    else:
        # Default to complete
        logger.warning("unclear_validation_status", status=status)
        return "complete"


def build_graph() -> StateGraph:
    """
    Build and compile the complete LangGraph workflow.
    
    Flow:
    1. strategist -> (condition: profile complete?) -> job_scout
    2. job_scout -> matchmaker
    3. matchmaker -> validator
    4. validator -> (condition: needs re-search?) -> job_scout or END
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create graph with AgentState
    workflow = StateGraph(AgentState)
    
    # Add all nodes
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("job_scout", job_scout_node)
    workflow.add_node("matchmaker", matchmaker_node)
    workflow.add_node("validator", validator_node)
    
    # Set entry point
    workflow.set_entry_point("strategist")
    
    # Add conditional edge from strategist
    # If profile is complete -> job_scout, else -> END (awaiting user input)
    workflow.add_conditional_edges(
        "strategist",
        should_continue_to_job_scout,
        {
            "continue": "job_scout",
            "await_info": END,  # End and wait for user to provide missing info
        }
    )
    
    # Add edge from job_scout to matchmaker
    workflow.add_edge("job_scout", "matchmaker")
    
    # Add edge from matchmaker to validator
    workflow.add_edge("matchmaker", "validator")
    
    # Add conditional edge from validator
    # If violations found -> job_scout (re-search), else -> END
    workflow.add_conditional_edges(
        "validator",
        should_research,
        {
            "research": "job_scout",  # Loop back to job_scout with adjusted params
            "complete": "final_validation",  # Go to final validation checkpoint
        }
    )
    
    # Add final validation node (adds human-readable summary)
    workflow.add_node("final_validation", final_validation_node)
    
    # Final validation goes to END
    workflow.add_edge("final_validation", END)
    
    # Compile graph with interrupt before final_validation
    # Note: LangGraph API handles persistence automatically, so no custom checkpointer needed
    app = workflow.compile(
        interrupt_before=["final_validation"]  # Interrupt before final validation for Studio inspection
    )
    
    logger.info("graph_built_and_compiled with interrupt before final_validation")
    
    return app


# Create the graph instance
graph = build_graph()
