# Test of Resilience: Zero Results Handling

## Overview

This test suite verifies that the system properly handles cases where search returns 0 results, ensuring:
1. The agent receives a clear message about 0 results
2. Filters are relaxed on retry
3. The system doesn't enter an infinite loop (max 2-3 retries)

## Changes Made

### 1. Added Retry Limit to `job_scout_node` (`apps/orchestrator/graph/nodes.py`)

**Key Changes:**
- Added `max_search_retries = 3` constant to limit retry attempts
- Check retry limit at the start of `job_scout_node` to prevent infinite loops
- Track `_search_attempt` in `search_params` to count retries
- Enhanced relaxation instruction to include attempt count
- Added retry limit check after search to prevent further retries

**Behavior:**
- When `_search_attempt >= max_search_retries`, the node stops retrying and returns `awaiting_info` status with a message asking the user to relax criteria
- The relaxation instruction now includes attempt count: `"attempt {search_attempt + 1}/{max_search_retries}"`
- When 0 results are detected, the message explicitly states: `"Search returned 0 results. Please relax filters. (Attempt X/3)"`

### 2. Enhanced Zero Results Detection

The node already had logic to detect 0 results from `ToolMessage`:
- Checks if the last message is a `ToolMessage` with `count: 0`
- Sets `needs_filter_relaxation = True` when 0 results are detected
- Adds a relaxation instruction to `conversation_history` for the LLM

### 3. Test Suite (`tests/test_resilience_zero_results.py`)

**Test Cases:**
1. `test_zero_results_triggers_filter_relaxation`: Verifies that 0 results trigger filter relaxation and a new tool call with relaxed filters
2. `test_retry_limit_prevents_infinite_loop`: Verifies that the system stops retrying after max retries
3. `test_filter_relaxation_removes_salary_min`: Verifies that `salary_min` is removed or lowered when filters are relaxed
4. `test_zero_results_message_format`: Verifies that the correct message format is generated
5. `test_retry_limit_enforced`: Verifies that the retry limit is enforced and search is not called when limit is reached

## Flow Diagram

```
User Query → job_scout_node
    ↓
Search with strict filters (salary_min=200k, remote_only, etc.)
    ↓
Tool returns count: 0
    ↓
job_scout_node detects 0 results
    ↓
Adds relaxation instruction to conversation_history
    ↓
Status: awaiting_info, missing_info: "Search returned 0 results. Please relax filters. (Attempt 1/3)"
    ↓
[Graph flow back to job_scout_node if user provides input or system retries]
    ↓
job_scout_node runs again with relaxation instruction
    ↓
LLM sees relaxation instruction and makes new tool call with relaxed filters (e.g., removes salary_min)
    ↓
Tool returns results OR count: 0 again
    ↓
If count: 0 again and attempt < 3: Repeat
If attempt >= 3: Stop and ask user to relax criteria manually
```

## Transition Between job_scout_node and validator_node

The `validator_node` has logic to skip validation when:
- `candidate_pool` is empty
- `_filters_relaxed` flag is not set

This prevents the validator from interfering with the retry logic when 0 results are returned.

## Key Messages

**When 0 results are detected:**
```
"Search returned 0 results. Please relax filters. (Attempt 1/3)"
```

**When retry limit is reached:**
```
"Search attempted 3 times with no results. Please relax your search criteria 
(e.g., lower salary expectations, expand company stages, or remove location restrictions)."
```

## Running the Tests

```bash
# Run all resilience tests
pytest tests/test_resilience_zero_results.py -v

# Run specific test
pytest tests/test_resilience_zero_results.py::TestResilienceZeroResults::test_zero_results_triggers_filter_relaxation -v
```

## Notes

- The retry limit in `job_scout_node` (`_search_attempt`, max 3) is separate from the retry limit in `should_research` (`_research_iterations`, max 2)
- The `_search_attempt` counter is incremented each time `job_scout_node` runs
- The relaxation instruction is added to `conversation_history`, which the LLM sees in subsequent calls
- The LLM is responsible for actually relaxing filters based on the instruction
