#!/bin/bash
# Script to run integration tests for the funds-search API

set -e

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Running integration tests..."
echo "Project root: $PROJECT_ROOT"

# Run integration tests
cd "$PROJECT_ROOT"
pytest tests/integration -v

echo "Integration tests completed!"

