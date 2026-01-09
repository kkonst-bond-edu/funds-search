"""
Test script to verify ClassificationAgent with a single vacancy.

Usage:
    python tests/test_classification_single.py
    
    Or with custom title and description:
    python tests/test_classification_single.py "Senior Backend Engineer" "We are looking for a senior backend engineer with 5+ years of experience in Python and microservices. Remote work available."
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.orchestrator.agents.classification import ClassificationAgent


async def test_classification(title: str, description: str):
    """
    Test classification with a single vacancy.
    
    Args:
        title: Job vacancy title
        description: Job vacancy description
    """
    print("=" * 60)
    print("ClassificationAgent Test")
    print("=" * 60)
    print(f"\nTitle: {title}")
    print(f"\nDescription:\n{description}")
    print("\n" + "-" * 60)
    print("Running classification...")
    print("-" * 60 + "\n")
    
    try:
        # Initialize the classification agent
        classifier = ClassificationAgent()
        
        # Classify the vacancy
        result = await classifier.classify(title, description)
        
        # Print the result as formatted JSON
        print("Classification Result:")
        print(json.dumps(result, indent=2))
        print("\n" + "=" * 60)
        print("Test completed successfully!")
        print("=" * 60)
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during classification: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main entry point."""
    # Default test vacancy if no arguments provided
    default_title = "Senior Backend Engineer"
    default_description = """We are looking for a Senior Backend Engineer to join our AI startup. 
You will be responsible for building scalable microservices using Python and FastAPI.
The role requires 5+ years of experience with distributed systems and cloud infrastructure.
Remote work is available for qualified candidates.
We are a Series A company in the AI industry, focused on enterprise solutions."""

    # Parse command line arguments
    if len(sys.argv) >= 3:
        title = sys.argv[1]
        description = sys.argv[2]
    elif len(sys.argv) == 2:
        print("Usage: python test_classification_single.py <title> <description>")
        print("Or: python test_classification_single.py (uses default example)")
        sys.exit(1)
    else:
        title = default_title
        description = default_description
        print("Using default test vacancy. Provide title and description as arguments to test custom vacancy.\n")

    # Run the async test
    result = asyncio.run(test_classification(title, description))
    
    # Exit with error code if classification failed
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    main()
