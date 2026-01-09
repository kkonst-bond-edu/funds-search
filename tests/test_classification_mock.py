"""
Test script to verify ClassificationAgent with a mock vacancy.

This script tests the classification agent with a specific mock vacancy
and verifies that the classification returns valid (non-default) values.
"""

import asyncio
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from apps.orchestrator.agents.classification import ClassificationAgent


async def test_classification():
    """
    Test classification with a mock vacancy and verify the results.
    """
    # Define mock vacancy
    title = "Senior Lead Software Engineer, Backend"
    description = "We are looking for a Python expert to build scalable AI infrastructure using Kubernetes and AWS. Experience with FastAPI and PostgreSQL is required."
    
    print("=" * 70)
    print("ClassificationAgent Test - Mock Vacancy")
    print("=" * 70)
    print(f"\nTitle: {title}")
    print(f"\nDescription: {description}")
    print("\n" + "-" * 70)
    print("Loading environment variables...")
    print("-" * 70)
    
    # Load environment variables from .env file
    load_dotenv()
    print("✓ Environment variables loaded\n")
    
    print("-" * 70)
    print("Initializing ClassificationAgent...")
    print("-" * 70)
    
    try:
        # Initialize the classification agent
        agent = ClassificationAgent()
        print("✓ ClassificationAgent initialized successfully\n")
        
        print("-" * 70)
        print("Running classification...")
        print("-" * 70)
        
        # Call classify method
        result = await agent.classify(title, description)
        
        # Print the raw JSON response
        print("\n" + "=" * 70)
        print("Raw JSON Response:")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        print()
        
        # Verify the results
        print("=" * 70)
        print("Verification:")
        print("=" * 70)
        
        category = result.get("category", "Unknown")
        industry = result.get("industry", "Unknown")
        experience_level = result.get("experience_level", "Unknown")
        remote_option = result.get("remote_option", False)
        
        # Check each field
        category_valid = category not in ["Other", "Unknown"]
        industry_valid = industry not in ["Other", "Unknown"]
        experience_level_valid = experience_level not in ["Other", "Unknown"]
        
        print(f"Category: {category} {'✓' if category_valid else '✗ (Invalid: should not be Other or Unknown)'}")
        print(f"Industry: {industry} {'✓' if industry_valid else '✗ (Invalid: should not be Other or Unknown)'}")
        print(f"Experience Level: {experience_level} {'✓' if experience_level_valid else '✗ (Invalid: should not be Other or Unknown)'}")
        print(f"Remote Option: {remote_option} {'✓' if isinstance(remote_option, bool) else '✗ (Invalid: should be boolean)'}")
        
        # Overall result
        all_valid = category_valid and industry_valid and experience_level_valid and isinstance(remote_option, bool)
        
        print("\n" + "=" * 70)
        if all_valid:
            print("✓ All verifications passed!")
            print("=" * 70)
            return True
        else:
            print("✗ Some verifications failed!")
            print("=" * 70)
            return False
            
    except Exception as e:
        print(f"\n❌ Error during classification: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    success = asyncio.run(test_classification())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
