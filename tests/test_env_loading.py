"""
Test script to verify environment variable loading.
"""

import os
from dotenv import load_dotenv

load_dotenv()
print(f"Pinecone Key exists: {bool(os.getenv('PINECONE_API_KEY'))}")
print(f"Pinecone Index Name: {os.getenv('PINECONE_INDEX_NAME', 'Not set')}")
print(f"Embedding Service URL: {os.getenv('EMBEDDING_SERVICE_URL', 'Not set')}")

