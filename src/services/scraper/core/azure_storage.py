"""
Azure Blob Storage integration for storing raw HTML content.
Supports both Connection String and Service Principal (Identity-based) authentication.
"""

import os
import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
import re

try:
    from azure.storage.blob import BlobServiceClient, BlobClient
    from azure.core.exceptions import AzureError
except ImportError:
    BlobServiceClient = None
    BlobClient = None
    AzureError = None

try:
    from azure.identity import DefaultAzureCredential
except ImportError:
    DefaultAzureCredential = None

logger = logging.getLogger(__name__)


def upload_html_to_azure_blob(
    fund_name: str,
    url: str,
    html_content: str,
    container_name: Optional[str] = None,
) -> Optional[str]:
    """
    Upload raw HTML content to Azure Blob Storage.
    
    Supports two authentication methods (in priority order):
    1. Connection String (AZURE_STORAGE_CONNECTION_STRING) - for backward compatibility
    2. Service Principal via DefaultAzureCredential (AZURE_STORAGE_ACCOUNT_NAME) - identity-based
    
    Args:
        fund_name: Name of the fund (e.g., "a16z")
        url: URL of the scraped page
        html_content: Raw HTML content to upload
        container_name: Azure Blob Storage container name (defaults to AZURE_STORAGE_CONTAINER_NAME or "raw-scrapes")
        
    Returns:
        URL to the uploaded blob, or None if upload failed or Azure is not configured
    """
    # Check if Azure Blob Storage is configured
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    
    # Check if required packages are installed
    if BlobServiceClient is None:
        logger.warning("azure-storage-blob package not installed. Install with: pip install azure-storage-blob")
        return None
    
    blob_service_client = None

    # Priority 1: Use Connection String if available (backward compatibility)
    if connection_string:
        try:
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            logger.debug("Using Connection String authentication for Azure Blob Storage")
        except Exception as e:
            logger.error(f"Failed to initialize BlobServiceClient with connection string: {str(e)}")

    # Priority 2: Use Service Principal via DefaultAzureCredential
    if blob_service_client is None and account_name:
        if DefaultAzureCredential is None:
            logger.warning(
                "azure-identity package not installed. Install with: pip install azure-identity. "
                "Falling back to Connection String method."
            )
            return None
        
        try:
            # Build account URL from account name
            account_url = f"https://{account_name}.blob.core.windows.net"
            
            # Initialize DefaultAzureCredential
            # This will automatically use AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_CLIENT_SECRET
            # or other available authentication methods (Managed Identity, Azure CLI, etc.)
            credential = DefaultAzureCredential()
            
            # Create BlobServiceClient with account URL and credential
            blob_service_client = BlobServiceClient(
                account_url=account_url,
                credential=credential
            )
            logger.debug(f"Using Service Principal authentication for Azure Blob Storage (account: {account_name})")
        except Exception as e:
            logger.error(f"Failed to initialize BlobServiceClient with Service Principal: {str(e)}")
            return None
    
    # No authentication method available
    if blob_service_client is None:
        logger.warning(
            "Azure Blob Storage not configured. Set either:\n"
            "  - AZURE_STORAGE_CONNECTION_STRING (Connection String method), or\n"
            "  - AZURE_STORAGE_ACCOUNT_NAME (Service Principal method via DefaultAzureCredential)\n"
            "For Service Principal, also ensure AZURE_CLIENT_ID, AZURE_TENANT_ID, and AZURE_CLIENT_SECRET are set."
        )
        return None
    
    try:
        # Use default container name if not provided
        if not container_name:
            container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "raw-scrapes")
        
        # Ensure container exists
        container_client = blob_service_client.get_container_client(container_name)
        try:
            container_client.create_container()
            logger.info(f"Created Azure Blob Storage container: {container_name}")
        except Exception:
            # Container might already exist, which is fine
            pass
        
        # Extract job ID from URL
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip("/").split("/")
        job_id = path_parts[-1] if path_parts else "unknown"
        
        # Sanitize job_id for blob name
        job_id = re.sub(r'[^\w\-_]', '_', job_id)
        
        # Create date-based blob path: fund_name/YYYY-MM-DD/job_id.html
        date_str = datetime.now().strftime("%Y-%m-%d")
        blob_name = f"{fund_name}/{date_str}/{job_id}.html"
        
        # Upload HTML content
        blob_client = blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        blob_client.upload_blob(
            html_content,
            overwrite=True,
            content_settings={"content_type": "text/html; charset=utf-8"}
        )
        
        # Get the blob URL
        blob_url = blob_client.url
        
        logger.info(f"Uploaded HTML to Azure Blob Storage: {blob_url}")
        return blob_url
        
    except AzureError as e:
        logger.error(f"Azure Blob Storage error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Failed to upload HTML to Azure Blob Storage: {str(e)}", exc_info=True)
        return None
