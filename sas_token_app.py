#!/usr/bin/env python
import os
import logging
from datetime import datetime, timedelta
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient, generate_container_sas, ContainerSasPermissions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sas_token():
    """Generate a SAS token for the specified container with 24-hour validity"""
    try:
        # Configuration
        account_name = "capozzol02storage"
        container_name = "bigolpotofdater"
        
        # Try to use Managed Identity first (for production)
        try:
            credential = ManagedIdentityCredential()
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=credential
            )
            logger.info("Successfully authenticated using Managed Identity")
        except Exception as e:
            logger.warning(f"Managed Identity authentication failed: {e}")
            logger.info("Falling back to DefaultAzureCredential")
            
            # Fall back to DefaultAzureCredential (works locally with az login)
            credential = DefaultAzureCredential()
            blob_service_client = BlobServiceClient(
                account_url=f"https://{account_name}.blob.core.windows.net",
                credential=credential
            )
        
        # Get account key (needed for SAS generation)
        account_key = blob_service_client.get_account_information()["account_key"]
        
        # Generate SAS token with 24-hour validity
        start_time = datetime.utcnow()
        expiry_time = start_time + timedelta(days=1)
        
        # Define permissions (read, list)
        permissions = ContainerSasPermissions(read=True, list=True)
        
        # Generate the SAS token
        sas_token = generate_container_sas(
            account_name=account_name,
            container_name=container_name,
            account_key=account_key,
            permission=permissions,
            expiry=expiry_time,
            start=start_time
        )
        
        # Format the token with the leading question mark
        formatted_token = f"?{sas_token}"
        
        logger.info(f"Generated SAS token valid from {start_time} to {expiry_time}")
        return formatted_token
        
    except Exception as e:
        logger.error(f"Error generating SAS token: {e}")
        # Return a default empty token or raise exception based on your error handling strategy
        return ""

if __name__ == "__main__":
    # Generate the token
    sas_token = generate_sas_token()
    
    # Print the token (for debugging or capturing in startup script)
    print(f"SAS_TOKEN={sas_token}")
    
    # Optionally write to a file that can be sourced
    with open("/tmp/sas_token.env", "w") as f:
        f.write(f"SAS_TOKEN={sas_token}")
    
    logger.info("SAS token generation complete")
