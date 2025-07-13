# Automated SAS Token Generation Plan for Azure Container

Based on your requirements and the information provided, I'll outline a comprehensive plan to automate the SAS token generation for your Azure Container. This solution will focus on the files you mentioned (main.py and rag_assistant_with_history_copy.py) and will work within the constraints of a containerized app in Azure Web App Services.

## Current Understanding

1. You need to regenerate the SAS token daily
2. You have access to both Azure CLI and Azure PowerShell on the server
3. The application is containerized and runs in Azure Web App Services
4. The target Azure Storage account is in resource group `rg-sbx-19-switzerlandnorth-usr-capozzol`
5. The container name is `bigolpotofdater`

## Solution Architecture

I propose a solution with two main components:

1. __SAS Token Generation Script__: A Python script that uses the Azure SDK to generate a new SAS token
2. __Integration with Web App__: A mechanism to make the generated token available to the application

### Option 1: Azure Function with Managed Identity

This approach uses an Azure Function with a timer trigger to generate the SAS token daily and store it in an environment variable accessible to your web app.

#### Components:

- Azure Function with timer trigger (daily execution)
- Managed Identity for secure access to Azure Storage
- App Configuration or Key Vault to store the generated token
- Web App configuration to read from App Configuration/Key Vault

#### Flow:

1. Azure Function runs daily, generates a new SAS token
2. Token is stored in App Configuration or Key Vault
3. Web App reads the token at startup or on-demand
4. No .env file modification needed - environment variables are used

### Option 2: Web App Startup Script

This approach incorporates the SAS token generation into the web app's startup process.

#### Components:

- Python script to generate SAS token
- Startup command in the web app configuration
- Managed Identity for the web app

#### Flow:

1. When the container starts, a startup script runs
2. Script generates a new SAS token using Azure SDK
3. Token is stored in an environment variable accessible to the application
4. Application reads the token from the environment variable

## Detailed Implementation Plan

Let's proceed with Option 2 as it's simpler and doesn't require additional Azure resources:

### 1. Create SAS Token Generation Script

Create a new file `generate_sas_token.py` with the following content:

```python
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
```

### 2. Modify main.py to Use Dynamic SAS Token

Update the main.py file to dynamically get the SAS token:

```python
# Add this function to main.py
def get_sas_token():
    """Get the SAS token from environment or generate a new one if needed"""
    sas_token = os.getenv("SAS_TOKEN", "")
    
    # If no token is available or it's expired, generate a new one
    if not sas_token:
        try:
            # Import the generation function
            from generate_sas_token import generate_sas_token
            sas_token = generate_sas_token()
            # Optionally set the environment variable for future use
            os.environ["SAS_TOKEN"] = sas_token
            logger.info("Generated new SAS token")
        except Exception as e:
            logger.error(f"Error generating SAS token: {e}")
            # Return empty string or a default token
            sas_token = ""
    
    return sas_token

# Then replace the current sas_token loading with:
# sas_token = os.getenv("SAS_TOKEN", "")
# with:
sas_token = get_sas_token()
```

### 3. Update Dockerfile to Include the Script

Ensure your Dockerfile includes the new script:

```dockerfile
# Add this to your Dockerfile
COPY generate_sas_token.py /app/generate_sas_token.py
RUN chmod +x /app/generate_sas_token.py

# Install Azure SDK dependencies
RUN pip install azure-identity azure-storage-blob
```

### 4. Configure Web App Startup Command

In your Azure Web App configuration, set the startup command to generate the token before starting the application:

```bash
python /app/generate_sas_token.py && python /app/main.py
```

### 5. Configure Managed Identity

1. Enable System Assigned Managed Identity for your Web App
2. Grant the Managed Identity the "Storage Blob Data Contributor" role on your storage account

## Deployment Steps

1. Create the `generate_sas_token.py` script
2. Modify `main.py` to use the dynamic token generation
3. Update your Dockerfile to include the script and dependencies
4. Build and push the updated container image
5. Configure the Web App with the startup command
6. Enable and configure Managed Identity permissions

## Testing Plan

1. Local testing:

   - Run the script locally with Azure CLI credentials
   - Verify token generation and validity

2. Container testing:

   - Build the container locally
   - Test the startup flow
   - Verify the application can access the token

3. Production testing:

   - Deploy to Azure Web App
   - Verify token generation on startup
   - Test document downloads to confirm token works

## Monitoring and Maintenance

1. Add logging to track token generation and expiry
2. Set up alerts for token generation failures
3. Consider implementing a fallback mechanism for token generation failures

Would you like me to make any adjustments to this plan before we proceed with implementation? Once you're satisfied with the approach, please toggle to

Act Mode (⌘⇧A) so I can help you implement this solution.

So I cant use System Assigned Managed Identity. I have to use the other option, API Key? Key Access?

**Checkpoint

Compare

Restore

API Request$0.0000

# Revised SAS Token Generation Plan Using Storage Account Key

Since System Assigned Managed Identity is not an option, we’ll switch to using your storage account’s access key (API key) to generate SAS tokens. The high-level flow remains similar to Option 2 (startup script), but credentials come from an environment variable.

## Key Changes

1. __Credential Source__

   - Instead of ManagedIdentityCredential, the script will read the storage account key from an environment variable (e.g., `AZURE_STORAGE_ACCOUNT_KEY`).

2. __Azure Web App Configuration__

   - In your Azure Web App’s __Configuration > Application Settings__, add:

     - `AZURE_STORAGE_ACCOUNT_NAME` = `capozzol02storage`
     - `AZURE_STORAGE_ACCOUNT_KEY` = `<your-storage-account-key>`

3. __generate_sas_token.py__\
   Update to use the account key directly:

   ```python
   import os
   from datetime import datetime, timedelta
   from azure.storage.blob import generate_container_sas, ContainerSasPermissions

   def generate_sas_token():
       account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
       account_key  = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
       container    = "bigolpotofdater"
       if not account_name or not account_key:
           raise RuntimeError("Missing storage account name or key")

       # Valid for 24 hours
       start = datetime.utcnow()
       expiry = start + timedelta(days=1)
       permissions = ContainerSasPermissions(read=True, list=True)

       sas = generate_container_sas(
           account_name=account_name,
           container_name=container,
           account_key=account_key,
           permission=permissions,
           start=start,
           expiry=expiry
       )
       return f"?{sas}"

   if __name__ == "__main__":
       token = generate_sas_token()
       # Write to temp file for startup sourcing
       with open("/tmp/sas_token.env", "w") as f:
           f.write(f"SAS_TOKEN={token}")
       print(token)
   ```

4. __Dockerfile__

   - Ensure `generate_sas_token.py` is copied into the image.
   - Install `azure-storage-blob`.

5. __Startup Command__\
   In your Web App’s Startup Command:

   ```bash
   python /app/generate_sas_token.py && source /tmp/sas_token.env && python /app/main.py
   ```

6. __main.py__\
   Replace direct `os.getenv("SAS_TOKEN")` with:

   ```python
   import os, subprocess

   def get_sas_token():
       token = os.getenv("SAS_TOKEN")
       if not token:
           # If unset, source the temp file
           env_file = "/tmp/sas_token.env"
           if os.path.exists(env_file):
               for line in open(env_file):
                   key, _, val = line.strip().partition("=")
                   os.environ[key] = val
               token = os.getenv("SAS_TOKEN", "")
       return token

   sas_token = get_sas_token()
   ```

## Next Steps

1. Confirm you have the __storage account key__ for `capozzol02storage`.
2. Set `AZURE_STORAGE_ACCOUNT_NAME` and `AZURE_STORAGE_ACCOUNT_KEY` in your Web App’s Application Settings.
3. Review the updated `generate_sas_token.py` script.
4. Toggle to __Act mode__ when ready, and I will implement these changes.

Let me know if you need any adjustments before proceeding.
