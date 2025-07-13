import os
from dotenv import load_dotenv
load_dotenv()
from datetime import datetime, timedelta
from azure.storage.blob import generate_container_sas, ContainerSasPermissions

def generate_sas_token():
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
    account_key  = os.getenv("AZURE_STORAGE_ACCOUNT_KEY")
    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "sage-prod-datalake")
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
    with open("/tmp/sas_token.env", "w") as f:
        f.write(f"SAS_TOKEN={token}")
    print(token)
