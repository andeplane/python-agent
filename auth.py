from getpass import getpass
from cognite.client import CogniteClient, ClientConfig
from cognite.client.credentials import OAuthClientCredentials
import os

def create_client():
    CLUSTER_NAME="api"
    COGNITE_PROJECT="andershaf"
    COGNITE_BASE_URL=f"https://{CLUSTER_NAME}.cognitedata.com"
    COGNITE_TENANT_ID=f"5409fe5e-39ad-4448-90ad-6688455011bf"
    COGNITE_TOKEN_URL=f"https://login.microsoftonline.com/{COGNITE_TENANT_ID}/oauth2/v2.0/token"
    COGNITE_TOKEN_SCOPES=f"https://{CLUSTER_NAME}.cognitedata.com/.default"
    COGNITE_CLIENT_ID=os.getenv("ANDERSHAF_CLIENTID")
    COGNITE_CLIENT_SECRET=os.getenv("ANDERSHAF_CLIENTSECRET")

    oauth_provider = OAuthClientCredentials(
        token_url=f"https://login.microsoftonline.com/{COGNITE_TENANT_ID}/oauth2/v2.0/token",
        client_id=COGNITE_CLIENT_ID,
        client_secret=COGNITE_CLIENT_SECRET,
        scopes=[f"https://{CLUSTER_NAME}.cognitedata.com/.default"]
    )
    config = ClientConfig(
        project=COGNITE_PROJECT,
        client_name="Cognite Python Tutorial",
        credentials=oauth_provider,
        base_url=f"https://{CLUSTER_NAME}.cognitedata.com",
        timeout=120
    )
    client = CogniteClient(config=config)
    return client