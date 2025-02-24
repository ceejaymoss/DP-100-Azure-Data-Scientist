# Run in jpyter notebooks on azure machine learning studio
# a single hash indents a single notebook code snippet

#
import azureml.core
from azureml import Workspace, Datastore 

ws = Workspace.from_config()
#

#
# Azure blob storage details
blob_account_name = os.getenv("BLOB_ACCOUNTNAME", "Replace me")
blob_account_key = os.getenv("BLOB_ACCOUNT_KEY", "Replace me")
blob_container_name = os.getenv("BLOB_CONTAINER", "Replace me")

blob_datastore = Datastore.register_azure_blob_container(
    workspace = ws,
    datastore_name = "dp100_1303_blob",
    container_name = blob_container_name,
    account_name = blob_account_name,
    account_key = blob_account_key)
#

#
# Azure file share storage details
file_account_name = os.getenv("FILE_SHARE_ACCOUNTNAME", "Replace me")
file_account_key = os.getenv("FILE_SHARE_ACCOUNT_KEY", "Replace me")
file_container_name = os.getenv("FILE_SHARE_CONTAINER", "Replace me")

file_datastore = Datastore.register_azure_file_share(
    workspace = ws,
    datastore_name = "dp100_1303_blob",
    file_share_name = file_container_name,
    account_name = file_account_name,
    account_key = file_account_key)
#

#
# Data lake resource details
subscription_id = os.getenv("ADL_SUBSCRIPTION", "Replace me")
resource_group = os.getenv("ADL_RESOURCE_GROUP", "Replace me")
datalake_account_name = os.getenv("ADLSGEN2_ACCOUNTNAME", "Replace me")

# Service principle (app registration) details
tenant_id = os.getenv("ADLSGEN2_TENANT", "Replace me")
client_id = os.getenv("ADLSGEN2_CLIENTID", "Replace me")
client_secret = os.getenv("ADLSGEN2_CLIENT_SECRET", "Replace me")

# Register the data lake
adlsgen2_datastore = Datastore.register_azure_data_lake_gen2(
    workspace = ws,
    datastore_name = "dp100_1303_datalake",
    account_name = datalake_account_name,
    file_system = 'dlfileshare',
    tenant_id = tenant_id,
    client_id = client_id,
    client_secret = client_secret)
#
