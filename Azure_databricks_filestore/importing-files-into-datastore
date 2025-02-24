# Run in jpyter notebooks on azure machine learning studio
# a single hash indents a single notebook code snippet

#
import azureml.core
from azureml import Workspace, Datastore 

ws = Workspace.from_config()
#

#
from azureml.core import Dataset
datastore_blob = Datastore_get(ws, 'dp100_1304')
csv_paths = [
    (datastore_blob, 'data/files/current_data.csv'),
    (datastore_blob, data/files/archive/*.csv)]

dataset_tab = Dataset.Tabular.from_delimited_files(path=csv.paths)
dataset_tab = dataset_tab.register(workspace=ws, name='dp100_1304_tabular')
#

#
from azureml.core import Dataset

datastore_file = Datastore.get(ws, 'dp100_1303_file')

dataset_file = Dataset.File.from_files(path=(datastore_file, 'data/files/images*.jpg'))
dataset_file = dataset_file.register(workspace=ws, name='dp100_1304_file')
#
