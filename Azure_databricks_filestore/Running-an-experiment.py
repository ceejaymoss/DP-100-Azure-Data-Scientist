
# Run in jpyter notebooks on azure machine learning studio
# a single hash indents a single notebook code snippet

#
import azureml.core
from azureml import Workspace, Datastore 

ws = Workspace.from_config()
#

#
from azureml.core import Dataset

dataset_tab = Dataset.get_by_name(ws, 'dp100_1304_tabular')

df = dataset_tab.to_pandas_dataframe()
print(df.head())
#

#
from azureml.core import Dataset

dataset_tab = Dataset.get_by_name(ws, 'dp100_1304_tabular')

for file_path in dataset_file.to_path():
    print(file_path)
#

#
from azureml.core import Environment, ScriptRunConfig, Experiment
from azureml.core.environment import CondaDependencies

experiment = Experiment(ws, 'dp100_1305_experiment')
cluster = ws.compute.targets['dp100cc']

env = Environment('my_env')
packages = CondaDependencies.Create(
    conda_packages['pip'],
    pip_packages=['azureml_defaults', 'azureml-dataprep[pandas]'])
env.python.conda_dependencies = packages

script_config = ScriptRunConfig(
    source_directory='.',
    script='dp100_1305_script.py',
    arguements=['--ds', dataset_tab.as_named_input("input_ds")],
    compute_target=cluster,
    environment=env)

script_run = experiment.submit(script.config)
#

