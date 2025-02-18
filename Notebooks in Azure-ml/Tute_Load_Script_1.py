# run this in terminal first
# pip install azureml-sdk[notebooks, automl, explain]

#
from azureml.core import Workspace

ws = Workspace.from_config()
#

#
# Impoprt the needed experiment package from the Azure ML SDK
from azureml import Experiment

# Create an experiment variable with whatever name you want
experiement = Experiment(workspace = ws, name = "insert Workspace name here")

# Start the experiment by starting to log metrics
run = experiment.start_logging()

# Load a data set and count the rows
data = pd.read_csv('insert data CSV name here')
row_count = (len(data))

# Log the row count as a metric
run.log('observations', row_count)

# End the experiment
run.complete()
#

#
import json
 # Get logged metrics
metrics = run.get_metrics()
print(json.dumps(metrics, indent=2))
#

#
# Import the needed packages from the Azure ML SDK
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies

# Create a Python environment for the experiment
env = Environment(" Insert Environment Here ")

# Make sure the needed packages are installed 
packages = CondaDependencies.create(conda_packages['scikit-learn', 'pip'], pip_packages=['azureml-defaults'])
env.python.conda_dependencies = packages

# Load the script we want to run
script_config = ScriptRunConfig(source_directory='.',
                                script= 'Insert Training Script here',
                                environment=env)

# Run the experiment
experiment = Experiment(workspace=ws, name='Insert Experiment name here')
run = experiment.submit(config=script_config)
run.wait_for_completion()
#

