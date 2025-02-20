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

## MODIFIED script for regrate edit
# Load the script we want to run
#script_config = ScriptRunConfig(source_directory='.',
#                                script= 'Insert Training Script here',
#                                arguements = ['reg-rate', 0.1],
#                                environment=env)
