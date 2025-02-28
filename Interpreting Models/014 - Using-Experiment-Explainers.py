#
# Adding explainability to a model training experiment
import azureml.core
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to use Azure ML {} to work with {}'.format(azureml.core.VERSION, ws.name))
#

#
# Train and explain a model using an experiment
import os, shutil
from azureml.core import Experiment

# Create a folder for the experiment files
experiment_folder = 'diabetes_train_and_explain'
os.makedirs(experiment_folder, exist_ok=True)

# Copy the data file into the experiment folder
shutil.copy('data/diabetes.csv', os.path.join(experiment_folder, "diabetes.csv"))
#

#
# Create a training script
%%writefile $experiment_folder/diabetes_training.py
# Import libraries
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# Import Azure ML run library
from azureml.core.run import Run

# Import libraries for model explanation
from azureml.interpret import ExplanationClient
from interpret.ext.blackbox import TabularExplainer

# Get the experiment run context
run = Run.get_context()

# load the diabetes dataset
print("Loading Data...")
data = pd.read_csv('diabetes.csv')

features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
labels = ['not-diabetic', 'diabetic']

# Separate features and labels
X, y = data[features].values, data['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a decision tree model
print('Training a decision tree model')
model = DecisionTreeClassifier().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
run.log('AUC', np.float(auc))

os.makedirs('outputs', exist_ok=True)
# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=model, filename='outputs/diabetes.pkl')

# Get explanation
explainer = TabularExplainer(model, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(explanation, comment='Tabular Explanation')

# Complete the run
run.complete()
#

#
# Run the experiment
from azureml.core import Experiment, ScriptRunConfig, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.widgets import RunDetails


# Create a Python environment for the experiment
explain_env = Environment("explain-env")

# Create a set of package dependencies (including the azureml-interpret package)
packages = CondaDependencies.create(conda_packages=['scikit-learn','pandas','pip'],
                                    pip_packages=['azureml-defaults','azureml-interpret'])
explain_env.python.conda_dependencies = packages

# Create a script config
script_config = ScriptRunConfig(source_directory=experiment_folder,
                      script='diabetes_training.py',
                      environment=explain_env) 

# submit the experiment
experiment_name = 'mslearn-diabetes-explain'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)
RunDetails(run).show()
run.wait_for_completion()
#

#
# retireve feature importance values
from azureml.interpret import ExplanationClient

# Get the feature explanations
client = ExplanationClient.from_run(run)
engineered_explanations = client.download_model_explanation()
feature_importances = engineered_explanations.get_feature_importance_dict()

# Overall feature importance
print('Feature\tImportance')
for key, value in feature_importances.items():
    print(key, '\t', value)
#
