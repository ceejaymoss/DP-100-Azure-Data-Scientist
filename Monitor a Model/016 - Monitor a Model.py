#
# Load Workspace
from azureml.core import Workspace

# Load the workspace from the saved config file
ws = Workspace.from_config()
print('Ready to work with', ws.name)
#

#
# Prepare a model for deployment
from azureml.core import Experiment
from azureml.core import Model
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from azureml.core import Dataset

# Upload data files to the default datastore
default_ds = ws.get_default_datastore()
default_ds.upload_files(files=['./data/diabetes.csv', './data/diabetes2.csv'],
                       target_path='diabetes-data/',
                       overwrite=True,
                       show_progress=True)

#Create a tabular dataset from the path on the datastore
print('Creating dataset...')
data_set = Dataset.Tabular.from_delimited_files(path=(default_ds, 'diabetes-data/*.csv'))

# Register the tabular dataset
print('Registering dataset...')
try:
    data_set = data_set.register(workspace=ws, 
                               name='diabetes dataset',
                               description='diabetes data',
                               tags = {'format':'CSV'},
                               create_new_version=True)
except Exception as ex:
    print(ex)

# Create an Azure ML experiment in your workspace
experiment = Experiment(workspace=ws, name='mslearn-train-diabetes')
run = experiment.start_logging()
print("Starting experiment:", experiment.name)

# load the diabetes dataset
print("Loading Data...")
diabetes = data_set.to_pandas_dataframe()

# Separate features and labels
X, y = diabetes[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values, diabetes['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a decision tree model
print('Training a decision tree model')
model = DecisionTreeClassifier().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

# Save the trained model
model_file = 'diabetes_model.pkl'
joblib.dump(value=model, filename=model_file)
run.upload_file(name = 'outputs/' + model_file, path_or_stream = './' + model_file)

# Complete the run
run.complete()

# Register the model
print('Registering model...')
run.register_model(model_path='outputs/diabetes_model.pkl', model_name='diabetes_model',
                   tags={'Training context':'Inline Training'},
                   properties={'AUC': run.get_metrics()['AUC'], 'Accuracy': run.get_metrics()['Accuracy']})

# Get the registered model
model = ws.models['diabetes_model']

print('Model trained and registered.')
#

#
# Deploy a model as a Web Service
import os

folder_name = 'diabetes_service'

# Create a folder for the web service files
experiment_folder = './' + folder_name
os.makedirs(experiment_folder, exist_ok=True)

print(folder_name, 'folder created.')

# Set path for scoring script
script_file = os.path.join(experiment_folder,"score_diabetes.py")
#

#
# Entry script used to score new data
%%writefile $script_file
import json
import joblib
import numpy as np
from azureml.core.model import Model

# Called when the service is loaded
def init():
    global model
    # Get the path to the deployed model file and load it
    model_path = Model.get_model_path('diabetes_model')
    model = joblib.load(model_path)

# Called when a request is received
def run(raw_data):
    # Get the input data as a numpy array
    data = json.loads(raw_data)['data']
    np_data = np.array(data)
    # Get a prediction from the model
    predictions = model.predict(np_data)
    
    # print the data and predictions (so they'll be logged!)
    log_text = 'Data:' + str(data) + ' - Predictions:' + str(predictions)
    print(log_text)
    
    # Get the corresponding classname for each prediction (0 or 1)
    classnames = ['not-diabetic', 'diabetic']
    predicted_classes = []
    for prediction in predictions:
        predicted_classes.append(classnames[prediction])
    # Return the predictions as JSON
    return json.dumps(predicted_classes)
#

#
# Conda configuration file
from azureml.core.conda_dependencies import CondaDependencies 

# Add the dependencies for our model (AzureML defaults is already included)
myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

# Save the environment config as a .yml file
env_file = folder_name + "/diabetes_env.yml"
with open(env_file,"w") as f:
    f.write(myenv.serialize_to_string())
print("Saved dependency info in", env_file)

# Print the .yml file
with open(env_file,"r") as f:
    print(f.read())
#

#
# Deploy the service
from azureml.core.webservice import AciWebservice, Webservice
from azureml.core.model import Model
from azureml.core.model import InferenceConfig

# Configure the scoring environment
inference_config = InferenceConfig(runtime= "python",
                                   entry_script=script_file,
                                   conda_file=env_file)

service_name = "diabetes-service-app-insights"
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1, memory_gb = 1)
aci_service = Model.deploy(workspace=ws,
                           name= service_name,
                           models= [model],
                           inference_config= inference_config,
                           deployment_config=deployment_config)
aci_service.wait_for_deployment(show_output = True)
print(aci_service.state)
#

#
# Enable app insights
# Enable AppInsights
aci_service.update(enable_app_insights=True)
print('AppInsights enabled!')
#

#
# Use the web service
endpoint = aci_service.scoring_uri
print(endpoint)
#

#
# HTTP request
import requests
import json

# Create new data for inferencing
x_new = [[2,180,74,24,21,23.9091702,1.488172308,22],
         [0,148,58,11,179,39.19207553,0.160829008,45]]

# Convert the array to a serializable list in a JSON document
input_json = json.dumps({"data": x_new})

# Set the content type
headers = { 'Content-Type':'application/json' }

# Get the predictions
predictions = requests.post(endpoint, input_json, headers = headers)
print(predictions.status_code)
if predictions.status_code == 200:
    predicted_classes = json.loads(predictions.json())
    for i in range(len(x_new)):
        print ("Patient {}".format(x_new[i]), predicted_classes[i] )
#

# You can now view the data logged for the service endpoint
# On the overview page, click the associated Applications Insights resource.
# On app insights page click logs
# Paste the following query and click run
## traces
##  | where message == "STDOUT"
##    and customDimensions.["Service Name"] == "diabetes-service-app-insights"
##  | project timestamp, customDimensions.Content

# Can take up to 5 minutes for logs to load

#
# Delete the service
try:
    aci_service.delete()
    print('Service deleted.')
except Exception as ex:
    print(ex.message)
#

