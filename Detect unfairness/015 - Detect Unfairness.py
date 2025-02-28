# In Terminal
# !pip show azureml-contrib-fairness
# !pip install --upgrade fairlearn==0.5.0

#
# Train a model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# load the diabetes dataset
print("Loading Data...")
data = pd.read_csv('data/diabetes.csv')

# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
X, y = data[features].values, data['Diabetic'].values

# Get sensitive features
S = data[['Age']].astype(int)
# Change value to represent age groups
S['Age'] = np.where(S.Age > 50, 'Over 50', '50 or younger')

# Split data into training set and test set
X_train, X_test, y_train, y_test, S_train, S_test = train_test_split(X, y, S, test_size=0.20, random_state=0, stratify=y)

# Train a classification model
print("Training model...")
diabetes_model = DecisionTreeClassifier().fit(X_train, y_train)

print("Model trained.")
#

#
# Fairlean package setup
from fairlearn.metrics import selection_rate, MetricFrame
from sklearn.metrics import accuracy_score, recall_score, precision_score

# Get predictions for the witheld test data
y_hat = diabetes_model.predict(X_test)

# Get overall metrics
print("Overall Metrics:")
# Get selection rate from fairlearn
overall_selection_rate = selection_rate(y_test, y_hat) # Get selection rate from fairlearn
print("\tSelection Rate:", overall_selection_rate)
# Get standard metrics from scikit-learn
overall_accuracy = accuracy_score(y_test, y_hat)
print("\tAccuracy:", overall_accuracy)
overall_recall = recall_score(y_test, y_hat)
print("\tRecall:", overall_recall)
overall_precision = precision_score(y_test, y_hat)
print("\tPrecision:", overall_precision)

# Get metrics by sensitive group from fairlearn
print('\nMetrics by Group:')
metrics = {'selection_rate': selection_rate,
           'accuracy': accuracy_score,
           'recall': recall_score,
           'precision': precision_score}

group_metrics = MetricFrame(metrics,
                             y_test, y_hat,
                             sensitive_features=S_test['Age'])

print(group_metrics.by_group)
#

#
# Fairness Dashboard
from fairlearn.widget import FairlearnDashboard

# View this model in Fairlearn's fairness dashboard, and see the disparities which appear:
FairlearnDashboard(sensitive_features=S_test, 
                   sensitive_feature_names=['Age'],
                   y_true=y_test,
                   y_pred={"diabetes_model": diabetes_model.predict(X_test)})
#

#
# Exclude age from the training model
# Separate features and labels
ageless = features.copy()
ageless.remove('Age')
X2, y2 = data[ageless].values, data['Diabetic'].values

# Split data into training set and test set
X_train2, X_test2, y_train2, y_test2, S_train2, S_test2 = train_test_split(X2, y2, S, test_size=0.20, random_state=0, stratify=y2)

# Train a classification model
print("Training model...")
ageless_model = DecisionTreeClassifier().fit(X_train2, y_train2)
print("Model trained.")

# View this model in Fairlearn's fairness dashboard, and see the disparities which appear:
FairlearnDashboard(sensitive_features=S_test2, 
                   sensitive_feature_names=['Age'],
                   y_true=y_test2,
                   y_pred={"ageless_diabetes_model": ageless_model.predict(X_test2)})
#

#
# Register the model
from azureml.core import Workspace, Experiment, Model
import joblib
import os

# Load the Azure ML workspace from the saved config file
ws = Workspace.from_config()
print('Ready to work with', ws.name)

# Save the trained model
model_file = 'diabetes_model.pkl'
joblib.dump(value=diabetes_model, filename=model_file)

# Register the model
print('Registering model...')
registered_model = Model.register(model_path=model_file,
                                  model_name='diabetes_classifier',
                                  workspace=ws)
model_id= registered_model.id


print('Model registered.', model_id)
#

#
# Binary classification group metric sets
from fairlearn.metrics._group_metric_set import _create_group_metric_set
from azureml.contrib.fairness import upload_dashboard_dictionary, download_dashboard_by_upload_id

#  Create a dictionary of model(s) you want to assess for fairness 
sf = { 'Age': S_test.Age}
ys_pred = { model_id:diabetes_model.predict(X_test) }
dash_dict = _create_group_metric_set(y_true=y_test,
                                    predictions=ys_pred,
                                    sensitive_features=sf,
                                    prediction_type='binary_classification')

exp = Experiment(ws, 'mslearn-diabetes-fairness')
print(exp)

run = exp.start_logging()

# Upload the dashboard to Azure Machine Learning
try:
    dashboard_title = "Fairness insights of Diabetes Classifier"
    upload_id = upload_dashboard_dictionary(run,
                                            dash_dict,
                                            dashboard_name=dashboard_title)
    print("\nUploaded to id: {0}\n".format(upload_id))

    # To test the dashboard, you can download it
    downloaded_dict = download_dashboard_by_upload_id(run, upload_id)
    print(downloaded_dict)
finally:
    run.complete()
#

#
#
from azureml.widgets import RunDetails

RunDetails(run).show()
#

#
# Mitigate unfairness in the model
from fairlearn.reductions import GridSearch, EqualizedOdds
import joblib
import os

print('Finding mitigated models...')

# Train multiple models
sweep = GridSearch(DecisionTreeClassifier(),
                   constraints=EqualizedOdds(),
                   grid_size=20)

sweep.fit(X_train, y_train, sensitive_features=S_train.Age)
models = sweep.predictors_

# Save the models and get predictions from them (plus the original unmitigated one for comparison)
model_dir = 'mitigated_models'
os.makedirs(model_dir, exist_ok=True)
model_name = 'diabetes_unmitigated'
print(model_name)
joblib.dump(value=diabetes_model, filename=os.path.join(model_dir, '{0}.pkl'.format(model_name)))
predictions = {model_name: diabetes_model.predict(X_test)}
i = 0
for model in models:
    i += 1
    model_name = 'diabetes_mitigated_{0}'.format(i)
    print(model_name)
    joblib.dump(value=model, filename=os.path.join(model_dir, '{0}.pkl'.format(model_name)))
    predictions[model_name] = model.predict(X_test)
#

#
# Fairlearn Dashboard - Decide on if age is a factor
FairlearnDashboard(sensitive_features=S_test, 
                   sensitive_feature_names=['Age'],
                   y_true=y_test,
                   y_pred=predictions)
#



