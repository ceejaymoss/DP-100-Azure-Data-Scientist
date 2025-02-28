# Input into terminal
#!pip show azureml-explain-model azureml-interpret

#
# Explain a model
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

# load the diabetes dataset
print("Loading Data...")
data = pd.read_csv('data/diabetes.csv')

# Separate features and labels
features = ['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']
labels = ['not-diabetic', 'diabetic']
X, y = data[features].values, data['Diabetic'].values

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Train a decision tree model
print('Training a decision tree model')
model = DecisionTreeClassifier().fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))

print('Model trained.')
#

#
# Get an explainer for the model
from interpret.ext.blackbox import TabularExplainer

# "features" and "classes" fields are optional
tab_explainer = TabularExplainer(model,
                             X_train, 
                             features=features, 
                             classes=labels)
print(tab_explainer, "ready!")
#

#
# Get global feature importance
# you can use the training data or the test data here
global_tab_explanation = tab_explainer.explain_global(X_train)

# Get the top features by importance
global_tab_feature_importance = global_tab_explanation.get_feature_importance_dict()
for feature, importance in global_tab_feature_importance.items():
    print(feature,":", importance)
#

#
# Get local feature importance
# Get the observations we want to explain (the first two)
X_explain = X_test[0:2]

# Get predictions
predictions = model.predict(X_explain)

# Get local explanations
local_tab_explanation = tab_explainer.explain_local(X_explain)

# Get feature names and importance for each possible label
local_tab_features = local_tab_explanation.get_ranked_local_names()
local_tab_importance = local_tab_explanation.get_ranked_local_values()

for l in range(len(local_tab_features)):
    print('Support for', labels[l])
    label = local_tab_features[l]
    for o in range(len(label)):
        print("\tObservation", o + 1)
        feature_list = label[o]
        total_support = 0
        for f in range(len(feature_list)):
            print("\t\t", feature_list[f], ':', local_tab_importance[l][o][f])
            total_support += local_tab_importance[l][o][f]
        print("\t\t ----------\n\t\t Total:", total_support, "Prediction:", labels[predictions[o]])
#

