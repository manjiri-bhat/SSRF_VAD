# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
train_data = pd.read_csv("./CSVs_for_testing/combined/three_class_data/combined_three_class_train.csv")
train_data.head()

# %%
data_shuffeled = train_data.sample(frac=1).reset_index(drop=True) ##data shuffling for randomized input
X_train = data_shuffeled.drop('Class', axis=1)   
y_train = data_shuffeled['Class']

# %%
test_data = pd.read_csv("./CSVs_for_testing/combined/three_class_data/combined_three_class_test.csv")
test_data.head()

# %%
data_shuffeled_test = test_data.sample(frac=1).reset_index(drop=True) ##data shuffling for randomized input
X_test = data_shuffeled_test.drop('Class', axis=1)   
y_test = data_shuffeled_test['Class']

# %%
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay 

#y_train = LabelEncoder().fit_transform(y_train)
#y_test = LabelEncoder().fit_transform(y_test)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {'n_estimators': randint(10,100),
              'max_depth': randint(1,10)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf, 
                                 param_distributions = param_dist, 
                                 n_iter=5, 
                                 cv=5)

# Fit the random search object to the data
rand_search.fit(X_train, y_train)

# %%
# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

# %%
y_pred_rf = best_rf.predict(X_test)


# %%
cm = confusion_matrix(y_test, y_pred_rf)
print(confusion_matrix(y_test,y_pred_rf))  
print(classification_report(y_test,y_pred_rf, digits=4)) 

# Save the confusion matrix plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure(figsize=(8, 6))
disp.plot(cmap='Blues')
plt.title(f'S-RF')
plt.show()

# %%
import shap

# %%
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# %%
#Plotting model-agnostic variable importance
shap.summary_plot(shap_values, X_test)

# %%
rf_importances = list(best_rf.feature_importances_)
feature_list = list(X_train.columns)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, rf_importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
