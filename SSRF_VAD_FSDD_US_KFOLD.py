# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint

# %%
# Load training and test data
train_data = pd.read_csv("./CSVs_for_testing/combined/two_class_data/combined_twoclass_train.csv")
test_data = pd.read_csv("./CSVs_for_testing/combined/two_class_data/combined_twoclass_test.csv")

# Shuffle and split the data
data_shuffled = train_data.sample(frac=1).reset_index(drop=True)
X_train = data_shuffled.drop('Class', axis=1)
y_train = data_shuffled['Class']

data_shuffled_test = test_data.sample(frac=1).reset_index(drop=True)
X_test = data_shuffled_test.drop('Class', axis=1)
y_test = data_shuffled_test['Class']

# Labelled and unlabelled data
#X_label = X_train[:int(X_train.shape[0]*4/5)]
#y_label = y_train[:int(X_train.shape[0]*4/5)]
#X_unlabelled = X_train[int(X_train.shape[0]*4/5):]
X_label = X_train
y_label = y_train


# %%
semi_model = RandomForestClassifier()

count_iter = 0
# Self Learning Algorithm rf
while True:
    

    # Train the model
    semi_model.fit(X_label, y_label)

    # Get probability predictions on unlabeled data
    X_unlabelled.reset_index(drop=True, inplace=True)
    y_pred = semi_model.predict_proba(X_unlabelled)
    
    # get samples where probability >0.9 for atleast one class
    index = [ index for index,x in enumerate(np.max(y_pred,axis=1)) if x > 0.90]
    if len(index)==0:
        break
        
    temp = X_unlabelled.iloc[index]
    
    # drop high probability samples from unlabeled data and append to labeled data
    X_unlabelled.drop(index,inplace=True)
    
    pred = pd.Series(semi_model.predict(temp))
    
    X_label=X_label.append(temp,ignore_index=True)
    y_label=y_label.append(pred,ignore_index=True)
    count_iter+=1

# %%
print("No of iterations for pseudo-labeling:", count_iter)
print("Unlabelled samples = ", len(X_unlabelled))

# %%
# Randomized Search for n_estimators and max_depth with Cross-validation
param_dist = {'n_estimators': [25, 50, 75, 100], 'max_depth': randint(1, 10)}

# Perform randomized search with 5-fold cross-validation
random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), 
                                   param_distributions=param_dist, 
                                   n_iter=5, 
                                   cv=5, 
                                   scoring='accuracy', 
                                   n_jobs=-1, 
                                   return_train_score=True)

# Fit the model
random_search.fit(X_label, y_label)

# Extracting the results
results = random_search.cv_results_

# Plot sensitivity analysis (n_estimators vs accuracy)
mean_test_scores = results['mean_test_score']
n_estimators = param_dist['n_estimators']

plt.figure(figsize=(8, 6))
plt.plot(n_estimators, mean_test_scores[:len(n_estimators)], marker='o', linestyle='-', color='b', label='Accuracy')
plt.xlabel('n_estimators')
plt.ylabel('Accuracy')
plt.title('Random Forest Sensitivity Analysis on n_estimators')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Best model from random search
best_rf = random_search.best_estimator_
print("Best hyperparameters:", random_search.best_params_)
print("Best cross-validation accuracy:", random_search.best_score_)

# Test set prediction using the best model
y_pred_rf = best_rf.predict(X_test)

# Confusion matrix and classification report
cm = confusion_matrix(y_test, y_pred_rf)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf, digits=4))

# Plot the confusion matrix
ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')

# %%
# Extract cross-validation results
cv_results = random_search.cv_results_

# Extract the mean test scores for each parameter combination
mean_test_scores = cv_results['mean_test_score']

# Calculate the average CV accuracy
average_cv_accuracy = np.mean(mean_test_scores)

print(f'Average CV Accuracy: {average_cv_accuracy:.4f}')

# %%



