# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import shap

# %%
# Load and shuffle data
train_data = pd.read_csv("./CSVs_for_testing/combined/two_class_data/combined_twoclass_train.csv")
data_shuffeled = train_data.sample(frac=1).reset_index(drop=True)
X_train = data_shuffeled.drop('Class', axis=1)
y_train = data_shuffeled['Class']

test_data = pd.read_csv("./CSVs_for_testing/combined/two_class_data/combined_twoclass_test.csv")
data_shuffeled_test = test_data.sample(frac=1).reset_index(drop=True)
X_test = data_shuffeled_test.drop('Class', axis=1)
y_test = data_shuffeled_test['Class']

# %%
# Split data into labeled and unlabeled
X_label = X_train[:int(X_train.shape[0] / 5)]
y_label = y_train[:int(X_train.shape[0] / 5)]
X_unlabelled = X_train[int(X_train.shape[0] / 5):]

# %%
# Semi-supervised learning
semi_model = RandomForestClassifier()
while True:
    semi_model.fit(X_label, y_label)
    X_unlabelled.reset_index(drop=True, inplace=True)
    y_pred = semi_model.predict_proba(X_unlabelled)
    index = [index for index, x in enumerate(np.max(y_pred, axis=1)) if x > 0.90]
    if len(index) == 0:
        break
    temp = X_unlabelled.iloc[index]
    X_unlabelled.drop(index, inplace=True)
    pred = pd.Series(semi_model.predict(temp))
    X_label = X_label.append(temp, ignore_index=True)
    y_label = y_label.append(pred, ignore_index=True)

# %%
print(X_label.shape)
print(X_unlabelled.shape)

# %%
# Hyperparameter tuning
param_dist = {'n_estimators': randint(10, 100), 'max_depth': randint(1, 10)}
rf = RandomForestClassifier()
rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
rand_search.fit(X_label, y_label)
best_rf = rand_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)


# %%
# Baseline metrics with all features
baseline_accuracy = accuracy_score(y_test, y_pred_rf)
baseline_f1 = f1_score(y_test, y_pred_rf)

# %%
print("baseline accuracy and f1-score", baseline_accuracy,baseline_f1)

# %%

# Feature importance using Gini impurity
gini_importances = list(best_rf.feature_importances_)
gini_sorted_features = sorted(zip(gini_importances, X_label.columns), reverse=True)

# %%
# Output the sorted feature importances for verification
print("Sorted gini feature importances:")
for importance, feature in gini_sorted_features:
    print(f"Feature: {feature}, Importance: {importance}")

# %%
# Feature importance using TreeSHAP
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test)

# If shap_values is a list (multi-class), sum across all classes to get a single importance value per feature
if isinstance(shap_values, list):
    # Sum or average the absolute SHAP values across all classes for each feature
    shap_importances = np.sum(np.abs(shap_values), axis=0).mean(axis=0)
else:
    # For binary classification, shap_values is already a 2D array, so just take the mean
    shap_importances = np.mean(np.abs(shap_values), axis=0)

# Now, shap_importances should be a 1D array where each entry corresponds to the importance of a feature
shap_sorted_features = sorted(zip(shap_importances, X_test.columns), reverse=True)

# Output the sorted feature importances for verification
print("Sorted SHAP feature importances:")
for importance, feature in shap_sorted_features:
    print(f"Feature: {feature}, Importance: {importance}")


# %%
# Function to evaluate performance with a subset of features
def evaluate_subset(features):
    X_train_subset = X_label[features]
    X_test_subset = X_test[features]
    # Initialize and train a new RandomForest model
    param_dist = {'n_estimators': randint(10, 100), 'max_depth': randint(1, 10)}
    rf = RandomForestClassifier()
    model = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
    model.fit(X_train_subset, y_label)
    y_pred_subset = model.predict(X_test_subset)
    acc = accuracy_score(y_test, y_pred_subset)
    f1 = f1_score(y_test, y_pred_subset)
    return acc, f1



# %%
# Iterative feature addition and performance evaluation
def performance_vs_features(sorted_features, method_name):
    accuracies = []
    f1_scores = []
    features_selected = []
    for i in range(1, len(sorted_features) + 1):
        current_features = [f for _, f in sorted_features[:i]]
        acc, f1 = evaluate_subset(current_features)
        accuracies.append(acc)
        f1_scores.append(f1)
        features_selected.append(current_features)
        if acc >= baseline_accuracy and f1 >= baseline_f1:
            break
    return accuracies, f1_scores, features_selected


# %%
# Evaluate performance with Gini and TreeSHAP feature selection
gini_accuracies, gini_f1_scores, gini_selected_features = performance_vs_features(gini_sorted_features, "Gini")
shap_accuracies, shap_f1_scores, shap_selected_features = performance_vs_features(shap_sorted_features, "TreeSHAP")


# %%
# Plotting the results
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(gini_accuracies) + 1), gini_accuracies, label="Gini Accuracy")
plt.plot(range(1, len(shap_accuracies) + 1), shap_accuracies, label="TreeSHAP Accuracy")
plt.axhline(baseline_accuracy, color='r', linestyle='--', label="Baseline Accuracy")
plt.xlabel('Number of Features')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Features')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(gini_f1_scores) + 1), gini_f1_scores, label="Gini F1-Score")
plt.plot(range(1, len(shap_f1_scores) + 1), shap_f1_scores, label="TreeSHAP F1-Score")
plt.axhline(baseline_f1, color='r', linestyle='--', label="Baseline F1-Score")
plt.xlabel('Number of Features')
plt.ylabel('F1-Score')
plt.title('F1-Score vs Number of Features')
plt.legend()
plt.show()

# %%
# Print selected features for both methods
print("Gini selected features matching baseline metrics:", gini_selected_features[-1])
print("TreeSHAP selected features matching baseline metrics:", shap_selected_features[-1])

# %%

print("Gini final accuracy:", gini_accuracies[-1] )
print("Gini final F1-score:", gini_f1_scores[-1])

print("TreeSHAP final accuracy:", shap_accuracies[-1])
print("TreeSHAP final F1-score:", shap_f1_scores[-1])


# %%
# Plot Gini and SHAP importances for selected and additional features
def plot_importances(sorted_features, importances, selected_features, method_name, extra_features=5):
    selected_count = len(selected_features)
    features_to_plot = sorted_features[:selected_count + extra_features]
    
    feature_names = [f[1] for f in features_to_plot]
    feature_importances = [f[0] for f in features_to_plot]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances, align='center')
    plt.yticks(range(len(feature_importances)), feature_names)
    plt.xlabel(f'{method_name} Importance')
    plt.title(f'{method_name} Importances for Selected and Additional Features')
    plt.gca().invert_yaxis()
    plt.show()

# Plot Gini importances
plot_importances(gini_sorted_features, gini_importances, gini_selected_features[-1], "Gini")

# Plot SHAP importances
plot_importances(shap_sorted_features, shap_importances, shap_selected_features[-1], "TreeSHAP")


# %%
# Plot Gini and SHAP importances side-by-side for selected and additional features
def plot_side_by_side_importances(gini_sorted_features, shap_sorted_features, selected_features, extra_features=5):
    selected_count = len(selected_features)
    features_to_plot = [f for _, f in gini_sorted_features[:selected_count + extra_features]]

    gini_importances = {f: imp for imp, f in gini_sorted_features}
    shap_importances = {f: imp for imp, f in shap_sorted_features}
    
    gini_values = [gini_importances[f] for f in features_to_plot]
    shap_values = [shap_importances[f] for f in features_to_plot]

    x = np.arange(len(features_to_plot))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 8))
    rects1 = ax.bar(x - width/2, gini_values, width, label='Gini Importance')
    rects2 = ax.bar(x + width/2, shap_values, width, label='TreeSHAP Importance')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Gini vs TreeSHAP Feature Importance')
    ax.set_xticks(x)
    ax.set_xticklabels(features_to_plot, rotation=90)
    ax.legend()

    fig.tight_layout()
    plt.show()

# Plot side-by-side importances
plot_side_by_side_importances(gini_sorted_features, shap_sorted_features, gini_selected_features[-1])


# %%
def plot_feature_group_importances(gini_sorted_features, shap_sorted_features, group_names, extra_features=5):
    selected_count = len(group_names)
    features_to_plot = [f for _, f in gini_sorted_features[:selected_count + extra_features]]

    # Extract importance scores for Gini and SHAP
    gini_importances = {f: imp for imp, f in gini_sorted_features}
    shap_importances = {f: imp for imp, f in shap_sorted_features}

    gini_values = [gini_importances[f] for f in features_to_plot]
    shap_values = [shap_importances[f] for f in features_to_plot]

    x = np.arange(len(features_to_plot))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 8))
    rects1 = ax.bar(x - width/2, gini_values, width, label='Gini Importance')
    rects2 = ax.bar(x + width/2, shap_values, width, label='TreeSHAP Importance')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Gini vs TreeSHAP Feature Importance')
    ax.set_xticks(x)
    ax.set_xticklabels(features_to_plot, rotation=90)
    ax.legend()

    fig.tight_layout()
    plt.show()

# Assuming you have already identified group names
group_names = X_train.columns  # or a list of feature names if they differ
plot_feature_group_importances(gini_sorted_features, shap_sorted_features, group_names)



