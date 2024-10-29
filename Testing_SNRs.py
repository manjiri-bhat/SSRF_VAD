# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.stats import randint
from itertools import combinations
from itertools import permutations

# %%
# Load your SNR data from different CSV files
snr_paths = [
"./CSVs_for_testing/combined/combined_10_snr.csv", 
"./CSVs_for_testing/combined/combined_5_snr.csv", 
"./CSVs_for_testing/combined/combined_0_snr.csv", 
"./CSVs_for_testing/combined/combined_min5_snr.csv",
"./CSVs_for_testing/combined/combined_min10_snr.csv" ]

# Load SNR data
snr_data = []
for path in snr_paths:
    df = pd.read_csv(path)
    print(f"Loaded SNR data from {path}. Columns: {df.columns.tolist()}")
    # Check if 'Class' column exists
    if 'Class' not in df.columns:
        raise KeyError(f"'Class' column not found in SNR data file: {path}")
    snr_data.append(df)

# Define SNR levels
snr_levels = [10, 5, 0, -5, -10]
snr_data_dict = dict(zip(snr_levels, snr_data))



# %%
# Load noise data
noise_path = "./CSVs_for_testing/combined/combined_us8K_only.csv"

# %%
# Load noise data
noise_data = pd.read_csv(noise_path)
X_noise = noise_data.drop('Class', axis=1)
y_noise = noise_data['Class']  # Assuming 'Class' column is also in the noise data

# %%
# Split noise data
num_samples_per_set = 1000
X_noise_train = X_noise.iloc[:num_samples_per_set]
y_noise_train = y_noise.iloc[:num_samples_per_set]

X_noise_test = X_noise.iloc[num_samples_per_set:num_samples_per_set*2]
y_noise_test = y_noise.iloc[num_samples_per_set:num_samples_per_set*2]

X_noise_unlabelled = X_noise.iloc[num_samples_per_set*2:]
y_noise_unlabelled = y_noise.iloc[num_samples_per_set*2:]

# Define the number of SNR levels to use for training
num_unlabelled_snr = 3

# %%
# Initialize a set to track processed combinations
processed_combinations = set()

# Generate all combinations of SNR levels (unique combinations)
for train_snr in snr_levels:
    remaining_snr = [s for s in snr_levels if s != train_snr]
    for unlabelled_snr_set in combinations(remaining_snr, 3):
        test_snr = [s for s in remaining_snr if s not in unlabelled_snr_set][0]

        # Avoid duplicate combinations by sorting the unlabelled set
        combination = (train_snr, tuple(sorted(unlabelled_snr_set)), test_snr)

        if combination in processed_combinations:
            continue

        processed_combinations.add(combination)

        # Load the data
        X_train = snr_data_dict[train_snr].drop('Class', axis=1)
        y_train = snr_data_dict[train_snr]['Class']

        # Add noise samples to the training data
        X_train = pd.concat([X_train, X_noise_train], ignore_index=True)
        y_train = pd.concat([y_train, y_noise_train], ignore_index=True)

        # Combine unlabelled data
        X_unlabelled = pd.concat([snr_data_dict[snr].drop('Class', axis=1) for snr in unlabelled_snr_set], ignore_index=True)
        y_unlabelled = pd.concat([snr_data_dict[snr]['Class'] for snr in unlabelled_snr_set], ignore_index=True)

        # Add noise samples to the unlabelled data
        X_unlabelled = pd.concat([X_unlabelled, X_noise_unlabelled], ignore_index=True)
        y_unlabelled = pd.concat([y_unlabelled, y_noise_unlabelled], ignore_index=True)

        # Testing data
        X_test = snr_data_dict[test_snr].drop('Class', axis=1)
        y_test = snr_data_dict[test_snr]['Class']

        # Add noise samples to the testing data
        X_test = pd.concat([X_test, X_noise_test], ignore_index=True)
        y_test = pd.concat([y_test, y_noise_test], ignore_index=True)

        # Initialize the self-supervised random forest model
        semi_model = RandomForestClassifier()

        # Self-Learning Algorithm
        X_label, y_label = X_train, y_train

        count_iter = 0
        while True:
            # Train the model
            semi_model.fit(X_label, y_label)

            # Get probability predictions on unlabeled data
            y_pred_proba = semi_model.predict_proba(X_unlabelled)

            # Get samples where probability > 0.9 for at least one class
            max_probs = np.max(y_pred_proba, axis=1)
            index = np.where(max_probs > 0.90)[0]
            if len(index) == 0:
                break

            # Handle index properly with reset_index
            temp = X_unlabelled.iloc[index].reset_index(drop=True)
            pred = pd.Series(semi_model.predict(temp)).reset_index(drop=True)

            # Drop high probability samples from unlabeled data and append to labeled data
            X_unlabelled = X_unlabelled.drop(index).reset_index(drop=True)
            X_label = pd.concat([X_label, temp], ignore_index=True)
            y_label = pd.concat([y_label, pred], ignore_index=True)
            count_iter += 1

        print(f"Training on SNR: {train_snr}, Unlabelled on SNRs: {unlabelled_snr_set}, Testing on SNR: {test_snr}")
        print("No of iterations for pseudo-labeling:", count_iter)
        print("Unlabelled samples = ", len(X_unlabelled))

        # Randomized Search for hyperparameters with Cross-validation
        param_dist = {'n_estimators': randint(1,100), 'max_depth': randint(1, 10)}
        random_search = RandomizedSearchCV(estimator=RandomForestClassifier(), 
                                           param_distributions=param_dist, 
                                           n_iter=5, 
                                           cv=5, 
                                           scoring='accuracy', 
                                           n_jobs=-1, 
                                           return_train_score=True)

        # Fit the model
        random_search.fit(X_label, y_label)

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

        # Save the confusion matrix plot
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        plt.figure(figsize=(8, 6))
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix - Train SNR: {train_snr}, Test SNR: {test_snr}')
        plt.show()

        # Extract cross-validation results
        cv_results = random_search.cv_results_

        # Extract the mean test scores for each parameter combination
        mean_test_scores = cv_results['mean_test_score']

        # Calculate the average CV accuracy
        average_cv_accuracy = np.mean(mean_test_scores)
        print(f'Average CV Accuracy: {average_cv_accuracy:.4f}')



