# %%
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# %%
df = {
    'Classifier': ['MarbleNet', 'S-RF', 'SSRF', 'SSRF', 'MarbleNet', 'S-RF', 'SSRF', 'SSRF'],
    'DatasetSize': [88000, 88000, 44000, 17600, 89600, 89600, 44800, 17920],  # 100% for CNN and RandomForest, 50% and 20% for Self-Supervised RF
    'FeatureReps': [1, 5, 5, 5, 1, 5, 5, 5],  # CNN uses 1 feature, RF models use 5 features
    'Dataset': ['GSC', 'GSC', 'GSC', 'GSC', 'GSC-Freesound', 'GSC-Freesound', 'GSC-Freesound', 'GSC-Freesound'],
    'Accuracy': [99.8, 99.79, 99.72, 99.65, 98.83, 99.13, 98.93, 98.15]  # Example F1-scores, replace with your actual values
}

# %%
# Step 1: Create the ANOVA model, encoding categorical variables
formula = 'Accuracy ~ C(Classifier) + C(DatasetSize) + C(FeatureReps) + C(Dataset)'
model = ols(formula, data=df).fit()

# Step 2: Perform ANOVA
anova_results = anova_lm(model, typ=2)  # Type 2 ANOVA
print(anova_results)

# %%
# Step 3: Tukey's HSD test
# For Tukey's HSD, we need to group the factors and F1 scores
tukey_data = pd.DataFrame(df)

# %%
# Example: Applying Tukey's HSD to the 'Classifier' factor
tukey = pairwise_tukeyhsd(endog=tukey_data['Accuracy'], groups=tukey_data['DatasetSize'], alpha=0.05)

# Display results
print(tukey)

# %%
# Tukey's HSD for classifiers
tukey_classifier = pairwise_tukeyhsd(endog=tukey_data['Accuracy'], groups=tukey_data['Classifier'], alpha=0.05)
print(tukey_classifier)

# To perform ranking, we extract the results from Tukey's HSD
results = pd.DataFrame(data=tukey_classifier._results_table.data[1:], columns=tukey_classifier._results_table.data[0])

# Display results for classifier comparisons
print(results)

# Sort based on mean difference (descending) to rank
ranked_results = results[results['reject'] == True].sort_values('meandiff', ascending=False)

# Display ranked classifiers by significance
print(ranked_results)

# %%
# 1. Tukey's HSD for DatasetSize
tukey_size = pairwise_tukeyhsd(endog=tukey_data['Accuracy'], groups=tukey_data['DatasetSize'], alpha=0.05)
print(tukey_size)

# Extract results for DatasetSize Tukey's HSD
size_results = pd.DataFrame(data=tukey_size._results_table.data[1:], columns=tukey_size._results_table.data[0])

# Rank based on mean difference (descending) and filter significant comparisons for DatasetSize
ranked_size = size_results[size_results['reject'] == True].sort_values('meandiff', ascending=False)
print("\nRanked DatasetSize based on Tukey's HSD mean differences:")
print(ranked_size)

# 2. Tukey's HSD for FeatureReps
tukey_feature = pairwise_tukeyhsd(endog=tukey_data['Accuracy'], groups=tukey_data['FeatureReps'], alpha=0.05)
print(tukey_feature)

# Extract results for FeatureReps Tukey's HSD
feature_results = pd.DataFrame(data=tukey_feature._results_table.data[1:], columns=tukey_feature._results_table.data[0])

# Rank based on mean difference (descending) and filter significant comparisons for FeatureReps
ranked_feature = feature_results[feature_results['reject'] == True].sort_values('meandiff', ascending=False)
print("\nRanked FeatureReps based on Tukey's HSD mean differences:")
print(ranked_feature)

# 3. Ranking groups based on F1-scores
# Calculate mean F1-scores for each group (Classifier, DatasetSize, FeatureReps)
ranked_by_f1_classifier = tukey_data.groupby('Classifier')['Accuracy'].mean().sort_values(ascending=False)
ranked_by_f1_size = tukey_data.groupby('DatasetSize')['Accuracy'].mean().sort_values(ascending=False)
ranked_by_f1_feature = tukey_data.groupby('FeatureReps')['Accuracy'].mean().sort_values(ascending=False)

# Display the ranking based on F1-scores
print("\nRanking Classifiers based on Accuracy:")
print(ranked_by_f1_classifier)

print("\nRanking DatasetSize based on Accuracy:")
print(ranked_by_f1_size)

print("\nRanking FeatureReps based on Accuracy:")
print(ranked_by_f1_feature)


