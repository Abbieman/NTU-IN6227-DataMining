# NTU-IN6227-DataMining

### Setup

Before running the notebook, install the required packages:

```bash
pip install -r requirements.txt
```

### Assignment 1: Classification — Comparing Decision Tree and Random Forest Models

This assignment focuses on supervised learning through classification algorithms.
The goal is to predict whether an individual’s income exceeds $50K per year based on demographic and employment features from the Census Income dataset.

##### Objectives

1. Implement Decision Tree and Random Forest classifiers using the scikit-learn library.
2. Preprocess and encode the dataset (handle missing values, categorical encoding, normalization).
3. Split the data into training and test sets and train both models.
4. Evaluate model performance using accuracy, precision, recall, and F1-score.
5. Compare the two models in terms of predictive accuracy and computational efficiency.

##### Summary

- Dataset: Census Income Dataset (UCI Machine Learning Repository)
- Algorithms: Decision Tree, Random Forest
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score
- Tools: Python, scikit-learn, pandas, matplotlib

This assignment demonstrates how ensemble learning (Random Forest) improves prediction stability and accuracy over a single Decision Tree by reducing overfitting.

### Assignment 2: Association Rule Mining — Comparing Apriori Algorithm with Brute-Force Approach

This assignment aims to gain practical experience with association rule mining algorithms and to compare the performance of an optimized algorithm (Apriori) with a brute-force baseline.

##### Objectives

1. Implement the Apriori algorithm using the mlxtend library.
2. Prepare multiple datasets (D1–D5) of varying transaction and item sizes from the Groceries Dataset.
3. Measure the execution time for frequent itemset and rule generation.
4. Estimate the expected runtime of a brute-force approach using theoretical complexity.
5. Plot and compare the measured (Apriori) and estimated (Brute-Force) execution times.

##### Summary

- Dataset: Groceries Dataset (Kaggle)
- Algorithms: Apriori, Brute-Force (estimated)
- Evaluation Metric: Execution Time
- Tools: Python, pandas, mlxtend, matplotlib

This assignment shows that Apriori scales efficiently due to its pruning and candidate generation mechanisms, while brute-force methods grow quadratically with data size.
