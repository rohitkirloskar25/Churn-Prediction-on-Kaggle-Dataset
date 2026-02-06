This project explores the complete machine learning pipeline for predicting customer churn, with a specific focus on data preprocessing, feature encoding, and advanced strategies for handling imbalanced datasets.

# 📂 Project Structure
## 1. Data Source
WA_Fn-UseC_-Telco-Customer-Churn.csv: The primary dataset containing customer demographics, service details, and the target Churn label.

## 2. Jupyter Notebooks
*churn_tutorial.ipynb*: 
  - The end-to-end master pipeline.
  - Data Cleaning & EDA: Handling missing values and visualizing trends.
  - Preprocessing: Implementation of Label Encoding and data splitting.
  - Balancing: Applying SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance.
  - Tuning: Hyperparameter optimization using GridSearchCV.

*imbalance_churn.ipynb*: 
  - A comparative study on sampling techniques.
  - Explores multiple strategies including Oversampling (SMOTE), Undersampling (RandomUnderSampler, NearMiss), and Hybrid methods (SMOTETomek).

*un-encoded_churn.ipynb*: 
  - Tests the feasibility of training models directly on raw categorical data.
  - Concludes why numerical encoding is a mathematical necessity for standard Machine Learning algorithms.

## 3. Model Artifacts
*customer_churn_model.pkl* and *encoders_under.pkl*: The saved version of the final trained Random Forest classifier.

# 🚀 How to Understand
Pipeline: Start with *churn_tutorial.ipynb* to understand the full workflow.
Experimentation: Refer to imbalance_churn.ipynb for different class balancing strategy
