# Pure Health Project Code

This repository contains code for constructing cohorts of Danish adults aged 50–69 and training machine learning models to predict early mortality using linked national register data.

The project integrates **R** (cohort construction and feature aggregation) and **Python** (model training, threshold selection, and evaluation)

---

## Overview

**Objective**  
To predict mortality among Danish adults aged 50–69 and to evaluate model performance under strong class imbalance, with particular emphasis on recall-oriented metrics.

**Current scope**  
The repository currently supports:
- Cohort construction and feature aggregation
- XGBoost model training using randomized hyperparameter search
- Threshold selection based on F2-score
- Evaluation on held-out test data

Model interpretation (e.g. SHAP, decomposition) is not yet included in this version of the repository.

---

## Data and Cohort Design

- Population: Danish adults aged **50–69**
- Age bands:
  - **50–54**
  - **55–59**
  - **60–64**
  - **65–69**
- Observation period: **1995–2023**
- Outcome: `early_death`  
  (1 if death occurs within the age band, 0 otherwise)

Cohorts are constructed separately.

---

## Script Descriptions

### 1. `01_cohort_data_prep.R`

Builds the analysis cohorts of Danish adults aged **50–69**, stratified into four age bands:

- **50–54**
- **55–59**
- **60–64**
- **65–69**

The script loads yearly Danish panel data covering **1995–2023** and identifies individuals belonging to each age band in the relevant years.

For each age band, the script:
- Filters individuals by age
- Defines the outcome variable `early_death`  
  (1 if death occurs within the age band, 0 otherwise)
- Constructs long-term aggregated features using rolling windows:
  - **0–5 years (late)**
  - **6–10 years (mid)**
  - **11–15 years (early)**  
  covering diagnoses, hospitalizations, medication use, and socioeconomic characteristics

After cohort-specific processing, the script outputs a **single combined dataset** containing all four age bands, spanning the full observation period **1995–2023**.

The final dataset is saved in **Parquet format** and serves as the input for downstream modeling.

---

### 2. `02_split_format_function.py`

Defines the function `split_and_format_data()` for preparing cohort data for machine learning.

The function:
- Loads a Parquet dataset
- Optionally filters individuals by sex
- Separates ID and outcome variables
- Splits data into training and test sets (default 70/30), optionally stratified by outcome and calendar year

Variable types are automatically detected and preprocessing is applied as follows:
- **Categorical variables**:
  - Imputed with the constant value `"missing"`
  - One-hot encoded (unknown categories ignored)
- **Numeric variables**:
  - Passed through without transformation. Numeric variables are intentionally not scaled or imputed, allowing XGBoost to handle missingness and feature scaling internally.

The function returns:
- Training and test feature matrices
- Training and test outcome vectors
- ID columns for train and test sets
- A fitted preprocessing pipeline (`ColumnTransformer`) for use in modeling

---

### 3. `03_train_xgboost_function.py`

Implements `train_xgboost_model_random()` for training an XGBoost classifier using **randomized hyperparameter search**.

Key features:
- Uses a scikit-learn `Pipeline` combining preprocessing and an `XGBClassifier`
- Trains models using **RandomizedSearchCV**
- Employs **Stratified K-fold cross-validation**
- Supports multi-metric evaluation with refitting based on a user-specified metric:
  - `f1`, `f2`, `recall`, `precision`, `roc_auc`, or `pr_auc`
- Handles class imbalance via the `scale_pos_weight` argument

The function returns:
- The best fitted model
- The selected hyperparameters
- A default classification threshold (0.5)

---

### 4. `04_run_xgboost.ipynb`

Implements the end-to-end modeling workflow for a specific cohort (e.g. sex × age band).

The notebook performs the following steps:

1. **Data loading and splitting**
   - Loads a cohort-specific Parquet dataset
   - Use the split format functtion to filter by sex and split data into training and test sets stratified by outcome and year

2. **Class imbalance handling**
   - Computes `scale_pos_weight` from the training data.

3. **Model training**
   - Trains an XGBoost model using `train_xgboost_model_random()`
   - Optimizes hyperparameters via randomized search
   - Selects the best model using **F2-score**

4. **Threshold selection**
   - Computes precision–recall curves on the training data
   - Selects the classification threshold that maximizes **F2-score**
   - Restricts the threshold to the interval **[0.3, 0.7]**

5. **Evaluation**
   - Evaluates performance on the test set using:
     - F1 and F2 scores
     - Precision and recall
     - ROC-AUC and PR-AUC
     - Accuracy and balanced accuracy
     - Confusion matrix and specificity

6. **Predicted probability gap**
   - Computes mean predicted mortality risk among:
     - survivors (`y = 0`)
     - deaths (`y = 1`)
   - Reports the difference as a model-implied risk gap

---

## Outputs

- Trained XGBoost models
- Performance metrics 
- Confusion matrix and derived statistics
- Estimated predicted mortality risk gaps

---

## Notes

- All random processes are seeded to ensure reproducibility within cohorts.
