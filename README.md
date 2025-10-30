# Pure Health Project Code

This repository contains scripts that build and analyze a cohort of Danish adults aged 60–69 to predict mortality using national register data and machine learning.  
The workflow integrates R and Python scripts to prepare data, train models, and interpret results.

---

## Overview

**Objective**  
To predict mortality among Danish adults aged 60–69 using linked national registers and to quantify how socioeconomic, health, and demographic factors contribute to predicted risk.

**Approach**  
- Cohort construction and variable aggregation in R  
- Model preparation, training, and evaluation in Python  
- Model interpretation through SHAP feature importance and a Shapley–Owens decomposition  
- Final visualization and domain-level summaries in R

**Outputs**  
Interpretable model explanations, feature importance plots, and decompositions of inequality in predicted mortality risk.

---

## Workflow Summary

1. **Cohort construction (R):** Identify individuals aged 50–69, define the mortality outcome, and compute long-term aggregated features (e.g., diagnoses, income, hospitalizations).  
2. **Data preparation (Python):** Format data for modeling, split into train/test sets, and handle class imbalance.  
3. **Model training:** Logistic Regression and XGBoost trained via grid search and stratified cross-validation.  
4. **Evaluation:** Compute precision, recall, and F1-score; create confusion matrices and precision–recall curves.  
5. **Interpretation:** Generate SHAP feature importance and Shapley–Owens decomposition analyses.  
6. **Visualization (R):** Produce domain- and time-level visualizations of SHAP and decomposition results.

---

---

## Repository Structure

###  preprocessing/
- **cohort_data_prep.R** — Build cohort (ages 50–69), define outcome, aggregate long-term variables.  
- **splitformat.py** — Split data and handle preprocessing (scaling, encoding, imputation).  
- **resample.py** — Functions for class imbalance (upsample / downsample).  

###  models/
- **xgboost_model.py** — XGBoost grid-search training and model selection.  
- **logistic.py** — Logistic Regression with grid search and coefficient extraction.  

###  explain/
- **evaluate_model.py** — Evaluate model performance and save metrics/plots.  
- **feature_importance.py** — Compute and visualize SHAP feature importances.  
- **decompose.py** — Shapley–Owens decomposition of predicted risk gap.  
- **model_interpretation_plots.R** — Visualize SHAP and decomposition outputs.  

###  run/
- **cohort_run.py** — Main pipeline integrating all components.  

---


## Script Descriptions

### 1. `preprocessing/cohort_data_prep.R`
Builds the analysis cohort of people aged 50–69 and prepares data for mortality prediction.  
Loads yearly Danish panel data, filters for individuals in the age range, and creates yearly Parquet files.  
Generates rolling variables summarizing 5-, 10-, and 15-year histories of diagnoses, hospitalizations, and socioeconomic data.  
Defines the outcome variable `early_death` (1 if death occurs between 50–69, 0 otherwise) and saves the prepared dataset.

---

### 2. `preprocessing/splitformat.py`
Defines the function `split_and_format_data()` for preparing data for machine learning.  
Loads a Parquet dataset, removes unwanted columns, and splits data into train/test sets (optionally stratified by year).  
Automatically detects numeric and categorical variables and applies preprocessing:
- Imputation (mean for numeric, "missing" for categorical)  
- Standardization of numeric variables  
- One-hot encoding of categorical variables  
Returns training and test data, ID columns, and a fitted preprocessing pipeline.

---

### 3. `preprocessing/resample.py`
Provides two functions to address class imbalance:
- `upsample_minority()` – increases the minority class by random duplication (with replacement).  
- `downsample_majority()` – reduces the majority class by random removal (without replacement).  

---

### 4. `models/xgboost_model.py`
Implements `train_xgboost_model()` for grid-search training of an XGBoost classifier.  
Tests all hyperparameter combinations using stratified K-fold cross-validation.  
Computes F1, precision, and recall scores, and selects the best-performing model based on a chosen metric (default: F1).  
Returns the trained model and its best parameters.

---

### 5. `models/logistic.py`
Provides utilities for Logistic Regression training and interpretation.  
- `train_logistic_model()` tests combinations of regularization penalties (L1/L2) and strengths (C values) using stratified cross-validation.  
  The function averages F1, precision, and recall scores and selects the best configuration.  
- `extract_coefficients()` extracts and ranks model coefficients by absolute magnitude to identify key predictors.

---

### 6. `explain/evaluate_model.py`
Defines a function for evaluating trained binary classifiers on test data.  
Predicts class probabilities and labels, computes precision, recall, and F1-score, and saves:  
- Test metrics (CSV)  
- Individual predictions (CSV)  
- Confusion matrix (PDF)  
- Precision–recall curve (PDF)

---

### 7. `explain/feature_importance.py`
Calculates and visualizes SHAP feature importances for a trained classifier.  
Computes SHAP values for four subsets: all cases, accurate predictions, true positives, and all deaths.  
Saves both beeswarm and bar plots (top 20 features) for each subset and returns a summary DataFrame of mean absolute SHAP values.

---

### 8. `explain/decompose.py`
Implements a simplified Shapley–Owens decomposition to explain the average difference in predicted risk between survivors and deaths.  
The function:
1. Splits data into outcome groups (y=0, y=1).  
2. Counterfactually replaces feature values in the positive group with those from the negative group.  
3. Measures how predicted probabilities change.  
4. Aggregates feature-level effects as the explained part of the risk gap.  
Outputs a CSV (`decomp.csv`) and returns results for further visualization.

---

### 9. `run/cohort_run.py`
The master pipeline script that orchestrates the full workflow:
1. Load and split the prepared cohort data.  
2. Rebalance the training set using up/down-sampling.  
3. Train the XGBoost model via grid search (optimizing F1 with precision constraint).  
4. Evaluate model performance.  
5. Compute SHAP feature importances.  
6. Run Shapley–Owens decomposition to explain predicted risk inequality.  
All outputs are saved under `results/cohort/`.

---

### 10. `explain/model_interpretation_plots.R`
Loads and visualizes SHAP and Shapley–Owens results.  
Plots how feature importance differs across model versions and aggregates decomposition results by conceptual domains (healthcare, socioeconomic, family) and temporal dimensions (proximal vs distal).  
Generates histograms and polar (pie-style) charts to visualize how each variable group contributes to the explained difference in predicted mortality risk.

---

## Outputs

- Preprocessed cohort data (Parquet files)  
- Model evaluation metrics (CSV + plots)  
- SHAP feature importance plots (PDF)  
- Shapley–Owens decomposition results (`decomp.csv`)  
- Domain-level visualizations (R outputs)

---




