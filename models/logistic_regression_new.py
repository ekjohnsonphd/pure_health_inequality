"""
Utility for training a Logistic Regression model with manual grid-search
and cross-validation.

SUMMARY
--------
This script defines two functions:
1) train_logistic_model():
   - Performs grid search over Logistic Regression hyperparameters.
   - Uses stratified K-fold cross-validation for each (penalty, C) pair.
   - Computes average F1, precision, and recall across folds.
   - Selects the best-performing model based on F1 or recall and refits on full data.

2) extract_coefficients():
   - Extracts and sorts model coefficients by absolute magnitude.

CHANGES MADE
------------
1) Moved model fit/eval INSIDE the CV loop (so not only the final fold’s metrics are used).
2) Enforced row indexing with .iloc to select rows and not columns.
3) Added zero_division=0 to precision/recall to avoid warnings.
4) After selecting best params, refit a fresh model on the FULL dataset.
5) Ensured solver–penalty compatibility: 'liblinear' for L1, 'lbfgs' for L2.
6) Added optional class_weight="balanced" (commented for future use).
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score


# Function to train a logistic regression model using cross-validation to find the best penalty and C value.
def train_logistic_model(
    X,                                      # feature matrix
    y,                                      # binary target
    penalties=("l2",),                      # penalties to test ("l1", "l2")
    C_values=(0.01, 0.1, 1.0, 10.0),        # inverse regularization strengths
    random_state=42,                        # seed for reproducibility
    cv_folds=3,                             # number of stratified folds
    maximize="f1",                          # metric to optimize ("f1" or "recall")
    min_precision=None,                     # min precision if maximizing recall
):
    # Normalize to pandas for reliable .iloc row selection
    X = pd.DataFrame(X).reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)

    best_model = None
    best_score = -np.inf
    best_params = {}

    # Loop over each penalty type
    for penalty in penalties:
        # Use correct solver depending on penalty type
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        # Loop over each regularization strength
        for C in C_values:
            metrics = []

            # Define stratified K-fold cross-validation
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

            # Cross-validation loop
            for train_idx, val_idx in skf.split(X, y):
                # Split into training and validation folds
                X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

                # Train logistic regression model on the current fold
                model = LogisticRegression(
                    penalty=penalty,
                    C=C,
                    solver=solver,
                    max_iter=1000,
                    random_state=random_state,
                    # Uncomment below to handle class imbalance
                    # class_weight="balanced",
                )
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)

                # Collect metrics for this fold (robust to zero-division)
                metrics.append({
                    "f1": f1_score(y_val_fold, y_pred, zero_division=0),
                    "precision": precision_score(y_val_fold, y_pred, zero_division=0),
                    "recall": recall_score(y_val_fold, y_pred, zero_division=0),
                })

            # Compute average metrics across folds
            avg = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()

            # Choose which metric to maximize (F1 or recall)
            if maximize == "f1":
                score, valid = avg["f1"], True
            elif maximize == "recall":
                score = avg["recall"]
                valid = (avg["precision"] >= min_precision) if (min_precision is not None) else False
            else:
                raise ValueError("maximize must be 'f1' or 'recall'")

            # If current setting is better, remember params
            if valid and score > best_score:
                best_score = score
                best_params = {"penalty": penalty, "C": C, "solver": solver}

    if not best_params:
        raise RuntimeError("No valid hyperparameter combination satisfied the criteria.")

    # Refit best model on the FULL dataset with the chosen hyperparameters
    best_model = LogisticRegression(
        penalty=best_params["penalty"],
        C=best_params["C"],
        solver=best_params["solver"],
        max_iter=1000,
        random_state=random_state,
        # Uncomment below to handle class imbalance in final fit
        # class_weight="balanced",
    ).fit(X, y)

    return best_model, best_params


# Function to extract model coefficients and sort by importance
def extract_coefficients(model, feature_names):
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": model.coef_.flatten()})
        .sort_values(by="coefficient", key=np.abs, ascending=False)
        .reset_index(drop=True)
    )