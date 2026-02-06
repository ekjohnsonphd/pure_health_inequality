"""
Utility for training an XGBoost classifier with grid-search.

This function:
1. Tests all combinations of hyperparameters from `param_grid`.
2. Uses stratified K-fold cross-validation. CV_folds=3
3. Calculates F1, precision, and recall for each model.
4. Keeps the model with the best score (default: F1).
5. Optionally enforces a minimum precision when optimizing recall.
6. Returns the best model and parameters.

Changes:
- Added zero_division=0 to precision/recall to avoid warnings when a fold has no positives.
- After selecting best hyperparameters, re-fit the final model on the full (X, y) before returning.
"""

import os                                   # for creating output folders
from itertools import product               # for all parameter combinations
import numpy as np                          # for numeric operations
import pandas as pd                         # for data handling
from xgboost import XGBClassifier           # XGBoost model
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def train_xgboost_model(
    X,                                      # feature matrix (pandas DataFrame)
    y,                                      # binary target vector (pandas Series)
    param_grid,                             # dict of hyperparameters to test
    output_dir,                             # where to save results (folder is created if missing)
    cv_folds=3,                             # number of cross-validation folds
    maximize="f1",                          # metric to optimize ('f1' or 'recall')
    min_precision=None,                     # minimum precision if maximizing recall
    random_state=42,                        # seed for reproducibility
):
    """Train an XGBoost classifier with grid search."""

    # 1) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2) Generate all combinations of hyperparameters
    param_names = list(param_grid.keys())
    grid = list(product(*param_grid.values()))

    # Prepare tracking variables
    best_model = None
    best_score = -np.inf
    best_params = None
    results = []

    # 3) Convert text columns to 'category' dtype (XGBoost supports this)
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    # 4) Set up stratified K-fold CV (preserves class balance)
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    # 5) Loop over all hyperparameter combinations
    for combo in grid:
        params_dict = dict(zip(param_names, combo))   # map names to values
        metrics = []                                  # store metrics per fold

        # Perform cross-validation
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Initialize model with current parameters
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                enable_categorical=True,
                random_state=random_state,
                **params_dict,
            )

            # Fit model on training fold
            model.fit(X_train_fold, y_train_fold)

            # Predict on validation fold
            y_pred = model.predict(X_val_fold)"""
Utility for training an XGBoost classifier with exhaustive grid-search.

This function:
1. Tests all combinations of hyperparameters from `param_grid`.
2. Uses stratified K-fold cross-validation.
3. Calculates F1, precision, and recall for each model.
4. Keeps the model with the best score (default: F1).
5. Optionally enforces a minimum precision when optimizing recall.
6. Returns the best model and parameters.

Changes made after summary:
- Added zero_division=0 to precision/recall to avoid warnings when a fold has no positives.
- After selecting best hyperparameters, re-fit the final model on the full (X, y) before returning.
"""

import os                                   # for creating output folders
from itertools import product               # for all parameter combinations
import numpy as np                          # for numeric operations
import pandas as pd                         # for data handling
from xgboost import XGBClassifier           # XGBoost model
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def train_xgboost_model(
    X,                                      # feature matrix (pandas DataFrame)
    y,                                      # binary target vector (pandas Series)
    param_grid,                             # dict of hyperparameters to test
    output_dir,                             # where to save results (folder is created if missing)
    cv_folds=3,                             # number of cross-validation folds
    maximize="f1",                          # metric to optimize ('f1' or 'recall')
    min_precision=None,                     # minimum precision if maximizing recall
    random_state=42,                        # seed for reproducibility
):
    """Train an XGBoost classifier with manual grid search."""

    # 1) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2) Generate all combinations of hyperparameters
    param_names = list(param_grid.keys())
    grid = list(product(*param_grid.values()))

    # Prepare tracking variables
    best_model = None
    best_score = -np.inf
    best_params = None
    results = []

    # 3) Convert text columns to 'category' dtype (XGBoost supports this)
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    # 4) Set up stratified K-fold CV (preserves class balance)
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    # 5) Loop over all hyperparameter combinations
    for combo in grid:
        params_dict = dict(zip(param_names, combo))   # map names to values
        metrics = []                                  # store metrics per fold

        # Perform cross-validation
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # Initialize model with current parameters
            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                enable_categorical=True,
                random_state=random_state,
                **params_dict,
            )

            # Fit model on training fold
            model.fit(X_train_fold, y_train_fold)

            # Predict on validation fold
            y_pred = model.predict(X_val_fold)

            # Store metrics for this fold
            metrics.append(
                {
                    "f1": f1_score(y_val_fold, y_pred),
                    # FIX 3: avoid zero-division warnings by returning 0 when undefined
                    "precision": precision_score(y_val_fold, y_pred, zero_division=0),
                    "recall": recall_score(y_val_fold, y_pred, zero_division=0),
                }
            )

        # 6) Average metrics across folds
        avg = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()
        avg.update(params_dict)
        results.append(avg)

        # 7) Check if this combination is the best so far
        if maximize == "f1":
            score = avg["f1"]
            valid = True
        elif maximize == "recall":
            score = avg["recall"]
            valid = (
                avg["precision"] >= min_precision if min_precision is not None else False
            )
        else:
            raise ValueError("maximize must be 'f1' or 'recall'")

        # If valid and better score → save as best (keep a reference model)
        if valid and score > best_score:
            best_score = score
            best_params = params_dict
            best_model = model

    # FIX 4: Refit the best model on the entire dataset to produce the final estimator
    if best_params is None:
        raise RuntimeError("No valid hyperparameter combination satisfied the criteria.")
    best_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=random_state,
        **best_params,
    )
    best_model.fit(X, y)

    # Return the final model and its parameters
    return best_model, best_params


            # Store metrics for this fold
            metrics.append(
                {
                    "f1": f1_score(y_val_fold, y_pred),
                    # FIX : avoid zero-division warnings by returning 0 when undefined
                    "precision": precision_score(y_val_fold, y_pred, zero_division=0),
                    "recall": recall_score(y_val_fold, y_pred, zero_division=0),
                }
            )

        # 6) Average metrics across folds
        avg = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()
        avg.update(params_dict)
        results.append(avg)

        # 7) Check if this combination is the best so far
        if maximize == "f1":
            score = avg["f1"]
            valid = True
        elif maximize == "recall":
            score = avg["recall"]
            valid = (
                avg["precision"] >= min_precision if min_precision is not None else False
            )
        else:
            raise ValueError("maximize must be 'f1' or 'recall'")

        # If valid and better score → save as best (keep a reference model)
        if valid and score > best_score:
            best_score = score
            best_params = params_dict
            best_model = model

    # FIX : Refit the best model on the entire dataset to produce the final estimator
    if best_params is None:
        raise RuntimeError("No valid hyperparameter combination satisfied the criteria.")
    best_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        enable_categorical=True,
        random_state=random_state,
        **best_params,
    )
    best_model.fit(X, y)

    # Return the final model and its parameters
    return best_model, best_params
