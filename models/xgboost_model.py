"""
Utility for training an XGBoost classifier with exhaustive grid‑search.

The function:

* Builds a grid of hyper‑parameters from ``param_grid``.
* Performs stratified k‑fold cross‑validation for each combination.
* Computes average metrics (F1, precision, recall) per fold.
* Keeps the model that maximises a chosen metric (default: F1).
* Optionally enforces a minimum precision when the optimisation target is recall.
* Returns the best model and the best‑found hyper‑parameters.

All artefacts are written to ``output_dir`` (currently just a directory
creation step – no artefacts are written inside the function, but the
caller may write metrics/plots if desired).
"""

import os
from itertools import product

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


def train_xgboost_model(
    X,
    y,
    param_grid,
    output_dir,
    cv_folds=3,
    maximize="f1",
    min_precision=None,
    random_state=42,
):
    """
    Train an XGBoost classifier with hyper‑parameter optimisation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.  Categorical columns are automatically cast to
        ``category`` dtype so that XGBoost can handle them natively.
    y : pd.Series
        Binary target vector.
    param_grid : dict
        Mapping from hyper‑parameter names to lists of values to try.
        Example:
            {
                "max_depth": [3, 4],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [100, 200],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
            }
    output_dir : str
        Path where output artefacts will be stored.  The function only
        ensures that the directory exists; artefact writing is up to the
        caller.
    cv_folds : int, default 3
        Number of folds for stratified k‑fold CV.
    maximize : str, default "f1"
        Which metric to optimise over the grid.
        Must be one of `"f1"` or `"recall"`.
    min_precision : float or None, default None
        When ``maximize`` is `"recall"`, the candidate model is only
        considered if its precision exceeds this threshold.
    random_state : int, default 42
        Seed for reproducibility of the CV splits and the XGBoost model.

    Returns
    -------
    best_model : XGBClassifier
        Trained model that achieved the best score.
    best_params : dict
        Hyper‑parameter combination that yielded the best model.
    """
    # ------------------------------------------------------------------
    # 1️⃣  Ensure the output directory exists
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 2️⃣  Enumerate all parameter combinations
    # ------------------------------------------------------------------
    param_names = list(param_grid.keys())
    grid = list(product(*param_grid.values()))

    best_model = None
    best_score = -np.inf
    best_params = None
    results = []

    # ------------------------------------------------------------------
    # 3️⃣  Prepare categorical columns for XGBoost
    # ------------------------------------------------------------------
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    # ------------------------------------------------------------------
    # 4️⃣  Setup stratified k‑fold CV
    # ------------------------------------------------------------------
    skf = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    # ------------------------------------------------------------------
    # 5️⃣  Grid‑search over hyper‑parameters
    # ------------------------------------------------------------------
    for combo in grid:
        params_dict = dict(zip(param_names, combo))
        metrics = []

        # Iterate over CV folds
        for train_idx, val_idx in skf.split(X, y):
            X_train_fold = X.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_train_fold = y.iloc[train_idx]
            y_val_fold = y.iloc[val_idx]

            # ------------------------------------------------------------------
            # 5a. Initialise and fit the model
            # ------------------------------------------------------------------
            model = XGBClassifier(
                use_label_encoder=False,  # avoid warning for XGBoost 1.7+
                eval_metric="logloss",
                enable_categorical=True,
                random_state=random_state,
                **params_dict,
            )
            model.fit(X_train_fold, y_train_fold)

            # ------------------------------------------------------------------
            # 5b. Predict on the validation fold and collect metrics
            # ------------------------------------------------------------------
            y_pred = model.predict(X_val_fold)
            metrics.append(
                {
                    "f1": f1_score(y_val_fold, y_pred),
                    "precision": precision_score(y_val_fold, y_pred),
                    "recall": recall_score(y_val_fold, y_pred),
                }
            )

        # ------------------------------------------------------------------
        # 5c. Compute average metrics across folds
        # ------------------------------------------------------------------
        avg = pd.DataFrame(metrics).mean().to_dict()
        avg.update(params_dict)
        results.append(avg)

        # ------------------------------------------------------------------
        # 5d. Decide if this combination is the new best
        # ------------------------------------------------------------------
        if maximize == "f1":
            score = avg["f1"]
            valid = True
        elif maximize == "recall":
            score = avg["recall"]
            # enforce minimum precision only if a threshold is supplied
            valid = (
                avg["precision"] >= min_precision if min_precision is not None else False
            )
        else:
            raise ValueError("maximize must be 'f1' or 'recall'")

        if valid and score > best_score:
            best_score = score
            best_params = params_dict
            best_model = model

    return best_model, best_params