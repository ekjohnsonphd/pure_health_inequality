import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, make_scorer, fbeta_score)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV


def train_xgboost_model_random(
    X,
    y,
    preprocessor,
    param_grid,
    cv_folds=3,
    maximize="f2",  # or recall, precision, roc_auc, pr_auc
    random_state=42,
    n_iter: int = 60,
    n_jobs=3,
    scale_pos_weight=None,
):
    X = X.copy()
    y = np.asarray(y).astype(int)

    # Stratified k fold
    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state)

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        enable_categorical=False,
        n_jobs=n_jobs,
        random_state=random_state,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
    )

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", base_model),
    ])

    param_grid_prefixed = {f"model__{k}": v for k, v in param_grid.items()}

    scoring = {
        "f1":        make_scorer(f1_score,       zero_division=0),
        "f2":        make_scorer(fbeta_score,     beta=2, zero_division=0),
        "recall":    make_scorer(recall_score,    zero_division=0),
        "precision": make_scorer(precision_score, zero_division=0),
        "roc_auc":   "roc_auc",
        "pr_auc":    "average_precision",
    }

    if maximize not in scoring:
        raise ValueError("maximize must be one of: f1, f2, recall, precision, roc_auc, pr_auc")

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid_prefixed,
        n_iter=n_iter,
        scoring=scoring,
        refit=maximize,
        cv=cv,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        return_train_score=False,
    )

    search.fit(X, y)

    best_model = search.best_estimator_
    best_params = search.best_params_

    print("Best params:", best_params)
    print("Best CV score:", search.best_score_)

    # Threshold is set after training — see run.ipynb
    chosen_threshold = None

    return best_model, best_params, chosen_threshold
