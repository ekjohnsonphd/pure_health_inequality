import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (f1_score, precision_score, recall_score, make_scorer, 
roc_auc_score, average_precision_score, precision_recall_curve, fbeta_score)
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

def train_xgboost_model_random(
    X,
    y,
    preprocessor,
    param_grid,
    #output_dir,
    cv_folds=3,
    maximize="f1", # Or recall, precision, roc_auc, pr_auc
    #min_precision=0.3,
    random_state=42,
    n_iter: int =80, # Numer of combinations 
    n_jobs=3,
    scale_pos_weight=None,
):
    X=X.copy()
    y=np.asarray(y).astype(int)

    # Stratified k cold
    cv = StratifiedKFold(
       n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Base model:

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr", # Or logloss, error, auc, aucpr
        enable_categorical=False,
        n_jobs=3,
        random_state=random_state,
        tree_method="hist", 
        scale_pos_weight=scale_pos_weight,
        )

    pipe=Pipeline(steps=[
        ("preprocess",preprocessor),
        ("model", base_model),
    ])

    param_grid_prefixed= {f"model__{k}": v for k, v in param_grid.items()} 


    #Scoring dictionary
    scoring ={
        "f1": make_scorer(f1_score, zero_division=0),
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "recall": make_scorer(recall_score,zero_division=0),
        "precision": make_scorer(precision_score,zero_division=0),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        
    } 


    refit_metric=maximize
    if refit_metric not in scoring:
        raise ValueError("maximize must be f1, f2, recall, precision, roc_auc or pr_auc,")

    # Randomized search

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid_prefixed,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_metric, 
        cv=cv, 
        n_jobs=3,
        random_state=random_state,
        verbose=1,
        return_train_score=False,
    )

    search.fit(X,y)


    best_model=search.best_estimator_
    best_params=search.best_params_

    print("Best params:",search.best_params_)
    print("Best CV score:", search.best_score_)


    #Choosen threshold (no threshold tuning)
    chosen_threshold=0.5


    return best_model, best_params, chosen_threshold









