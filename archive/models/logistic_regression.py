import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


def train_logistic_model(
    X,
    y,
    penalties=[ "l2"],
    C_values=[0.01, 0.1, 1.0, 10],
    random_state=42,
    cv_folds=3,
    maximize="f1",
    min_precision=None,
):
    """_summary_

    Args:
        penalty (str, optional): "l1" or "l2. Defaults to "l2".
        C (float, optional): inverse regularization strength. Defaults to 1.0.
    """
    best_model = None
    best_score = -np.inf
    best_params = {}

    #X = pd.DataFrame(X).reset_index(drop=True)
    #y = pd.Series(y).reset_index(drop=True)

    for penalty in penalties:
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        for C in C_values:
            metrics = []
            skf = StratifiedKFold(
                n_splits=cv_folds, shuffle=True, random_state=random_state
            )

            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
                

            model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=1000,
                random_state=random_state,
            )

            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)

            metrics.append(
                {
                    "f1": f1_score(y_val_fold, y_pred),
                    "precision": precision_score(y_val_fold, y_pred),
                    "recall": recall_score(y_val_fold, y_pred),
                }
            )

        avg = pd.DataFrame(metrics).mean().to_dict()

        if maximize == "f1":
            score = avg["f1"]
            valid = True
        elif maximize == "recall":
            score = avg["recall"]
            valid = (
                avg["precision"] >= min_precision
                if min_precision is not None
                else False
            )
        else:
            raise ValueError("maximize must be f1 or recall")

        if valid and score > best_score:
            best_score = score
            best_params = {"penalty": penalty, "C": C, "solver": solver}
            best_model = model

    return best_model, best_params


def extract_coefficients(model, feature_names):
    return pd.DataFrame(
        {"feature": feature_names, "coefficient": model.coef_.flatten()}
    ).sort_values(by="coefficient", key=np.abs, ascending=False)
