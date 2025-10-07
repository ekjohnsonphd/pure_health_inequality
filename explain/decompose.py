"""
Utility functions for performing a simplified Shapley‑Owens style
decomposition of a binary classification model.

The workflow is:

1. Separate the data into the two outcome groups (label 0 and 1).
2. For each feature in ``top_features``:
   - If numeric, replace the feature value in the positive group with the
     mean from the negative group and compute how much the predicted
     probability changes.
   - If categorical, for every category level:
     * Replace that level in the positive group.
     * Compute the change in mean prediction.
     * Weight the change by how often that level occurs in the negative group.
   - The summed weighted change is the feature’s contribution to closing
     the gap between the two groups.
3. Sum all contributions to get the explained part of the baseline
   probability gap; the remainder is the “Residual”.
4. Output a CSV with the sorted contributions.

The code is intentionally lightweight and does **not** perform a full
Shapley value estimation – it uses a single‑path Owens approach that
is easy to understand and fast to compute.
"""

import pandas as pd


def decompose_categorical_variable(
    model, group_1: pd.DataFrame, group_0: pd.DataFrame, feature: str
):
    """
    Compute the contribution of a categorical feature to the
    difference in predicted probabilities between two outcome groups.

    Parameters
    ----------
    model : sklearn‑compatible estimator
        Must expose ``predict_proba``.
    group_1 : pd.DataFrame
        Data of the positive outcome group (label==1).
    group_0 : pd.DataFrame
        Data of the negative outcome group (label==0).
    feature : str
        Column name of the categorical variable.

    Returns
    -------
    tuple
        ``(feature, contribution)`` where ``contribution`` is the
        weighted reduction in the probability gap when the positive
        group is “counter‑factually” set to the distribution of
        the negative group.
    """
    # All unique levels in the negative group (used for weighting)
    levels = group_0[feature].dropna().unique()

    # Baseline mean probability for the positive group
    group_1_preds = model.predict_proba(group_1)[:, 1]
    baseline_mean = group_1_preds.mean()

    # Accumulate the weighted contribution
    contrib = 0
    for level in levels:
        X_cf = group_1.copy()
        X_cf[feature] = level
        X_cf = enforce_category_types(X_cf, group_1)
        cf_preds = model.predict_proba(X_cf)[:, 1]

        # Change in mean prediction when the feature is set to this level
        delta = baseline_mean - cf_preds.mean()

        # Weight by the prevalence of this level in the negative group
        weight = (group_0[feature] == level).mean()
        contrib += weight * delta
    
    return feature, contrib


def decompose_numeric_variable(
    model, group_1: pd.DataFrame, group_0: pd.DataFrame, feature: str
):
    """
    Compute the contribution of a numeric feature to the probability gap.

    For numeric variables we simply replace the feature in the positive
    group with the mean value from the negative group and observe the
    change in predicted probability.

    Returns
    -------
    tuple
        ``(feature, contribution)``
    """
    baseline_mean = model.predict_proba(group_1)[:, 1].mean()

    X_cf = group_1.copy()
    X_cf[feature] = group_0[feature].mean()

    cf_preds = model.predict_proba(X_cf)[:, 1]
    return feature, baseline_mean - cf_preds.mean()


def enforce_category_types(df: pd.DataFrame, reference_df: pd.DataFrame):
    """
    Ensure that all columns that are categorical in ``reference_df``
    are cast to the ``category`` dtype in ``df``.

    XGBoost requires categorical columns to be encoded as ``category``.
    """
    for col in reference_df.select_dtypes(include="category").columns:
        df[col] = df[col].astype("category")
    return df


# ----------------------------------------------------------------------
# Simplified Shapley‑Owens decomposition
# ----------------------------------------------------------------------
def shapley_owens_decomp(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    top_features: list,
    outdir: str,
    method: str = "mean",
):
    """
    Perform a single‑path Owens decomposition for the top features.

    Parameters
    ----------
    model : sklearn‑compatible estimator
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Binary target vector.
    top_features : list[str]
        Features to include in the decomposition.
    outdir : str
        Directory where ``decomp.csv`` will be written.
    method : str, default "mean"
        Currently unused; placeholder for future extensions
        (e.g. median or other aggregation).

    Returns
    -------
    tuple
        (decomp_df, base_gap, yhat_0, yhat_1)

        * ``decomp_df`` – DataFrame with feature contributions.
        * ``base_gap`` – Raw difference in mean predicted probability
          between the two outcome groups.
        * ``yhat_0`` – Model predictions for the negative group.
        * ``yhat_1`` – Model predictions for the positive group.
    """
    df = X.copy()
    df["early_death"] = y

    # Separate positive (1) and negative (0) outcome groups
    group_0 = X[y == 0]
    group_1 = X[y == 1]

    # Average predicted probability for each group
    yhat_0 = model.predict_proba(group_0)[:, 1]
    yhat_1 = model.predict_proba(group_1)[:, 1]

    # Collect contributions for each top feature
    results = []
    for feat in top_features:
        if pd.api.types.is_numeric_dtype(df[feat]):
            result = decompose_numeric_variable(model, group_1, group_0, feat)
        else:
            result = decompose_categorical_variable(model, group_1, group_0, feat)
        results.append(result)

    # Residual – the portion of the gap that is not explained by the selected features
    base_gap = yhat_1.mean() - yhat_0.mean()
    explained = sum(r[1] for r in results)
    residual = base_gap - explained
    results.append(("Residual", residual))

    # Build and sort the decomposition DataFrame
    decomp = pd.DataFrame(results, columns=["Feature", "Contribution"])
    decomp.sort_values(by="Contribution", ascending=False, inplace=True)

    # Persist to CSV for later inspection
    decomp.to_csv(f"{outdir}/decomp.csv", index=False)

    return decomp, base_gap, yhat_0, yhat_1
