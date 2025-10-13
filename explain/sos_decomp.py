"""
Shapley-Owens-Shorrocks decomposition for XGBoost mortality model.

Decomposes the mortality risk gap between those who die early (y=1) 
and survivors (y=0) by estimating each feature's contribution.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


def _ensure_categorical(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to match XGBoost requirements."""
    cat_cols = reference.select_dtypes(include="category").columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
    return df


def _decompose_numeric(
    model, 
    group_1: pd.DataFrame, 
    group_0: pd.DataFrame, 
    feature: str
) -> Tuple[str, float]:
    """
    Contribution of a numeric feature to the mortality gap.
    
    Replaces group_1's feature values with group_0's mean and measures
    the change in predicted probability.
    """
    baseline = model.predict_proba(group_1)[:, 1].mean()
    
    X_cf = group_1.copy()
    X_cf[feature] = group_0[feature].mean()
    
    counterfactual = model.predict_proba(X_cf)[:, 1].mean()
    
    return feature, baseline - counterfactual


def _decompose_categorical(
    model,
    group_1: pd.DataFrame,
    group_0: pd.DataFrame, 
    feature: str
) -> Tuple[str, float]:
    """
    Contribution of a categorical feature to the mortality gap.
    
    For each category level, replaces group_1 with that level and 
    weights the effect by its prevalence in group_0.
    """
    levels = group_0[feature].dropna().unique()
    baseline = model.predict_proba(group_1)[:, 1].mean()
    
    contribution = 0.0
    for level in levels:
        X_cf = group_1.copy()
        X_cf[feature] = level
        X_cf = _ensure_categorical(X_cf, group_1)
        
        cf_proba = model.predict_proba(X_cf)[:, 1].mean()
        delta = baseline - cf_proba
        weight = (group_0[feature] == level).mean()
        
        contribution += weight * delta
    
    return feature, contribution


def shapley_owens_decomposition(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    output_path: str = None
) -> pd.DataFrame:
    """
    Perform Shapley-Owens-Shorrocks decomposition.
    
    Parameters
    ----------
    model : XGBoost classifier
        Trained model with predict_proba method
    X : pd.DataFrame
        Feature matrix
    y : pd.Series  
        Binary outcome (0 = survivor, 1 = early death)
    features : List[str]
        Features to decompose
    output_path : str, optional
        Path to save decomposition CSV
        
    Returns
    -------
    pd.DataFrame
        Decomposition with columns: Feature, Contribution
        Sorted by absolute contribution, includes Residual
    """
    # Split by outcome
    group_0 = X[y == 0]
    group_1 = X[y == 1]
    
    # Base gap
    yhat_0 = model.predict_proba(group_0)[:, 1].mean()
    yhat_1 = model.predict_proba(group_1)[:, 1].mean()
    gap = yhat_1 - yhat_0
    
    # Decompose each feature
    results = []
    for feat in features:
        if pd.api.types.is_numeric_dtype(X[feat]):
            result = _decompose_numeric(model, group_1, group_0, feat)
        else:
            result = _decompose_categorical(model, group_1, group_0, feat)
        results.append(result)
    
    # Calculate residual
    explained = sum(r[1] for r in results)
    residual = gap - explained
    results.append(("Residual", residual))
    
    # Format output
    decomp = pd.DataFrame(results, columns=["Feature", "Contribution"])
    decomp = decomp.sort_values("Contribution", ascending=False, key=abs)
    
    # Add summary statistics
    decomp.loc[len(decomp)] = ["Total_Gap", gap]
    decomp.loc[len(decomp)] = ["Total_Explained", explained]
    decomp.loc[len(decomp)] = ["Pct_Explained", 100 * explained / gap]
    
    if output_path:
        decomp.to_csv(output_path, index=False)
        print(f"Decomposition saved to {output_path}")
        print(f"Explained: {100*explained/gap:.1f}% of {gap:.4f} gap")
    
    return decomp
