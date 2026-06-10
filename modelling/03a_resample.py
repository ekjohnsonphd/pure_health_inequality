
import pandas as pd
from sklearn.utils import resample


# ─────────────────────────────────────────────────────────────────────────────
# Upsample minority class
#
# This increases the number of positive cases by sampling them with replacement.
# Used when deaths/positive cases are rare.
# ─────────────────────────────────────────────────────────────────────────────

def upsample_minority(X, y, factor=5, random_state=42):
    
    # Combine features and outcome into one dataframe
    df = pd.concat([X, y], axis=1)

    # Split into minority class and majority class
    # Here: 1 = early death, 0 = survivor
    df_minority = df[df[y.name] == 1]
    df_majority = df[df[y.name] == 0]

    # Sample minority cases with replacement
    # Example: factor=5 means create 5 times as many positive cases
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_minority) * factor,
        random_state=random_state
    )

    # Combine majority cases with upsampled minority cases
    # Then shuffle the rows
    df_upsampled = pd.concat(
        [df_majority, df_minority_upsampled]
    ).sample(
        frac=1,
        random_state=random_state
    )

    # Return features and outcome separately
    return df_upsampled.drop(columns=y.name), df_upsampled[y.name]


# ─────────────────────────────────────────────────────────────────────────────
# Downsample majority class
#
# This reduces the number of negative cases by sampling survivors without
# replacement.
# ─────────────────────────────────────────────────────────────────────────────

def downsample_majority(X, y, factor=2, random_state=42):
    
    # Combine features and outcome into one dataframe
    df = pd.concat([X, y], axis=1)

    # Split into minority and majority class
    df_minority = df[df[y.name] == 1]
    df_majority = df[df[y.name] == 0]

    # Sample fewer majority cases without replacement
    # Example: factor=2 keeps about half of the majority class
    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_majority) // factor,
        random_state=random_state
    )

    # Combine minority cases with downsampled majority cases
    # Then shuffle the rows
    df_downsampled = pd.concat(
        [df_minority, df_majority_downsampled]
    ).sample(
        frac=1,
        random_state=random_state
    )

    # Return features and outcome separately
    return df_downsampled.drop(columns=y.name), df_downsampled[y.name]


# ─────────────────────────────────────────────────────────────────────────────
# Resample data so positive cases make up about 25%
#
# This keeps all positive cases and downsamples negative cases.
# Target structure:
# - 1 positive case
# - 3 negative cases
#
# This gives approximately 25% positive cases.
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_25pct(X, y, random_state=42):
    
    # Combine features and outcome
    df = pd.concat([X, y], axis=1)

    # Split into positive and negative cases
    df_pos = df[df[y.name] == 1]
    df_neg = df[df[y.name] == 0]

    # Count positive cases
    n_pos = len(df_pos)

    # For 25% positives, we want 3 negative cases per positive case
    n_neg_target = 3 * n_pos

    # Do not sample more negative cases than actually exist
    n_neg_target = min(n_neg_target, len(df_neg))

    # Downsample negative cases
    df_neg_down = resample(
        df_neg,
        replace=False,
        n_samples=n_neg_target,
        random_state=random_state
    )

    # Combine all positives with downsampled negatives and shuffle
    df_out = pd.concat(
        [df_pos, df_neg_down]
    ).sample(
        frac=1,
        random_state=random_state
    )

    # Return features and outcome separately
    return df_out.drop(columns=y.name), df_out[y.name]


# ─────────────────────────────────────────────────────────────────────────────
# Resample data to any target positive rate
#
# Example:
# target_rate = 0.25 means 25% positive cases.
# target_rate = 0.10 means 10% positive cases.
#
# This keeps all positive cases and downsamples negative cases.
# ─────────────────────────────────────────────────────────────────────────────

def resample_to_target_rate(X, y, target_rate=0.25, random_state=42):

    # Check that target_rate is valid
    if not (0 < target_rate < 1):
        raise ValueError("target rate must be between 0 and 1")

    # Combine features and outcome
    df = pd.concat([X, y], axis=1)

    # Split into positive and negative cases
    df_pos = df[df[y.name] == 1]
    df_neg = df[df[y.name] == 0]

    # Count positive and negative cases
    n_pos = len(df_pos)
    n_neg = len(df_neg)

    # Error checks
    if n_pos == 0:
        raise ValueError("No positive cases")
    if n_neg == 0:
        raise ValueError("No negative cases")

    # Calculate how many negative cases are needed
    # to reach the target positive rate
    n_neg_target = int(round(n_pos * (1 - target_rate) / target_rate))

    # Do not sample more negatives than available
    n_neg_target = min(n_neg_target, n_neg)

    # Downsample negative cases
    df_neg_down = resample(
        df_neg,
        replace=False,
        n_samples=n_neg_target,
        random_state=random_state
    )

    # Combine all positives with selected negatives and shuffle
    df_out = pd.concat(
        [df_pos, df_neg_down]
    ).sample(
        frac=1,
        random_state=random_state
    )

    # Return features and outcome separately
    return df_out.drop(columns=y.name), df_out[y.name]

