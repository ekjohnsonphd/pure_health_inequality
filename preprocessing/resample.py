

# --------------------------------------------------------------
# SUMMARY:
# This script provides two functions for handling class imbalance:
# 1) upsample_minority() – makes the minority class five times larger than its original size
#    by randomly duplicating minority samples (with replacement).
# 2) downsample_majority() – makes the majority class half its original size
#    by randomly removing majority samples (without replacement).
# --------------------------------------------------------------


import pandas as pd
from sklearn.utils import resample  # Used to randomly resample rows from a DataFrame


# We define the function: upsample_minority

def upsample_minority(X, y, factor=5, random_state=42):
    # Combine feature matrix (X) and target vector (y) into one DataFrame
    df = pd.concat([X, y], axis=1)

    # Split the data into minority and majority classes
    df_minority = df[df[y.name] == 1]  # All rows where target = 1 (deaths)
    df_majority = df[df[y.name] == 0]  # All rows where target = 0 (survivors)

    # Randomly resample (duplicate) the minority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,                        # Sampling with replacement (same row can appear multiple times)
        n_samples=len(df_minority) * factor, # Create 'factor' times more minority samples, here 5 as decided earlier.
        random_state=random_state            # Ensures reproducibility
    )

    # Combine the upsampled minority class with the original majority class
    # and shuffle the rows to mix them
    df_upsampled = pd.concat(
        [df_majority, df_minority_upsampled]
    ).sample(frac=1, random_state=random_state)

    # Separate X (features) and y (target) again before returning. This restores the standard format expected by ML models (X for inputs, y for labels)
    return df_upsampled.drop(columns=y.name), df_upsampled[y.name]


# We define the function: downsample_majority
def downsample_majority(X, y, factor=2, random_state=42):
    # Combine feature matrix (X) and target vector (y) into one DataFrame
    df = pd.concat([X, y], axis=1)

    # Split the data into minority and majority classes
    df_minority = df[df[y.name] == 1]  # All rows where target = 1
    df_majority = df[df[y.name] == 0]  # All rows where target = 0

    # Randomly select a subset of the majority class
    df_majority_downsampled = resample(
        df_majority,
        replace=False,                       # Sampling without replacement (no row removed twice)
        n_samples=len(df_majority) // factor, # Reduce majority class by 'factor', here 2. 
        random_state=random_state            # Ensures reproducibility
    )

    # Combine the downsampled majority class with the minority class
    # and shuffle the rows to mix them
    df_downsampled = pd.concat(
        [df_minority, df_majority_downsampled]
    ).sample(frac=1, random_state=random_state)

    # Separate X (features) and y (target) again before returning
    return df_downsampled.drop(columns=y.name), df_downsampled[y.name]
