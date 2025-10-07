import pandas as pd
from sklearn.utils import resample


def upsample_minority(X, y, factor=5, random_state=42):
    df = pd.concat([X, y], axis=1)

    df_minority = df[df[y.name] == 1]
    df_majority = df[df[y.name] == 0]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_minority) * factor,
        random_state=random_state
    )

    df_upsampled = pd.concat(
        [df_majority, df_minority_upsampled]).sample(
            frac=1, random_state=random_state)
    return df_upsampled.drop(columns=y.name), df_upsampled[y.name]


def downsample_majority(X, y, factor=2, random_state=42):
    df = pd.concat([X, y], axis=1)

    df_minority = df[df[y.name] == 1]
    df_majority = df[df[y.name] == 0]

    df_majority_downsampled = resample(
        df_majority,
        replace=False,
        n_samples=len(df_majority) // factor,
        random_state=random_state
    )

    df_downsampled = pd.concat(
        [df_minority, df_majority_downsampled]).sample(
            frac=1, random_state=random_state)
    return df_downsampled.drop(columns=y.name), df_downsampled[y.name]
