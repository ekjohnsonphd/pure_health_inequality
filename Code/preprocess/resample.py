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


def resample_to_25pct(X,y, random_state=42):
    df = pd.concat([X, y], axis=1)
    df_pos = df[df[y.name] == 1]
    df_neg = df[df[y.name] == 0]

    n_pos = len(df_pos)
    n_neg_target=3*n_pos

    n_neg_target=min(n_neg_target, len(df_neg))

    df_neg_down=resample(
        df_neg,
        replace=False,
        n_samples=n_neg_target,
        random_state=random_state
    )

    df_out=pd.concat([df_pos, df_neg_down]).sample(frac=1, random_state=random_state)
    return df_out.drop(columns=y.name), df_out[y.name] 


def resample_to_target_rate(X,y, target_rate=0.25, random_state=42):

    if not (0 < target_rate <1):
        raise ValueError("target rate must be between 0 and 1")

    df = pd.concat([X, y], axis=1)

    df_pos = df[df[y.name] == 1]
    df_neg = df[df[y.name] == 0]

    n_pos = len(df_pos)
    n_neg=len(df_neg)

    if n_pos==0:
        raise ValueError("No positive cases")
    if n_neg==0:
        raise ValueError("No negative cases")

    n_neg_target=int(round(n_pos*(1-target_rate)/target_rate))

    n_neg_target=min(n_neg_target, n_neg)

    df_neg_down=resample(
        df_neg,
        replace=False,
        n_samples=n_neg_target,
        random_state=random_state
    )

    df_out=pd.concat([df_pos, df_neg_down]).sample(frac=1, random_state=random_state)
    return df_out.drop(columns=y.name), df_out[y.name] 
