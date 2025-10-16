"""
SUMMARY
--------
This script defines one function: "split_and_format_data" that:
1. Loads a dataset from a Parquet file.
2. Drops ID and leakage columns.
3. Splits the data into train/test sets (optionally stratified by year).
4. Identifies numeric and categorical variables.
5. Builds and fits a preprocessing pipeline:
   - Imputes missing values
   - Scales numeric variables
   - One-hot encodes categorical variables
6. Returns train/test sets, IDs, and a fitted preprocessor.

CHANGES
---------------
- Safe handling of `drop_cols=None` (uses DEFAULT_DROP_COLS).
- Fixed typo in numeric pipeline step name: "scaler" (was "scalar").
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Global to the script: list of columns that should be dropped (e.g. IDs or leakage columns).
DEFAULT_DROP_COLS = []

# Define the function:
def split_and_format_data(
    data_path: str,             # Path to the input .parquet dataset
    drop_cols: list = None,     # Columns to drop (e.g., IDs)
    target_col: str = "early_death",  # Target column name (default: 'early_death')
    test_size: float = 0.3,     # Fraction of data used for testing (default: 30%)
    random_state: int = 42,     # Random seed for reproducibility
    stratify_on_year: bool = False,  # If True, stratify by both target and year
):
    # Load data
    df = pd.read_parquet(data_path)

    # Extract IDs (keep for later merging, evaluation, traceability)
    ids = df["pnr"]

    # Use default drop list if None
    drop_cols = drop_cols or DEFAULT_DROP_COLS

    # Drop target and unwanted columns from features
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]

    # Optional stratification by (target, year) combo
    if stratify_on_year:
        if "year" not in df.columns:
            raise ValueError("Year column not in data")
        strata = df[target_col].astype(str) + "_" + df["year"].astype(str)
    else:
        strata = y  # only target

    # Split data into train/test sets (keeping same class proportions)
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids,
        test_size=test_size,
        stratify=strata,
        random_state=random_state
    )

    # Identify variable types
    binary_vars = [col for col in X.columns if X[col].dropna().isin([0, 1]).all()]
    categorical_vars = [col for col in X.columns if X[col].dtype == "object" and col not in binary_vars]
    numeric_vars = [col for col in X.columns if col not in binary_vars + categorical_vars]
    numeric_vars += binary_vars  # include binaries with numeric vars

    # Print overview (optional for debugging)
    for col in categorical_vars:
        n_unique = X[col].nunique()
        print(f"{col}: {n_unique} unique values")
    print("Numeric cols:", numeric_vars)
    print("Categorical vars:", categorical_vars)

    # Check for any uncategorized columns
    missing_vars = set(X.columns) - set(numeric_vars) - set(categorical_vars)
    print("Missing vars", missing_vars)

    # Preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),       # fill missing numeric values with mean
        ("scaler", StandardScaler())                       # standardize numeric features
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),  # replace missing with 'missing'
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))  # encode categorical variables
    ])

    # Combine both preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_vars),
            ("cat", categorical_transformer, categorical_vars),
        ],
        remainder="passthrough",  # keep other columns as-is (if any)
    )

    # Fit preprocessor to training data
    preprocessor.fit(X_train)

    # Return processed splits and preprocessor
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        id_train.reset_index(drop=True),
        id_test.reset_index(drop=True),
        preprocessor
    )

"""
The function returns:

X_train, X_test : pandas.DataFrame
    Training and testing feature sets.
y_train, y_test : pandas.Series
    Target values for training and testing.
id_train, id_test : pandas.Series
    Corresponding person IDs (for later merging or evaluation).
preprocessor : sklearn ColumnTransformer
    Fitted preprocessing pipeline ready for model training.
"""
