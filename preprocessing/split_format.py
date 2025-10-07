import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DEFAULT_DROP_COLS = []


def split_and_format_data(
    data_path: str,
    drop_cols: list = None,
    target_col: str = "early_death",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify_on_year: bool = False,
):
    df = pd.read_parquet(data_path)
    
    ids = df["pnr"]

    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]

    if stratify_on_year:
        if "year" not in df.columns:
            raise ValueError("Year column not in data")
        strata = df[target_col].astype(str) + "_" + df["year"].astype(str)
    else:
        strata = y

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, test_size=test_size, stratify=strata, random_state=random_state
    )

    # identify variable types
    binary_vars = [col for col in X.columns if X[col].dropna().isin(
        [0, 1]).all()]
    categorical_vars = [col for col in X.columns if X[col].dtype == "object" and col not in binary_vars]
    numeric_vars = [
        col for col in X.columns if col not in binary_vars + categorical_vars
    ]
    numeric_vars += binary_vars
    
    for col in categorical_vars:
        n_unique = X[col].nunique()
        print(f"{col}: {n_unique} unique values")
        
    print("Numeric cols:", numeric_vars)
    print("Categorical vars:", categorical_vars)
    missing_vars = set(X.columns) - set(numeric_vars) - set(categorical_vars)
    print("Missing vars", missing_vars)

    # preprocessor to standardize for logistic regression
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scalar", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        ))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_vars),
            ("cat", categorical_transformer, categorical_vars),
        ],
        remainder="passthrough",
    )

    preprocessor.fit(X_train)

    return X_train, X_test, y_train, y_test, id_train.reset_index(drop=True), id_test.reset_index(drop=True), preprocessor
