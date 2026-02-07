import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

DEFAULT_DROP_COLS = []


def split_and_format_data(
    data_path: str,
    drop_cols: list =None,
    sex_filter: list =None, 
    sex_col: str="de_sex",
    target_col: str = "early_death",
    id_col: str= "pnr",
    test_size: float = 0.3,
    random_state: int = 42,
    stratify_on_year: bool = False,
    year_col: str ="year",
    
):
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split

    df = pd.read_parquet(data_path, engine="fastparquet")

    drop_cols=drop_cols or [] 

    if sex_filter is not None:
        df = df[df[sex_col].isin(sex_filter)].copy()


    #Keep IDs:
    ids = df[id_col].copy()   

    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col].astype(int)

    if stratify_on_year:
        if year_col not in df.columns:
            raise ValueError("Year column not in data")
        strata = df[target_col].astype(str) + "_" + df[year_col].astype(str)
    else:
        strata = y

    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, ids, 
        test_size=test_size, 
        stratify=strata, 
        random_state=random_state
    )

    # identify variable types

    categorical_vars = [
        col for col in X.columns 
        if X[col].dtype == "object" or str(X[col].dtype) =="category"
    ]
    numeric_vars = [
        col for col in X.columns 
        if col not in categorical_vars
    ]

        
    #print("Numeric cols:", numeric_vars)
    #print("Categorical vars:", categorical_vars)
    missing_vars = set(X.columns) - set(numeric_vars) - set(categorical_vars)
    print("Missing vars", missing_vars)

   
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        ))
    ])
   
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_vars),
        ],
        remainder="passthrough",
    )

    return (X_train, X_test, y_train, y_test, 
    id_train.reset_index(drop=True), 
    id_test.reset_index(drop=True), 
    preprocessor)



