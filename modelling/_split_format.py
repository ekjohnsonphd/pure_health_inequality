
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def split_and_format_data(
    data_path: str,
    drop_cols: list = None,
    sex_filter: list = None,
    sex_col: str = "de_sex",
    target_col: str = "early_death",
    id_col: str = "pnr",
    test_size: float = 0.3,
    cal_size_within_train: float = 0.3,
    random_state: int = 42,
    stratify_on_year: bool = False,
    year_col: str = "year",
):
    # ─────────────────────────────────────────────────────────────────────────
    # Load data
    # The input is a parquet file containing the cohort dataset.
    # ─────────────────────────────────────────────────────────────────────────
    
    df = pd.read_parquet(data_path, engine="fastparquet")


    # ─────────────────────────────────────────────────────────────────────────
    # Define columns to drop
    # If drop_cols is not given, use an empty list.
    # ─────────────────────────────────────────────────────────────────────────
    
    drop_cols = drop_cols or []


    # ─────────────────────────────────────────────────────────────────────────
    # Optional sex filter
    # For example, keep only females or only males if sex_filter is provided.
    # ─────────────────────────────────────────────────────────────────────────
    
    if sex_filter is not None:
        df = df[df[sex_col].isin(sex_filter)].copy()


    # ─────────────────────────────────────────────────────────────────────────
    # Save person IDs
    # IDs are kept separately so they are not used as model features.
    # ─────────────────────────────────────────────────────────────────────────
    
    ids = df[id_col].copy()


    # ─────────────────────────────────────────────────────────────────────────
    # Define features X and outcome y
    #
    # X = all predictor variables, except target and columns we want to drop.
    # y = outcome variable, converted to integer.
    # ─────────────────────────────────────────────────────────────────────────
    
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col].astype(int)


    # ─────────────────────────────────────────────────────────────────────────
    # Define stratification variable
    #
    # Stratification helps keep the outcome distribution similar
    # across train/test splits.
    #
    # If stratify_on_year = True:
    # stratify by both outcome and year.
    #
    # Otherwise:
    # stratify only by outcome.
    # ─────────────────────────────────────────────────────────────────────────
    
    if stratify_on_year:
        if year_col not in df.columns:
            raise ValueError("Year column not in data")
        
        strata = df[target_col].astype(str) + "_" + df[year_col].astype(str)
    
    else:
        strata = y


    # ─────────────────────────────────────────────────────────────────────────
    # First split: train/test
    #
    # The test set is kept separate and used for final evaluation.
    # The remaining data becomes the full training data.
    # IDs are split at the same time to preserve row matching.
    # ─────────────────────────────────────────────────────────────────────────
    
    X_train_full, X_test, y_train_full, y_test, id_train_full, id_test = train_test_split(
        X,
        y,
        ids,
        test_size=test_size,
        stratify=strata,
        random_state=random_state
    )


    # ─────────────────────────────────────────────────────────────────────────
    # Second split: fit/calibration
    #
    # The training data can be split again into:
    # - fit data: used to train the model
    # - calibration data: used to calibrate predicted probabilities
    #
    # If cal_size_within_train = 0, no calibration set is created.
    # ─────────────────────────────────────────────────────────────────────────
    
    if cal_size_within_train is not None and cal_size_within_train > 0:

        # Create stratification variable for the fit/calibration split
        if stratify_on_year:
            if year_col not in X_train_full.columns:
                raise ValueError("Year column not in training data")
            
            strata_train = y_train_full.astype(str) + "_" + X_train_full[year_col].astype(str)
        
        else:
            strata_train = y_train_full


        # Split training data into fit and calibration sets
        X_fit, X_cal, y_fit, y_cal, id_fit, id_cal = train_test_split(
            X_train_full,
            y_train_full,
            id_train_full,
            test_size=cal_size_within_train,
            stratify=strata_train,
            random_state=random_state
        )

    else:
        # Use all training data for fitting if no calibration set is requested
        X_fit = X_train_full
        y_fit = y_train_full
        id_fit = id_train_full

        X_cal = None
        y_cal = None
        id_cal = None


    # ─────────────────────────────────────────────────────────────────────────
    # Identify categorical and numeric variables
    #
    # Categorical variables are object/category columns.
    # Numeric variables are all remaining columns.
    # ─────────────────────────────────────────────────────────────────────────
    
    categorical_vars = [
        col for col in X_fit.columns
        if X_fit[col].dtype == "object" or str(X_fit[col].dtype) == "category"
    ]

    numeric_vars = [
        col for col in X_fit.columns
        if col not in categorical_vars
    ]


    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing for categorical variables
    #
    # First, missing values are filled with "missing".
    # Then categorical variables are one-hot encoded.
    #
    # handle_unknown="ignore" means that new categories in test data
    # will not cause an error.
    # ─────────────────────────────────────────────────────────────────────────
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        ))
    ])


    # ─────────────────────────────────────────────────────────────────────────
    # Create full preprocessor
    #
    # Categorical variables are transformed.
    # Numeric variables are passed through unchanged.
    #
    # Note: numeric_vars is identified above, but not directly used here because
    # remainder="passthrough" keeps all non-categorical columns.
    # ─────────────────────────────────────────────────────────────────────────
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_vars),
        ],
        remainder="passthrough",
    )


    # ─────────────────────────────────────────────────────────────────────────
    # Return all data splits and the preprocessor
    #
    # Returned objects:
    # - X_fit, y_fit: used for model fitting (training) 
    # - X_cal, y_cal: used for calibration
    # - X_test, y_test: used for final testing
    # - id_fit, id_cal, id_test: person IDs for each split
    # - preprocessor: used in the modelling pipeline
    # ─────────────────────────────────────────────────────────────────────────
    
    return (
        X_fit,
        X_cal,
        X_test,
        y_fit,
        y_cal,
        y_test,
        id_fit.reset_index(drop=True),
        None if id_cal is None else id_cal.reset_index(drop=True),
        id_test.reset_index(drop=True),
        preprocessor
    )

