# Pure Health Project Code

This repository contains code for constructing cohorts of Danish adults aged 50–69, training machine-learning models to predict early mortality, and decomposing the predicted mortality-probability gap between early deaths and survivors using SHAP-based methods.

The project integrates R (data preparation, comorbidities, several plots) and Python (modelling, SHAP, decomposition figures).

---

## Overview

**Objective**
Predict early mortality among Danish adults aged 50–69, evaluate model performance under strong class imbalance, and explain the predicted probability gap between early deaths and survivors using SHAP-based decomposition.

**Pipeline at a glance**
- **Cohort construction and rolling-window feature aggregation** in R.
- **Cohort design:** individuals are included at the **minimum age of the age band** (for example, age 50 for the 50–54 cohort), and predictors are measured by looking backwards from that index age.
- **Rolling feature windows:** variables are aggregated over 1–5, 6–10, and 11–15 years before the index age.
- **Model training:** XGBoost models are trained in Python using randomized hyperparameter search and F2-based threshold selection.
- **Model variants:** three modelling variants are compared — a raw model, a resampled model, and a probability-calibrated model.
- **Performance evaluation:** model performance is evaluated across all sex–age cohorts.
- **Explainability:** SHAP values are computed using tree-based SHAP for the raw model and model-agnostic SHAP for the calibrated model.
- **Decomposition:** the predicted mortality gap is decomposed into feature-group and temporal contributions.
- **Figures:** final figures and supplementary prediction plots are produced from the saved model outputs.

---

## Data and cohort design

- **Population:** Danish adults aged **50–69**
- **Age bands** (cohorts are built and modelled separately for each): **50–54, 55–59, 60–64, 65–69**, each further split by **sex**, giving **8 cohorts** in total.
- **Observation period:** yearly register panel data, **2000–2023**
- **Outcome:** `early_death` — 1 if death occurs within the age band, 0 otherwise
- **Index age:** the lower bound of each age band. For example, individuals in the 50–54 cohort are indexed at age 50, individuals in the 55–59 cohort at age 55, and so on.
- **Feature timing:** one row per individual is kept at index age. Rolling predictors are measured by looking backwards from this index age.
- **Rolling feature windows:**
  - **1–5 years** before index age (*proximal*)
  - **6–10 years** before index age (*distal*)
  - **11–15 years** before index age (*distal*)

Variables measured at the index age itself are named as *immediate*. Numeric variables are summarized as rolling mean and standard deviation; diagnosis and medication flags as rolling `max` (ever present in the window); categorical variables as "value present in the window".

Because the project uses restricted Danish register data, the raw data are **not** publicly available. A detailed overview of variables and data structure is in [`data_dictionary.xlsx`](data_dictionary.xlsx).

> **Paths:** All scripts reference absolute paths inside the secure register environment (e.g. `/Data_files/...`, `/XBoost_results/...`). These are illustrative of the data layout and will need to be adapted to any other environment.

---

## How to read this repository

Scripts are numbered in the order they are meant to be read. The number is the pipeline **step**; a **letter suffix** (e.g. `02a`) marks a supporting module that the numbered step uses.

```
preprocess/
  01_cohort_data_prep.R              Build the cohort and rolling features
  01a_generate_rolling_variables.R     └─ rolling-window helper used by 01

modelling/
  02_run_raw_xgboost.ipynb           Train + evaluate the raw XGBoost model
  02a_split_format.py                  └─ train/calibration/test split + preprocessing
  02b_xgboost_model.py                 └─ randomized-search XGBoost training
  03_run_resampling.ipynb            Same pipeline on a resampled training set
  03a_resample.py                      └─ resampling helpers
  04_run_calibration.ipynb           Train, then isotonic-calibrate probabilities
  05_combined_evaluation_plots.py    ROC/PR curves aggregated across all cohorts

shap_analysis/
  06a_compute_shap_values_raw_model.py    SHAP for the raw model (TreeExplainer)
  06b_compute_shap_values_calibrated.py   SHAP for the calibrated model (model-agnostic)
  07_shap_gap_decomposition.qmd           Aggregate SHAP & decompose the mortality gap

figures/
  08_plot_mortality_gap_decomposition.py  Figure 1: gap decomposition (2×2 panels)
  09_plot_cumulative_contribution.R       Figure 2: cumulative top-feature contribution
```

**Supporting / supplementary scripts** (not numbered — see [Supporting scripts](#supporting-scripts) below):
`preprocess/cohort_data_prep_expanded.R`, `preprocess/calculate_charlson.R`, `preprocess/DEX_ICD_map_v148.csv`, `preprocess/DEX_causelist_v148.csv`, `descriptives/descriptive.py`, the three `prediction_plots/*.R`, and `data_dictionary.xlsx`.

> **Note on numbering and imports:** the run notebooks `import` the helper modules by their plain names (`from split_format import ...`, `from xgboost_model import ...`). Python cannot import a module whose filename starts with a digit, so the numeric prefixes (`02a_`, `02b_`, …) are for **reading order only** and intentionally do not match the import statements inside the notebooks.

---

## Script descriptions

### Preprocessing

#### `01_cohort_data_prep.R`
Builds one analysis cohort; the age band is set at the top of the script (`min_age`/`max_age`), so it is rerun once per band. For the chosen band it:
- reads the yearly register panel files, keeping people resident in Denmark within the age range;
- defines cohort membership (one row per person), keeping individuals observed across the band and dropping deaths that fall outside the data range;
- writes per-year population-panel files for the cohort members;
- builds a rolling-feature configuration over the 1–5, 6–10 and 11–15 year windows — ICD-10 diagnosis flags, numeric variables (healthcare costs, hospitalisations, long-term sick leave, and personal/household socio-economic variables) and marital status — and generates them with `generate_rolling_variables()`;
- merges the rolling features back, keeps one row per person at index age, defines `early_death`, and writes the cohort dataset (partitioned by year).

A fuller earlier version, with more feature groups and explicit censoring handling, is kept for reference as `cohort_data_prep_expanded.R` (see [Supporting scripts](#supporting-scripts)).

#### `01a_generate_rolling_variables.R`
Helper sourced by `01`. Given a variable configuration, it uses DuckDB window functions (partitioned by person, ordered by year) to compute rolling statistics over each requested window: `avg`/`sd`/`min`/`max`/`sum`/`median` for numeric variables and "value ever present" for categorical variables. Returns one merged `data.table` of rolling variables.

---

### Modelling

#### `02a_split_format.py` — `split_and_format_data()`
Prepares one cohort dataset for modelling. The function loads the cohort Parquet file, optionally filters by sex, removes ID and leakage columns, and splits the data into **fit**, **calibration**, and **test** sets.

The calibration split is optional: `cal_size_within_train` is set above zero when a calibration set is needed, and set to `0` for the raw and resampled models. Splits can be stratified by outcome and calendar year.

Categorical variables are filled with `"missing"` and one-hot encoded. Numeric variables are left unchanged because XGBoost can handle missing values internally. The function returns the feature matrices, outcome vectors, ID columns, and the fitted preprocessing transformer.

#### `02b_xgboost_model.py` — `train_xgboost_model_random()`
Trains an `XGBClassifier` inside a preprocessing pipeline using **RandomizedSearchCV** with **stratified k-fold** CV. Supports multi-metric scoring (`f1`, `f2`, `recall`, `precision`, `roc_auc`, `pr_auc`) with refit on a chosen metric (default **F2**) and class-imbalance handling via `scale_pos_weight`. Returns the best fitted model and its hyperparameters (the decision threshold is selected later, in the run notebooks).

#### `02_run_raw_xgboost.ipynb`
End-to-end workflow for one cohort (sex × age band, set at the top). It splits and preprocesses the data, computes `scale_pos_weight` from the training set, trains the model via randomized search (refit on F2), and selects the classification **threshold that maximizes F2** on the training precision–recall curve. It then evaluates on train and test (F1, F2, precision, recall, ROC-AUC, PR-AUC, balanced accuracy, accuracy, MCC, confusion matrix, specificity), computes the **predicted mortality-probability gap** (mean predicted risk among deaths minus survivors) with a bootstrap 95% CI, and saves the model JSON, processed/raw `X_test` and `y_test`, a predictions file, and a results JSON. This is the model whose SHAP values are computed in `06a`.

#### `03a_resample.py`
Offers **three resampling strategies** to choose from, all operating on the training set: **upsample the minority** class (`upsample_minority`), **downsample the majority** class (`downsample_majority`), or **downsample to a target positive rate** (`resample_to_target_rate`, with a fixed-25% preset `resample_to_25pct`). Used only by the resampling run, which calls `resample_to_target_rate`.

#### `03_run_resampling.ipynb`
Identical pipeline to `02`, except the training set is rebalanced with `resample_to_target_rate` (target positive rate 0.5) and the model is trained **without** `scale_pos_weight` — class imbalance is handled by resampling instead. Evaluation, threshold selection, gap computation, and saved artifacts mirror the raw run.

#### `04_run_calibration.ipynb`
Trains the raw model (with `scale_pos_weight`) on the fit set, then fits `CalibratedClassifierCV(method="isotonic", cv="prefit")` on the held-out **calibration set** (`cal_size_within_train=0.3`). It produces a raw-vs-calibrated reliability curve, selects the F2-maximizing threshold on the calibration set, and saves the calibrated model (`joblib`), calibrated predictions, the calibration data, and a results JSON. This is the model whose SHAP values are computed in `06b`.

#### `05_combined_evaluation_plots.py`
Aggregates model performance **across all cohorts**: loads each cohort's saved model and test data, computes ROC and precision–recall curves, and saves combined overlay figures (`combined_roc.png`, `combined_pr.png`).

---

### SHAP analysis

#### `06a_compute_shap_values_raw_model.py`
Loads a raw model (JSON) and its processed test set, splits by outcome, and reports the predicted-probability gap. It builds an interventional **`shap.TreeExplainer`** (`model_output="probability"`) against a background of 5,000 sampled survivors, computes SHAP values for early deaths and survivors, and saves `shap_values.csv` (SHAP values plus `y`, `pred`, `baseline`).

#### `06b_compute_shap_values_calibrated.py`
SHAP for the **calibrated** model. Because the calibrated estimator is not a plain tree model, it uses the **model-agnostic** `shap.Explainer` with a custom prediction function (encoding/decoding categoricals), a background of up to 500 survivors, and a sample of up to 100 deaths and 100 survivors from the calibration set. Saves `calibrated_shap_values.csv`.

#### `07_shap_gap_decomposition.qmd`
Quarto/R document that aggregates SHAP values across the 8 cohorts and decomposes the predicted mortality gap. For each feature it computes the **gap contribution = mean(SHAP | deaths) − mean(SHAP | survivors)**, assigns each feature to an interpretable **feature group** (`group1`: healthcare costs & utilization, disease history, psychiatric medications, economic characteristics, demographics/household, year & month of birth, parish characteristics, …) and a **temporal group** (`group2`: Immediate, Proximal `_1_5`, Distal `_6_10`/`_11_15`), and records cohort mortality rates. It writes:
- `shap_results_all.csv` — raw-model decomposition (input to both figures),
- `calibrated_shap_results_all.csv` — calibrated-model decomposition,
- `shap_results_baseline_sensitivity.csv` — sensitivity of results to SHAP background size.

---

### Figures

#### `08_plot_mortality_gap_decomposition.py` (Figure 1)
Reads `shap_results_all.csv` and builds a **2 × 2** figure: feature-group decomposition (females, males) and temporal decomposition (females, males). Group contributions are converted to shares and scaled so each stacked bar sums to that cohort's total predicted gap. Saves `figures/figure1_mortality_gap_decomposition.png`.

#### `09_plot_cumulative_contribution.R` (Figure 2)
Reads `shap_results_all.csv`, ranks features by absolute contribution within each sex–age cohort, and plots the **cumulative percentage of the predicted gap** explained by the top 50 features (colour = age group, line type = sex). Saves `figures/figure2_cumulative_contribution.png`.

---

## Supporting scripts

These are not part of the main numbered sequence but support or supplement it.

#### `preprocess/cohort_data_prep_expanded.R`
An earlier, more detailed version of the cohort-building script (`01`). It configures a broader feature set, including medication flags, employment-status categories, and family death events.

This version also allows the **index year** to be defined as **the year before the minimum age of the age band** (`min_age - 1`). This can be useful when features should be measured up to, but not including, the first year of the age band.

It also handles left- and right-censoring explicitly and blanks hospitalisation counts in each person's entry year. The script is retained for reference; the streamlined `01_cohort_data_prep.R` is the current script.

#### `preprocess/calculate_charlson.R`
Reference implementation showing **how the Charlson comorbidity index and the Elixhauser–van Walraven index can be computed** from ICD-10 diagnosis codes in the LPR hospital register (monthly, 5-year lookback; ICD definitions from Quan 2005, weights from Charlson 1984 / Quan 2011 and van Walraven 2009, with optional per-disease indicators). The comorbidity features used in the cohort are produced upstream while building the diagnosis features; this script is included to document the methodology and is not necessarily the exact script run for this project. *(Authored by Nicolai Simonsen.)*

#### `preprocess/DEX_ICD_map_v148.csv` and `preprocess/DEX_causelist_v148.csv`
Reference lookups defining how ICD-10 codes map to diagnosis/cause categories (`acause`) and their human-readable cause/family names. They are used **upstream, before preprocessing, to construct all of the diagnosis features** in the panel data, and again by the cause-of-death prediction plot below.

#### `descriptives/descriptive.py`
Builds a descriptive table for one cohort (set `COHORT_NAME`/`SEX` at the top), comparing survivors vs. deceased: continuous variables as mean ± sd with standardized mean differences; binary and categorical variables as percentages with risk ratios for death. Saved as a multi-sheet Excel workbook.

#### `prediction_plots/predictions_death_cause.R`
Joins test-set predictions to registered cause of death (ICD-10 mapped to causes via the DEX lookups), labels true-positives vs. false-negatives among deaths, and summarizes the model's true-positive share by cause (top-15 plot, plus full and top-50 cause summaries).

#### `prediction_plots/false_positives_age_at_death.R`
Among **false positives** (predicted to die but survived the window), plots the distribution of subsequent age at death.

#### `prediction_plots/predicted_probability_density.R`
Density of the predicted probability of early death by true outcome, with the decision threshold and per-group medians marked.

---

## Outputs

- **Trained models** — raw model JSON and calibrated model (`joblib`) per cohort
- **Processed test/calibration data** — `X_test_<cohort>.csv`, `y_test_<cohort>.csv` (and calibration equivalents)
- **Predictions** — `predictions.parquet` / `calibrated_predictions.parquet`
- **Performance metrics & results JSON** — F1/F2, precision, recall, ROC-AUC, PR-AUC, accuracy, balanced accuracy, MCC, confusion matrix, specificity, predicted-probability gap with bootstrap CI
- **Combined evaluation plots** — `combined_roc.png`, `combined_pr.png`
- **SHAP values** — `shap_values.csv` and `calibrated_shap_values.csv` per cohort
- **Decomposition tables** — `shap_results_all.csv`, `calibrated_shap_results_all.csv`, `shap_results_baseline_sensitivity.csv`
- **Figures** — `figures/figure1_mortality_gap_decomposition.png`, `figures/figure2_cumulative_contribution.png`

---

## Example figures

### Figure 1. Decomposition of the predicted mortality gap
Generated by `08_plot_mortality_gap_decomposition.py`.

![Figure 1](figures/figure1_mortality_gap_decomposition.png)

### Figure 2. Cumulative contribution of top features
Generated by `09_plot_cumulative_contribution.R`.

![Figure 2](figures/figure2_cumulative_contribution.png)

---

## Notes

- Cohort, sex, and age-band selections are set at the top of the relevant scripts/notebooks and rerun per cohort.
- Random processes are seeded for reproducibility within cohorts.
- Class imbalance is addressed three ways across the runs: `scale_pos_weight` (raw), training-set resampling (resampling), and isotonic probability calibration (calibration).
</content>
</invoke>
