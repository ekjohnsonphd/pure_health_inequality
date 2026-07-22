# Deprecated / superseded scripts (archived 2026-07-21)

Moved here during the pipeline cleanup. None are part of the primary analysis.
Kept for reference and reproducibility of earlier results.

| Archived file | Was | Why archived | Superseded by |
|---|---|---|---|
| `shap_raw.py` | `shap_analysis/06a_compute_shap_values_raw_model.py` | Raw (uncalibrated) model SHAP no longer used | `shap_analysis/05_shap_calibrated.py` |
| `resample.py` | `modelling/03a_resample.py` | Imbalance handled by `scale_pos_weight`, not resampling | `scale_pos_weight` in `modelling/03_train_and_calibrate.ipynb` |
| `resample_run.ipynb` | `modelling/03_run_resampling.ipynb` | Resampling model variant dropped (no imbalance-handling sensitivity) | `modelling/03_train_and_calibrate.ipynb` |
| `run_raw_xgboost.ipynb` | `modelling/02_run_raw_xgboost.ipynb` | Raw-only run; the combined train+calibrate notebook trains one shared base model and calibrates it | `modelling/03_train_and_calibrate.ipynb` |

Note: the raw-model SHAP block was also removed from `shap_analysis/06_shap_decomposition.qmd`
(it consumed `shap_values.csv` produced by `shap_raw.py`).
