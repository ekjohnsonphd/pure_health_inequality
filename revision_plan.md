# Analysis revision plan

Working notes for revising Anne's pipeline. Primary model = **scale_pos_weight** (no
resampling); primary SHAP = **calibrated model on the test set**. Nothing in the primary
path (02 → 04) is resampled; calibration and test stay at the natural early-death rate.

Split roles: **train** = fit model; **calibration** = fit isotonic map + pick F2 threshold;
**test** = report only (metrics, mortality gap + CI, SHAP). "Test unseen" means unseen during
training/tuning — reporting *on* test is its intended use.

---

## List A — by change (and why)

**A1. ~~Drop leakage column `i.de_age_at_death`~~ — RESOLVED, no action.**
Verified: `i.de_age_at_death` is not present in the cohort files read for calibration modeling,
so there is no leakage and the current calibration model results stand. The drop in 03 was
defensive against a column that isn't in the current data. No drop-list change needed.

**A2. Move calibrated SHAP (`05_shap_calibrated.py`) from calibration set → test set; replace the 100/100 sample.**
Calibration set was used to fit the isotonic map (in-sample); test is the correct held-out
basis. Reads `X_test_raw`/`y_test_raw` saved by the run notebook (A8).
**Compute constraint (DECISION NEEDED):** the calibrated model uses model-agnostic *permutation*
SHAP; the test set is ~230k rows × ~1,635 features, so explaining *all* rows is infeasible
(~10¹¹–10¹² model evals). The decomposition only needs stable group means, and deaths are rare
(~3.7k in test). Plan: explain **all deaths + a random survivor sample** (background ~100–200)
— removes the arbitrary 100/100 balance, keeps it tractable. Sizes TBD with Emily.

**A8. Make the calibrated pipeline self-sufficient: save the test data from 04.**
The raw test features are currently saved only by the old raw-model notebook (02 cell 29), so
the calibrated path secretly depends on running 02. Add `X_test_raw.to_parquet(...)` + `y_test`
to 04 (mirroring 02 cell 29). Then 06b reads 04's outputs and 02 is no longer a dependency — it
can be archived alongside 03 (confirm you don't report the uncalibrated metrics; 04's
calibration curve already shows raw-vs-calibrated).

**A3. Add a bootstrap CI for the *calibrated* mortality gap on test (in 04).**
Bootstrap CIs currently exist only for the uncalibrated (02) and resampled (03) gaps. The
primary result is the calibrated gap (04 cell 10), which has no CI. `y_prob_cal` is already
computed, so this is `bootstrap_gap(y_test, y_prob_cal)`.

**A4. Report calibrated performance metrics on test, not calibration (in 04).**
04 computes F1/F2/AUC/PR-AUC on the same cal data used to fit the calibrator → optimistic.
Threshold stays chosen on cal. *(Only needed if these metrics are reported.)*

**A5. Deprecate the non-calibrated SHAP path (06a + raw block of 07); document, then archive.**
Raw SHAP no longer used. Header note explaining why, then `git mv` 06a to `archive/` and pull
the raw-model block out of 07, leaving the calibrated block as the primary decomposition.

**A6. Archive the resampling experiment (03_run_resampling.ipynb); document.**
No imbalance-handling sensitivity wanted. Document + `git mv` to `archive/`.

**A7. Housekeeping: stop committing `.DS_Store`.**
(The drop-list sync is no longer needed now that A1 is resolved and 04 saves its own test data.)

*No change now:* baseline-N sensitivity (07 block 3) — needed for the paper, generation code
lives on the server, leave as-is.

---

## List B — by file (what changes)

| File | Change |
|---|---|
| `modelling/02_run_raw_xgboost.ipynb` | **No change.** Candidate for archive once 04 saves its own test data (A8) — confirm the uncalibrated metrics aren't reported. |
| `modelling/04_run_calibration.ipynb` | (a) **Save `X_test_raw` + `y_test`** to parquet (mirror 02 cell 29) so the calibrated path is self-contained (A8); (b) add bootstrap CI on calibrated gap using test `y_prob_cal` (A3); (c) evaluation block → compute metrics on `X_test_raw`/`y_test`, keep cal-derived threshold (A4). |
| `shap_analysis/06b_compute_shap_values_calibrated.py` | Repoint inputs `X_cal_raw`/`y_cal` → `X_test_raw`/`y_test_raw` (saved by 04); remove 100/100 sampling, explain all test rows; keep survivor baseline; note compute cost / `max_evals` is a single permutation. |
| `shap_analysis/06a_compute_shap_values_raw_model.py` | Deprecation header, then `git mv` to `archive/`. |
| `shap_analysis/07_shap_gap_decomposition.qmd` | Remove/relocate raw-model block (block 1) with a deprecation note; keep calibrated block (2, primary) and baseline-sensitivity block (3, supplementary). |
| `modelling/03_run_resampling.ipynb` | Deprecation note, then `git mv` to `archive/`. |
| `.gitignore` / `.DS_Store` | Re-add `.DS_Store` to `.gitignore`, untrack committed copies. |

Each is a small, self-contained, separately-committed edit so the `git diff` maps cleanly onto
the server changes.

---

## Raw data sources (inputs to the preprocess pipeline)

For assembling a self-contained directory.

### Core raw input (required)
- **Yearly panel parquet files:** `/Data_files/data_panel/data_panel{YEAR}.parquet`, YEAR = 2000–2023 (24 files).
  Read by `preprocess/01_cohort_data_prep.R` (and `cohort_data_prep_expanded.R`). `data_panel2018.parquet` is
  additionally read alone for column names. This panel already contains the person-year records with `de_*`
  (demographics/death), `se_*` (socioeconomic), `hc_*` (healthcare incl. ICD flags, costs, comorbidities),
  `fa_*` (family). **This is the true raw source** — everything downstream is derived from it.

### Code dependency (required to run preprocess)
- **`generate_rolling_variables.R`** — sourced by 01 from `Nicolai/ExpBoD-data/functions/generate_rolling_variables.R`.
  The repo's `preprocess/01a_generate_rolling_variables.R` is a copy of this function; 01 sources the Nicolai
  path, so that is the one actually used on the server. Copy the sourced file, not just the local copy.

### Optional (only if regenerating comorbidities)
- **LPR register datasets:** `/Emily/data/rawdata/lpr_adm`, `/Emily/data/rawdata/lpr_diag`
  (+ commented `lpr_sksopr` / `lpr_sksube` / `lpr_bes`). Read by `preprocess/calculate_charlson.R`, which is
  headed *"Example. Not used in our process."* Comorbidity scores are already in the panel; only needed to
  rebuild Charlson/Elixhauser from scratch.

### Reference maps (downstream, NOT preprocess)
- `/data/DEX_ICD_map_v148.csv`, `/data/DEX_causelist_v148.csv` — read by
  `prediction_plots/predictions_death_cause.R` for cause-of-death mapping (stored in repo as the `_v148.xlsx`
  files under `preprocess/`). Not part of model preprocessing.

### Pipeline intermediates (generated, then consumed)
- `/Data_files/population_panel_{cohort}/population_panel{year}.parquet` — cohort-restricted panels (01 → rolling vars).
- Final cohort files consumed by modelling: `cohort_{min}_to_{max}.parquet`. **If you skip re-running cohort
  construction, these are the only data files you need to copy** (one per age band), and you can ignore the panel.

### Path discrepancies to reconcile when assembling the self-contained dir
- 01 writes `write_dataset("Anne/Data_files/cohort_data")` **partitioned by year, with no cohort name** — but the
  modelling notebooks read a single `/Data_files/cohort_data/cohort_50_to_54.parquet`, and `descriptive.py` reads
  `/Anne/Data_files/cohort_data/cohort_{COHORT_NAME}.parquet`. The write path/naming doesn't match the read
  path, and running 01 for a second age band would overwrite the same `cohort_data` folder. Confirm how the
  per-cohort `cohort_{age}.parquet` files are actually produced on the server before relying on 01.
- All input/output paths are hardcoded absolute server paths (`/Data_files/...`, `/Anne/...`, `Nicolai/...`) —
  they'll need repointing to the self-contained directory.
