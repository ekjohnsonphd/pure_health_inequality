# Supplementary tables

Full tables accompanying the manuscript "The predictable and explainable gap in early mortality risk: a nationwide register-based cohort study". They are too large for the journal's Word supplement and are released here instead. All values are aggregate outputs from the Statistics Denmark research server; no individual-level data are included.

## `top50_specifications_by_cohort.csv`

The 50 characteristic specifications contributing most to the predicted five-year mortality gap between early decedents and survivors, for each of the eight age–sex cohorts, ranked by absolute contribution.

| Column | Content |
|---|---|
| Sex, Age band | Cohort |
| Rank | Rank within the cohort by absolute contribution |
| Variable | Specification name in the analysis code |
| Level | For multi-level categorical specifications, one row per level follows the specification row |
| Characteristic, Observation window, Transformation | The specification's parent characteristic, its lookback window, and how it was computed |
| Domain | Health or Sociodemographics |
| Kind | Continuous, binary or categorical |
| % of gap | Contribution as a percentage of the cohort's predicted mortality gap |
| Cumulative % | Cumulative contribution in rank order |
| SMD | Standardized mean difference between early decedents and survivors (continuous specifications) |
| RR | Risk ratio (binary specifications and categorical levels) |

## `feature_dictionary.csv`

All 1,631 candidate features used in the prediction models, with each feature's characteristic, domain (Health or Sociodemographics), observation window and transformation.
