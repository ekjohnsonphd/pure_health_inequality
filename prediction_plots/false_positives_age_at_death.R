
library(data.table)
library(arrow)
library(ggplot2)


# ─────────────────────────────────────────────────────────────────────────────
# Define file paths
# pred_file contains model predictions.
# cohort_file contains the original cohort data.
# out_dir is where the plot will be saved.
# ─────────────────────────────────────────────────────────────────────────────

pred_file   <- "/XBoost_results/Female_65-69/predictions.parquet"
cohort_file <- "/Data_files/cohort_data/cohort65_to_69.parquet"
out_dir     <- "/XBoost_results/Female_65-69"


# ─────────────────────────────────────────────────────────────────────────────
# Load prediction data and cohort data
# ─────────────────────────────────────────────────────────────────────────────

pred   <- read_parquet(pred_file, as_data_frame = TRUE)
cohort <- read_parquet(cohort_file, as_data_frame = TRUE)


# ─────────────────────────────────────────────────────────────────────────────
# Convert to data.table for faster data manipulation
# ─────────────────────────────────────────────────────────────────────────────

setDT(pred)
setDT(cohort)


# ─────────────────────────────────────────────────────────────────────────────
# Keep only relevant cohort variables
#
# pnr is used for merging.
# de_age_at_death is used to describe when false positives died later.
# de_sex and early_death are kept as extra cohort information.
# ─────────────────────────────────────────────────────────────────────────────

cohort_small <- cohort[, .(
  pnr,
  de_age_at_death,
  de_sex,
  early_death
)]


# ─────────────────────────────────────────────────────────────────────────────
# Merge predictions with cohort information
# all.x = TRUE keeps all rows from the prediction data.
# ─────────────────────────────────────────────────────────────────────────────

dt <- merge(
  pred,
  cohort_small,
  by = "pnr",
  all.x = TRUE
)


# ─────────────────────────────────────────────────────────────────────────────
# Remove pnr after merging
# This avoids keeping the person ID in the analysis dataset.
# ─────────────────────────────────────────────────────────────────────────────

dt[, pnr := NULL]


# ─────────────────────────────────────────────────────────────────────────────
# Identify false positives
#
# False positive means:
# - true outcome y_test == 0, so the person survived during the target window
# - predicted outcome y_pred == 1, so the model predicted early death
# ─────────────────────────────────────────────────────────────────────────────

fp <- dt[y_test == 0 & y_pred == 1]


# ─────────────────────────────────────────────────────────────────────────────
# Create age-at-death groups for false positives
#
# If de_age_at_death is missing, the person is labelled Alive/censored.
# Otherwise, age at death is grouped into 70-74, 75-79, 80-84, and 85+.
# ─────────────────────────────────────────────────────────────────────────────

fp[, age_group := fifelse(
  is.na(de_age_at_death),
  "Alive/censored",
  as.character(cut(
    de_age_at_death,
    breaks = c(70, 75, 80, 85, Inf),
    right  = FALSE,
    labels = c("70-74", "75-79", "80-84", "85+")
  ))
)]


# ─────────────────────────────────────────────────────────────────────────────
# Set order of age groups in the plot
# ─────────────────────────────────────────────────────────────────────────────

fp[, age_group := factor(
  age_group,
  levels = c("70-74", "75-79", "80-84", "85+", "Alive/censored")
)]


# ─────────────────────────────────────────────────────────────────────────────
# Remove rows where age group could not be assigned
# ─────────────────────────────────────────────────────────────────────────────

fp_plot_dt <- fp[!is.na(age_group)]


# ─────────────────────────────────────────────────────────────────────────────
# Plot number of false positives by later age at death group
# ─────────────────────────────────────────────────────────────────────────────

p <- ggplot(fp_plot_dt, aes(x = age_group)) +
  geom_bar(fill = "blue", alpha = 0.7) +
  labs(
    title = "False positives: age at death",
    x     = "Age at death",
    y     = "Count"
  ) +
  theme_minimal(base_size = 13)


# ─────────────────────────────────────────────────────────────────────────────
# Save plot
# ─────────────────────────────────────────────────────────────────────────────

ggsave(
  filename = file.path(out_dir, "false_positives_age_at_death.png"),
  plot     = p
)


# Show plot
p

