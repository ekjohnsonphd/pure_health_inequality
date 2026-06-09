library(data.table)
library(arrow)
library(ggplot2)

pred_file   <- "/XBoost_results/Female_65-69/predictions.parquet"
cohort_file <- "/Data_files/cohort_data/cohort65_to_69.parquet"
out_dir     <- "/XBoost_results/Female_65-69"

# Load data
pred   <- read_parquet(pred_file, as_data_frame = TRUE)
cohort <- read_parquet(cohort_file, as_data_frame = TRUE)

setDT(pred)
setDT(cohort)

cohort_small <- cohort[, .(pnr, de_age_at_death, de_sex, early_death)]
dt <- merge(pred, cohort_small, by = "pnr", all.x = TRUE)
dt[, pnr := NULL]

# False positives: predicted to die but survived
fp <- dt[y_test == 0 & y_pred == 1]

# Age at death groups
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

fp[, age_group := factor(
  age_group,
  levels = c("70-74", "75-79", "80-84", "85+", "Alive/censored")
)]

fp_plot_dt <- fp[!is.na(age_group)]

p <- ggplot(fp_plot_dt, aes(x = age_group)) +
  geom_bar(fill = "blue", alpha = 0.7) +
  labs(
    title = "False positives: age at death",
    x     = "Age at death",
    y     = "Count"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = file.path(out_dir, "false_positives_age_at_death.png"),
  plot     = p,
)
p
