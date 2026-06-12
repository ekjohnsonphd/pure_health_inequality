library(data.table)
library(ggplot2)
library(scales)

# Figure 2:
# Shows how quickly the top SHAP features accumulate toward explaining
# the predicted mortality risk gap between deceased and surviving individuals.
# Each line shows one sex-age group.
# Steeper lines mean fewer features explain more of the gap.

# --- Load SHAP decomposition data ---
shap <- fread("/XBoost_results/shap_results_all.csv")

# --- Extract sex and age group ---
shap[, sex := fifelse(grepl("^male", age_bin), "Male", "Female")]
shap[, age_label := sub("^(male|female)_", "", age_bin)]

# --- Keep relevant values per feature and group ---
concentration_data <- shap[, .(
  variable,
  shap,
  mean_pred_0 = first(mean_pred_0),
  mean_pred_1 = first(mean_pred_1)
), by = .(age_bin, sex, age_label)]

# --- Calculate total predicted risk gap ---
concentration_data[, total_gap := mean_pred_1 - mean_pred_0, by = .(age_bin, sex)]

# --- Convert SHAP values to percent of total gap ---
concentration_data[, pct_contribution := (shap / total_gap) * 100]

# --- Rank features by absolute SHAP contribution ---
concentration_data[, abs_shap := abs(shap)]
setorder(concentration_data, age_bin, sex, -abs_shap)
concentration_data[, rank := seq_len(.N), by = .(age_bin, sex)]

# --- Calculate cumulative percent contribution ---
concentration_data[, cumulative_pct := cumsum(pct_contribution), by = .(age_bin, sex)]

# --- Plot cumulative contribution for top 50 features ---
p_cumulative <- ggplot(
  concentration_data[rank <= 50],
  aes(
    x = rank,
    y = cumulative_pct,
    color = age_label,
    linetype = sex
  )
) +
  geom_line(linewidth = 1.2) +

  # Highlight selected feature ranks
  geom_point(
    data = concentration_data[rank %in% c(5, 10, 20, 30, 50)],
    size = 2.5
  ) +

  # Reference lines at 50%, 75%, and 90%
  geom_hline(
    yintercept = c(50, 75, 90),
    linetype = "dashed",
    alpha = 0.3
  ) +

  # Labels
  labs(
    x = "Number of Top Features",
    y = "Cumulative % of Total Gap Explained",
    title = "Cumulative Contribution of Top Features to Mortality Gap",
    color = "Age Group",
    linetype = "Sex"
  ) +

  # Format y-axis as percent
  scale_y_continuous(
    labels = percent_format(scale = 1),
    breaks = seq(0, 100, by = 10)
  ) +

  # Set x-axis breaks
  scale_x_continuous(
    breaks = c(1, 5, 10, 20, 30, 40, 50)
  ) +

  # Use readable colors
  scale_color_brewer(palette = "Set1") +

  # Clean theme
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

# --- Save figure ---
ggsave(
  "figures/figure2_cumulative_contribution.png",
  plot = p_cumulative,
  width = 12,
  height = 7,
  dpi = 300
)

# --- Display figure ---
p_cumulative