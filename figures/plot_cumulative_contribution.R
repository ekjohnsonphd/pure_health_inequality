library(data.table)
library(ggplot2)
library(scales)

# Figure 2: cumulative contribution of top features to the predicted mortality gap

# --- Load data ---
shap <- fread("data/20260212/shap_results_all.csv")

# --- Minimal cleaning ---
shap[, sex := fifelse(grepl("^Male", age_bin), "Male", "Female")]
shap[, age_label := sub("^(Male|Female)_", "", age_bin)]

# --- Calculate cumulative contribution for each age-sex group ---
concentration_data <- shap[, .(
  variable,
  shap,
  mean_pred_0 = first(mean_pred_0),
  mean_pred_1 = first(mean_pred_1)
), by = .(age_bin, sex, age_label)]

# --- Calculate total gap for each age-sex group ---
concentration_data[, total_gap := mean_pred_1 - mean_pred_0, by = .(age_bin, sex)]

# --- Calculate percent contribution ---
concentration_data[, pct_contribution := (shap / total_gap) * 100]

# --- Rank features by absolute contribution within each age-sex group ---
concentration_data[, abs_shap := abs(shap)]
setorder(concentration_data, age_bin, sex, -abs_shap)
concentration_data[, rank := seq_len(.N), by = .(age_bin, sex)]

# --- Calculate cumulative percentage contribution ---
concentration_data[, cumulative_pct := cumsum(pct_contribution), by = .(age_bin, sex)]

# --- Plot ---
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
  geom_point(
    data = concentration_data[rank %in% c(5, 10, 20, 30, 50)],
    size = 2.5
  ) +
  geom_hline(
    yintercept = c(50, 75, 90),
    linetype = "dashed",
    alpha = 0.3
  ) +
  labs(
    x = "Number of Top Features",
    y = "Cumulative % of Total Gap Explained",
    title = "Cumulative Contribution of Top Features to Mortality Gap",
    color = "Age Group",
    linetype = "Sex"
  ) +
  scale_y_continuous(
    labels = percent_format(scale = 1),
    breaks = seq(0, 100, by = 10)
  ) +
  scale_x_continuous(
    breaks = c(1, 5, 10, 20, 30, 40, 50)
  ) +
  scale_color_brewer(palette = "Set1") +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    panel.grid.minor = element_blank()
  )

# --- Save figure ---
ggsave(
  "paper/figures/figure2_cumulative_contribution.png",
  plot = p_cumulative,
  width = 12,
  height = 7,
  dpi = 300
)

# --- Display figure ---
p_cumulative