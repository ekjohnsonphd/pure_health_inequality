
library(data.table)
library(arrow)
library(ggplot2)


# ─────────────────────────────────────────────────────────────────────────────
# Define file paths
# pred_file contains model predictions.
# out_dir is where the plot will be saved.
# ─────────────────────────────────────────────────────────────────────────────

# Project data root — set to your server data path when replicating.
data_dir <- "../data"
cohort   <- "female_65-69"

pred_file <- file.path(data_dir, cohort, "calibrated_predictions.parquet")
out_dir   <- file.path(data_dir, cohort)


# ─────────────────────────────────────────────────────────────────────────────
# Load predictions
# The prediction file should contain:
# - y_test: true outcome
# - y_proba: predicted probability of early death
# - calibrated_threshold: classification threshold
# ─────────────────────────────────────────────────────────────────────────────

pred <- read_parquet(pred_file, as_data_frame = TRUE)
setDT(pred)


# ─────────────────────────────────────────────────────────────────────────────
# Extract threshold and calculate median prediction by outcome group
#
# thr is the model threshold used to classify people.
# vlines stores the median predicted probability for each true outcome group.
# ─────────────────────────────────────────────────────────────────────────────

thr <- unique(pred$calibrated_threshold)

vlines <- pred[
  ,
  .(x = median(y_proba, na.rm = TRUE)),
  by = y_test
]


# ─────────────────────────────────────────────────────────────────────────────
# Plot predicted probability distributions
#
# The density curves compare predicted probabilities for:
# - y_test = 0: survived
# - y_test = 1: early death
# ─────────────────────────────────────────────────────────────────────────────

p_combined <- ggplot(pred, aes(x = y_proba, colour = factor(y_test))) +
  geom_density(linewidth = 1.1, adjust = 1.2) +

  # Add median lines for each outcome group
  geom_vline(
    data      = vlines,
    aes(xintercept = x, colour = factor(y_test)),
    linetype  = "dashed",
    linewidth = 1
  ) +

  # Add model threshold line
  geom_vline(
    xintercept = thr,
    linetype   = "dashed",
    colour     = "black",
    linewidth  = 1
  ) +

  # Set colours and labels for outcome groups
  scale_colour_manual(
    values = c("0" = "blue", "1" = "red"),
    labels = c("0" = "y_test=0", "1" = "y_test=1"),
    name   = NULL
  ) +

  # Add plot labels
  labs(
    title = "Predicted p(early death) by true outcome",
    x     = "Predicted p(early death)",
    y     = "Density"
  ) +

  theme_minimal(base_size = 13) +
  coord_cartesian(clip = "off") +

  # Add text label for threshold
  annotate(
    "text",
    x      = thr + 0.03,
    y      = Inf,
    label  = paste0("Threshold= ", round(thr, 2)),
    vjust  = 6,
    hjust  = 0,
    colour = "black",
    size   = 3.5
  ) +

  # Add text label for survivor median
  annotate(
    "text",
    x      = Inf,
    y      = 2.6,
    label  = "Median y_test=0",
    hjust  = 1.1,
    colour = "blue",
    size   = 3.5
  ) +

  # Add text label for early-death median
  annotate(
    "text",
    x      = Inf,
    y      = 2.43,
    label  = "Median y_test=1",
    hjust  = 1.1,
    colour = "red",
    size   = 3.5
  )


# ─────────────────────────────────────────────────────────────────────────────
# Save plot
# ─────────────────────────────────────────────────────────────────────────────

ggsave(
  filename = file.path(out_dir, "predicted_probability_density.png"),
  plot     = p_combined
)


# Show plot
p_combined

