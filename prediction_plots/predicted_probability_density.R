library(data.table)
library(arrow)
library(ggplot2)

pred_file <- "/XBoost_results/Female_65-69/predictions.parquet"
out_dir   <- "/XBoost_results/Female_65-69"

# Load predictions
pred <- read_parquet(pred_file, as_data_frame = TRUE)
setDT(pred)

thr    <- unique(pred$threshold)
vlines <- pred[, .(x = median(y_proba, na.rm = TRUE)), by = y_test]

p_combined <- ggplot(pred, aes(x = y_proba, colour = factor(y_test))) +
  geom_density(linewidth = 1.1, adjust = 1.2) +

  # Median lines per group
  geom_vline(
    data      = vlines,
    aes(xintercept = x, colour = factor(y_test)),
    linetype  = "dashed",
    linewidth = 1
  ) +

  # Threshold line
  geom_vline(
    xintercept = thr,
    linetype   = "dashed",
    colour     = "black",
    linewidth  = 1
  ) +

  scale_colour_manual(
    values = c("0" = "blue", "1" = "red"),
    labels = c("0" = "y_test=0", "1" = "y_test=1"),
    name   = NULL
  ) +
  labs(
    title = "Predicted p(early death) by true outcome",
    x     = "Predicted p(early death)",
    y     = "Density"
  ) +
  theme_minimal(base_size = 13) +
  coord_cartesian(clip = "off") +
  annotate("text", x = thr + 0.03, y = Inf,
           label  = paste0("Threshold= ", round(thr, 2)),
           vjust  = 6, hjust = 0, colour = "black", size = 3.5) +
  annotate("text", x = Inf, y = 2.6,
           label  = "Median y_test=0",
           hjust  = 1.1, colour = "blue", size = 3.5) +
  annotate("text", x = Inf, y = 2.43,
           label  = "Median y_test=1",
           hjust  = 1.1, colour = "red", size = 3.5)

ggsave(
  filename = file.path(out_dir, "predicted_probability_density.png"),
  plot     = p_combined,
)
p_combined
