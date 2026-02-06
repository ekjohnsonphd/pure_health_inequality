library(tidyverse)
library(data.table)
library(arrow)

shap <- fread("results/cohort/xgb/f1_optimization/all_shap.csv")
shap[, V1 := NULL]

wide_shap <- dcast(shap, feature~version, value.var = "mean_abs_shap")

ggplot(shap) + 
  geom_histogram(aes(x = mean_abs_shap, fill = version)) + 
  facet_wrap(.~version) + 
  scale_x_log10()

ggplot(shap[mean_abs_shap > 0.15]) + 
  geom_col(aes(x = feature, y = mean_abs_shap, fill = mean_abs_shap)) + 
  facet_wrap(.~version, scales = "free")


decomp <- fread("results/cohort/xgb/f1_optimization/decomp.csv")
decomp <- decomp[Feature != "in_dk"]
ggplot(decomp[Feature != "Residual"]) + 
  geom_histogram(aes(x = Contribution))

decomp[str_detect(Feature, "hc_cost|hc_util|hc_hospitalizations|se_long_term"), group1 := "Healthcare costs & utilization"]
decomp[str_detect(Feature, "hc_diag|hc_icd|hc_charlson|hc_elixhauser"), group1 := "Heathcare dx"]
decomp[str_detect(Feature, "se_hh|se_pers|se_educ"), group1 := "Economic characteristics"]
# decomp[str_detect(Feature, "se_pers"), group1 := "Personal economic characteristics"]
# decomp[str_detect(Feature, "se_long_term"), group1 := "Long term sickness absense"]
decomp[Feature == "year", group1 := "Year of birth"][Feature == "de_sex", group1 := "Sex"]
decomp[str_detect(Feature, "de_marital"), group1 := "Marital status"]
decomp[Feature == "de_imigration_status", group1 := "Immigration status"]
decomp[str_detect(Feature, "fa_num_child|fa_hh_size|_died"), group1 := "Family characteristics"]
decomp[str_detect(Feature, "imigration|ethnicity"), group1 := "Immigration"]
decomp[Feature == "Residual", group1 := "Residual"]
decomp[is.na(group1)]$Feature

decomp[, group2 := "Proximal"]
decomp[str_detect(Feature,"6_10|11_15"), group2 := "Distal"]
decomp[Feature == "Residual", group2 := "Residual"]

ggplot(decomp) + 
  stat_summary(aes(x = "", y = Contribution, fill = group1), geom = "bar", fun = sum, width = 1) + 
  coord_polar(theta = "y") + 
  theme_void() + 
  labs(fill = "Group")

ggplot(decomp) + 
  stat_summary(aes(x = "", y = Contribution, fill = group2), geom = "bar", fun = sum, width = 1) + 
  coord_polar(theta = "y") + 
  theme_void() + 
  labs(fill = "Group")
