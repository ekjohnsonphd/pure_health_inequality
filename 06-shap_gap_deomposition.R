library(data.table)
library(stringr)
library(magrittr)

age_bins <- c(
  "Female_50-54", "Female_55-59", "Female_60-64", "Female_65-69",
  "Male_50-54", "Male_55-59", "Male_60-64", "Male_65-69"
)

shap <- lapply(age_bins, function(age) {
  shap <- fread(file.path("XBoost_results", age, "shap_values.csv"))

  shap <- melt(shap, id.vars = c("pred", "y", "baseline"))
  shap <- shap[, .(
    shap_val = mean(value),
    mean_pred = mean(pred),
    count = .N
  ), by = c("y", "variable", "baseline")]

  shap <- dcast(shap, variable + baseline ~ y, value.var = c("shap_val", "mean_pred", "count"))
  shap[, shap := shap_val_1 - shap_val_0]

  setorder(shap, -shap)

  shap[str_detect(variable, "hc_cost|hc_util|hc_hospitalizations|se_long_term"), group1 := "Healthcare costs & utilization"]
  shap[str_detect(variable, "hc_di|hc_charlson|hc_elix"), group1 := "Healthcare dx"]
  shap[str_detect(variable, "med_|medication_count"), group1 := "Psychiatric medications"]
  shap[str_detect(variable, "se_hh|se_pers|se_educ"), group1 := "Economic characteristics"]
  shap[str_detect(variable, "year|month"), group1 := "Year & month of birth"]
  shap[variable == "de_sex", group1 := "Sex"]
  shap[str_detect(variable, "de_marital"), group1 := "Marital status"]
  shap[variable == "de_imigration_status", group1 := "Immigration status"]
  shap[str_detect(variable, "fa_num_child|fa_hh_size|_died"), group1 := "Family characteristics"]
  shap[str_detect(variable, "imigration|ethnicity"), group1 := "Immigration"]
  shap[str_detect(variable, "parish|region"), group1 := "Parish characteristics"]
  shap[variable == "Residual", group1 := "Residual"]

  print(shap[is.na(group1), unique(variable)])

  shap[, group2 := "Immediate"]
  shap[str_detect(variable, "_2_6$"), group2 := "Proximal"]
  shap[str_detect(variable, "_7_15$"), group2 := "Distal"]
  shap[variable == "Residual", group2 := "Residual"]

  shap[, age_bin := age]
  shap[, mort_rate := count_1 / (count_1 + count_0)]

  return(shap)
}) %>% rbindlist()

print(unique(shap[, .(age_bin, mean_pred_0, mean_pred_1, count_0, count_1, mort_rate)]))

fwrite(shap, file.path("XBoost_results", "shap_results_all.csv"))