
library(data.table)
library(arrow)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# Define cohort, sex, input files, mapping files, and output folder.
# ─────────────────────────────────────────────────────────────────────────────

COHORT_NAME <- "65_to_69"
SEX         <- "Female"   # "Female" or "Male"

# Project data root — set to your server data path when replicating.
data_dir <- "../data"
cohort   <- "female_65-69"

cohort_file <- "/cohort_data/cohort65_to_69.parquet"   # raw cohort input (server register path)
pred_file   <- file.path(data_dir, cohort, "calibrated_predictions.parquet")
death_file  <- "/data/Nicolai/ExpBoD-data/rawdata2/Grunddata/dodsaasg2022.parquet"

dex_icd_map <- "/data/DEX_ICD_map_v148.csv"
dex_causes  <- "/data/DEX_causelist_v148.csv"

out_dir     <- file.path(data_dir, cohort)


# ─────────────────────────────────────────────────────────────────────────────
# Helper function: clean ICD-10 codes
#
# Converts codes to character, removes extra spaces, makes them uppercase,
# and turns empty strings into NA.
# ─────────────────────────────────────────────────────────────────────────────

clean_icd10 <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- toupper(x)
  x[x == ""] <- NA
  x
}


# ─────────────────────────────────────────────────────────────────────────────
# Helper function: map ICD-10 codes to cause groups
#
# First tries exact ICD-10 match.
# If no match, it tries shorter versions of the code.
# This is useful because some ICD codes are more detailed than the lookup table.
# ─────────────────────────────────────────────────────────────────────────────

map_icd10_to_acause <- function(icd_vec, lookup_dt) {
  
  # Clean ICD codes
  icd_vec <- clean_icd10(icd_vec)
  
  # Create named lookup vector: ICD-10 code -> acause
  v0 <- setNames(lookup_dt$acause, lookup_dt$icd10)
  
  # First try exact match
  out <- v0[icd_vec]
  
  # If no match, trim last character and try again
  na1 <- which(is.na(out) & !is.na(icd_vec))
  
  if (length(na1)) {
    icd_trim1 <- substr(
      icd_vec[na1],
      1,
      pmax(nchar(icd_vec[na1]) - 1, 0)
    )
    
    out[na1] <- v0[icd_trim1]
  }
  
  # If still no match, trim two characters and try again
  na2 <- which(is.na(out) & !is.na(icd_vec))
  
  if (length(na2)) {
    icd_trim2 <- substr(
      icd_vec[na2],
      1,
      pmax(nchar(icd_vec[na2]) - 2, 0)
    )
    
    out[na2] <- v0[icd_trim2]
  }
  
  return(out)
}


# ─────────────────────────────────────────────────────────────────────────────
# Load model predictions
# Prediction file include pnr, y_test, y_pred, and predicted probability.
# ─────────────────────────────────────────────────────────────────────────────

pred <- read_parquet(pred_file, as_data_frame = TRUE)
setDT(pred)


# ─────────────────────────────────────────────────────────────────────────────
# Load cohort data
# ─────────────────────────────────────────────────────────────────────────────

cohort <- read_parquet(cohort_file, as_data_frame = TRUE)
setDT(cohort)


# ─────────────────────────────────────────────────────────────────────────────
# Keep people in this sex group who died during the outcome window
# Only pnr is needed for linking to the death register.
# ─────────────────────────────────────────────────────────────────────────────

cohort_dead <- cohort[
  de_sex == SEX & early_death == 1,
  .(pnr)
]

cohort_dead <- unique(cohort_dead[!is.na(pnr)])


# ─────────────────────────────────────────────────────────────────────────────
# Load death register
# Keep pnr and underlying cause of death ICD-10 code.
# ─────────────────────────────────────────────────────────────────────────────

death <- read_parquet(death_file, as_data_frame = TRUE)
setDT(death)

death <- death[
  ,
  .(
    pnr,
    icd10 = clean_icd10(c_dodtilgrundl_acme)
  )
]

# Remove rows without ICD-10 cause
death <- death[!is.na(icd10)]


# ─────────────────────────────────────────────────────────────────────────────
# Restrict death register to deaths in this cohort
# ─────────────────────────────────────────────────────────────────────────────

setkey(cohort_dead, pnr)
setkey(death, pnr)

death <- death[cohort_dead, nomatch = 0]


# ─────────────────────────────────────────────────────────────────────────────
# Load ICD-10 to DEX cause mapping
# This maps ICD-10 codes to broader cause categories.
# ─────────────────────────────────────────────────────────────────────────────

icd_map <- fread(dex_icd_map)

icd_map <- icd_map[
  code_system == "icd10",
  .(
    icd10  = clean_icd10(icd_code),
    acause = as.character(acause)
  )
]

icd_map <- icd_map[!is.na(icd10)]


# ─────────────────────────────────────────────────────────────────────────────
# Map ICD-10 codes to acause groups
# Unknown is used if no match is found.
# ─────────────────────────────────────────────────────────────────────────────

death[, acause := map_icd10_to_acause(icd10, icd_map)]
death[is.na(acause), acause := "unknown"]


# ─────────────────────────────────────────────────────────────────────────────
# Load readable cause names
# This adds cause_name and family_name to the acause codes.
# ─────────────────────────────────────────────────────────────────────────────

causes <- fread(dex_causes)

causes <- causes[
  ,
  .(
    acause      = as.character(acause),
    cause_name  = as.character(cause_name),
    family      = as.character(family),
    family_name = as.character(family_name)
  )
]


# ─────────────────────────────────────────────────────────────────────────────
# Merge cause names onto death data
# ─────────────────────────────────────────────────────────────────────────────

setkey(causes, acause)
setkey(death, acause)

death <- causes[death]


# ─────────────────────────────────────────────────────────────────────────────
# Merge predictions with death-cause information
# all.x = TRUE keeps all prediction rows.
# ─────────────────────────────────────────────────────────────────────────────

setkey(pred, pnr)
setkey(death, pnr)

dt <- merge(
  pred,
  death,
  by = "pnr",
  all.x = TRUE
)


# Remove person ID after merging
dt[, pnr := NULL]



# ─────────────────────────────────────────────────────────────────────────────
# Check cause information for true deaths
# y_test == 1 means the person died in the outcome window.
# ─────────────────────────────────────────────────────────────────────────────

dt[y_test == 1, .(
  N          = .N,
  NA_acause  = sum(is.na(acause)),
  unknown    = sum(acause == "unknown", na.rm = TRUE)
)]


# ─────────────────────────────────────────────────────────────────────────────
# Check cause information for survivors
# y_test == 0 should usually not have death cause information.
# ─────────────────────────────────────────────────────────────────────────────

dt[y_test == 0, .(
  N          = .N,
  NA_acause  = sum(is.na(acause)),
  unknown    = sum(acause == "unknown", na.rm = TRUE)
)]


# ─────────────────────────────────────────────────────────────────────────────
# Restrict to true deaths with cause information
# ─────────────────────────────────────────────────────────────────────────────

dt_dead <- dt[y_test == 1 & !is.na(acause)]


# ─────────────────────────────────────────────────────────────────────────────
# Label true positives and false negatives
#
# TP = died and model predicted death
# FN = died but model predicted survival
# ─────────────────────────────────────────────────────────────────────────────

dt_dead[, outcome := fifelse(y_pred == 1, "TP", "FN")]


# ─────────────────────────────────────────────────────────────────────────────
# Count totals
# ─────────────────────────────────────────────────────────────────────────────

total_tp     <- dt_dead[outcome == "TP", .N]
total_fn     <- dt_dead[outcome == "FN", .N]
total_deaths <- nrow(dt_dead)


# ─────────────────────────────────────────────────────────────────────────────
# Summarize true positives and false negatives by cause of death
# ─────────────────────────────────────────────────────────────────────────────

cause_summary <- dt_dead[
  ,
  .(
    n_tp     = sum(outcome == "TP"),
    n_fn     = sum(outcome == "FN"),
    n_deaths = .N
  ),
  by = .(acause, cause_name, family, family_name)
]


# ─────────────────────────────────────────────────────────────────────────────
# Add proportions
#
# prop_of_all_TP: among all true positives, share from this cause
# prop_of_all_FN: among all false negatives, share from this cause
# prop_cause_TP: within this cause, share correctly predicted as death
# prop_of_all_deaths: share of all deaths from this cause
# ─────────────────────────────────────────────────────────────────────────────

cause_summary[
  ,
  `:=`(
    prop_of_all_TP     = n_tp / total_tp,
    prop_of_all_FN     = n_fn / total_fn,
    prop_cause_TP      = n_tp / (n_tp + n_fn),
    prop_of_all_deaths = n_deaths / total_deaths
  )
]


# ─────────────────────────────────────────────────────────────────────────────
# Remove garbage-code causes
# ─────────────────────────────────────────────────────────────────────────────

cause_summary <- cause_summary[
  cause_name != "Garbage code" &
    family_name != "Garbage code"
]


# ─────────────────────────────────────────────────────────────────────────────
# Save cause summaries
# ─────────────────────────────────────────────────────────────────────────────

fwrite(
  cause_summary,
  file.path(out_dir, "cause_summary_all.csv")
)


# ─────────────────────────────────────────────────────────────────────────────
# Save top 50 causes by number of deaths
# ─────────────────────────────────────────────────────────────────────────────

cause_top50 <- cause_summary[
  order(-n_deaths)
][1:50]

fwrite(
  cause_top50,
  file.path(out_dir, "cause_top50.csv")
)


# ─────────────────────────────────────────────────────────────────────────────
# Select top 15 causes for plotting
# ─────────────────────────────────────────────────────────────────────────────

cause_top15 <- cause_summary[
  order(-n_deaths)
][1:15]


# ─────────────────────────────────────────────────────────────────────────────
# Plot TP share within each cause of death
# ─────────────────────────────────────────────────────────────────────────────

library(ggplot2)

p_top_15 <- ggplot(
  cause_top15,
  aes(
    x = reorder(cause_name, prop_cause_TP),
    y = prop_cause_TP
  )
) +
  geom_col(fill = "#4c78a8") +
  coord_flip() +
  labs(
    x     = NULL,
    y     = "TP share within cause (TP / (TP + FN))",
    title = "Model performance by cause of death (Top 15 causes)"
  ) +
  theme_bw()


# ─────────────────────────────────────────────────────────────────────────────
# Save plot
# ─────────────────────────────────────────────────────────────────────────────

ggsave(
  filename = file.path(out_dir, "top15_tp_share_by_cause.png"),
  plot     = p_top_15
)

