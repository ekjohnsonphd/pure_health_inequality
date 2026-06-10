
library(data.table)
library(tidyverse)
library(arrow)
library(parallel)

# source Nicolai's functions
source("Nicolai/ExpBoD-data/functions/generate_rolling_variables.R")


# ─────────────────────────────────────────────────────────────────────────────
# Define cohort
# ─────────────────────────────────────────────────────────────────────────────

min_age <- 50
max_age <- 54
cohort_name <- paste0(min_age, "_to_", max_age)

data_path <- "/Data_files/data_panel/"
out_path  <- paste0("/Data_files/population_panel_", cohort_name, "/")

time_periods <- list(c(1,5), c(6,10), c(11,15))


# ─────────────────────────────────────────────────────────────────────────────
# Read one year of panel data to get column names
# ─────────────────────────────────────────────────────────────────────────────

tdf <- open_dataset(paste0(data_path, "data_panel2018.parquet")) %>% 
  collect() %>% 
  setDT()

head(tdf)

# ─────────────────────────────────────────────────────────────────────────────
# Read all yearly panel files and keep people in the age range
# ─────────────────────────────────────────────────────────────────────────────

panel_files <- Sys.glob(paste0(data_path,"data_panel*.parquet"))

dt <- lapply(panel_files, function(file){
  data <- open_dataset(file) %>%
    filter(in_dk == 1 & de_age %in% min_age:max_age) %>%
    select(-contains("rank")) %>%
    collect() %>% 
    setDT()
  
  return(data)
}) %>% 
  rbindlist(fill = TRUE)


# ─────────────────────────────────────────────────────────────────────────────
# Remove people who died before entering the cohort age range
# ─────────────────────────────────────────────────────────────────────────────

dt <- dt[is.na(de_age_at_death) | de_age_at_death >= min_age]

# ─────────────────────────────────────────────────────────────────────────────
# Create one row per person to define the cohort population
# ─────────────────────────────────────────────────────────────────────────────

population <- dt[
  ,
  .(
    year_max = max(year),
    age_max = max(de_age),
    year_min = min(year),
    age_min = min(de_age),
    de_age_at_death
  ),
  by = "pnr"
] %>% 
  unique()


# ─────────────────────────────────────────────────────────────────────────────
# Keep people who enter at min_age and are observed until max_age
# or die within the cohort age range
# ─────────────────────────────────────────────────────────────────────────────

population <- population[
  age_min == min_age & 
    (age_max == max_age | de_age_at_death <= max_age)
]


# ─────────────────────────────────────────────────────────────────────────────
# Estimate death year and remove deaths outside available data
# ─────────────────────────────────────────────────────────────────────────────

population[, death_year := year_max + (floor(de_age_at_death) - age_max)]

population <- population[
  (is.na(de_age_at_death) | de_age_at_death >= max_age) | death_year <= 2023
]


# Quick check: count people who died before max_age
count(population, (de_age_at_death < max_age & !is.na(de_age_at_death)))


# ─────────────────────────────────────────────────────────────────────────────
# Restrict dt to the final cohort population
# ─────────────────────────────────────────────────────────────────────────────

dt <- dt[pnr %in% population$pnr]

population_ids <- unique(dt$pnr)


# ─────────────────────────────────────────────────────────────────────────────
# Create cohort-specific yearly population panel files
# ─────────────────────────────────────────────────────────────────────────────

lapply(2000:2023, function(yr){
  file <- paste0(data_path, "data_panel", yr, ".parquet")
  
  data <- open_dataset(file) %>%
    filter(pnr %in% population$pnr) %>%
    write_parquet(paste0(out_path, "population_panel", yr, ".parquet"))
})


# ─────────────────────────────────────────────────────────────────────────────
# Create configuration lists for rolling variables
# ─────────────────────────────────────────────────────────────────────────────


# ICD diagnosis variables: binary variable
# stats = "max" means: diagnosis occurred at least once in the window

icd_columns <- lapply(
  grep("^hc_icd_(?!.*NA)", names(tdf), value = TRUE, perl = TRUE), 
  function(variable){
    list(
      var_name = variable,
      value = "1",
      periods = time_periods,
      stats = "max",
      dataset = "population_panel"
    )
  }
)


# Numeric variables
# stats = c("avg", "sd") means average and standard deviation in each window

numeric_variables <- c(
  # grep("^hc_util_", names(tdf), value = TRUE),
  grep("^hc_cost_", names(tdf), value = TRUE),
  grep("^hc_hospitalizations_", names(tdf), value = TRUE),
  "se_long_term_sick_leave_weeks",
  
  grep(
    "^se_pers(?!.*employment_status)", 
    names(tdf), 
    value = TRUE, 
    perl = TRUE
  ),
  
  grep(
    "^se_hh(?!.*employment_status)", 
    names(tdf), 
    value = TRUE, 
    perl = TRUE
  )
)

numeric_columns <- lapply(numeric_variables, function(variable){
  list(
    var_name = variable,
    periods = time_periods,
    stats = c("avg","sd"),
    dataset = "population_panel"
  )
})


# Full variable configuration
# Includes categorical variables such as marital status, binary variables such as ICD diagnosis variables, and the numeric variables

var_config <- c(
  list(
    list(
      var_name = "de_marital_status",
      dataset = "population_panel",
      periods = time_periods,
      value = "Married"
    ),
    
    list(
      var_name = "de_marital_status",
      dataset = "population_panel",
      periods = time_periods,
      value = "Divorced"
    ),
    
    list(
      var_name = "de_marital_status",
      dataset = "population_panel",
      periods = time_periods,
      value = "Widow"
    )
  ),
  
  icd_columns,
  numeric_columns
)


# ─────────────────────────────────────────────────────────────────────────────
# Generate rolling variables
# ─────────────────────────────────────────────────────────────────────────────

data <- generate_rolling_variables(
  var_config = var_config, 
  data_path = out_path
)


# ─────────────────────────────────────────────────────────────────────────────
# Merge rolling variables back onto the cohort data
# ─────────────────────────────────────────────────────────────────────────────

dt <- merge(
  dt, 
  data, 
  by = c("pnr","year"), 
  all.x = TRUE
)


# ─────────────────────────────────────────────────────────────────────────────
# Keep one row per person at min_age
# This is the index age for the cohort.
# ─────────────────────────────────────────────────────────────────────────────

final_data <- dt[de_age == min_age]


# ─────────────────────────────────────────────────────────────────────────────
# Define outcome: early_death
#
# early_death = 1 if the person dies between min_age and max_age
# early_death = 0 otherwise
# ─────────────────────────────────────────────────────────────────────────────

final_data[, early_death := 0]

final_data[
  de_age_at_death >= min_age & de_age_at_death <= max_age, 
  early_death := 1
]


# ─────────────────────────────────────────────────────────────────────────────
# Save final cohort data
# ─────────────────────────────────────────────────────────────────────────────

final_data %>% 
  group_by(year) %>%
  write_dataset("Anne/Data_files/cohort_data")
