library(data.table)
library(tidyverse)
library(arrow)
library(parallel)

# source Nicolai's functions
source("../../Nicolai/ExpBoD-data/functions/generate_rolling_variables.R")
source("../../Nicolai/ExpBoD-data/functions/generate_hierarchical_variables.R")

min_age <- 50
max_age <- 69

data_path <- "../../Nicolai/ExpBoD-data/outputdata/releases/v1.0.0/"
time_periods <- list(c(0,5), c(6,10), c(11,15))
# time_periods <- list(c(0,5))


# read in a single year of panel data so I can get the colnames 
tdf <- open_dataset(paste0(data_path, "99_panel2018.parquet")) %>% collect() %>% setDT()
head(tdf)

panel_files <- Sys.glob(paste0(data_path,"99_panel2*.parquet"))
dt <- lapply(panel_files, function(file){
  data <- open_dataset(file) %>%
    filter(in_dk == 1 & de_age %in% min_age:max_age) %>%
    select(-contains("rank")) %>%
    collect() %>% setDT()
  return(data)
}) %>% rbindlist(fill = TRUE)

dt <- dt[is.na(de_age_at_death) | de_age_at_death >= min_age]

population <- dt[,.(year_max = max(year), age_max = max(de_age), year_min = min(year), 
                    age_min = min(de_age), de_age_at_death), by = "pnr"] %>% unique()
population <- population[age_min == min_age & (age_max == max_age | de_age_at_death <= max_age)]
population[, death_year := year_max + (floor(de_age_at_death) - age_max)]
population <- population[(is.na(de_age_at_death) | de_age_at_death >= max_age) | death_year <= 2023]

count(population, (de_age_at_death < max_age & !is.na(de_age_at_death)))

dt <- dt[pnr %in% population$pnr]

population_ids <- unique(dt$pnr)

# generate rolling vars for the 69-year olds for each year
lapply(1986:2023, function(yr){
  file <- paste0(data_path, "99_panel", yr, ".parquet")
  data <- open_dataset(file) %>% 
    filter(pnr %in% population$pnr) %>%
    write_parquet(paste0("data/temp/population_panel",yr,".parquet"))
})

# create configuration lists to feed into the function
icd_columns <- lapply(grep("^hc_icd10_2_(?!.*NA)", names(tdf), value = TRUE, perl = TRUE), function(variable){
  list(var_name = variable, value = "1", periods = time_periods, stats = "max", dataset = "population_panel")
})

cci_columns <- lapply(grep("^hc_charlson_(?!.*NA)", names(tdf), value = TRUE, perl = TRUE), function(variable){
  list(var_name = variable, value = "1", periods = time_periods, stats = "max", dataset = "population_panel")
})

elix_columns <- lapply(grep("^hc_elixhauser_(?!.*NA)", names(tdf), value = TRUE, perl = TRUE), function(variable){
  list(var_name = variable, value = "1", periods = time_periods, stats = "max",  dataset = "population_panel")
})

numeric_variables <- c(#grep("^hc_util_", names(tdf), value = TRUE),
  grep("^hc_cost_", names(tdf), value = TRUE),
  grep("^hc_hospitalizations_", names(tdf), value = TRUE),
  "se_long_term_sick_leave_weeks" ,
  grep("^se_pers(?!.*rank)(?!.*employment_status)", names(tdf), value = TRUE, perl = TRUE),#)#,
  grep("^se_hh(?!.*rank)(?!.*employment_status)", names(tdf), value = TRUE, perl = TRUE))
numeric_columns <- lapply(numeric_variables, function(variable){
  list(var_name = variable, periods = time_periods, stats = c("avg","sd"), dataset = "population_panel")
})

var_config <- c(list(
  list(
    var_name = "de_marital_status",
    dataset = "population_panel", # dataset name without years
    periods = time_periods,  # time periods in years
    value = "Married" # for categorical variables
  ),
  list(
    var_name = "de_marital_status",
    dataset = "population_panel", # dataset name without years
    periods = time_periods,  # time periods in years
    value = "Divorced" # for categorical variables
  ),
  list(
    var_name = "de_marital_status",
    dataset = "population_panel", # dataset name without years
    periods = time_periods,  # time periods in years
    value = "Widow" # for categorical variables
  )), #,
  icd_columns,
  numeric_columns
)

data <- generate_rolling_variables(var_config = var_config, data_path = "data/temp/")
dt <- merge(dt, data, by = c("pnr","year"), all.x = TRUE)


final_data <- dt[de_age == age_min]
final_data[, early_death := 0]
final_data[de_age_at_death >= age_min & de_age_at_death <= age_max, early_death := 1]

final_data %>% 
  group_by(year) %>%
  write_dataset("data/ZZ_cohort_data")

