# Extended features, allowing for index age = min_age -1 and exclude immidiate hospitalisations

library(data.table)
library(tidyverse)
library(arrow)
library(parallel)
library(duckdb)


# source Nicolai's functions
source("Nicolai/ExpBoD-data/functions/generate_rolling_variables.R")


# Cohort definition (single place)
cohort_range <- c(60, 64)

# Derived values
min_age <- cohort_range[1]
max_age <- cohort_range[2]
index_age <- min_age
cohort_name <- paste0(min_age, "_to_", max_age)

# Yearly panel files including parish variables from 2002
data_path <- "/Data_files/data_panel/"

## Years we use:
data_years <- 2000:2023

# Rolling windows are measures backwards from index age
time_periods <- list(
  c(1, 5),
  c(6, 10),
  c(11, 15)
)

max_back <- max(sapply(time_periods, function(x) x[2]))
min_age_needed <- index_age - max_back # Needed for raw + rolling variables

# read in a single year of panel data so I can get the colnames for config list later
tdf <- open_dataset(paste0(data_path, "data_panel2018.parquet")) 
colnames_tdf <- names(tdf)


# Read panel files and build dt for the age band
message("Step 1: reading panel file and buiding dt")

panel_files <- paste0(data_path, "data_panel", data_years, ".parquet")

# This data set is for defining the cohort
dt_band <- lapply(panel_files, function(file) {
  dat <- open_dataset(file) |> 
    filter(in_dk == 1 & de_age %in% min_age:max_age) |> # in Denmark and age range
    collect()
    setDT(dat)
  dat
}) |> 
  rbindlist(fill = TRUE) # Row bind all years

message("Step 1 done")

# Keep people alive at the record year. 
dt_band <- dt_band[alive == 1 & (is.na(de_age_at_death) | de_age_at_death >= min_age)]


message("Step 2: creating population table")
# Collapse to one row per person to define cohort
population <- dt_band[,
                 .(
                   year_max = max(year), # Last observed year in age band
                   age_max = max(de_age), # highest observed age
                   year_min = min(year), # first observed year
                   age_min = min(de_age), # lowest oberved age
                   de_age_at_death = max(de_age_at_death, na.rm = TRUE) # age at death if any
                 ),
                 by = "pnr"
] 

# Replace inf when Na in de_age_at death
population[is.infinite(de_age_at_death), de_age_at_death :=NA_real_] 

message("Step 2 done: population has: ",nrow(population), "persons.")

# Define cohort, death year and drop future deaths. 
population <- population[
  age_min == min_age & 
    (age_max == max_age | (!is.na(de_age_at_death) & de_age_at_death <= max_age))
]
#Death year estimate
population[, death_year := year_max + (floor(de_age_at_death) - age_max)]

# Drop future deaths:
population <- population[
  is.na(de_age_at_death) | death_year <= max(data_years)
]

# Ensure the fill age window fits inside the data range: removes left and right cencoring. 
 
population <- population[
  year_max >= min(data_years) + (max_age - min_age)]
population <- population[
  year_min <= max(data_years) - (max_age - min_age)]


message("Step 2a done: final cohort population has: ",nrow(population), "persons.")

message("year_min_range:", min(population$year_min), "to" , max(population$year_min))
message("year_max_range:", min(population$year_max), "to" , max(population$year_max))


# 2b. Read feature window data: index_age and back: what we use for raw variables and rolling variables

message("Step 2b: reading feature window data ")

dt_feat <- lapply(panel_files, function(file) {
  dat <- open_dataset(file) |>
    filter(in_dk==1 & de_age %in% min_age_needed:index_age)|>
    collect()

  setDT(dat)
  dat
}) |>
  rbindlist(fill = TRUE)

message("Step 2b: : dt_feat has: ", nrow(dt_feat), "rows before cohort restriction")

# Restrict to individuals in cohort

dt_feat <- dt_feat[pnr %in% population$pnr] 
message("Step 2b: : dt_feat restricted by pnr has: ", nrow(dt_feat))


# Exclude hospitalisation counts for the entry year (first observed year) for each person
# Identify hospitalisation columns
hosp_cols <- names(dt_feat)[grepl("^hc_hospitalizations_", names(dt_feat))]
# Compute entry year (minimum year) for each person
entry_year_dt <- dt_feat[, .(entry_year = min(year)), by = pnr]
# Merge entry year back to dt_feat
dt_feat <- merge(dt_feat, entry_year_dt, by = "pnr", all.x = TRUE)

# Set hospitalisation columns to NA for the entry year
for (col in hosp_cols) {
  dt_feat[year == entry_year, (col) := NA_real_]
}
# Remove temporary entry_year column
dt_feat[, entry_year := NULL]

message(" Step 3: Writing population panel files")
out_path <- paste0("/Data_files/population_panel_",cohort_name, "/")
dir.create(out_path, showWarnings = FALSE, recursive = TRUE)

lapply(seq_along(data_years), function(i) {
  yr <- data_years[i]
    message(" - Year",yr)
  
  dat <- open_dataset(panel_files[i]) |>
    filter (in_dk ==1 )|> 
    collect()
  setDT(dat)

  # Keep people in the cohort 
  dat <- dat[pnr %in% population$pnr] 


 write_parquet(dat, paste0(out_path, "population_panel", yr, ".parquet")) 
 rm(dat)
 gc() 
 invisible(NULL)
  

}) 
message("step 3 done")


rm(dt_band)
gc()


message("Step 4: creating confinguration lists")
# create configuration lists to feed into the function
#Icd diagnoses flags:
icd_columns <- lapply(
  grep("^hc_di_(?!count$)(?!.*NA)", colnames_tdf, value = TRUE, perl = TRUE),
  function(variable) {
    list(
      var_name = variable,
      value = "1",
      periods = time_periods,
      stats = "max",
      dataset = "population_panel"
    )
  }
)

# Medication flags:
med_columns <- lapply(
  grep("^med_(?!num_meds$)(?!.*NA)", colnames_tdf, value = TRUE, perl = TRUE),
  function(variable) {
    list(
      var_name = variable,
      value = "1",
      periods = time_periods,
      stats = "max",
      dataset = "population_panel"
    )
  }
)

# Numeric_variables

numeric_variables <- c(
  grep("^hc_util_", colnames_tdf, value = TRUE),
  grep("^hc_cost_", colnames_tdf, value = TRUE),
  grep("^hc_hospitalizations_", colnames_tdf, value = TRUE),
  "se_long_term_sick_leave_weeks",
  
  # All socio economic variables expept employment status
  grep(
    "^se_pers(?!.*.*employment_status)",
    colnames_tdf,
    value = TRUE,
    perl = TRUE), # 
  grep(
    "^se_hh(?!.*employment_status)",
    colnames_tdf,
    value = TRUE,
    perl = TRUE),
  # Add education rank variables: 
  "se_educ_rank", 
  "se_educ_rank_age_gender",
  # add comorbidies
  "hc_charlson_comorbidities",
  "hc_elixhauser_comorbidities",
  # Add diagnoses and medication count
  "med_num_meds",
  "hc_di_count"
  
)

numeric_columns <- lapply(numeric_variables, function(variable) {
  list(
    var_name = variable,
    periods = time_periods,
    stats = c("avg", "sd"),
    dataset = "population_panel"
  )
})


# Config of categorical variables
var_config <- c(
  list(
    # Marital status
    list(
      var_name = "de_marital_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "Married" # for categorical variables
    ),
    list(
      var_name = "de_marital_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "Divorced" # for categorical variables
    ),
      list(
        var_name = "de_marital_status",
        dataset = "population_panel", # dataset name without years
        periods = time_periods, # time periods in years
        value = "Widow" # for categorical variables
    ),
  
# Employment status (pre_socio), pers
        list(
      var_name = "se_pers_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "on_welfare" # for categorical variables
    ),
           list(
      var_name = "se_pers_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "disab_recipient" # for categorical variables
    ),
            list(
      var_name = "se_pers_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "employed" # for categorical variables
         ),
              list(
      var_name = "se_pers_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "in_school" # for categorical variables
    ),
            list(
      var_name = "se_pers_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "on_leave" # for categorical variables
          ),
              list(
      var_name = "se_pers_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "unemployed" # for categorical variables
    ),
    # Employment status (pre_socio), hh
        list(
      var_name = "se_hh_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "on_welfare" # for categorical variables
    ),
           list(
      var_name = "se_hh_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "disab_recipient" # for categorical variables
    ),
            list(
      var_name = "se_hh_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "employed" # for categorical variables

         ),
              list(
      var_name = "se_hh_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "in_school" # for categorical variables
    ),
            list(
      var_name = "se_hh_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "on_leave" # for categorical variables
          ),
              list(
      var_name = "se_hh_employment_status",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "unemployed" # for categorical variables
      ),

    # Family death events
    list(
      var_name = "fa_partner_died",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "1" # for categorical variables
    ),
    list(
      var_name = "fa_child_died",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "1" # for categorical variables
    ),
    list(
      var_name = "fa_parent_died",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "1" # for categorical variables
    ),
    list(
      var_name = "fa_hh_member_died",
      dataset = "population_panel", # dataset name without years
      periods = time_periods, # time periods in years
      value = "1" # for categorical variables
    )
  ), 
  icd_columns,
  med_columns,
  numeric_columns
)

message("step 4 done")

message("Step 5: Generating rolling variables...")

# Generate rolling variables
data_roll <- generate_rolling_variables(
  var_config = var_config,
  data_path = out_path
)

message("Step 5 done: rolling  dataset has: " , nrow(data_roll), "rows")


#Check duplicates 
dup_roll <- data_roll[, .N, by = .(pnr, year)] [N>1]
message("Number of duplicated pnr-year rows in data roll:", nrow(dup_roll))



#Merge rolling variables back onto dt
dt <- merge(dt_feat, data_roll, by = c("pnr", "year"), all.x = TRUE)

#Step 6
# Create final data and outcome:

# Keep one row per person at index age in the ageband
final_data <- dt[de_age == index_age]

# Check one row per person 
dup_final <- final_data[, .N, by = pnr] [N>1]
message("Number of duplicated persons in final data:", nrow(dup_final))

#Outcome: death between min_age and max_age (inclusive)
final_data[, early_death := 0L]
final_data[
  de_age_at_death >= min_age & de_age_at_death <= max_age,
  early_death := 1L]

# Write parquet.
final_parquet_path <- file.path(
  "Anne/Data_files/cohort_data",
  paste0("cohort_", cohort_name, ".parquet")
)
    write_parquet(final_data, final_parquet_path)

message("Finished: cohort file saved for ",  cohort_name, ".")
 
