# Function used in the script "cohort_data_prep.R"


#' Generate rolling statistics for specified time periods and variables
#'
#' @param var_config List of variable configurations with the following structure
#'    list(
#'      list(
#'        var_name = "de_age",
#'        dataset = "1_pop", # dataset name without years
#'        periods = list(c(0, 5), c(6, 10)),  # time periods in years
#'        stats = c("avg", "median"), # could be of the following: avg, median, min, max, sum, sd
#'        value = NULL # not applicable for numeric variables
#'      ),
#'      list(
#'        var_name = "de_marital_status",
#'        dataset = "1_pop", # dataset name without years
#'        periods = list(c(0, 5)),  # time periods in years
#'        stats = NULL, # not applicable for categorical variables
#'        value = "Married" # for categorical variables
#'        )
#'      )
#' @param data_path Optional path to data. Default to workdata folder
#' 
#' @return data.table with new rolling variables



generate_rolling_variables <- function(var_config, data_path = "D:/Nicolai/data") {
  suppressPackageStartupMessages({
    library(DBI)
    library(duckdb)
    library(data.table)
    library(future)
    library(future.apply)
    library(glue)
    library(magrittr)
  })
  # Create data.table with datasets, var_name, periods, and value from var_config
  config_dt <- rbindlist(lapply(var_config, function(config) {
    if (is.null(config$stats) & is.null(config$value)) config$stats <- "avg"
    rbindlist(lapply(config$periods, function(period) {
      data.table(
        dataset = config$dataset,
        var_name = config$var_name,
        period_start = period[1],
        period_end = period[2],
        stats = config$stats,
        value = config$value
      )
    }))
  }),
  fill = TRUE)
  
  # Loop through each variable and create all rolling variables
  unique_vars <- unique(config_dt$var_name)
  
  # Set up parallel processing
  plan(multisession, workers = min(5, length(unique_vars)))
  
  result_list <- future_lapply(unique_vars, function(var) {
    data_path <- data_path
    
    # Create DuckDB connection if not already active
    if (!exists("con") || !DBI::dbIsValid(con)){
      con <- dbConnect(duckdb())
      dbExecute(con, "SET memory_limit = '150GB'")
    }
    
    # Get all configs for this variable
    var_configs <- config_dt[var_name == var]
    
    # Build DuckDB query
    select_clauses <- c()
    for (i in seq_len(nrow(var_configs))) {
      config_row <- var_configs[i]
      dataset_name <- config_row$dataset
      var_name <- config_row$var_name
      period_start <- config_row$period_start
      period_end <- config_row$period_end
      stat <- config_row$stats
      value <- config_row$value
      
      
      if (is.null(value) || is.na(value)) {
        # Numeric variable: calculate rolling stats
        new_var_name <- sprintf("%s_roll_%s_%d_%d", var_name, stat, period_start, period_end)
        if (stat == "avg"){
          select_clauses <- c(select_clauses, glue("AVG({var_name}) OVER (PARTITION BY pnr ORDER BY year ROWS BETWEEN {period_end} PRECEDING AND {period_start} PRECEDING) AS {new_var_name}"))
        }
        if (stat == "median"){
          select_clauses <- c(select_clauses, glue("MEDIAN({var_name}) OVER (PARTITION BY pnr ORDER BY year ROWS BETWEEN {period_end} PRECEDING AND {period_start} PRECEDING) AS {new_var_name}"))
        }
        if (stat == "min"){
          select_clauses <- c(select_clauses, glue("MIN({var_name}) OVER (PARTITION BY pnr ORDER BY year ROWS BETWEEN {period_end} PRECEDING AND {period_start} PRECEDING) AS {new_var_name}"))
        }
        if (stat == "max"){
          select_clauses <- c(select_clauses, glue("MAX({var_name}) OVER (PARTITION BY pnr ORDER BY year ROWS BETWEEN {period_end} PRECEDING AND {period_start} PRECEDING) AS {new_var_name}"))
        }
        if (stat == "sum"){
          select_clauses <- c(select_clauses, glue("SUM({var_name}) OVER (PARTITION BY pnr ORDER BY year ROWS BETWEEN {period_end} PRECEDING AND {period_start} PRECEDING) AS {new_var_name}"))
        }
        if (stat == "sd"){
          select_clauses <- c(select_clauses, glue("STDDEV_POP({var_name}) OVER (PARTITION BY pnr ORDER BY year ROWS BETWEEN {period_end} PRECEDING AND {period_start} PRECEDING) AS {new_var_name}"))
        }
        
      } else {
        # Categorical variable: check if value is present within period
        new_var_name <- sprintf("%s_%s_%d_%d", var_name, value, period_start, period_end)
        select_clauses <- c(select_clauses, glue("BOOL_OR({var_name} = '{value}') OVER (PARTITION BY pnr ORDER BY year ROWS BETWEEN {period_end} PRECEDING AND {period_start} PRECEDING) AS {new_var_name}"))
      }
    }
    
    # Run query
    dt <- glue("
              SELECT
                pnr,
                year,
                {paste(select_clauses, collapse = ',\n')}
              FROM
                read_parquet('{file.path(data_path, paste0(dataset_name, '*'))}', union_by_name = true)
              ") %>% 
      dbGetQuery(con, .) %>% 
      setDT()
    
    # Return
    return(dt)
  })
  
  # Close workers
  plan(sequential)
  
  # Merge all datasets
  data <- Reduce(function(...) merge(..., by = c("pnr", "year"), all = TRUE), result_list)
  
  # Return final data
  return(data)
}
