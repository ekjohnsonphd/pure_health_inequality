library(data.table)
library(arrow)


#Config

cohort_file <- "/cohort_data/cohort65_to_69.parquet"
pred_file <-"/XBoost_results/Female_65-69/prediction/predictions.parquet"
death_file <-"/data/Nicolai/ExpBoD-data/rawdata2/Grunddata/dodsaasg2022.parquet"

dex_icd_map <- "/data/DEX_ICD_map_v148.csv"
dex_causes <- "/data/DEX_causelist_v148.csv"

out_dir <- "/data/XBoost_results/Female_65-69/prediction"




#Helper: standardize strings
clean_icd10 <- function(x) {
  x <- as.character(x)
  x <- trimws(x)
  x <- toupper(x)
  x[x==""] <- NA
  x
}


#Progressive mapping: exact, then trim -1, then trim -2

map_icd10_to_acause <-function(icd_vec, lookup_dt) {
  # lookup_dt must have columns icd10, acause
  icd_vec <- clean_icd10(icd_vec)
  
  #named vector for fast lookup
  v0 <- setNames(lookup_dt$acause,lookup_dt$icd10)
  
  out <- v0[icd_vec]
  
  #Trim -1
  na1 <- which(is.na(out) & !is.na(icd_vec))
  if (length(na1)) {
    icd_trim1 <- substr(icd_vec[na1],1,pmax(nchar(icd_vec[na1])-1,0))
    out[na1] <-v0[icd_trim1]
  }
  
  #Trim -2
  na2 <- which(is.na(out) & !is.na(icd_vec))
  if (length(na2)) {
    icd_trim2 <- substr(icd_vec[na2],1,pmax(nchar(icd_vec[na2])-2,0))
    out[na2] <-v0[icd_trim2]
  }
  
  out
  
}


#Load predictions
pred <- read_parquet(pred_file, as_data_frame = TRUE)
setDT(pred)


#Load cohort and restrict to females who die in the window
cohort <- read_parquet(cohort_file, as_data_frame = TRUE)
setDT(cohort)


cohort_dead <- cohort[de_sex=="Female" & early_death ==1,.(pnr)]
cohort_dead <- unique(cohort_dead[!is.na(pnr)])

# load death and restrict to cohort deaths only
death <- read_parquet(death_file, as_data_frame = TRUE)
setDT(death)

death <- death[, .(
  pnr,
  icd10 = clean_icd10(c_dodtilgrundl_acme)
)]

death <- death[!is.na(icd10)] 

# Restrict:
setkey(cohort_dead, pnr)
setkey(death, pnr)

#Merge
death <-death[cohort_dead, nomatch = 0] 


# Dex ICD mappings
icd_map <- fread(dex_icd_map)
icd_map <- icd_map [code_system=="icd10", .(
  icd10= clean_icd10(icd_code),
  acause=as.character(acause)
)]
icd_map <- icd_map[!is.na(icd10)]


#Map
death[, acause:=map_icd10_to_acause(icd10, icd_map)]
death[is.na(acause),acause:="unknown"]

#Better cause names
causes <- fread(dex_causes)
causes <- causes[, .(
  acause=as.character(acause),
  cause_name=as.character(cause_name),
  family = as.character(family),
  family_name= as.character(family_name)
  
)]

# Merge 
setkey(causes,acause)
setkey(death,acause)
death <- causes[death] # left join death onto causes

# Merge in predictions
setkey(pred,pnr)
setkey(death, pnr)

dt <-merge(pred,death ,by ="pnr", all.x=TRUE)

dt[, pnr:=NULL]

#Save
#fwrite(dt, file.path(out_dir,"dt_predictions_with_causes.csv"))



dt[y_test==1, .(
  N= .N, 
  NA_acause=sum(is.na(acause)),
  unknown =sum(acause=="unknown", na.rm = TRUE)
)]

dt[y_test==0, .(
  N= .N, 
  NA_acause=sum(is.na(acause)),
  unknown =sum(acause=="unknown", na.rm = TRUE)
)]

##
# Restrict to deaths with cause
dt_dead <- dt[y_test==1 & !is.na(acause)]

#label TP/FN
dt_dead[, outcome := fifelse(y_pred==1, "TP", "FN")]

#Totals
total_tp <- dt_dead[outcome=="TP", .N]
total_fn <- dt_dead[outcome=="FN", .N]
total_deaths <- nrow(dt_dead)

# Summarize per cause
cause_summary <- dt_dead[
  ,
  .(
    n_tp=sum(outcome == "TP"),
    n_fn=sum(outcome =="FN"),
    n_deaths =.N
  ),
  by =.(acause, cause_name,family, family_name)
]

# Add proportions 
cause_summary[
  ,
  `:=`(
    prop_of_all_TP = n_tp/ total_tp,
    prop_of_all_FN = n_fn/ total_fn,
    prop_cause_TP = n_tp / (n_tp+n_fn),
    prop_of_all_deaths = n_deaths /total_deaths
  )
]

#Remove garbage code
cause_summary <-cause_summary[cause_name !="Garbage code" & family_name !="Garbage code"]
# save
fwrite(cause_summary, file.path(out_dir,"cause_summary_all.csv"))

# Top 50 by frequency
cause_top50 <- cause_summary[
  order(-n_deaths)
] [1:50]

fwrite(cause_top50, file.path(out_dir,"cause_top50.csv"))


# Top 15 by frequency
cause_top15 <- cause_summary[
  order(-n_deaths)
] [1:15]
### Plots 



library(ggplot2)

p_top_15 <-ggplot(cause_top15, aes(x=reorder(cause_name,prop_cause_TP),y=prop_cause_TP))+
  geom_col(fill= "#4c78a8")+
  coord_flip()+
  labs(
    x=NULL,
    y="TP share within cause (TP/ (TP+FN))",
    title="Model performance by cause of death (Top 15 causes)"
  ) +
  theme_bw()

# Save
ggsave(
  filename=file.path(out_dir, "top15_tp_share_by_cause.png"),
  plot=p_top_15
)

