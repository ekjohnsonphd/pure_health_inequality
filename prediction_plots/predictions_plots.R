library(data.table)
library(arrow)

pred_file <-"/XBoost_results/Female_65-69/prediction/predictions.parquet"
cohort_file <- "/data/Data_files/cohort_data/cohort65_to_69.parquet"

out_dir <- "/XBoost_results/Female_65-69/prediction"

#Load data
pred <- read_parquet(pred_file, as.data.frame=TRUE)
cohort <- read_parquet(cohort_file, as_data_frame = TRUE)

setDT(pred)
setDT(cohort)


cohort_small <- cohort[, .(pnr, de_age_at_death, de_sex, early_death)] 


dt <- merge(pred, cohort_small, by ="pnr", all.x = TRUE)

dt[, pnr:=NULL]

#Save
#fwrite(dt, file.path(out_dir,"dt_y_test_age_of_death.csv"))


## False positives
library(data.table)
library(ggplot2)

# Filter false positives
fp <- dt[y_test==0 & y_pred ==1]

# Make age groups 65+

fp[, age_group :=fifelse(
  is.na(de_age_at_death),
  "Alive/censored",
  as.character(cut(
    de_age_at_death,
    breaks=c(70,75,80,85,Inf),
    right= FALSE,
    labels=c("70-74","75-79","80-84","85+")
  ))
)]

# Correct order
fp[, age_group :=factor(
  age_group,
  levels =c("70-74","75-79","80-84","85+", "Alive/censored")
  )]


# Remove NA: 
fp_plot_dt <-fp [!is.na(age_group)]


# Plot
p <- ggplot(fp_plot_dt, aes(x=age_group)) +
  geom_bar(fill="blue", alpha=0.7)+
  labs(
    title = "False positives: age at death",
    x="Age at death",
    y="Count"
  ) +
  theme_minimal(base_size = 13)



ggsave(
  filename=file.path(out_dir,"false_positives_age_at_death.png"),
  plot=p,
)
p



# Density plot: 

# Median + threshold

thr=unique(pred$threshold)

vlines <-pred[, .(
  x=median(y_proba, na.rm=TRUE)
), by =y_test]

p_combined <- ggplot(pred, aes(x=y_proba, colour=factor(y_test)))+
  geom_density(linewidth=1.1, adjust =1.2) +
  
  # Median lines
  
  geom_vline(
    data=vlines, 
    aes(xintercept=x, colour=factor(y_test)),
    linetype="dashed",
    linewidth=1
)+
  
  # Threshold line
  
  geom_vline(
    xintercept=thr,
    linetype="dashed",
    colour="black",
    linewidth=1
  )+
  
  scale_colour_manual(values = c("0"= "blue", "1"="red"),
                      labels=c("0"="y_test=0","1" ="y_test=1"),
                      name=NULL)+
  labs(
    title = "Predicted p(early death) by true outcome",
    x="Predicted p(early death)",
    y= "Density"
  )+
  theme_minimal(base_size = 13)+
  coord_cartesian(clip = "off") +

  annotate(
    "text",
    x=thr+0.03, 
    y=Inf, 
    label= paste0("Threshold= ", round(thr,2)),
    vjust=6,
    hjust=0,
    colour="black",
    size=3.5
  )+

  annotate(
    "text",
    x=Inf, 
    y=2.6, 
    label= "Median y_test=0",
    hjust=1.1,
    colour="blue",
    size=3.5
  )+
  annotate(
    "text",
    x=Inf, 
    y=2.43, 
    label= "Median y_test=1",
    hjust=1.1,
    colour="red",
    size=3.5
  )
  
  
  

ggsave(
  filename= file.path(out_dir,"p_early_death_density_median_threshold.png"),
  plot=p_combined,
)


