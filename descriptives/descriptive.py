import pandas as pd
import numpy as np
import re


# Standardized mean difference for countinous variables
def smd_cont(x0,x1):
    m0=np.nanmean(x0)
    m1=np.nanmean(x1)
    s0=np.nanstd(x0,ddof=1)
    s1=np.nanstd(x1,ddof=1)

    pooled= np.sqrt((s0**2 + s1**2) /2)

    if pooled ==0 or np.isnan(pooled):
        return np.nan
    
    return ((m1 - m0) / pooled)

# Standardized mean difference for binary variables (and categorical levels)
def smd_binary(p0,p1):
    pbar= (p0 + p1) /2 
    denom = np.sqrt(pbar * (1-pbar))
    
    if denom ==0 or np.isnan(denom):
        return np.nan
    
    return ((p1-p0) / denom)

# Risk ratio for death from 2x2 counts:
def risk_ratio_from_counts(a,b,c,d):
    # a = dead among exposed
    #b = alive among exposed
    #c = dead among unexposed
    # d= alaive among unexposed

        #Haldane-Anscome correction if any zero cell
    if (a==0) or (b==0) or (c==0) or(d==0):
        a+=0.5
        b+=0.5
        c+=0.5
        d+=0.5

    #Risks
    risk_exposed= a /(a+b)
    risk_unexposed = c /(c+d)

    if risk_unexposed == 0 or np.isnan(risk_unexposed):
        return np.nan
    return risk_exposed / risk_unexposed



file_path = "/Anne/Data_files/cohort_dat/cohort60_to_64.parquet"
out_file= "/Anne/descriptive//descriptives_cohort_60-64_male.xlsx"

OUTCOME= "early_death"

#Load data
df=pd.read_parquet(file_path)

df = df[df["de_sex"] =="Male"].copy() 

# Remove id, parish etc. variables


keep_cols =[]
for c in df.columns:
    if c in {"pnr", "in_dk", "alive", "de_sex", "de_age"}: 
        continue
    if c.startswith(("de_parish", "de_municipality", "family_id", "de_region")):
        continue
    keep_cols.append(c)
 

# Ensure outcome i kept
if OUTCOME not in keep_cols: 
    keep_cols.append(OUTCOME)

df= df[keep_cols.copy()]
# Remove duplicates
df = df.loc[:, ~df.columns.str.endswith((".x", ".y"))] 

#Split groups

alive=df[df[OUTCOME]==0].copy()
dead= df[df[OUTCOME]==1].copy() 

print("Alive N:", len(alive)," | Dead N:", len(dead), "| Total:", len(df))
print("Columns kept:", df.shape[1])

# Find numeric cols 
numeric_cols = df.select_dtypes(include=["int", "float", "bool"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c!=OUTCOME] 

# Split numeric into continous and binary
binary_cols= []
cont_cols= []  
for c in numeric_cols: 
    vals=df[c].dropna().unique()
    if len(vals) > 0 and set(vals).issubset({0,1}):
        binary_cols.append(c)
    else:
        cont_cols.append(c) 

# Categorical cols
cat_cols =df.select_dtypes(include=["object", "category"]).columns.tolist()

print("Continuous:", len(cont_cols), "| Binary 0/1:", len(binary_cols), "| Categorical:", len(cat_cols))

# Build tabels

alive_label=f"Alive (N={len(alive):,} )"
dead_label =f"Deceased (N={ len(dead):,})"

# a) Continous: mean ±  sd

rows_cont=[]
for c in cont_cols: 
    rows_cont.append({
        "Feature":c, 
        alive_label: f"{alive[c].mean():.3f} ± {alive[c].std():.2f}",
        dead_label: f"{dead[c].mean():.3f} ± {dead[c].std():.2f}",
        "SMD": round(smd_cont(alive[c], dead[c]),3),
        "abs_SMD": round(abs(smd_cont(alive[c], dead[c])),3)
    })

tab_cont=pd.DataFrame(rows_cont)

#b) Binary (%==1) + RR_death
rows_bin = []
for c in binary_cols:

    p0=alive[c].mean()
    p1 = dead[c].mean()

    a= ((df[OUTCOME]==1) & (df[c] ==1)).sum()
    b= ((df[OUTCOME]==0) & (df[c] ==1)).sum()
    c0= ((df[OUTCOME]==1) & (df[c] ==0)).sum()
    d0= ((df[OUTCOME]==0) & (df[c] ==0)).sum()


    N_exposed = a+b

    rr= risk_ratio_from_counts(a,b,c0,d0)

    if N_exposed < 3:
        rr=np.nan

    rows_bin.append({
        "Feature": c,
        alive_label: f"{alive[c].mean()*100:.2f}%", 
        dead_label: f"{dead[c].mean()*100:.2f}%",
        "N_exposed": N_exposed,
        "SMD": round(smd_binary(p0,p1),3),
        "abs_SMD": round(abs(smd_binary(p0,p1)),3),
        "Risk ratio for death": round(rr,3) if not np.isnan(rr) else np.nan
    })

tab_bin = pd.DataFrame(rows_bin)
tab_bin = tab_bin[tab_bin["N_exposed"]>=3]     

# c) Categorical: % per level + RR_death
rows_cat=[]
for c in cat_cols:
    p0=alive[c].value_counts(normalize=True, dropna=False)*100
    p1=dead[c].value_counts(normalize=True, dropna=False)*100
    levels=sorted(set(p0.index).union(set(p1.index)), key=lambda x:"" if x is None else str(x))
    for lev in levels:

        p0_level = p0.get(lev,0) /100
        p1_level = p1.get(lev, 0)/100

        exposed= df[c].eq(lev) 

        a= ((df[OUTCOME]==1) & exposed).sum()
        b= ((df[OUTCOME]==0) & exposed).sum()
        c0= ((df[OUTCOME]==1) & (~exposed)).sum()
        d0= ((df[OUTCOME]==0) & (~exposed)).sum()

        N_exposed= a+b

        rr= risk_ratio_from_counts(a,b,c0,d0)

        if N_exposed <3:
            rr=np.nan


        rows_cat.append({
            "Feature": f"{c}:{lev} ",
            alive_label: f"{p0.get(lev,0):.2f}%",
            dead_label: f"{p1.get(lev,0):.2f}%",
            "N_exposed": N_exposed,
            "SMD": round(smd_binary(p0_level, p1_level),3),
            "abs_SMD": round(abs(smd_binary(p0_level, p1_level)),3),
            "Risk ratio for death": round(rr,3) if not np.isnan(rr) else np.nan

        })   
tab_cat=pd.DataFrame(rows_cat)
tab_cat = tab_cat[tab_cat["N_exposed"]>=3]

# save excel
with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    tab_cont.to_excel(writer, sheet_name="continuous", index=False)
    tab_bin.to_excel(writer, sheet_name="binary_0_1", index=False)
    tab_cat.to_excel(writer, sheet_name="categorical", index=False)

print("Saved:", out_file)
