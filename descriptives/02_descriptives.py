
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# Define cohort, sex filter, outcome, input file, and output Excel file.
# ─────────────────────────────────────────────────────────────────────────────

COHORT_NAME = "60_to_64"
SEX         = "Male"   # "Male" or "Female" or None to keep both
OUTCOME     = "early_death"

file_path = f"/Anne/Data_files/cohort_data/cohort_{COHORT_NAME}.parquet"
out_file  = f"/Anne/descriptive/descriptives_cohort_{COHORT_NAME}_{SEX}.xlsx"


# ─────────────────────────────────────────────────────────────────────────────
# Standardized mean difference for continuous variables
#
# This compares the mean difference between alive and deceased,
# scaled by the pooled standard deviation.
# ─────────────────────────────────────────────────────────────────────────────

def smd_cont(x0, x1):
    m0 = np.nanmean(x0)
    m1 = np.nanmean(x1)
    s0 = np.nanstd(x0, ddof=1)
    s1 = np.nanstd(x1, ddof=1)
    
    pooled = np.sqrt((s0**2 + s1**2) / 2)
    
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    
    return (m1 - m0) / pooled


# ─────────────────────────────────────────────────────────────────────────────
# Standardized mean difference for binary/categorical level variables
#
# This compares percentages between alive and deceased.
# ─────────────────────────────────────────────────────────────────────────────

def smd_binary(p0, p1):
    pbar = (p0 + p1) / 2
    denom = np.sqrt(pbar * (1 - pbar))
    
    if denom == 0 or np.isnan(denom):
        return np.nan
    
    return (p1 - p0) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Risk ratio for death from 2x2 counts
#
# Compares death risk among exposed vs unexposed.
# Adds 0.5 to all cells if any cell is zero.
# ─────────────────────────────────────────────────────────────────────────────

def risk_ratio_from_counts(a, b, c, d):
    # a = dead among exposed
    # b = alive among exposed
    # c = dead among unexposed
    # d = alive among unexposed

    # Correction if any cell is zero
    if (a == 0) or (b == 0) or (c == 0) or (d == 0):
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    risk_exposed   = a / (a + b)
    risk_unexposed = c / (c + d)

    if risk_unexposed == 0 or np.isnan(risk_unexposed):
        return np.nan
    
    return risk_exposed / risk_unexposed


# ─────────────────────────────────────────────────────────────────────────────
# Load cohort data
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_parquet(file_path)


# ─────────────────────────────────────────────────────────────────────────────
# Optional sex filter
# If SEX is "Male" or "Female", keep only that group.
# If SEX is None, keep everyone.
# ─────────────────────────────────────────────────────────────────────────────

if SEX is not None:
    df = df[df["de_sex"] == SEX].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Remove ID and geography columns
#
# These columns are not used in the descriptive tables.
# ─────────────────────────────────────────────────────────────────────────────

keep_cols = []

for c in df.columns:
    if c in {"pnr", "in_dk", "alive", "de_sex", "de_age"}:
        continue
    
    if c.startswith(("de_parish", "de_municipality", "family_id", "de_region")):
        continue
    
    keep_cols.append(c)


# Make sure the outcome column is kept
if OUTCOME not in keep_cols:
    keep_cols.append(OUTCOME)


# Keep selected columns only
df = df[keep_cols]


# Remove duplicated suffix columns, for example columns ending in .x or .y
df = df.loc[:, ~df.columns.str.endswith((".x", ".y"))]


# ─────────────────────────────────────────────────────────────────────────────
# Split data by outcome
#
# alive = early_death == 0
# dead  = early_death == 1
# ─────────────────────────────────────────────────────────────────────────────

alive = df[df[OUTCOME] == 0].copy()
dead  = df[df[OUTCOME] == 1].copy()

print("Alive N:", len(alive), " | Dead N:", len(dead), "| Total:", len(df))
print("Columns kept:", df.shape[1])


# ─────────────────────────────────────────────────────────────────────────────
# Classify columns as continuous, binary, or categorical
# ─────────────────────────────────────────────────────────────────────────────

# Start with all numeric columns
numeric_cols = df.select_dtypes(include=["int", "float", "bool"]).columns.tolist()

# Remove outcome from numeric feature list
numeric_cols = [c for c in numeric_cols if c != OUTCOME]

binary_cols = []
cont_cols   = []

# Binary columns are numeric columns that only contain 0 and 1
for c in numeric_cols:
    vals = df[c].dropna().unique()
    
    if len(vals) > 0 and set(vals).issubset({0, 1}):
        binary_cols.append(c)
    else:
        cont_cols.append(c)

# Categorical columns are object/category columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

print("Continuous:", len(cont_cols), "| Binary 0/1:", len(binary_cols), "| Categorical:", len(cat_cols))


# Labels used in the output tables
alive_label = f"Alive (N={len(alive):,})"
dead_label  = f"Deceased (N={len(dead):,})"


# ─────────────────────────────────────────────────────────────────────────────
# A) Continuous variables
#
# For each continuous variable, calculate:
# - mean ± standard deviation among alive
# - mean ± standard deviation among deceased
# - standardized mean difference
# ─────────────────────────────────────────────────────────────────────────────

rows_cont = []

for c in cont_cols:
    smd = smd_cont(alive[c], dead[c])
    
    rows_cont.append({
        "Feature":    c,
        alive_label:  f"{alive[c].mean():.3f} ± {alive[c].std():.2f}",
        dead_label:   f"{dead[c].mean():.3f} ± {dead[c].std():.2f}",
        "SMD":        round(smd, 3),
        "abs_SMD":    round(abs(smd), 3),
    })

tab_cont = pd.DataFrame(rows_cont)


# ─────────────────────────────────────────────────────────────────────────────
# B) Binary variables
#
# For each binary variable, calculate:
# - percent with value 1 among alive
# - percent with value 1 among deceased
# - number exposed
# - SMD
# - risk ratio for death
# ─────────────────────────────────────────────────────────────────────────────

rows_bin = []

for c in binary_cols:
    p0 = alive[c].mean()
    p1 = dead[c].mean()

    # 2x2 table counts
    a  = ((df[OUTCOME] == 1) & (df[c] == 1)).sum()
    b  = ((df[OUTCOME] == 0) & (df[c] == 1)).sum()
    c0 = ((df[OUTCOME] == 1) & (df[c] == 0)).sum()
    d0 = ((df[OUTCOME] == 0) & (df[c] == 0)).sum()

    N_exposed = a + b
    rr  = risk_ratio_from_counts(a, b, c0, d0)
    smd = smd_binary(p0, p1)

    # Suppress risk ratio if very few exposed people
    if N_exposed < 3:
        rr = np.nan

    rows_bin.append({
        "Feature":             c,
        alive_label:           f"{p0 * 100:.2f}%",
        dead_label:            f"{p1 * 100:.2f}%",
        "N_exposed":           N_exposed,
        "SMD":                 round(smd, 3),
        "abs_SMD":             round(abs(smd), 3),
        "Risk ratio for death": round(rr, 3) if not np.isnan(rr) else np.nan,
    })

tab_bin = pd.DataFrame(rows_bin)

# Keep only rows with at least 3 exposed people
tab_bin = tab_bin[tab_bin["N_exposed"] >= 3]


# ─────────────────────────────────────────────────────────────────────────────
# C) Categorical variables
#
# For each category level, calculate:
# - percent in alive group
# - percent in deceased group
# - number exposed
# - SMD
# - risk ratio for death
# ─────────────────────────────────────────────────────────────────────────────

rows_cat = []

for c in cat_cols:
    # Percent distribution for each level
    p0 = alive[c].value_counts(normalize=True, dropna=False) * 100
    p1 = dead[c].value_counts(normalize=True, dropna=False) * 100

    # Get all levels that appear in either group
    levels = sorted(
        set(p0.index).union(set(p1.index)),
        key=lambda x: "" if x is None else str(x)
    )
    
    for lev in levels:
        p0_level = p0.get(lev, 0) / 100
        p1_level = p1.get(lev, 0) / 100
        
        # Exposed means belonging to this category level
        exposed = df[c].eq(lev)

        # 2x2 table counts
        a  = ((df[OUTCOME] == 1) & exposed).sum()
        b  = ((df[OUTCOME] == 0) & exposed).sum()
        c0 = ((df[OUTCOME] == 1) & (~exposed)).sum()
        d0 = ((df[OUTCOME] == 0) & (~exposed)).sum()

        N_exposed = a + b
        rr  = risk_ratio_from_counts(a, b, c0, d0)
        smd = smd_binary(p0_level, p1_level)

        # Suppress risk ratio if very few exposed people
        if N_exposed < 3:
            rr = np.nan

        rows_cat.append({
            "Feature":             f"{c}:{lev}",
            alive_label:           f"{p0.get(lev, 0):.2f}%",
            dead_label:            f"{p1.get(lev, 0):.2f}%",
            "N_exposed":           N_exposed,
            "SMD":                 round(smd, 3),
            "abs_SMD":             round(abs(smd), 3),
            "Risk ratio for death": round(rr, 3) if not np.isnan(rr) else np.nan,
        })

tab_cat = pd.DataFrame(rows_cat)

# Keep only rows with at least 3 exposed people
tab_cat = tab_cat[tab_cat["N_exposed"] >= 3]


# ─────────────────────────────────────────────────────────────────────────────
# Save descriptive tables to Excel
#
# The Excel file has three sheets:
# - continuous
# - binary_0_1
# - categorical
# ─────────────────────────────────────────────────────────────────────────────

with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
    tab_cont.to_excel(writer, sheet_name="continuous", index=False)
    tab_bin.to_excel(writer, sheet_name="binary_0_1", index=False)
    tab_cat.to_excel(writer, sheet_name="categorical", index=False)

print("Saved:", out_file)

