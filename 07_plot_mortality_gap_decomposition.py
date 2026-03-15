import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")

plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# --- Load data ---
shap = pd.read_csv("data/20260212/shap_results_all.csv")

# --- Minimal cleaning (match your R logic) ---
shap["group1"] = shap["group1"].replace({
    "Heathcare dx": "Disease history",
    "Healthcare dx": "Disease history",
    "Immigration": "Demographics and household",
    "Marital status": "Demographics and household",
    "Family characteristics": "Demographics and household",
})

shap["sex"] = shap["age_bin"].str.extract(r"^(Male|Female)")
shap["age_label"] = shap["age_bin"].str.replace(r"^(Male|Female)_", "", regex=True)

# --- Orders (as in your R code) ---
age_order = ["50-54", "55-59", "60-64", "65-69"]
group_order = [
    "Healthcare costs & utilization",
    "Disease history",
    "Economic characteristics",
    "Demographics and household",
    "Psychiatric medications",
    "Year & month of birth",
    "Parish characteristics",
]
temporal_order = ["Immediate", "Proximal", "Distal"]

# --- Colors ---
cat_colors = {
    "Healthcare costs & utilization": "#8DD3C7",
    "Disease history": "#FFFFB3",
    "Economic characteristics": "#BEBADA",
    "Demographics and household": "#FB8072",
    "Psychiatric medications": "#80B1D3",
    "Year & month of birth": "#FDB462",
    "Parish characteristics": "#B3DE69",
}
temp_colors = {
    "Immediate": "#E74C3C",
    "Proximal":  "#F39C12",
    "Distal":    "#3498DB",
}

# --- Helper: get total gap per sex-age (mean_pred_1 - mean_pred_0) ---
# Assumes shap_results_all.csv contains mean_pred_0 and mean_pred_1 columns.
if not {"mean_pred_0", "mean_pred_1"}.issubset(shap.columns):
    raise ValueError(
        "Expected columns mean_pred_0 and mean_pred_1 in shap_results_all.csv "
        "to scale bars to the risk gap, but they were not found."
    )

gap_df = (
    shap.groupby(["sex", "age_label"], as_index=False)[["mean_pred_0", "mean_pred_1"]]
        .first()
)
gap_df["total_gap"] = gap_df["mean_pred_1"] - gap_df["mean_pred_0"]

# --- Helper: percent composition + scaled heights that sum to total_gap ---
def prep_scaled(df, group_col, group_levels, gap_df):
    agg = (
        df.groupby(["sex", "age_label", group_col], as_index=False)["shap"]
          .sum()
          .rename(columns={"shap": "value"})
    )

    # Ensure all groups exist (fill missing as 0)
    idx = pd.MultiIndex.from_product(
        [agg["sex"].unique(), age_order, group_levels],
        names=["sex", "age_label", group_col],
    )
    agg = agg.set_index(["sex", "age_label", group_col]).reindex(idx, fill_value=0).reset_index()

    # Percent within each sex-age (robust to signs)
    agg["abs_value"] = agg["value"].abs()
    agg["total_abs"] = agg.groupby(["sex", "age_label"])["abs_value"].transform("sum").replace(0, np.nan)
    agg["pct"] = 100 * agg["abs_value"] / agg["total_abs"]
    agg["pct"] = agg["pct"].fillna(0)

    # Merge in total_gap and scale heights so bars sum to the gap
    agg = agg.merge(gap_df[["sex", "age_label", "total_gap"]], on=["sex", "age_label"], how="left")
    agg["height"] = (agg["pct"] / 100.0) * agg["total_gap"]

    # Top-3 labels per sex-age (still in %)
    agg["rank"] = agg.groupby(["sex", "age_label"])["abs_value"].rank(method="first", ascending=False)
    agg["label"] = np.where(agg["rank"] <= 3, agg["pct"].round(1).astype(str) + "%", "")

    # Ordering
    agg["age_label"] = pd.Categorical(agg["age_label"], categories=age_order, ordered=True)
    agg[group_col] = pd.Categorical(agg[group_col], categories=group_levels, ordered=True)
    agg = agg.sort_values(["sex", "age_label", group_col])

    return agg

cat_scaled = prep_scaled(shap, "group1", group_order, gap_df)
temp_scaled = prep_scaled(shap, "group2", temporal_order, gap_df)

# --- Plotting helper (stacked bars whose total height = total_gap) ---
def stacked_panel(ax, df_scaled, group_col, group_levels, colors, title,
                  show_legend=True, legend_title=None, ymax=None):
    x = np.arange(len(age_order))
    width = 0.72
    bottom = np.zeros(len(age_order), dtype=float)

    for g in group_levels:
        sub = df_scaled[df_scaled[group_col] == g].copy()
        sub = sub.set_index("age_label").reindex(age_order).reset_index()

        vals = sub["height"].to_numpy()  # <- scaled to total_gap
        ax.bar(x, vals, width=width, bottom=bottom, label=g,
               color=colors[g], edgecolor="white", linewidth=0.4)

        # % labels (top-3)
        labels = sub["label"].to_numpy()
        for i, (v, lab) in enumerate(zip(vals, labels)):
            if lab != "" and v > 0:
                ax.text(x[i], bottom[i] + v/2, lab, ha="center", va="center",
                        fontsize=9, color="black")

        bottom += vals

    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(age_order, rotation=45, ha="right")

    # y-axis in probability units (gap), not 0–100
    if ymax is None:
        ymax = max(bottom) * 1.15 if max(bottom) > 0 else 1.0
    ax.set_ylim(0, ymax)

    ax.set_ylabel("Mortality risk gap (predicted probability)")
    ax.set_xlabel("Age Group")
    ax.grid(axis="y", alpha=0.25)

    if show_legend:
        ax.legend(title=legend_title, bbox_to_anchor=(1.02, 0.5),
                  loc="center left", frameon=False)

# Determine a shared y-limit so panels are comparable (optional but usually nicer)
overall_max_gap = gap_df["total_gap"].max()
shared_ymax = overall_max_gap * 1.15

# --- Build 2x2 figure ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

# A: Feature Categories - Females
stacked_panel(
    axes[0, 0],
    cat_scaled[cat_scaled["sex"] == "Female"],
    "group1",
    group_order,
    cat_colors,
    "A. Feature Categories – Females",
    show_legend=True,
    legend_title="Feature Group",
    ymax=shared_ymax,
)

# B: Feature Categories - Males
stacked_panel(
    axes[0, 1],
    cat_scaled[cat_scaled["sex"] == "Male"],
    "group1",
    group_order,
    cat_colors,
    "B. Feature Categories – Males",
    show_legend=False,
    ymax=shared_ymax,
)

# C: Time Period - Females
stacked_panel(
    axes[1, 0],
    temp_scaled[temp_scaled["sex"] == "Female"],
    "group2",
    temporal_order,
    temp_colors,
    "C. Time Period – Females",
    show_legend=True,
    legend_title="Time Period",
    ymax=shared_ymax,
)

# D: Time Period - Males
stacked_panel(
    axes[1, 1],
    temp_scaled[temp_scaled["sex"] == "Male"],
    "group2",
    temporal_order,
    temp_colors,
    "D. Time Period – Males",
    show_legend=False,
    ymax=shared_ymax,
)

out_path = "paper/figures/figure1_mortality_gap_decomposition.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight", transparent=True)
plt.show()