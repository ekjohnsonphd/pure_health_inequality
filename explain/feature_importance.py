import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

# Beeswarm plot -------------------------------
def _plot_beeswarm(values, name, title, outdir):
    plt.figure()
    shap.plots.beeswarm(values, max_display=20, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{outdir}shap_summary_beeswarm_{name}.pdf")
    plt.close()

# Bar plot of shap features -------------------------------
def _plot_bar(values, columns, name, title, outdir):
    mean_abs_shap = pd.DataFrame(
        {
            "feature": columns,
            "mean_abs_shap": pd.DataFrame(values.values, columns=columns).abs().mean(),
        }
    ).sort_values(by="mean_abs_shap", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(
        mean_abs_shap["feature"][:20][::-1], mean_abs_shap["mean_abs_shap"][:20][::-1]
    )
    plt.xlabel("Mean absolute SHAP")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{outdir}shap_mean_barplot_{name}.pdf")
    plt.close()

def _process_shap_outputs(shap_values, columns):
    return(pd.DataFrame(
       {
        "feature": columns,
        "mean_abs_shap": pd.DataFrame(shap_values.values, columns=columns).abs().mean()
       } 
    ).sort_values(by="mean_abs_shap", ascending=False))

def run_shap_analysis(
    model, X_test, y_test, output_dir, threshold = 0.5, random_state: int = 42
):
    os.makedirs(output_dir, exist_ok=True)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    results_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    
    accurate_mask = results_df.y_true == results_df.y_pred
    true_positive_mask = (results_df.y_true == 1) & (results_df.y_pred == 1)
    deaths_mask = results_df.y_true == 1
    
    X_accurate = X_test[accurate_mask.values]
    X_tp = X_test[true_positive_mask.values]
    X_deaths = X_test[deaths_mask.values]
    
    # Estimate shap values
    explainer = shap.Explainer(model)
    shap_all = explainer(X_test)
    shap_accurate = explainer(X_accurate)
    shap_tp = explainer(X_tp)
    shap_deaths = explainer(X_deaths)

    _plot_bar(shap_all, X_test.columns, "all", "Top 20 feature importance: all", output_dir)
    _plot_bar(shap_tp, X_tp.columns, "true_positive", "Top 20 feature importance: true positives", output_dir)
    _plot_bar(shap_accurate, X_accurate.columns, "accurate", "Top 20 feature importance: accurate predictions", output_dir)
    _plot_bar(shap_deaths, X_deaths.columns, "deaths", "Top 20 feature importance: all deaths", output_dir)

    _plot_beeswarm(shap_all, "all", "Top 20 feature importance: all", output_dir)
    _plot_beeswarm(shap_tp, "true_positive", "Top 20 feature importance: true positives", output_dir)
    _plot_beeswarm(shap_accurate, "accurate", "Top 20 feature importance: accurate predictions", output_dir)
    _plot_beeswarm(shap_deaths, "deaths", "Top 20 feature importance: all deaths", output_dir)

    shap_all_df = _process_shap_outputs(shap_all, X_test.columns)
    shap_all_df["version"] = "all"
    shap_tp_df = _process_shap_outputs(shap_tp, X_tp.columns)
    shap_tp_df["version"] = "tp"
    shap_accurate_df = _process_shap_outputs(shap_accurate, X_accurate.columns)
    shap_accurate_df["version"] = "accurate"
    shap_deaths_df = _process_shap_outputs(shap_deaths, X_deaths.columns)
    shap_deaths_df["version"] = "deaths"

    shap_df = pd.concat([shap_all_df, shap_tp_df, shap_accurate_df, shap_deaths_df], ignore_index = True)
    return shap_df