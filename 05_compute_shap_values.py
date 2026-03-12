from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# Paths
RESULTS_DIR = Path("XBoost_results/Female_50-54")

MODEL_PATH = RESULTS_DIR / "model5_best_model_female_50_54.json"
X_PATH = RESULTS_DIR / "X_test_female_50_54.csv"
Y_PATH = RESULTS_DIR / "y_test_female_50_54.csv"

# Load model
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

X_test = pd.read_csv(X_PATH)
y_test = pd.read_csv(Y_PATH)["y_test"]

# Split groups
early_deaths = X_test[y_test == 1]
survivors = X_test[y_test == 0]

# Predictions
p_early_deaths = model.predict_proba(early_deaths)[:, 1]
p_survivors = model.predict_proba(survivors)[:, 1]

print(f"Early deaths prediction: {p_early_deaths.mean()}")
print(f"Survivors prediction: {p_survivors.mean()}")
print(f"Gap: {p_early_deaths.mean() - p_survivors.mean()}")

# SHAP baseline
np.random.seed(42)

baseline_size = min(500, survivors.shape[0])
idx = np.random.choice(survivors.shape[0], size=baseline_size, replace=False)
baseline = survivors.iloc[idx]


explainer = shap.TreeExplainer(
    model,
    data=baseline,
    feature_perturbation="interventional",
    model_output="probability"
)


# SHAP values
shap_early_deaths = explainer.shap_values(early_deaths)
shap_survivors = explainer.shap_values(survivors)
base_value = explainer.expected_value


# Save SHAP dataframe
feature_names = list(X_test.columns)

df_early_deaths = pd.DataFrame(shap_early_deaths, columns=feature_names)
df_early_deaths["y"] = 1
df_early_deaths["pred"] = p_early_deaths
df_early_deaths["baseline"] = base_value

df_survivors = pd.DataFrame(shap_survivors, columns=feature_names)
df_survivors["y"] = 0
df_survivors["pred"] = p_survivors
df_survivors["baseline"] = base_value

df_shap = pd.concat([df_early_deaths, df_survivors], ignore_index=True)

df_shap.to_csv(RESULTS_DIR / "shap_values.csv", index=False)