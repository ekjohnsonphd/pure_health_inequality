from pathlib import Path
from numpy import int_
import pandas as pd
import xgboost as xgb
import numpy as np
import shap

#Paths


cohort= "female_50-54"

RESULTS_DIR = Path(f"/data/XBoost_results/{cohort}")

MODEL_PATH = RESULTS_DIR / f"model5_best_model_{cohort} .json"
X_PATH= RESULTS_DIR /f"X_test_{cohort}.csv"
Y_PATH = RESULTS_DIR / "y_test_{cohort}.csv"

# Load model and data
model=xgb.XGBClassifier()
model.load_model(MODEL_PATH)

X_test=pd.read_csv(X_PATH)
y_test=pd.read_csv(Y_PATH)["y_test"] 

# SHAP decomposition:


early_deaths = X_test[y_test == 1]
survivors = X_test[y_test == 0]

p_early_deaths = model.predict_proba(early_deaths)[:, 1]
p_survivors = model.predict_proba(survivors)[:, 1]

print(f"Early deaths prediction: {p_early_deaths.mean()}")
print(f"Survivors prediction: {p_survivors.mean()}")
print(f"Gap: {p_early_deaths.mean() - p_survivors.mean()}")

idx = np.random.choice(survivors.shape[0], size=5000, replace=False)
baseline = survivors.iloc[idx]
explainer = shap.TreeExplainer(
    model,
    data = baseline,
    feature_perturbation="interventional",
    model_output = "probability"
)
shap_early_deaths = explainer.shap_values(early_deaths)
shap_survivors = explainer.shap_values(survivors)
base_value = explainer.expected_value

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



df_shap.to_csv(RESULTS_DIR/ "shap_values.csv", index=False)


