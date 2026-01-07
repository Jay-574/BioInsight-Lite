import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# Load data (small sample for speed)
# -----------------------------
df = pd.read_csv("model_dataset.csv").sample(5000, random_state=57)

features = [
    "molecular_weight", "alogp", "hba", "hbd",
    "tpsa", "rotatable_bonds",
    "heavy_atom_count", "aromatic_rings"
]

X = df[features]
y = df["active"]

model = joblib.load("logistic_model_500000.pkl")

# Extract preprocessing + classifier
imputer = model.named_steps["imputer"]
scaler = model.named_steps["scaler"]
clf = model.named_steps["clf"]

X_imp = imputer.transform(X)
X_scaled = scaler.transform(X_imp)

explainer = shap.LinearExplainer(clf, X_scaled)
shap_values = explainer.shap_values(X_scaled)

shap.summary_plot(
    shap_values,
    X,
    feature_names=features,
    show=False
)

plt.savefig("shap_logistic_summary.png", bbox_inches="tight")
plt.close()

print("Saved shap_logistic_summary.png")
