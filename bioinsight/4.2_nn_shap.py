import pandas as pd
import numpy as np
import shap
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt

df = pd.read_csv("model_dataset.csv").sample(300, random_state=42)

features = [
    "molecular_weight", "alogp", "hba", "hbd",
    "tpsa", "rotatable_bonds",
    "heavy_atom_count", "aromatic_rings"
]

X = df[features].values
y = df["active"].values

imputer = joblib.load("nn_imputer.pkl")
scaler = joblib.load("nn_scaler.pkl")
model = tf.keras.models.load_model("nn_model.keras")

X = imputer.transform(X)
X = scaler.transform(X)

background = X[np.random.choice(X.shape[0], 50, replace=False)]

explainer = shap.KernelExplainer(model.predict, background)
shap_values = explainer.shap_values(X[:200])

# Fix: shap_values often comes as a list or with an extra dimension for the output class
if isinstance(shap_values, list):
    # If list (e.g., [class0_shap, class1_shap]), take the one for the positive class if binary
    print(f"SHAP values is list of length {len(shap_values)}")
    shap_values = shap_values[0]
elif len(shap_values.shape) == 3:
    # If returned as (samples, features, outputs), take the first output
    print(f"SHAP values shape {shap_values.shape}, selecting first output dimension.")
    shap_values = shap_values[:, :, 0]

plt.figure(figsize=(10, 6))

shap.summary_plot(
    shap_values,
    X[:200],
    feature_names=features,
    plot_type="dot",
    max_display=8,
    show=False
)

plt.title("Neural Network SHAP Summary (Bioactivity Prediction)")
plt.xlabel("SHAP value (impact on model output)")
plt.tight_layout()

plt.savefig("shap_nn_summary.png", dpi = 300)
plt.close()

print("Saved shap_nn_summary.png")

# Mean absolute SHAP values (global importance)
mean_shap = np.abs(shap_values).mean(axis=0)

# Sort features by importance
sorted_idx = np.argsort(mean_shap)
sorted_features = np.array(features)[sorted_idx]
sorted_shap = mean_shap[sorted_idx]

plt.figure(figsize=(10, 6))
bars = plt.barh(sorted_features, sorted_shap, color='skyblue', edgecolor='navy')
plt.xlabel("mean(|SHAP value|) (average impact on model output probability)")
plt.title("Global Feature Importance (Neural Network)")
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add values to the bars
for bar in bars:
    width = bar.get_width()
    plt.text(
        width, 
        bar.get_y() + bar.get_height()/2, 
        f'{width:.4f}', 
        ha='left', 
        va='center', 
        fontsize=10,
        color='black'
    )

plt.tight_layout()
plt.savefig("shap_nn_bar.png", dpi=300)
plt.close()

print("Saved shap_nn_bar.png")
