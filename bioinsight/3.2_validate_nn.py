import tensorflow as tf
import joblib
import numpy as np

# Load artifacts
model = tf.keras.models.load_model("nn_model.keras")
imputer = joblib.load("nn_imputer.pkl")
scaler = joblib.load("nn_scaler.pkl")

X_new = np.array([[
    350.2,   # molecular_weight
    2.4,     # alogp
    6,       # hba
    1,       # hbd
    75.3,    # tpsa
    4,       # rotatable_bonds
    24,      # heavy_atom_count
    2        # aromatic_rings
]])

# Preprocess
X_new = imputer.transform(X_new)
X_new = scaler.transform(X_new)

# Predict
prob = model.predict(X_new)[0][0]
pred = int(prob >= 0.5)

print("Predicted probability:", prob)
print("Predicted class:", pred)