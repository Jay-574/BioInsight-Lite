import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
print("Loading dataset...")
df = pd.read_csv("model_dataset.csv")

features = [
    "molecular_weight", "alogp", "hba", "hbd",
    "tpsa", "rotatable_bonds",
    "heavy_atom_count", "aromatic_rings"
]

X = df[features]

# We should ideally fit on the training data used for the NN. 
# Assuming standard 80/20 split with random_state=42 as seen in other scripts.
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

print("Fitting imputer and scaler on estimated training set...")
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)

print("Saving new artifacts...")
joblib.dump(imputer, "nn_imputer.pkl")
joblib.dump(scaler, "nn_scaler.pkl")

print("Done. Artifacts refreshed.")
