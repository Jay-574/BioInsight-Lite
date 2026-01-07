import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def train_model(n_samples):
    print(f"\n--- Training with {n_samples} samples ---")
    # -----------------------------
    # Load & sample dataset
    # -----------------------------
    df = pd.read_csv("model_dataset.csv")

    if n_samples and len(df) > n_samples:
        df, _ = train_test_split(
            df,
            train_size=n_samples,
            stratify=df["active"],
            random_state=42
        )

    features = [
        "molecular_weight", "alogp", "hba", "hbd",
        "tpsa", "rotatable_bonds",
        "heavy_atom_count", "aromatic_rings"
    ]

    X = df[features]
    y = df["active"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print(f"Logistic Regression Results ({n_samples} samples)")
    print("Accuracy :", accuracy_score(y_test, preds))
    print("ROC-AUC  :", roc_auc_score(y_test, probs))
    print("F1-score :", f1_score(y_test, preds))

    # -----------------------------
    # Save model
    # -----------------------------
    filename = f"logistic_model_{n_samples}.pkl"
    joblib.dump(model, filename)
    print(f"Saved model to {filename}")

if __name__ == "__main__":
    train_model(300000)
    train_model(500000)
