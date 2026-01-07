import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("model_dataset.csv", nrows=300000)

print("Shape:", df.shape)
print(df.describe())

# -------------------------------
# Class distribution
# -------------------------------
class_counts = df["active"].value_counts(normalize=True)
print("\nClass distribution:\n", class_counts)

plt.figure()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Active vs Inactive Distribution")
plt.ylabel("Proportion")
plt.xlabel("Active Label")
plt.savefig("eda_class_distribution.png")
plt.close()

# -------------------------------
# pChEMBL distribution
# -------------------------------
plt.figure()
sns.histplot(df["pchembl_value"], bins=50)
plt.title("pChEMBL Value Distribution")
plt.xlabel("pChEMBL Value")
plt.savefig("eda_pchembl_distribution.png")
plt.close()

# -------------------------------
# Feature correlation heatmap
# -------------------------------
feature_cols = [
    "molecular_weight", "alogp", "hba", "hbd",
    "tpsa", "rotatable_bonds",
    "heavy_atom_count", "aromatic_rings"
]

plt.figure(figsize=(10, 8))
sns.heatmap(df[feature_cols].corr(), cmap="coolwarm", center=0)
plt.title("Molecular Descriptor Correlation Heatmap")
plt.savefig("eda_correlation_heatmap.png")
plt.close()

print("EDA plots saved successfully.")
