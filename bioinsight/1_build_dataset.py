import sqlite3
import pandas as pd

DB_PATH = r"C:\Users\Deep Ghadiyali\Desktop\Aragen_Hackathon\chembl_36\chembl_36\chembl_36_sqlite\chembl_36.db"

conn = sqlite3.connect(DB_PATH)

query = """
SELECT
    a.activity_id,
    a.molregno,
    a.pchembl_value,
    t.chembl_id AS target_chembl_id,
    t.pref_name AS target_name,

    cp.full_mwt AS molecular_weight,
    cp.alogp,
    cp.hba,
    cp.hbd,
    cp.psa AS tpsa,
    cp.rtb AS rotatable_bonds,
    cp.heavy_atoms AS heavy_atom_count,
    cp.aromatic_rings

FROM activities a
JOIN assays ass ON a.assay_id = ass.assay_id
JOIN target_dictionary t ON ass.tid = t.tid
JOIN compound_properties cp ON a.molregno = cp.molregno

WHERE
    a.pchembl_value IS NOT NULL
    AND ass.assay_type = 'B'          -- binding assays (standard)
    AND t.organism = 'Homo sapiens'
"""

df = pd.read_sql(query, conn)
conn.close()

print("Raw rows:", len(df))

# Binary bioactivity label
df["active"] = (df["pchembl_value"] >= 6.0).astype(int)

# Drop rows with missing descriptors
feature_cols = [
    "molecular_weight", "alogp", "hba", "hbd",
    "tpsa", "rotatable_bonds",
    "heavy_atom_count", "aromatic_rings"
]

df = df.dropna(subset=feature_cols)

print("After cleaning:", len(df))
print(df["active"].value_counts())

df.to_csv("model_dataset.csv", index=False)
print("Saved model_dataset.csv")
