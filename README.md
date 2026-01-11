# BioInsight Lite – Data Explorer & Bioactivity Predictor

## 1. What is Bioactivity?

In drug discovery, **bioactivity** refers to the ability of a chemical compound to produce a biological effect by interacting with a biological target such as a protein, enzyme, or receptor.  
In experimental databases like ChEMBL, bioactivity is quantified using assay measurements such as IC50, Ki, Kd, or EC50, which indicate how strongly a compound binds to or modulates a target.

To standardize these measurements, ChEMBL provides the **pChEMBL value**, defined as:

pChEMBL = −log10(activity value in molar units)

Higher pChEMBL values indicate **stronger biological activity**.  
In this project, bioactivity prediction is framed as a **binary classification problem**:
- **Active (1)**: pChEMBL ≥ 6.0  
- **Inactive (0)**: pChEMBL < 6.0  

This threshold approximately corresponds to micromolar or better potency and is commonly used in cheminformatics studies.

---

## 2. Project Objective

**BioInsight Lite** is a mini end-to-end machine learning application designed to:
- Explore large-scale chemical and biological data from ChEMBL
- Train predictive models for compound bioactivity
- Explain model predictions using modern explainability techniques
- Provide an interactive interface for exploration and prediction

The project was developed under hackathon constraints with a focus on **correct scientific assumptions, interpretability, and usability**.

---

## 3. Dataset Description

### Data Source
- **ChEMBL version 36**
- Publicly available bioactivity database maintained by EMBL-EBI

### Database Format
- **SQLite database (`chembl_36.db`)**
- Selected over HDF5 because it provides full relational access to bioactivity, assay, and molecular descriptor tables

### Primary and Supporting Tables Used
- `activities` – bioactivity measurements
- `assays` – assay metadata
- `target_dictionary` – biological target information
- `compound_properties` – computed molecular descriptors

### Filtering Criteria
- Only **binding assays** (`assay_type = 'B'`)
- Only **human targets** (`organism = 'Homo sapiens'`)
- Only rows with valid `pchembl_value`

After cleaning, the final dataset contained approximately **1.76 million records**.

---

## 4. Feature Engineering

### Input Features (Independent Variables)
The following physicochemically meaningful descriptors were used:

- Molecular weight
- ALogP (lipophilicity)
- Hydrogen bond acceptors (HBA)
- Hydrogen bond donors (HBD)
- Topological polar surface area (TPSA)
- Number of rotatable bonds
- Heavy atom count
- Number of aromatic rings

These descriptors are widely used in QSAR and structure–activity relationship (SAR) modeling.

### Target Variable (Dependent Variable)
- `active` (binary)
  - 1 if pChEMBL ≥ 6.0
  - 0 otherwise

---

## 5. Exploratory Data Analysis (EDA)

EDA was performed to understand data quality and distributions:

- Summary statistics for all features
- Class distribution analysis (active vs inactive)
- pChEMBL value distribution
- Correlation heatmap of molecular descriptors

### Key Observations
- The dataset is moderately imbalanced (~71% active)
- Strong correlations exist between molecular weight and heavy atom count
- Lipophilicity and molecular size are dominant contributors to bioactivity
- Excessive polarity and flexibility are generally associated with reduced activity

EDA visualizations were saved and later integrated into the application workflow.

---

## 6. Model Development

Two models were trained as required by the hackathon rules.

### 6.1 Baseline Model – Logistic Regression
- Scikit-learn implementation
- Standardized inputs
- Class-balanced loss
- Provides a strong, interpretable baseline

### 6.2 Advanced Model – Neural Network
- TensorFlow / Keras implementation
- Fully connected feed-forward architecture
- ReLU activations and sigmoid output
- Trained on Google Colab for faster computation
- Designed specifically for tabular chemical data (no convolution layers)

The neural network captures **non-linear interactions** between molecular descriptors while remaining simple and stable.

---

## 7. Model Evaluation

Both models were evaluated using:

- **Accuracy**
- **ROC-AUC**
- **F1-score**

Evaluation was performed on a stratified test set to preserve class balance.

### For logistic regration
![alt text](image-1.png)
### For Neural Network
![alt text](image.png)
![alt text](image-2.png)
### Summary
- Logistic Regression provided strong baseline performance with high interpretability
- The Neural Network achieved improved ROC-AUC by modeling non-linear effects
- Both models produced consistent and chemically meaningful results
---

## 8. Explainability (SHAP)

To ensure transparency and trust in predictions, **SHAP (SHapley Additive exPlanations)** was used.

### Logistic Regression Explainability
- SHAP values closely matched model coefficients
- Heavy atom count and lipophilicity positively influenced bioactivity
- High TPSA and excessive rotatable bonds negatively influenced activity

### Neural Network Explainability
- SHAP KernelExplainer was used
- Feature contributions were smoother and distributed
- Confirmed that the neural network learned chemically sensible patterns
- No evidence of overfitting or spurious correlations

Both global (feature importance) and local (per-sample) explanations were generated.

---

## 9. Deployment – Streamlit Application

A Streamlit web application was built to demonstrate the complete pipeline.

### Application Features
- Dataset exploration and statistics
- Single-compound bioactivity prediction
- Comparison between Logistic Regression and Neural Network outputs
- Visualization of SHAP explainability plots

### Technologies Used
- Streamlit
- Pandas, NumPy
- Scikit-learn
- TensorFlow
- SHAP

The app is deployable on **Streamlit Cloud** or **Hugging Face Spaces** without modification.

---

## 10. Assumptions and Limitations

### Assumptions
- pChEMBL threshold of 6.0 is an appropriate activity cutoff
- Molecular descriptors sufficiently capture bioactivity trends
- Binding assays provide reliable activity signals

### Limitations
- No target-specific modeling (global model across targets)
- Fingerprint-based similarity modeling was not included
- Class imbalance was handled but not fully eliminated

---

## 11. Conclusion

BioInsight Lite demonstrates a complete, explainable, and deployable bioactivity prediction pipeline using real-world chemical biology data.  
The project emphasizes **scientific correctness, interpretability, and usability**, making it suitable for both academic evaluation and practical deployment.

---

## 12. Tools & Resources

- ChEMBL Database (v36)
- Python, Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- SHAP
- Streamlit
- Google Colab

---

## 13. Future Improvements

- Target-specific models
- Fingerprint-based similarity search
- Natural language query (NLQ) for compound search
- Hyperparameter optimization
- External validation datasets

---
