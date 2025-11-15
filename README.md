# 💳 Credit Card Fraud Detection Dashboard

An end-to-end machine learning project to detect fraudulent credit card transactions.  
This dashboard provides **EDA, feature engineering, model training, evaluation, SMOTE, and SHAP explainability** in a clean, interactive Streamlit UI.

🔗 **Live App:** https://creditcard-fraud-detection-btdadmmkipajtn2aepbzkc.streamlit.app/  
📦 **Repository:** https://github.com/manishadharmik7/creditcard-fraud-detection  

---

## 🚀 Project Overview

Credit card fraud is highly imbalanced and difficult to detect.  
This project uses ML techniques to:

- 📊 Explore & visualize data  
- ⚙️ Perform feature engineering  
- ⚖ Handle class imbalance with **SMOTE**  
- 🤖 Train **Random Forest** and **XGBoost**  
- 📈 Evaluate with PR-AUC, ROC-AUC, MCC  
- 🔍 Explain predictions using **SHAP**  
- 🖥 Provide an interactive **Streamlit Dashboard**  

---

## 🗂 Dataset

- Source: Kaggle Credit Card Fraud Detection  
- Samples: **284,807**  
- Fraud cases: **492** (highly imbalanced)  
- Features: **30 PCA features (V1–V28), Time, Amount**  
- Engineered features: **Hour**, **Log_Amount**  

Dataset link: *(add attribution or remove)*

---

## 📁 Folder Structure
creditcard-fraud-detection/
│
├── app.py # Streamlit dashboard
├── README.md
├── requirements.txt
│
├── data/
│ └── creditcard.csv
│
├── models/
│ ├── fraud_rfc_model.pkl
│ └── fraud_xgb_model.pkl
│
├── src/
│ ├── data_preprocess.py
│ ├── evaluate.py
│ ├── explain.py
│ └── train.py
│
└── notebook/
└── credit_card_fraud_detection_RESUME.ipynb


---

## 🔧 Installation

```bash
git clone https://github.com/manishadharmik7/creditcard-fraud-detection
cd creditcard-fraud-detection

pip install -r requirements.txt
streamlit run app.py

⭐ Key Features
📊 Data Preview

Dataset info

Fraud vs Non-fraud visualization

Distribution plots

🤖 Model Training

Random Forest

XGBoost

Automatic preprocessing pipeline

📈 Evaluation Metrics

Accuracy, Precision, Recall, F1

MCC

PR Curve

ROC Curve

Confusion Matrix

🧠 Explainability

SHAP feature importance

Interactive summary plots

⚖ Imbalance Handling

Oversampling using SMOTE

📊 Model Performance
Model	PR-AUC	ROC-AUC
XGBoost	0.99	0.995
RandomForest	0.98	0.993
🛠 Future Improvements

Deploy on Streamlit Cloud (done ✔)

Add real-time transaction testing

Add hyperparameter tuning (Optuna/GridSearch)

Add LightGBM model



