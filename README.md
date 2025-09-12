# 💳 Credit Card Fraud Detection Dashboard

An end-to-end credit card fraud detection project using machine learning and an interactive dashboard built with Streamlit. The project handles imbalanced data, performs feature engineering, trains multiple models, evaluates performance, and provides model explainability with SHAP.

---

## **Project Overview**

Credit card fraud is a critical issue in the financial industry. This project aims to detect fraudulent transactions using machine learning techniques. It includes:

- Data preprocessing and feature engineering
- Handling class imbalance with SMOTE
- Model training: Random Forest and XGBoost
- Model evaluation: Confusion Matrix, PR Curve, ROC Curve
- Feature importance analysis using SHAP
- Interactive dashboard to explore dataset and model metrics

---

## **Dataset**

- Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- Features: 30 numeric features (V1–V28, Time, Amount) and engineered features (Hour, Log_Amount)
- Target: `Class` (0 = Legitimate, 1 = Fraud)
- Total samples: 284,807  
- Fraud cases: 492 (highly imbalanced)

---

## **Folder Structure**

creditcard-fraud-detection/
├── app.py                          # Streamlit dashboard
├── README.md
├── requirements.txt                # Python dependencies
├── data/
│   └── creditcard.csv              # Dataset
├── models/
│   ├── fraud_rfc_model.pkl
│   └── fraud_xgb_model.pkl
├── src/
│   ├── data_preprocess.py
│   ├── evaluate.py
│   ├── explain.py
│   └── train.py
└── notebook/
    └── credit_card_fraud_detection_RESUME.ipynb  # Optional: notebook for analysis/resume


---

## **Setup & Installation**

1. Clone the repository:

```bash
git clone <repo_url>
cd creditcard-fraud-detection

pip install -r requirements.txt
streamlit run app.py

Key Features

Data Preview: View first few rows, dataset shape, fraud vs valid cases

Evaluation: Accuracy, Precision, Recall, F1 Score, Matthews Correlation Coefficient, PR-AUC, Confusion Matrix, PR & ROC Curves

Explainability: SHAP feature importance for top contributing features

Feature Engineering: Hour extraction from Time, Log transformation of Amount

Handling Imbalance: Oversampling with SMOTE

Model Performance
Model	PR-AUC	ROC-AUC
XGBoost	0.99	0.995
RandomForest	0.98	0.993


Future Improvements :
Deploy the dashboard online using Streamlit Cloud or Heroku

Include real-time transaction testing

Add hyperparameter tuning and model comparison for multiple classifiers

.txt](requirements.txt)