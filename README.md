💳 Credit Card Fraud Detection Dashboard

An end-to-end machine learning project to detect fraudulent credit card transactions.
This dashboard provides EDA, feature engineering, model training, SMOTE balancing, evaluation metrics, and SHAP explainability — all inside an interactive Streamlit UI.

🔗 Live App: https://creditcard-fraud-detection-btdadmmkipajtn2aepbzkc.streamlit.app/

📦 Repository: https://github.com/manishadharmik7/creditcard-fraud-detection


🚀 Project Overview

Credit card fraud detection is challenging due to highly imbalanced data.
This project uses advanced ML techniques to:

📊 Explore and visualize data

⚙️ Perform feature engineering

⚖️ Handle imbalance with SMOTE

🤖 Train Random Forest & XGBoost

📈 Evaluate with PR-AUC, ROC-AUC, MCC

🔍 Explain predictions using SHAP

🖥️ Provide an intuitive Streamlit Dashboard

🗂 Dataset

Source: Kaggle Credit Card Fraud Detection

Samples: 284,807

Fraud cases: 492

Features: PCA features (V1–V28), Time, Amount

Engineered: Hour, Log_Amount

(Dataset attribution link may be added if needed.)

📁 Folder Structure
creditcard-fraud-detection/
│── app.py                         # Streamlit dashboard
│── README.md
│── requirements.txt
│── data/
│   └── creditcard.csv
│── models/
│   ├── fraud_rfc_model.pkl
│   └── fraud_xgb_model.pkl
│── src/
│   ├── data_preprocess.py
│   ├── evaluate.py
│   ├── explain.py
│   └── train.py
└── notebook/
    └── credit_card_fraud_detection_RESUME.ipynb

🔧 Installation
git clone https://github.com/manishadharmik7/creditcard-fraud-detection
cd creditcard-fraud-detection

pip install -r requirements.txt
streamlit run app.py

⭐ Key Features
📊 Data Preview

Dataset summary

Fraud vs. Non-fraud distribution

Amount & Time visualizations

🤖 Model Training

Random Forest

XGBoost

Automatic preprocessing pipeline

📈 Evaluation Metrics

Accuracy, Precision, Recall, F1

MCC

PR Curve & ROC Curve

Confusion Matrix

🧠 Explainability

SHAP global feature importance

SHAP summary plots

⚖ Imbalance Handling

Oversampling with SMOTE

📊 Model Performance

| Model        | PR-AUC | ROC-AUC |
| ------------ | ------ | ------- |
| XGBoost      | 0.99   | 0.995   |
| RandomForest | 0.98   | 0.993   |


🛠 Future Improvements

Real-time transaction prediction

Hyperparameter tuning (Optuna/GridSearch)

Add LightGBM model

Expand dashboard sections