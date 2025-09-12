# src/train.py
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib
from imblearn.over_sampling import SMOTE
import os
from data_preprocess import load_and_preprocess_data

def oversample(X_train, y_train, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("Before SMOTE:", y_train.value_counts().to_dict())
    print("After SMOTE:", y_res.value_counts().to_dict())
    return X_res, y_res

def train_random_forest(X_train, y_train):
    rfc = RandomForestClassifier(
        n_estimators=50,
        max_depth=8,
        min_samples_split=20,
        n_jobs=-1,
        random_state=42
    )
    rfc.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(rfc, "models/fraud_rfc_model.pkl")
    print("✅ Random Forest model saved")
    return rfc

def train_xgboost(X_train, y_train):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='aucpr',
        random_state=42,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/fraud_xgb_model.pkl")
    print("✅ XGBoost model saved")
    return model

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, summary = load_and_preprocess_data("data/creditcard.csv")
    X_train_res, y_train_res = oversample(X_train, y_train)
    rfc_model = train_random_forest(X_train_res, y_train_res)
    xgb_model = train_xgboost(X_train_res, y_train_res)
