# src/data_preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath: str):
    """
    Load dataset, feature engineering, split train/val/test.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, summary_dict
    """

    # 1. Load dataset
    df = pd.read_csv(filepath)

    # Fraud vs Valid cases
    fraud = df[df['Class'] == 1]
    valid = df[df['Class'] == 0]
    outlier_fraction = len(fraud) / float(len(valid))

    # 2. Feature Engineering
    df['Hour'] = (df['Time'] // 3600) % 24
    df['Log_Amount'] = np.log1p(df['Amount'])

    # 3. Train/Val/Test split
    X = df.drop('Class', axis=1)
    y = df['Class']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    # 4. Summary dict for Streamlit
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "fraud_cases": len(fraud),
        "valid_cases": len(valid),
        "outlier_fraction": round(outlier_fraction, 6),
        "train_size": X_train.shape,
        "val_size": X_val.shape,
        "test_size": X_test.shape,
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, summary


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, summary = load_and_preprocess_data(
        "data/creditcard.csv"
    )
    print(summary)
