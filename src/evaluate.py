# src/evaluate.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, average_precision_score,
    classification_report, precision_recall_curve, roc_curve, auc
)
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "Matthews Corr Coef": matthews_corrcoef(y_test, y_pred),
        "PR AUC": average_precision_score(y_test, y_pred)
    }

    st.write("📊 Model Evaluation Metrics", metrics)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, label=f'PR AUC={auc(recall, precision):.4f}')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision'); ax.legend()
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC AUC={auc(fpr, tpr):.4f}')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate'); ax.legend()
    st.pyplot(fig)


def detect_outliers(X, y, sample_size=30000, random_state=42):
    X_sample = X.sample(sample_size, random_state=random_state)
    y_sample = y.loc[X_sample.index]
    outlier_fraction = len(y_sample[y_sample == 1]) / float(len(y_sample[y_sample == 0]))

    classifiers = {
        'Isolation Forest': IsolationForest(max_samples=len(X_sample), contamination=outlier_fraction,
                                            random_state=random_state),
        'Local Outlier Factor': LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
    }

    results = {}
    for name, clf in classifiers.items():
        if name == 'Local Outlier Factor':
            y_pred = clf.fit_predict(X_sample)
        else:
            clf.fit(X_sample)
            y_pred = clf.predict(X_sample)
        y_pred = np.where(y_pred == 1, 0, 1)
        results[name] = classification_report(y_sample, y_pred, output_dict=True)

    return results
