# src/explain.py
import shap
import matplotlib.pyplot as plt
import streamlit as st

def shap_summary(model, X_test, sample_size=1000):
    explainer = shap.TreeExplainer(model)
    X_sample = X_test.sample(min(sample_size, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_sample)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_sample, show=False)
    st.pyplot(fig)
    plt.clf()

def find_best_threshold(model, X_test, y_test, target_precision=0.9):
    from sklearn.metrics import precision_recall_curve
    y_scores = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    for i, p in enumerate(precision):
        if p >= target_precision:
            best_threshold = thresholds[i]
            st.write(f"Chosen threshold = {best_threshold:.4f}, Recall = {recall[i]:.4f}")
            return best_threshold
    return 0.5
