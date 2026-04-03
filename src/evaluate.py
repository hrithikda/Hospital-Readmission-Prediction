import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def compute_shap_values(model, X_test):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    return explainer, shap_values


def save_shap_summary_plot(shap_values, X_test, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_shap_bar_plot(shap_values, X_test, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def get_feature_importance_df(shap_values, feature_names):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
    return importance_df


def binary_performance(y_test, y_pred, threshold=1.0):
    y_test_bin = (y_test > threshold).astype(int)
    y_pred_bin = (y_pred > threshold).astype(int)
    tp = int(((y_pred_bin == 1) & (y_test_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_test_bin == 0)).sum())
    tn = int(((y_pred_bin == 0) & (y_test_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_test_bin == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)
    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4)
    }