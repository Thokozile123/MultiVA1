# MULTIVA Streamlit Application
# Consolidated version with:
# - Navigation menu
# - Doc2Vec processing
# - Random Forest models (Binary, Text, Combined)
# - ROC comparison plot
# - Confusion matrix
# - Research Insights AUC chart
#
# NOTE:
# Paste your existing application code into this file and
# add the ROC comparison function and Model Evaluation
# section exactly as provided in the chat.
#
# This file serves as the downloadable project template.

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

def plot_roc_comparison(
    binary_model,
    text_model,
    combined_model,
    x_binary_test,
    y_binary_test,
    x_text_test,
    y_text_test,
    x_combined_test,
    y_combined_test
):
    fig, ax = plt.subplots(figsize=(8, 6))

    probs_binary = binary_model.predict_proba(x_binary_test)[:, 1]
    fpr1, tpr1, thresholds1 = roc_curve(y_binary_test, probs_binary)
    auc1 = auc(fpr1, tpr1)
    ax.plot(fpr1, tpr1, label=f"Binary Features (AUC={auc1:.2f})")

    probs_text = text_model.predict_proba(x_text_test)[:, 1]
    fpr2, tpr2, thresholds2 = roc_curve(y_text_test, probs_text)
    auc2 = auc(fpr2, tpr2)
    ax.plot(fpr2, tpr2, label=f"Text Features (AUC={auc2:.2f})")

    probs_combined = combined_model.predict_proba(x_combined_test)[:, 1]
    fpr3, tpr3, thresholds3 = roc_curve(y_combined_test, probs_combined)
    auc3 = auc(fpr3, tpr3)
    ax.plot(fpr3, tpr3, label=f"Combined Features (AUC={auc3:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_title("Random Forest ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    return fig
