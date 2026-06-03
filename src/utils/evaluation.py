import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

# Initialize module-level logger
logger = logging.getLogger(__name__)


# Label mapping for experience level ordinal encoding
EXPERIENCE_LABELS = {
    0: "Internship",
    1: "Entry level",
    2: "Associate",
    3: "Mid-Senior level",
    4: "Director",
    5: "Executive",
}


# =========================================================
# METRIC COMPUTATION
# =========================================================

def compute_metrics(y_true, y_pred, label_map=None):
    """
    Compute classification metrics for a single model.

    Because the target classes are heavily imbalanced (classes 0, 4, 5
    together account for ~8% of samples), both macro and weighted
    averages are reported. Macro treats every class equally regardless
    of size, while weighted accounts for class frequency.

    MAE is included because the target is ordinal (Internship < Entry <
    Associate < Mid-Senior < Director < Executive). Standard metrics
    like F1 treat all misclassifications equally, but predicting
    Executive (5) for an Internship (0) is a worse mistake than
    predicting Entry level (1). MAE captures this distance. Lower is
    better; a perfect classifier has MAE = 0.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        label_map (dict or None): Mapping from class int to readable name.
            Defaults to EXPERIENCE_LABELS if None.

    Returns:
        dict: Dictionary with overall and per-class metrics.
    """
    label_map = label_map or EXPERIENCE_LABELS
    labels = sorted(y_true.unique()) if hasattr(y_true, "unique") else sorted(set(y_true))
    target_names = [label_map.get(l, str(l)) for l in labels]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    logger.info(
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"MAE: {metrics['mae']:.4f} | "
        f"F1 macro: {metrics['f1_macro']:.4f} | "
        f"F1 weighted: {metrics['f1_weighted']:.4f}"
    )

    return metrics


# =========================================================
# CLASSIFICATION REPORT DISPLAY
# =========================================================

def print_classification_report(y_true, y_pred, label_map=None):
    """
    Print a formatted sklearn classification report.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        label_map (dict or None): Mapping from class int to readable name.
            Defaults to EXPERIENCE_LABELS if None.
    """
    label_map = label_map or EXPERIENCE_LABELS
    labels = sorted(y_true.unique()) if hasattr(y_true, "unique") else sorted(set(y_true))
    target_names = [label_map.get(l, str(l)) for l in labels]

    report_str = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )
    print(report_str)


# =========================================================
# CONFUSION MATRIX
# =========================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    title="Confusion Matrix",
    label_map=None,
    normalize=True,
    figsize=(8, 6),
    cmap="Blues",
    ax=None,
):
    """
    Plot a confusion matrix as a heatmap.

    Normalization is on by default (row-normalized, i.e. recall per
    class) so that minority classes are visible despite low counts.
    Raw counts are shown as annotations alongside percentages when
    normalized.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        title (str): Plot title.
        label_map (dict or None): Mapping from class int to readable name.
            Defaults to EXPERIENCE_LABELS if None.
        normalize (bool): If True, normalize by true class (rows sum to 1).
        figsize (tuple): Figure size if creating a new figure.
        cmap (str): Matplotlib colormap.
        ax (matplotlib.axes.Axes or None): Axes to plot on. Creates new
            figure if None.

    Returns:
        matplotlib.axes.Axes
    """
    label_map = label_map or EXPERIENCE_LABELS
    labels = sorted(y_true.unique()) if hasattr(y_true, "unique") else sorted(set(y_true))
    target_names = [label_map.get(l, str(l)) for l in labels]

    cm_raw = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm_raw.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        cm_display = cm_raw.astype(float) / row_sums

        # Annotation: percentage on top, raw count below
        annot = np.empty_like(cm_raw, dtype=object)
        for i in range(cm_raw.shape[0]):
            for j in range(cm_raw.shape[1]):
                annot[i, j] = f"{cm_display[i, j]:.1%}\n({cm_raw[i, j]})"
    else:
        cm_display = cm_raw
        annot = cm_raw

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_display,
        annot=annot,
        fmt="" if normalize else "d",
        cmap=cmap,
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        vmin=0,
        vmax=1 if normalize else None,
        linewidths=0.5,
        linecolor="white",
    )
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)

    return ax


# =========================================================
# FULL SINGLE-MODEL EVALUATION
# =========================================================

def evaluate_model(y_true, y_pred, model_name="Model", label_map=None, figsize=(8, 6)):
    """
    Run full evaluation for a single model: print classification report,
    plot confusion matrix, and return metrics dict.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        model_name (str): Display name for the model.
        label_map (dict or None): Mapping from class int to readable name.
            Defaults to EXPERIENCE_LABELS if None.
        figsize (tuple): Figure size for confusion matrix.

    Returns:
        dict: Metrics dictionary from compute_metrics.
    """
    label_map = label_map or EXPERIENCE_LABELS

    print(f"{'=' * 50}")
    print(f"  {model_name}")
    print(f"{'=' * 50}\n")

    metrics = compute_metrics(y_true, y_pred, label_map)

    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"MAE:             {metrics['mae']:.4f}")
    print(f"F1 (macro):      {metrics['f1_macro']:.4f}")
    print(f"F1 (weighted):   {metrics['f1_weighted']:.4f}")
    print()

    print_classification_report(y_true, y_pred, label_map)

    plot_confusion_matrix(
        y_true, y_pred,
        title=f"{model_name} - Confusion Matrix (normalized)",
        label_map=label_map,
        normalize=True,
        figsize=figsize,
    )
    plt.tight_layout()
    plt.show()

    return metrics


# =========================================================
# MODEL COMPARISON
# =========================================================

def compare_models(results, sort_by="f1_macro"):
    """
    Build a comparison DataFrame from multiple model results.

    Parameters:
        results (dict): Mapping of model_name -> metrics dict
            (as returned by compute_metrics or evaluate_model).
        sort_by (str): Column to sort by in descending order.

    Returns:
        pd.DataFrame: Comparison table sorted by the chosen metric.
    """
    summary_keys = [
        "accuracy",
        "mae",
        "f1_macro",
        "f1_weighted",
        "precision_macro",
        "recall_macro",
    ]
    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        for key in summary_keys:
            row[key] = metrics.get(key, np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values(sort_by, ascending=False)

    return df


def plot_model_comparison(comparison_df, figsize=(12, 5)):
    """
    Bar chart comparing models across key metrics.

    Parameters:
        comparison_df (pd.DataFrame): Output of compare_models.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.axes.Axes
    """
    plot_cols = ["accuracy", "f1_macro", "f1_weighted"]
    available = [c for c in plot_cols if c in comparison_df.columns]

    ax = comparison_df[available].plot(
        kind="bar",
        figsize=figsize,
        edgecolor="white",
        width=0.75,
    )

    ax.set_title("Model Comparison", fontsize=13, pad=12)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", rotation=0)

    # Value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    return ax