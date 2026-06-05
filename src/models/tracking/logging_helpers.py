import logging
import os
import pickle
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

logger = logging.getLogger(__name__)


def _log_tags(tags, run_name, model_type="lightgbm"):
    """Log user-supplied tags plus a few automatic ones."""
    auto_tags = {
        "model_type": model_type,
        "task": "multiclass_classification",
        "run_name": run_name,
    }
    if tags:
        auto_tags.update(tags)
    mlflow.set_tags(auto_tags)


def _log_dataset(X_train, y_train, name, source):
    """Log the training dataset via mlflow.log_input (populates the Datasets column)."""
    df = X_train.copy()
    df["_target"] = y_train.values
    dataset = mlflow.data.from_pandas(df, source=source, name=name, targets="_target")
    mlflow.log_input(dataset, context="training")


def _log_params(params, val_size, X_train, model_type="lightgbm"):
    """Log hyperparameters plus dataset metadata."""
    mlflow.log_param("model_type", model_type)
    mlflow.log_param("val_size", val_size)
    mlflow.log_param("n_features", X_train.shape[1])
    mlflow.log_param("n_train_samples", X_train.shape[0])

    for key, value in params.items():
        mlflow.log_param(key, value)


def _log_metrics(metrics):
    """
    Log scalar metrics. Skips nested dicts like classification_report
    but flattens per-class F1 scores into individual metric entries.
    """
    for key, value in metrics.items():
        if key == "classification_report":
            _log_per_class_metrics(value)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            mlflow.log_metric(key, float(value))


def _log_per_class_metrics(report_dict):
    """
    Extract per-class precision / recall / f1 from the sklearn
    classification_report dict and log each as a separate MLflow metric.
    """
    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    for class_name, class_metrics in report_dict.items():
        if class_name in skip_keys:
            continue
        if not isinstance(class_metrics, dict):
            continue
        safe_name = class_name.replace(" ", "_").lower()
        for metric_name in ("precision", "recall", "f1-score"):
            value = class_metrics.get(metric_name)
            if value is not None:
                mlflow.log_metric(
                    f"{safe_name}_{metric_name.replace('-', '_')}",
                    float(value),
                )


def _log_confusion_matrix(y_true, y_pred):
    """Save a confusion-matrix heatmap as a PNG artifact."""
    from sklearn.metrics import confusion_matrix as sk_cm
    from ...utils.evaluation import EXPERIENCE_LABELS

    labels = sorted(set(y_true) | set(y_pred))
    target_names = [EXPERIENCE_LABELS.get(l, str(l)) for l in labels]

    cm = sk_cm(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        ax=ax,
        vmin=0,
        vmax=1,
    )
    ax.set_title("Confusion Matrix (normalized)")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        mlflow.log_artifact(path, artifact_path="plots")
    plt.close(fig)


def _log_feature_importance(importance):
    """Save a feature-importance bar chart and CSV as artifacts."""
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "feature_importance.csv")
        importance.to_csv(csv_path, header=True)
        mlflow.log_artifact(csv_path, artifact_path="feature_importance")

    fig, ax = plt.subplots(figsize=(10, 6))
    importance.sort_values().plot.barh(ax=ax)
    ax.set_title(f"Top {len(importance)} Features (split importance)")
    ax.set_xlabel("Importance")
    plt.tight_layout()

    with tempfile.TemporaryDirectory() as tmp:
        png_path = os.path.join(tmp, "feature_importance.png")
        fig.savefig(png_path, dpi=150)
        mlflow.log_artifact(png_path, artifact_path="plots")
    plt.close(fig)


def _log_model(model, register_name, model_type="lightgbm"):
    """
    Log the trained model using the appropriate MLflow flavour.

    The model_type determines which mlflow.<flavour>.log_model is called,
    so the correct serialisation format is used and the model shows up
    with the right type in the MLflow Models UI.
    """
    log_kwargs = {"artifact_path": register_name or "model"}
    if register_name:
        log_kwargs["registered_model_name"] = register_name

    if model_type == "lightgbm":
        import mlflow.lightgbm
        mlflow.lightgbm.log_model(lgb_model=model, **log_kwargs)

    elif model_type == "xgboost":
        import mlflow.xgboost
        mlflow.xgboost.log_model(xgb_model=model, **log_kwargs)

    elif model_type == "catboost":
        import mlflow.catboost
        mlflow.catboost.log_model(cb_model=model, **log_kwargs)

    elif model_type == "sklearn":
        import mlflow.sklearn
        mlflow.sklearn.log_model(sk_model=model, **log_kwargs)

    elif model_type == "tabnet":
        _log_tabnet_model(model, log_kwargs)

    else:
        try:
            import mlflow.sklearn
            mlflow.sklearn.log_model(sk_model=model, **log_kwargs)
        except Exception:
            import mlflow.pyfunc
            logger.warning(
                f"Unknown model_type '{model_type}', logging as pyfunc."
            )
            mlflow.pyfunc.log_model(python_model=model, **log_kwargs)

    label = f" and registered as '{register_name}'" if register_name else ""
    logger.info(f"Model logged (type={model_type}){label}")


def _log_tabnet_model(model_bundle, log_kwargs):
    """
    Log a TabNet model bundle.

    If the bundle is a dict containing a 'model' key (as returned by
    train_tabnet), the actual TabNet classifier is extracted and saved.
    Otherwise the object is saved directly via sklearn if possible.
    """
    import mlflow.sklearn

    actual_model = model_bundle
    if isinstance(model_bundle, dict) and "model" in model_bundle:
        actual_model = model_bundle["model"]

    try:
        mlflow.sklearn.log_model(sk_model=actual_model, **log_kwargs)
    except Exception:
        with tempfile.TemporaryDirectory() as tmp:
            pkl_path = os.path.join(tmp, "tabnet_bundle.pkl")
            with open(pkl_path, "wb") as f:
                pickle.dump(model_bundle, f)
            mlflow.log_artifact(pkl_path, artifact_path="tabnet")
        logger.warning("TabNet logged as pickle artifact (sklearn flavour failed).")
