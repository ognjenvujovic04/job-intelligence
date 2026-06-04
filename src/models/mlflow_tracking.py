"""
MLflow Experiment Tracking
===========================
Wraps the training and evaluation pipeline with MLflow logging.

Designed to be imported from a notebook or used standalone:
    from tracking.mlflow_tracking import run_experiment, compare_runs

Handles:
    - Experiment and run lifecycle management
    - Hyperparameter logging
    - Metric logging (overall + per-class)
    - Model artifact logging and optional registry
    - Confusion matrix artifact saving
    - Feature importance artifact saving

Directory layout assumed (adjust paths in defaults if needed):
    src/
        models/train_lightgbm.py
        evaluation/evaluation.py
        tracking/mlflow_tracking.py      <-- this file
    data/
        processed/feature_matrix_train.csv
        processed/feature_matrix_test.csv
"""

import logging
import os
import tempfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for artifact plots
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT_NAME = "experience-level-classification"
DEFAULT_TARGET = "experience_level_ord"


def configure_mlflow(
    tracking_uri=None,
    experiment_name=None,
):
    """
    Set the MLflow tracking URI and active experiment.

    If the experiment does not exist yet it will be created automatically.

    Parameters:
        tracking_uri (str or None): File-system path or remote URI.
            Defaults to ``DEFAULT_TRACKING_URI``.
        experiment_name (str or None): Human-readable experiment name.
            Defaults to ``DEFAULT_EXPERIMENT_NAME``.

    Returns:
        str: The experiment ID for the active experiment.
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    name = experiment_name or DEFAULT_EXPERIMENT_NAME

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)

    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id

    logger.info(
        f"MLflow configured | uri={uri} | experiment='{name}' "
        f"(id={experiment_id})"
    )
    return experiment_id


# =========================================================
# CORE: RUN AN EXPERIMENT
# =========================================================

def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    train_fn,
    eval_fn,
    feature_importance_fn=None,
    params=None,
    run_name=None,
    tags=None,
    register_model_name=None,
    val_size=0.15,
    top_n_features=15,
):
    """
    Execute a single tracked training run inside MLflow.

    This is the main entry point. It:
        1. Starts an MLflow run.
        2. Logs hyperparameters.
        3. Trains the model via *train_fn*.
        4. Generates predictions on the test set.
        5. Computes and logs evaluation metrics via *eval_fn*.
        6. Saves the confusion matrix and feature importance as artifacts.
        7. Logs the model (and optionally registers it).

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test target.
        train_fn (callable): ``train_fn(X_train, y_train, val_size, params)``
            returning a fitted model.
        eval_fn (callable): ``eval_fn(y_true, y_pred)`` returning a dict
            of scalar metrics (and optionally a ``classification_report``
            nested dict).
        feature_importance_fn (callable or None):
            ``feature_importance_fn(model, feature_names, top_n)``
            returning a pd.Series.
        params (dict or None): Hyperparameter overrides forwarded to
            *train_fn*.
        run_name (str or None): Descriptive name shown in the MLflow UI.
            Auto-generated from a timestamp when omitted.
        tags (dict or None): Arbitrary key-value metadata attached to the
            run (e.g. ``{"author": "ov", "stage": "baseline"}``).
        register_model_name (str or None): If provided, the logged model
            is also registered under this name in the MLflow Model Registry.
        val_size (float): Validation fraction passed through to *train_fn*.
        top_n_features (int): Number of top features to log.

    Returns:
        dict: A summary with keys ``run_id``, ``metrics``, ``model``,
            and ``params``.
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"lgbm_{timestamp}"

    merged_params = dict(params or {})

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started | name='{run_name}' | id={run_id}")

        # -- tags ----------------------------------------------------------
        _log_tags(tags, run_name)

        # -- hyperparameters -----------------------------------------------
        _log_params(merged_params, val_size, X_train)

        # -- training ------------------------------------------------------
        model = train_fn(X_train, y_train, val_size=val_size, params=merged_params)

        # log early-stopping iteration if available
        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            mlflow.log_metric("best_iteration", best_iter)

        # -- evaluation ----------------------------------------------------
        y_pred = model.predict(X_test)
        metrics = eval_fn(y_test, y_pred)
        _log_metrics(metrics)

        # -- artifacts: confusion matrix -----------------------------------
        _log_confusion_matrix(y_test, y_pred)

        # -- artifacts: feature importance ---------------------------------
        if feature_importance_fn is not None:
            importance = feature_importance_fn(
                model, list(X_train.columns), top_n_features
            )
            _log_feature_importance(importance)

        # -- model ---------------------------------------------------------
        _log_model(model, register_model_name)

        logger.info(f"MLflow run finished | id={run_id}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "model": model,
        "params": merged_params,
    }


# =========================================================
# COMPARING RUNS
# =========================================================

def compare_runs(
    experiment_name=None,
    metric_keys=None,
    top_n=10,
):
    """
    Pull recent runs from an experiment and return a comparison DataFrame.

    Parameters:
        experiment_name (str or None): Experiment to query.
            Defaults to ``DEFAULT_EXPERIMENT_NAME``.
        metric_keys (list[str] or None): Metric columns to include.
            Defaults to a standard set of classification metrics.
        top_n (int): Maximum number of runs to return (most recent first).

    Returns:
        pd.DataFrame: One row per run, sorted by start time descending.
    """
    name = experiment_name or DEFAULT_EXPERIMENT_NAME
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        logger.warning(f"Experiment '{name}' not found.")
        return pd.DataFrame()

    if metric_keys is None:
        metric_keys = [
            "accuracy",
            "mae",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
        ]

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=top_n,
    )

    if runs.empty:
        logger.info("No runs found.")
        return runs

    # Build a tidy subset
    keep_cols = ["run_id", "start_time", "tags.mlflow.runName"]
    metric_cols = [f"metrics.{k}" for k in metric_keys]
    available = [c for c in keep_cols + metric_cols if c in runs.columns]

    result = runs[available].copy()

    # Rename for readability
    rename_map = {"tags.mlflow.runName": "run_name"}
    rename_map.update({f"metrics.{k}": k for k in metric_keys})
    result.rename(columns=rename_map, inplace=True)

    return result


def get_best_run(
    experiment_name=None,
    metric="f1_macro",
    higher_is_better=True,
):
    """
    Return the run_id and metric value for the best run so far.

    Parameters:
        experiment_name (str or None): Experiment to query.
        metric (str): Metric to rank by.
        higher_is_better (bool): Sort direction.

    Returns:
        dict or None: ``{"run_id": ..., "metric_value": ...}`` for the
            best run, or None if no runs exist.
    """
    df = compare_runs(experiment_name=experiment_name, metric_keys=[metric])
    if df.empty:
        return None

    if higher_is_better:
        best = df.loc[df[metric].idxmax()]
    else:
        best = df.loc[df[metric].idxmin()]

    return {
        "run_id": best["run_id"],
        "run_name": best.get("run_name", ""),
        "metric_value": best[metric],
    }


# =========================================================
# INTERNAL HELPERS
# =========================================================

def _log_tags(tags, run_name):
    """Log user-supplied tags plus a few automatic ones."""
    auto_tags = {
        "model_type": "lightgbm",
        "task": "multiclass_classification",
        "run_name": run_name,
    }
    if tags:
        auto_tags.update(tags)
    mlflow.set_tags(auto_tags)


def _log_params(params, val_size, X_train):
    """Log hyperparameters plus dataset metadata."""
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

    # import here to allow headless usage
    from src.utils.evaluation import EXPERIENCE_LABELS

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
    # CSV
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "feature_importance.csv")
        importance.to_csv(csv_path, header=True)
        mlflow.log_artifact(csv_path, artifact_path="feature_importance")

    # bar chart
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


def _log_model(model, register_name):
    """Log the LightGBM model, optionally registering it."""
    log_kwargs = {
        "lgb_model": model,
        "artifact_path": "model",
    }
    if register_name:
        log_kwargs["registered_model_name"] = register_name

    mlflow.lightgbm.log_model(**log_kwargs)
    logger.info(
        f"Model logged"
        + (f" and registered as '{register_name}'" if register_name else "")
    )


# =========================================================
# CONVENIENCE: FULL PIPELINE IN ONE CALL
# =========================================================

def run_default_experiment(
    train_path=None,
    test_path=None,
    target=DEFAULT_TARGET,
    params=None,
    run_name=None,
    tags=None,
    tracking_uri=None,
    experiment_name=None,
    register_model_name=None,
    val_size=0.15,
):
    """
    Load data, configure MLflow, train, evaluate, and log -- all in one call.

    This is handy for quick experiments from the command line or a notebook
    when you just want to change ``params`` and see the results show up
    in the MLflow UI.

    Parameters:
        train_path (str): Path to training CSV.
        test_path (str): Path to test CSV.
        target (str): Name of the target column.
        params (dict or None): Hyperparameter overrides.
        run_name (str or None): MLflow run name.
        tags (dict or None): Extra metadata tags.
        tracking_uri (str or None): MLflow tracking URI.
        experiment_name (str or None): Experiment name.
        register_model_name (str or None): Model-registry name.
        val_size (float): Validation split fraction.

    Returns:
        dict: Summary from ``run_experiment``.
    """
    # lazy imports so the module loads fast even without these on the path
    from src.models.train_lightgbm import train_lightgbm, get_feature_importance
    from src.utils.evaluation import compute_metrics

    # Resolve default paths relative to the repository root (two levels up
    # from this file: src/models -> ../.. -> repository root). If callers
    # passed explicit paths, use those instead.
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if train_path is None:
        train_path = os.path.join(base_dir, "data", "processed", "feature_matrix_train.csv")
    if test_path is None:
        test_path = os.path.join(base_dir, "data", "processed", "feature_matrix_test.csv")

    # -- data --
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train = df_train.dropna(subset=[target])
    df_test = df_test.dropna(subset=[target])

    y_train = df_train[target].astype(int)
    y_test = df_test[target].astype(int)
    X_train = df_train.drop(columns=[target])
    X_test = df_test.drop(columns=[target])

    # -- MLflow setup --
    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    # -- run --
    return run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_fn=train_lightgbm,
        eval_fn=compute_metrics,
        feature_importance_fn=get_feature_importance,
        params=params,
        run_name=run_name,
        tags=tags,
        register_model_name=register_model_name,
        val_size=val_size,
    )


# =========================================================
# STANDALONE EXECUTION
# =========================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Run a tracked LightGBM experiment."
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Descriptive run name for the MLflow UI.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override learning_rate hyperparameter.",
    )
    parser.add_argument(
        "--num-leaves", type=int, default=None,
        help="Override num_leaves hyperparameter.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=None,
        help="Override n_estimators hyperparameter.",
    )
    parser.add_argument(
        "--register", type=str, default=None,
        help="Register the model under this name in the Model Registry.",
    )
    args = parser.parse_args()

    # Build param overrides from CLI flags
    cli_params = {}
    if args.learning_rate is not None:
        cli_params["learning_rate"] = args.learning_rate
    if args.num_leaves is not None:
        cli_params["num_leaves"] = args.num_leaves
    if args.n_estimators is not None:
        cli_params["n_estimators"] = args.n_estimators

    result = run_default_experiment(
        params=cli_params or None,
        run_name=args.run_name,
        register_model_name=args.register,
        tags={"source": "cli"},
    )

    print(f"\nRun ID: {result['run_id']}")
    print(f"Accuracy:    {result['metrics']['accuracy']:.4f}")
    print(f"F1 (macro):  {result['metrics']['f1_macro']:.4f}")
    print(f"MAE:         {result['metrics']['mae']:.4f}")