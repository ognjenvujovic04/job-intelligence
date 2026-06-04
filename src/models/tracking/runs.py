import logging

import mlflow
import pandas as pd

from .config import DEFAULT_EXPERIMENT_NAME

logger = logging.getLogger(__name__)


def compare_runs(experiment_name=None, metric_keys=None, top_n=10):
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

    keep_cols = ["run_id", "start_time", "tags.mlflow.runName", "tags.model_type"]
    metric_cols = [f"metrics.{k}" for k in metric_keys]
    available = [c for c in keep_cols + metric_cols if c in runs.columns]

    result = runs[available].copy()

    rename_map = {
        "tags.mlflow.runName": "run_name",
        "tags.model_type": "model_type",
    }
    rename_map.update({f"metrics.{k}": k for k in metric_keys})
    result.rename(columns=rename_map, inplace=True)

    return result


def get_best_run(experiment_name=None, metric="f1_macro", higher_is_better=True):
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
        "model_type": best.get("model_type", ""),
        "metric_value": best[metric],
    }
