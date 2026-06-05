import logging
from datetime import datetime

import mlflow

from .logging_helpers import (
    _log_confusion_matrix,
    _log_dataset,
    _log_feature_importance,
    _log_metrics,
    _log_model,
    _log_params,
    _log_tags,
)

logger = logging.getLogger(__name__)


def run_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    train_fn,
    eval_fn,
    feature_importance_fn=None,
    predict_fn=None,
    model_type="lightgbm",
    params=None,
    run_name=None,
    tags=None,
    register_model_name=None,
    val_size=0.15,
    top_n_features=15,
    dataset=None,
    dataset_name=None,
    train_source=None,
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
            returning a fitted model (or model bundle).
        eval_fn (callable): ``eval_fn(y_true, y_pred)`` returning a dict
            of scalar metrics (and optionally a ``classification_report``
            nested dict).
        feature_importance_fn (callable or None):
            ``feature_importance_fn(model, feature_names, top_n)``
            returning a pd.Series. For models that need extra args
            (e.g. MLP permutation importance needs X_test/y_test),
            wrap the call in a lambda before passing it in.
        predict_fn (callable or None): Custom prediction function
            ``predict_fn(model, X_test)`` returning an array of
            predictions. Required for models like CatBoost or TabNet
            whose predict interface differs from ``model.predict(X)``.
            When None, ``model.predict(X_test)`` is called directly.
        model_type (str): One of "lightgbm", "xgboost", "catboost",
            "sklearn", "tabnet", or "generic". Controls which MLflow
            model-logging flavour is used and what appears in the UI.
        params (dict or None): Hyperparameter overrides forwarded to
            *train_fn*.
        run_name (str or None): Descriptive name shown in the MLflow UI.
            Auto-generated from model_type + timestamp when omitted.
        tags (dict or None): Arbitrary key-value metadata attached to the
            run (e.g. ``{"author": "ov", "stage": "baseline"}``).
        register_model_name (str or None): If provided, the logged model
            is also registered under this name in the MLflow Model Registry.
        val_size (float): Validation fraction passed through to *train_fn*.
        top_n_features (int): Number of top features to log.
        dataset (str or None): Relative dataset path (e.g. ``"v1/feature_matrix_train.csv"``)
            logged as an MLflow parameter so runs are comparable by dataset version.
        dataset_name (str or None): Display name for the training dataset logged
            via ``mlflow.log_input`` (shows in the MLflow Datasets column).
        train_source (str or None): Source URI / file path for the dataset.
            Defaults to *dataset_name* when omitted.

    Returns:
        dict: A summary with keys ``run_id``, ``metrics``, ``model``,
            and ``params``.
    """
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{model_type}_{timestamp}"

    merged_params = dict(params or {})

    # CatBoost treats iterations/n_estimators/num_boost_round/num_trees
    # as synonyms and raises an error if more than one is present.
    if model_type == "catboost" and "n_estimators" in merged_params:
        merged_params["iterations"] = merged_params.pop("n_estimators")

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow run started | name='{run_name}' | id={run_id}")

        _log_tags(tags, run_name, model_type)
        _log_params(merged_params, val_size, X_train, model_type)
        if dataset is not None:
            mlflow.log_param("dataset", dataset)
        if dataset_name is not None:
            _log_dataset(X_train, y_train, dataset_name, train_source or dataset_name)

        model = train_fn(X_train, y_train, val_size=val_size, params=merged_params)

        best_iter = getattr(model, "best_iteration_", None)
        if best_iter is not None:
            mlflow.log_metric("best_iteration", best_iter)

        if predict_fn is not None:
            y_pred = predict_fn(model, X_test)
        else:
            y_pred = model.predict(X_test)

        metrics = eval_fn(y_test, y_pred)
        _log_metrics(metrics)

        _log_confusion_matrix(y_test, y_pred)

        if feature_importance_fn is not None:
            importance = feature_importance_fn(
                model, list(X_train.columns), top_n_features
            )
            _log_feature_importance(importance)

        _log_model(model, register_model_name, model_type)

        logger.info(f"MLflow run finished | id={run_id}")

    return {
        "run_id": run_id,
        "metrics": metrics,
        "model": model,
        "params": merged_params,
    }
