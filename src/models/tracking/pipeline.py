import logging
import os

import pandas as pd

from .config import (
    DEFAULT_REGRESSION_EXPERIMENT_NAME,
    DEFAULT_REGRESSION_TARGET,
    DEFAULT_TARGET,
    configure_mlflow,
)
from .experiment import run_experiment

logger = logging.getLogger(__name__)


def run_default_experiment(
    dataset="v1/feature_matrix_train.csv",
    target=DEFAULT_TARGET,
    model_type="lightgbm",
    params=None,
    run_name=None,
    tags=None,
    tracking_uri=None,
    experiment_name=None,
    register_model_name=None,
    val_size=0.15,
    train_path=None,
    test_path=None,
):
    """
    Load data, configure MLflow, train, evaluate, and log in one call.

    This is handy for quick experiments from the command line or a notebook
    when you just want to change ``params`` or ``model_type`` and see the
    results show up in the MLflow UI.

    Parameters:
        dataset (str): Path to the training CSV relative to ``data/processed/``.
            The test CSV is derived by replacing ``"train"`` with ``"test"`` in
            this string.  Defaults to ``"v1/feature_matrix_train.csv"``.
            Example: ``"v2/feature_matrix_train_v2.csv"``.
        target (str): Name of the target column.
        model_type (str): One of "lightgbm", "xgboost", "catboost",
            "sklearn_logreg", "sklearn_mlp", "tabnet".
        params (dict or None): Hyperparameter overrides.
        run_name (str or None): MLflow run name.
        tags (dict or None): Extra metadata tags.
        tracking_uri (str or None): MLflow tracking URI.
        experiment_name (str or None): Experiment name.
        register_model_name (str or None): Model-registry name.
        val_size (float): Validation split fraction.
        train_path (str or None): Full path override for the training CSV.
            Supersedes ``dataset`` when provided.
        test_path (str or None): Full path override for the test CSV.
            Supersedes the auto-derived test path when provided.

    Returns:
        dict: Summary from ``run_experiment``.
    """
    train_fn, predict_fn, fi_fn, mlflow_type = _resolve_model_functions(model_type)
    mlp_model = model_type in ("sklearn_mlp", "mlp")

    from ...utils.evaluation import compute_metrics

    if register_model_name is None:
        register_model_name = f"{model_type}-exp-level-classifier"

    # Resolve paths relative to the repository root
    # (three levels up from this file: src/models/tracking -> ../../.. -> repo root)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    processed_dir = os.path.join(base_dir, "data", "processed")
    if train_path is None:
        train_path = os.path.join(processed_dir, *dataset.replace("\\", "/").split("/"))
    if test_path is None:
        test_dataset = dataset.replace("train", "test")
        test_path = os.path.join(processed_dir, *test_dataset.replace("\\", "/").split("/"))

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    df_train = df_train.dropna(subset=[target])
    df_test = df_test.dropna(subset=[target])

    y_train = df_train[target].astype(int)
    y_test = df_test[target].astype(int)
    X_train = df_train.drop(columns=[target])
    X_test = df_test.drop(columns=[target])

    # MLP's get_feature_importance requires X_test/y_test for permutation importance;
    # wrap it into the standard (model, feature_names, top_n) interface now that data is loaded.
    if mlp_model and fi_fn is not None:
        _base_fi = fi_fn
        fi_fn = lambda m, fn, n: _base_fi(m, X_test, y_test, fn, n)

    configure_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    return run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_fn=train_fn,
        eval_fn=compute_metrics,
        feature_importance_fn=fi_fn,
        predict_fn=predict_fn,
        model_type=mlflow_type,
        params=params,
        run_name=run_name,
        tags=tags,
        register_model_name=register_model_name,
        val_size=val_size,
        dataset=dataset,
        dataset_name=os.path.basename(train_path),
        train_source=train_path,
    )


def run_default_regression_experiment(
    dataset="v2/feature_matrix_train_v2.csv",
    target=DEFAULT_REGRESSION_TARGET,
    params=None,
    run_name=None,
    tags=None,
    tracking_uri=None,
    experiment_name=None,
    register_model_name=None,
    val_size=0.2,
    train_path=None,
    test_path=None,
    exclude_cols=None,
):
    """
    Load data, configure MLflow, train a LightGBM regressor, and log in one call.

    Parameters:
        dataset (str): Training CSV path relative to ``data/processed/``.
        target (str): Name of the continuous target column.
        params (dict or None): Hyperparameter overrides.
        run_name (str or None): MLflow run name.
        tags (dict or None): Extra metadata tags.
        tracking_uri (str or None): MLflow tracking URI.
        experiment_name (str or None): Experiment name.
            Defaults to ``DEFAULT_REGRESSION_EXPERIMENT_NAME``.
        register_model_name (str or None): Model-registry name.
        val_size (float): Validation split fraction.
        train_path (str or None): Full path override for the training CSV.
        test_path (str or None): Full path override for the test CSV.
        exclude_cols (list[str] or None): Extra columns to drop before training
            (in addition to the target). Useful for dropping leaky features
            such as target-encoded salary columns.

    Returns:
        dict: Summary from ``run_experiment``.
    """
    from ..train.train_lightgbm import (
        get_feature_importance,
        train_lgbm_regressor,
    )
    from ...utils.evaluation import compute_regression_metrics

    if register_model_name is None:
        register_model_name = "lgbm-salary-regressor"

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    processed_dir = os.path.join(base_dir, "data", "processed")
    if train_path is None:
        train_path = os.path.join(processed_dir, *dataset.replace("\\", "/").split("/"))
    if test_path is None:
        test_dataset = dataset.replace("train", "test")
        test_path = os.path.join(processed_dir, *test_dataset.replace("\\", "/").split("/"))

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    # Keep only rows with a known salary target
    df_train = df_train.dropna(subset=[target])
    df_test = df_test.dropna(subset=[target])

    drop_cols = [target] + (exclude_cols or [])
    y_train = df_train[target]
    y_test = df_test[target]
    X_train = df_train.drop(columns=[c for c in drop_cols if c in df_train.columns])
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])

    configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name or DEFAULT_REGRESSION_EXPERIMENT_NAME,
    )

    return run_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        train_fn=train_lgbm_regressor,
        eval_fn=compute_regression_metrics,
        feature_importance_fn=get_feature_importance,
        model_type="lightgbm",
        task="regression",
        params=params,
        run_name=run_name,
        tags=tags,
        register_model_name=register_model_name,
        val_size=val_size,
        dataset=dataset,
        dataset_name=os.path.basename(train_path),
        train_source=train_path,
    )


def _resolve_model_functions(model_type):
    """
    Return (train_fn, predict_fn, feature_importance_fn, mlflow_type)
    for the given model_type string.

    predict_fn is None when the standard model.predict() interface works.
    """
    if model_type == "lightgbm":
        from ..train.train_lightgbm import train_lgbm_classifier, get_feature_importance
        return train_lgbm_classifier, None, get_feature_importance, "lightgbm"

    elif model_type == "xgboost":
        from ..train.train_xgboost import train_xgboost, get_feature_importance
        return train_xgboost, None, get_feature_importance, "xgboost"

    elif model_type == "catboost":
        from ..train.train_catboost import train_catboost, predict as cb_predict, get_feature_importance
        return train_catboost, cb_predict, get_feature_importance, "catboost"

    elif model_type in ("sklearn_logreg", "logreg", "logistic_regression"):
        from ..train.train_logreg import build_logreg_pipeline, get_feature_importance
        def _logreg_train(X_train, y_train, val_size=None, params=None):
            return build_logreg_pipeline(X_train, y_train, **(params or {}))
        return _logreg_train, None, get_feature_importance, "sklearn"

    elif model_type in ("sklearn_mlp", "mlp"):
        from ..train.train_mlp import train_mlp, get_feature_importance
        return train_mlp, None, get_feature_importance, "sklearn"

    elif model_type == "tabnet":
        from ..train.train_tabnet import train_tabnet, predict_tabnet, get_feature_importance
        return train_tabnet, predict_tabnet, get_feature_importance, "tabnet"

    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Supported types: "
            "lightgbm, xgboost, catboost, sklearn_logreg, sklearn_mlp, tabnet"
        )
