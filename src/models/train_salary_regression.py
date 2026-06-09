"""
Salary Regression Trainer
=========================
Trains a gradient-boosted regressor to impute missing normalized_salary values.
Rows with a known salary (~37%) are used for training; the model is then used
to fill the ~63% of rows where salary is absent.
"""

import argparse
import logging
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

# Allow running as `python -m src.models.train_salary_regression` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.tracking.config import configure_mlflow, DEFAULT_REGRESSION_EXPERIMENT_NAME
from src.models.tracking.experiment import run_experiment
from src.models.train_lightgbm import (
    train_lgbm_regressor,
    get_feature_importance as lgbm_fi,
)
from src.models.train_xgboost import (
    train_xgboost_regressor,
    get_feature_importance as xgb_fi,
)
from src.models.train_catboost import (
    train_catboost_regressor,
    predict_regressor as catboost_predict,
    get_feature_importance as cb_fi,
)
from src.utils.evaluation import compute_regression_metrics

logger = logging.getLogger(__name__)

TARGET = "normalized_salary"

_ALWAYS_DROP = {
    TARGET
}

_MODEL_REGISTRY = {
    "lightgbm": {
        "train_fn": train_lgbm_regressor,
        "fi_fn": lgbm_fi,
        "predict_fn": None,
        "params": {
            "boosting_type": "gbdt",
            "learning_rate": 0.008,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 5000,
        },
    },
    "xgboost": {
        "train_fn": train_xgboost_regressor,
        "fi_fn": xgb_fi,
        "predict_fn": None,
        "params": {
            "learning_rate": 0.008,
            "max_depth": 0,
            "max_leaves": 63,
            "min_child_weight": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_estimators": 5000,
        },
    },
    "catboost": {
        "train_fn": train_catboost_regressor,
        "fi_fn": cb_fi,
        "predict_fn": catboost_predict,
        "params": {
            "iterations": 5000,
            "learning_rate": 0.008,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "subsample": 0.8,
        },
    },
}


def train(
    train_path: str,
    test_path: str,
    model_type: str = "lightgbm",
    drop_cols: list[str] | None = None,
    register_model_name: str | None = None,
):
    """
    Train and log a salary regression model.

    Parameters
    ----------
    train_path : str
        Path to the training feature matrix CSV.
    test_path : str
        Path to the test feature matrix CSV.
    model_type : str
        One of 'lightgbm', 'xgboost', 'catboost'.
    drop_cols : list[str] or None
        Additional columns to exclude beyond the fixed leakage columns.
        Pass the same list you would put in EXCLUDE_COLS in the notebook.
    register_model_name : str or None
        If provided, registers the model in the MLflow Model Registry.

    Returns
    -------
    dict
        run_experiment result dict (run_id, metrics, model, params).
    """
    if model_type not in _MODEL_REGISTRY:
        raise ValueError(f"model_type must be one of {list(_MODEL_REGISTRY)}; got '{model_type}'")

    exclude = _ALWAYS_DROP | set(drop_cols or [])

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    feature_cols = [c for c in df_train.columns if c not in exclude]

    has_salary = df_train[TARGET].notna()
    X_known = df_train.loc[has_salary, feature_cols]
    y_known = df_train.loc[has_salary, TARGET]

    logger.info(
        f"Regression dataset: {len(X_known):,} rows with known salary "
        f"({has_salary.mean():.1%} of train) | {len(feature_cols)} features"
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_known, y_known, test_size=0.2, random_state=42
    )

    cfg = _MODEL_REGISTRY[model_type]

    result = run_experiment(
        X_train=X_tr,
        y_train=y_tr,
        X_test=X_val,
        y_test=y_val,
        train_fn=cfg["train_fn"],
        eval_fn=compute_regression_metrics,
        feature_importance_fn=cfg["fi_fn"],
        predict_fn=cfg["predict_fn"],
        model_type=model_type,
        task="regression",
        params=cfg["params"],
        register_model_name=register_model_name or model_type,
        dataset_name=os.path.basename(train_path),
        train_source=train_path,
    )

    m = result["metrics"]
    print(
        f"\nRun ID : {result['run_id']}\n"
        f"Model  : {model_type}\n"
        f"Val set: {len(X_val):,} rows\n"
        f"  MAE  : ${m['mae']:,.0f}\n"
        f"  RMSE : ${m['rmse']:,.0f}\n"
        f"  R²   : {m['r2']:.4f}"
    )

    return result


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train a salary regression model and log it to MLflow."
    )
    parser.add_argument(
        "--train-path",
        default="data/processed/v1/feature_matrix_train.csv",
        help="Path to training feature matrix CSV.",
    )
    parser.add_argument(
        "--test-path",
        default="data/processed/v1/feature_matrix_test.csv",
        help="Path to test feature matrix CSV.",
    )
    parser.add_argument(
        "--model-type",
        default="lightgbm",
        choices=list(_MODEL_REGISTRY),
        help="Gradient boosting backend to use (default: lightgbm).",
    )
    parser.add_argument(
        "--drop-cols",
        nargs="*",
        default=[],
        metavar="COL",
        help=(
            "Extra columns to drop in addition to the default leakage columns "
            "(normalized_salary)."
        ),
    )
    parser.add_argument(
        "--register-name",
        default=None,
        help="MLflow Model Registry name (defaults to model type).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()
    configure_mlflow(experiment_name=DEFAULT_REGRESSION_EXPERIMENT_NAME)

    train(
        train_path=args.train_path,
        test_path=args.test_path,
        model_type=args.model_type,
        drop_cols=args.drop_cols,
        register_model_name=args.register_name,
    )
