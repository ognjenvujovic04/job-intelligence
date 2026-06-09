"""
Prepare clustering dataset (v6)
================================
Loads both v5 splits (train + test), concatenates them into a single
feature matrix, uses the best salary regression model from MLflow run
1b6a1be92d2c4f42aba0aee91677ea94 to fill missing normalized_salary values,
and writes a single feature_matrix.csv as dataset v6.

No train/test split is produced — clustering uses the full dataset.

Usage
-----
    python -m src.data.prepare_clustering_dataset
"""

import logging
import os
import sys

import mlflow
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.tracking.config import DEFAULT_TRACKING_URI

logger = logging.getLogger(__name__)

SALARY_RUN_ID = "1b6a1be92d2c4f42aba0aee91677ea94"
V5_PATHS = [
    "data/processed/v5/feature_matrix_train.csv",
    "data/processed/v5/feature_matrix_test.csv",
]
V6_DIR = "data/processed/v6"
V6_OUT = os.path.join(V6_DIR, "feature_matrix.csv")
TARGET = "normalized_salary"
_NON_FEATURE_COLS = {TARGET, "job_id", "salary_is_missing"}


def load_salary_model(run_id: str):
    mlflow.set_tracking_uri(DEFAULT_TRACKING_URI)

    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{run_id}'")
    if versions:
        mv = versions[0]
        model_uri = f"models:/{mv.name}/{mv.version}"
    else:
        model_uri = f"runs:/{run_id}/model"

    logger.info("Loading salary regression model from %s", model_uri)
    return mlflow.pyfunc.load_model(model_uri)


def fill_salary(df: pd.DataFrame, model) -> pd.DataFrame:
    df = df.copy()
    missing = df[TARGET].isna()
    n_missing = int(missing.sum())

    if n_missing == 0:
        logger.info("No missing salary values — nothing to impute.")
        return df

    logger.info(
        "Imputing salary for %d rows (%.1f%% of %d)",
        n_missing,
        missing.mean() * 100,
        len(df),
    )

    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    predictions = model.predict(df.loc[missing, feature_cols])
    df.loc[missing, TARGET] = predictions
    return df


def main():
    os.makedirs(V6_DIR, exist_ok=True)

    parts = []
    for path in V5_PATHS:
        part = pd.read_csv(path)
        logger.info("Loaded %s: %d rows, %d cols", path, *part.shape)
        parts.append(part)

    df = pd.concat(parts, ignore_index=True)
    logger.info("Combined: %d rows, %d cols", *df.shape)

    model = load_salary_model(SALARY_RUN_ID)
    df = fill_salary(df, model)

    n_remaining = int(df[TARGET].isna().sum())
    if n_remaining:
        logger.warning("%d salary values still NaN after imputation.", n_remaining)

    df.to_csv(V6_OUT, index=False)
    logger.info("Saved v6 feature matrix → %s  (%d rows, %d cols)", V6_OUT, *df.shape)
    logger.info("Dataset v6 ready in %s", V6_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
