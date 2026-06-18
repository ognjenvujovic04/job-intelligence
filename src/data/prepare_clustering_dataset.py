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

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.models.serve.salary_regression import TARGET, predict_salary

logger = logging.getLogger(__name__)

V5_PATHS = [
    "data/processed/v5/feature_matrix_train.csv",
    "data/processed/v5/feature_matrix_test.csv",
]
V6_DIR = "data/processed/v6"
V6_OUT = os.path.join(V6_DIR, "feature_matrix.csv")


def fill_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing normalized_salary using the shared salary regressor.

    Only rows where the target is NaN are overwritten; observed salaries are
    left untouched. Delegates model loading + prediction to
    src.models.serve.salary_regression so there is one salary-inference path.
    """
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

    predicted = predict_salary(df.loc[missing], verbose=False)
    df.loc[missing, TARGET] = predicted["predicted_salary"].values
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

    df = fill_salary(df)

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
