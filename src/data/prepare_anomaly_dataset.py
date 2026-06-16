"""
Prepare anomaly-detection dataset (v7)
======================================
Takes the v5 train/test split (job_id retained, salary mostly NaN) and replaces
``normalized_salary`` with v6's regressed salaries (joined on ``job_id``). Unlike
the clustering dataset (v6), the **80/20 train/test split is kept**: anomaly
detection fits its scaler on the train split and derives reconstruction
thresholds from the train scores, so it needs a genuine hold-out.

v6 already imputed every posting's salary via a logged regression run and is
unique on ``job_id`` covering all v5 ids, so this is a clean lookup — no need to
re-run the regression model here.

Writes ``data/processed/v7/feature_matrix_{train,test}.csv``.

Usage
-----
    python -m src.data.prepare_anomaly_dataset
"""

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

logger = logging.getLogger(__name__)

# Repo-root-anchored paths so this runs regardless of cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
V5_TRAIN = os.path.join(_REPO_ROOT, "data", "processed", "v5", "feature_matrix_train.csv")
V5_TEST = os.path.join(_REPO_ROOT, "data", "processed", "v5", "feature_matrix_test.csv")
V6_PATH = os.path.join(_REPO_ROOT, "data", "processed", "v6", "feature_matrix.csv")
V7_DIR = os.path.join(_REPO_ROOT, "data", "processed", "v7")

TARGET = "normalized_salary"


def fill_salary_from_v6(df: pd.DataFrame, v6_salary: pd.Series, name: str) -> pd.DataFrame:
    """Replace normalized_salary with v6's regressed values (joined on job_id)."""
    df = df.copy()
    missing_before = int(df[TARGET].isna().sum())
    df[TARGET] = df["job_id"].map(v6_salary)
    missing_after = int(df[TARGET].isna().sum())
    logger.info(
        "%s: normalized_salary NaNs %d -> %d (filled from v6 regression)",
        name,
        missing_before,
        missing_after,
    )
    return df


def main():
    os.makedirs(V7_DIR, exist_ok=True)

    v6_salary = (
        pd.read_csv(V6_PATH, usecols=["job_id", TARGET])
        .set_index("job_id")[TARGET]
    )
    logger.info("Loaded %d v6 salaries for lookup", len(v6_salary))

    for split, src in (("train", V5_TRAIN), ("test", V5_TEST)):
        df = pd.read_csv(src)
        logger.info("Loaded %s: %d rows, %d cols", src, *df.shape)

        df = fill_salary_from_v6(df, v6_salary, split.capitalize())

        n_remaining = int(df[TARGET].isna().sum())
        if n_remaining:
            logger.warning("%s: %d salary values still NaN after v6 lookup.", split, n_remaining)

        out = os.path.join(V7_DIR, f"feature_matrix_{split}.csv")
        df.to_csv(out, index=False)
        logger.info("Saved v7 %s feature matrix -> %s  (%d rows, %d cols)", split, out, *df.shape)

    logger.info("Dataset v7 ready in %s", V7_DIR)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
