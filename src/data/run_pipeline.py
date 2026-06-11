"""
Data pipelines: training (fit + persist) and inference (transform new data).

`train_pipeline` runs the full training flow (cleaning -> domain classification
-> feature engineering with a train/test split) and persists every fitted
artifact to data/precomputed/ so it can be reused.

`prepare_data` takes a raw postings DataFrame (same schema as
data/raw/postings.csv) and returns a model-ready feature matrix by reapplying
those persisted artifacts, without re-fitting anything.
"""

import json
import logging
import os

import joblib

from data_cleaning_preprocessing import clean_dataframe, preprocess_dataset
from fe_domain_classification import (
    main as compute_domain_similarities,
    compute_domain_sim_df,
    ensure_prototype_embeddings,
)
from feature_engineering import feature_engineering_pipeline, transform_features

logger = logging.getLogger(__name__)

RAW_DATA = "../../data/raw/postings.csv"
CLEANED_DATA = "../../data/processed/cleaned_job_postings.csv"
DOMAIN_DATA = "../../data/precomputed/domain_probabilities.csv"
TRAIN_OUTPUT = "../../data/processed/v5/feature_matrix_train.csv"
TEST_OUTPUT = "../../data/processed/v5/feature_matrix_test.csv"

PRECOMPUTED_DIR = "../../data/precomputed"
PREP_ARTIFACTS = os.path.join(PRECOMPUTED_DIR, "prep_artifacts.joblib")
FEATURE_COLUMNS = os.path.join(PRECOMPUTED_DIR, "feature_columns.json")


def _configure_root_logging(verbose=True):
    """Configure logging once for the entire pipeline."""
    root = logging.getLogger()
    root.handlers.clear()

    level = logging.INFO if verbose else logging.CRITICAL

    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
    formatter.default_msec_format = "%s.%03d"  # dot instead of comma
    handler.setFormatter(formatter)

    root.addHandler(handler)
    root.setLevel(level)


def train_pipeline(
    raw_path=RAW_DATA,
    cleaned_path=CLEANED_DATA,
    domain_path=DOMAIN_DATA,
    train_output=TRAIN_OUTPUT,
    test_output=TEST_OUTPUT,
    force_domain_recompute=False,
    test_size=0.20,
    random_state=42,
    smoothing=20,
    verbose=True,
):
    """
    Run the full training pipeline from raw data to train/test feature matrices.

    Steps:
        1. Data cleaning and preprocessing
        2. Domain similarity computation (skipped if precomputed file exists)
        3. Feature engineering with train/test split
        4. Persist fitted artifacts to data/precomputed/ for inference reuse

    Parameters:
        raw_path (str): Path to the raw postings CSV.
        cleaned_path (str): Output path for the cleaned CSV.
        domain_path (str): Path to domain similarity scores (read or written).
        train_output (str): Output path for the training feature matrix.
        test_output (str): Output path for the test feature matrix.
        force_domain_recompute (bool): Recompute domain similarities even if
            the precomputed file already exists.
        test_size (float): Fraction of data reserved for the test set.
        random_state (int): Random seed for reproducibility.
        smoothing (int): Smoothing factor for target encoding.
        verbose (bool): Enable step-by-step logging.

    Returns:
        (pd.DataFrame, pd.DataFrame): Train and test feature matrices.
    """
    _configure_root_logging(verbose)

    logger.info("=" * 60)
    logger.info("STAGE 1/3: Data Cleaning & Preprocessing")
    logger.info("=" * 60)

    _, salary_outlier_threshold = preprocess_dataset(
        input_path=raw_path,
        output_path=cleaned_path,
        verbose=verbose,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 2/3: Domain Classification")
    logger.info("=" * 60)

    if not force_domain_recompute and os.path.exists(domain_path):
        logger.info(
            f"Precomputed domain file found at '{domain_path}', skipping recomputation"
        )
    else:
        if force_domain_recompute:
            logger.info("Forced recomputation of domain similarities")
        else:
            logger.info(
                f"No precomputed file at '{domain_path}', computing domain similarities"
            )
        compute_domain_similarities()

    # Prototype embeddings are needed by the inference pipeline; generate them
    # even when the full domain recomputation above was skipped.
    ensure_prototype_embeddings()

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3/3: Feature Engineering")
    logger.info("=" * 60)

    df_train, df_test, artifacts = feature_engineering_pipeline(
        input_path=cleaned_path,
        domain_path=domain_path,
        train_output=train_output,
        test_output=test_output,
        test_size=test_size,
        random_state=random_state,
        smoothing=smoothing,
        verbose=verbose,
        return_artifacts=True,
    )

    # -- Persist fitted artifacts for inference reuse --
    artifacts["salary_outlier_threshold"] = salary_outlier_threshold
    os.makedirs(PRECOMPUTED_DIR, exist_ok=True)
    joblib.dump(artifacts, PREP_ARTIFACTS)
    with open(FEATURE_COLUMNS, "w") as f:
        json.dump(artifacts["feature_columns"], f, indent=2)
    logger.info(f"Saved preparation artifacts to '{PREP_ARTIFACTS}'")

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"Pipeline complete | train: {df_train.shape} | test: {df_test.shape}"
    )
    logger.info("=" * 60)

    return df_train, df_test


def prepare_data(df, verbose=True):
    """
    Prepare new raw data for ML models using persisted training artifacts.

    Takes a raw postings DataFrame (same schema as data/raw/postings.csv) and
    returns a model-ready feature matrix with the exact column schema produced
    at training time. No encoders are re-fit: salary caps, target encodings and
    frequency maps are all loaded from data/precomputed/.

    Duplicate removal is skipped on new data; the salary-outlier threshold is
    reused from training.

    Parameters:
        df (pd.DataFrame): Raw postings dataframe.
        verbose (bool): Enable step-by-step logging.

    Returns:
        pd.DataFrame: Feature matrix aligned to the training schema.

    Raises:
        FileNotFoundError: when the persisted artifacts are missing (run
            train_pipeline first).
    """
    _configure_root_logging(verbose)

    if not os.path.exists(PREP_ARTIFACTS):
        raise FileNotFoundError(
            f"Preparation artifacts not found at '{PREP_ARTIFACTS}'. "
            "Run train_pipeline first to generate the precomputed artifacts."
        )
    artifacts = joblib.load(PREP_ARTIFACTS)

    logger.info("=" * 60)
    logger.info("INFERENCE 1/3: Cleaning")
    logger.info("=" * 60)
    cleaned, _ = clean_dataframe(
        df.copy(),
        salary_threshold=artifacts["salary_outlier_threshold"],
        remove_dupes=False,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("INFERENCE 2/3: Domain Classification")
    logger.info("=" * 60)
    domain_df = compute_domain_sim_df(cleaned)

    logger.info("")
    logger.info("=" * 60)
    logger.info("INFERENCE 3/3: Feature Engineering")
    logger.info("=" * 60)
    df_ready = transform_features(cleaned, domain_df, artifacts)

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Data preparation complete | shape: {df_ready.shape}")
    logger.info("=" * 60)

    return df_ready


if __name__ == "__main__":
    train_pipeline()
