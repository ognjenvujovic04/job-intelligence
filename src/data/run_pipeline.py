"""
End-to-end pipeline: cleaning -> domain classification -> feature engineering.

Thin orchestration layer that delegates all work to the existing modules.
Domain similarities are only recomputed when the precomputed file is missing.
"""

import logging
import os

from data_cleaning_preprocessing import preprocess_dataset
from fe_domain_classification import main as compute_domain_similarities
from feature_engineering import feature_engineering_pipeline

logger = logging.getLogger(__name__)

RAW_DATA = "../../data/raw/postings.csv"
CLEANED_DATA = "../../data/processed/cleaned_job_postings.csv"
DOMAIN_DATA = "../../data/precomputed/domain_probabilities.csv"
TRAIN_OUTPUT = "../../data/processed/v5/feature_matrix_train.csv"
TEST_OUTPUT = "../../data/processed/v5/feature_matrix_test.csv"


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

def run_pipeline(
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
    Run the full pipeline from raw data to train/test feature matrices.

    Steps:
        1. Data cleaning and preprocessing
        2. Domain similarity computation (skipped if precomputed file exists)
        3. Feature engineering with train/test split

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

    preprocess_dataset(
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

    logger.info("")
    logger.info("=" * 60)
    logger.info("STAGE 3/3: Feature Engineering")
    logger.info("=" * 60)

    df_train, df_test = feature_engineering_pipeline(
        input_path=cleaned_path,
        domain_path=domain_path,
        train_output=train_output,
        test_output=test_output,
        test_size=test_size,
        random_state=random_state,
        smoothing=smoothing,
        verbose=verbose,
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"Pipeline complete | train: {df_train.shape} | test: {df_test.shape}"
    )
    logger.info("=" * 60)

    return df_train, df_test


if __name__ == "__main__":
    run_pipeline()