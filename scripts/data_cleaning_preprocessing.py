import logging
import numpy as np
import pandas as pd

# Initialize module-level logger
logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool):
    """Internal helper to enable or disable pipeline logging dynamically."""
    if verbose:
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter(
                    "[%(levelname)s] %(message)s"
                )
            )
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.CRITICAL)


# =========================
# COLUMN DROPPING
# =========================

def drop_columns(df, columns=None):
    """
    Drop unnecessary columns from dataframe.

    Parameters:
        df (pd.DataFrame): Input dataframe
        columns (list): Columns to drop

    Returns:
        pd.DataFrame
    """

    default_columns = [
        # Zero-variance
        'sponsored',
        'compensation_type',

        # Not useful for modeling
        'job_posting_url',
        'job_id',
        'application_url',

        # Duplicate information
        'work_type',

        # Nearly empty
        'closed_time',
        'skills_desc'
    ]

    columns = columns if columns is not None else default_columns
    
    # Identify which target columns actually exist in the dataframe
    existing_drops = [col for col in columns if col in df.columns]
    logger.info(f"Dropping columns: {existing_drops}")

    return df.drop(columns=columns, errors='ignore')


# =========================
# DUPLICATE HANDLING
# =========================

def remove_duplicates(df, exclude_columns=[]):
    """
    Remove duplicate rows.

    Parameters:
        df (pd.DataFrame): Input dataframe
        exclude_columns (list): Columns excluded from duplicate check

    Returns:
        pd.DataFrame
    """

    subset_cols = [c for c in df.columns if c not in exclude_columns]

    initial_rows = len(df)
    df_cleaned = df.drop_duplicates(subset=subset_cols, keep='first')
    
    logger.info(f"Removed {initial_rows - len(df_cleaned)} duplicate rows (ignoring columns: {exclude_columns})")

    return df_cleaned


# =========================
# TEXT CLEANING
# =========================

def clean_text_columns(df, columns=None):
    """
    Lowercase and strip whitespace from text columns.

    Parameters:
        df (pd.DataFrame): Input dataframe
        columns (list): Columns to clean

    Returns:
        pd.DataFrame
    """

    default_columns = [
        'title',
        'company_name',
        'location',
        'description'
    ]

    columns = columns if columns is not None else default_columns
    existing_cols = [col for col in columns if col in df.columns]
    
    logger.info(f"Normalizing text (lowercase & strip whitespace) in columns: {existing_cols}")

    for col in existing_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .str.strip()
        )

    return df

# =========================
# OUTLIER HANDLING
# =========================

def handle_salary_outliers(
    df,
    salary_column='normalized_salary',
    related_salary_columns=None,
    quantile_threshold=0.9989
):
    """
    Detect salary outliers and replace related salary values with NaN.

    Parameters:
        df (pd.DataFrame): Input dataframe
        salary_column (str): Column used for outlier detection
        related_salary_columns (list): Columns to nullify
        quantile_threshold (float): Quantile threshold

    Returns:
        pd.DataFrame
    """

    default_salary_cols = [
        'min_salary',
        'max_salary',
        'med_salary',
        'normalized_salary'
    ]

    related_salary_columns = (
        related_salary_columns
        if related_salary_columns is not None
        else default_salary_cols
    )

    if salary_column not in df.columns:
        return df

    salary = df[salary_column].dropna()
    salary = salary[salary > 0]

    threshold = salary.quantile(quantile_threshold)
    outlier_mask = df[salary_column] > threshold
    outlier_count = outlier_mask.sum()

    logger.info(
        f"Identified {outlier_count} outliers in '{salary_column}' (> {threshold:.2f}). "
        f"Setting related columns {related_salary_columns} to NaN."
    )

    df.loc[outlier_mask, related_salary_columns] = np.nan

    return df


# =========================
# SAVE DATASET
# =========================

def save_dataset(
    df,
    output_path="../data/processed/cleaned_job_postings.csv"
):
    """
    Save dataframe to CSV.

    Parameters:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Output path
    """
    logger.info(f"Saving cleaned dataset to: {output_path}")
    df.to_csv(output_path, index=False)


# =========================
# COMPLETE PIPELINE
# =========================

def preprocess_dataset(
    input_path="../data/raw/postings.csv",
    output_path="../data/processed/cleaned_job_postings.csv",
    verbose=True
):
    """
    Complete preprocessing pipeline.
    
    Parameters:
        input_path (str): Path to raw CSV data
        output_path (str): Target path for cleaned CSV data
        verbose (bool): If True, outputs step-by-step progress to stdout. Default is True.
    """
    # Toggle logging state
    _configure_logging(verbose)

    df = pd.read_csv(input_path)
    logger.info(f"Pipeline started. Initial dataset shape: {df.shape}")

    df = drop_columns(df)
    df = remove_duplicates(df)
    df = clean_text_columns(df)
    df = handle_salary_outliers(df)
    
    save_dataset(df, output_path)
    logger.info(f"Pipeline finished successfully. Final dataset shape: {df.shape}")

    return df


# =========================
# RUN SCRIPT
# =========================

if __name__ == "__main__":
    preprocess_dataset()