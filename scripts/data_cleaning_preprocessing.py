import pandas as pd
import numpy as np


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

        # Duplicate information
        'work_type',

        # Nearly empty
        'closed_time',
        'skills_desc'
    ]

    columns = columns if columns is not None else default_columns

    return df.drop(columns=columns, errors='ignore')


# =========================
# DUPLICATE HANDLING
# =========================

def remove_duplicates(df, exclude_columns=None):
    """
    Remove duplicate rows.

    Parameters:
        df (pd.DataFrame): Input dataframe
        exclude_columns (list): Columns excluded from duplicate check

    Returns:
        pd.DataFrame
    """

    exclude_columns = exclude_columns if exclude_columns is not None else ['job_id']

    subset_cols = [c for c in df.columns if c not in exclude_columns]

    return df.drop_duplicates(subset=subset_cols, keep='first')


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

    for col in columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.strip()
            )

    return df


# =========================
# URL CLEANING
# =========================

def remove_url_protocol(df, column='application_url'):
    """
    Remove http:// or https:// from URLs.

    Parameters:
        df (pd.DataFrame): Input dataframe
        column (str): URL column

    Returns:
        pd.DataFrame
    """

    if column in df.columns:
        df[column] = df[column].str.replace(
            r'^https?://',
            '',
            regex=True
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

    salary = df[salary_column].dropna()
    salary = salary[salary > 0]

    threshold = salary.quantile(quantile_threshold)

    outlier_mask = df[salary_column] > threshold

    df.loc[outlier_mask, related_salary_columns] = np.nan

    return df


# =========================
# SAVE DATASET
# =========================

def save_dataset(
    df,
    output_path="../data/processed/cleaned_job_dataset.csv"
):
    """
    Save dataframe to CSV.

    Parameters:
        df (pd.DataFrame): Dataframe to save
        output_path (str): Output path
    """

    df.to_csv(output_path, index=False)


# =========================
# COMPLETE PIPELINE
# =========================

def preprocess_dataset(
    input_path="../data/raw/postings.csv",
    output_path="../data/processed/cleaned_job_dataset.csv"
):
    """
    Complete preprocessing pipeline.
    """

    df = pd.read_csv(input_path)

    print(f"Initial shape: {df.shape}")

    df = drop_columns(df)
    print(f"Shape after dropping columns: {df.shape}")

    df = remove_duplicates(df)
    print(f"Shape after removing duplicates: {df.shape}")

    df = clean_text_columns(df)

    df = remove_url_protocol(df)

    df = handle_salary_outliers(df)

    save_dataset(df, output_path)

    print(f"Cleaned dataset saved to: {output_path}")

    return df


# =========================
# RUN SCRIPT
# =========================

if __name__ == "__main__":
    preprocess_dataset()