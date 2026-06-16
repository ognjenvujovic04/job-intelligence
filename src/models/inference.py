"""
Unified inference facade for the five modeling tracks.

This is the single module a serving/API layer imports: it never touches MLflow
or `prepare_data` directly. Each function takes a DataFrame of **raw postings**
and returns a copy with that track's predictions appended, so every track shares
one ``raw df -> df`` contract:

    from src.models.inference import classify, predict_salary, predict_clusters, \
        detect_anomalies, summarize_postings

Four of the five compose the shared feature-engineering pipeline
(`src.data.run_pipeline.prepare_data`) before scoring; summarization works on the
raw ``description`` text and skips it. The underlying model loaders cache their
models module-wide, so the first call per track pays the load cost and later
calls are cheap.

All inference depends on local-only, gitignored artifacts existing on the host
(mlflow.db, the mlruns/ logged models, data/precomputed/). T5 also downloads
~500 MB from HuggingFace on first use and is slow on CPU.
"""

import logging

logger = logging.getLogger(__name__)


def classify(df, verbose=True):
    """Raw postings -> predicted experience level.

    Returns a copy of the prepared feature matrix with
    'predicted_experience_level_ord' and 'predicted_experience_level' columns.
    """
    from src.data.run_pipeline import prepare_data
    from src.models.classification import run_classification

    return run_classification(prepare_data(df, verbose=verbose), verbose=verbose)


def predict_salary(df, verbose=True):
    """Raw postings -> predicted normalized salary.

    Returns a copy of the prepared feature matrix with a 'predicted_salary'
    column.
    """
    from src.data.run_pipeline import prepare_data
    from src.models.salary_regression import predict_salary as _predict_salary

    return _predict_salary(prepare_data(df, verbose=verbose), verbose=verbose)


def predict_clusters(df, verbose=True):
    """Raw postings -> cluster id.

    Returns a copy of the prepared feature matrix with a 'cluster' column.
    """
    from src.data.run_pipeline import prepare_data
    from src.models.clustering import predict_clusters as _predict_clusters

    return _predict_clusters(prepare_data(df, verbose=verbose), verbose=verbose)


def detect_anomalies(df, verbose=True):
    """Raw postings -> anomaly flags + 0-4 ensemble score.

    Returns a copy of the prepared feature matrix with the four per-model flag
    columns and an 'anomaly_score' column.
    """
    from src.data.run_pipeline import prepare_data
    from src.models.anomaly_detection import run_anomaly_detection

    return run_anomaly_detection(prepare_data(df, verbose=verbose), verbose=verbose)


def summarize_postings(df, **kwargs):
    """Raw postings -> abstractive T5 summary.

    Operates on the raw 'description' text (no feature engineering). Returns a
    copy of df[['job_id', 'description']] with a 'summary' column. Extra kwargs
    are forwarded to the T5 summarizer (e.g. batch_size, max_summary_tokens).
    """
    from src.models.summarize import t5_summarize_df

    return t5_summarize_df(df, **kwargs)


if __name__ == "__main__":
    # End-to-end smoke test: run every facade function over the synthetic
    # postings and print one result row per posting.
    import os
    import sys

    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, repo_root)

    synthetic_csv = os.path.join(repo_root, "data", "raw", "synthetic_postings.csv")
    df_raw = pd.read_csv(synthetic_csv)
    print(f"Loaded {len(df_raw):,} synthetic postings from {synthetic_csv}\n")

    classified = classify(df_raw, verbose=False)
    salaries = predict_salary(df_raw, verbose=False)
    clusters = predict_clusters(df_raw, verbose=False)
    anomalies = detect_anomalies(df_raw, verbose=False)
    summaries = summarize_postings(df_raw)

    titles = df_raw["title"] if "title" in df_raw.columns else pd.Series([""] * len(df_raw))
    print("\nPer-posting results:")
    for i in range(len(df_raw)):
        summary = str(summaries["summary"].iloc[i])
        print(
            f"  Posting {i} | {titles.iloc[i]}\n"
            f"    experience: {classified['predicted_experience_level'].iloc[i]}\n"
            f"    salary    : {salaries['predicted_salary'].iloc[i]:,.0f}\n"
            f"    cluster   : {int(clusters['cluster'].iloc[i])} "
            f"({clusters['cluster_label'].iloc[i]})\n"
            f"    anomaly   : {int(anomalies['anomaly_score'].iloc[i])}/4\n"
            f"    summary   : {summary[:100]}{'...' if len(summary) > 100 else ''}"
        )
