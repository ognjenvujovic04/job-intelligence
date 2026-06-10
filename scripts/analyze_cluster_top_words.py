"""
Analyze top-words artifacts from a clustering MLflow run.

Usage:
    python scripts/analyze_cluster_top_words.py <RUN_ID> [--column title|description]

Prints words that appear in more than one cluster, ranked by how many
clusters they appear in (descending), then by total word count.
"""

import argparse
import os
import sys

import mlflow
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from models.tracking.config import configure_mlflow


def load_top_words(run_id: str, column: str) -> pd.DataFrame:
    artifact_dir = mlflow.artifacts.download_artifacts(run_id=run_id)
    path = os.path.join(artifact_dir, f"top_words_{column}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artifact 'top_words_{column}.csv' not found in run {run_id}.\n"
            f"  Looked in: {artifact_dir}\n"
            f"  Available: {os.listdir(artifact_dir)}"
        )
    return pd.read_csv(path)


def analyze(df: pd.DataFrame, min_clusters: int = 2) -> pd.DataFrame:
    """Return words that appear in >= min_clusters clusters, sorted by cluster count desc."""
    agg = (
        df.groupby("word")
        .agg(cluster_count=("cluster", "nunique"), total_count=("count", "sum"))
        .reset_index()
    )
    shared = agg[agg["cluster_count"] >= min_clusters].sort_values(
        ["cluster_count", "total_count"], ascending=False
    )
    return shared


def print_results(shared: pd.DataFrame, df_raw: pd.DataFrame, column: str, run_id: str) -> None:
    n_clusters = df_raw["cluster"].nunique()
    print(f"\nRun : {run_id}")
    print(f"Column  : {column}")
    print(f"Clusters: {n_clusters}")
    print(f"Words in >1 cluster: {len(shared)}\n")

    print(f"{'Word':<25} {'Clusters':>8}  {'Total count':>12}  Appears in")
    print("-" * 70)

    for _, row in shared.iterrows():
        clusters_with_word = sorted(
            df_raw.loc[df_raw["word"] == row["word"], "cluster"].tolist()
        )
        cluster_str = ", ".join(map(str, clusters_with_word))
        print(f"{row['word']:<25} {int(row['cluster_count']):>8}  {int(row['total_count']):>12}  [{cluster_str}]")


def main():
    parser = argparse.ArgumentParser(description="Analyze shared top words across clusters.")
    parser.add_argument("run_id", help="MLflow run ID")
    parser.add_argument(
        "--column",
        choices=["title", "description"],
        default="title",
        help="Which word-cloud column to inspect (default: title)",
    )
    args = parser.parse_args()

    configure_mlflow(experiment_name="clustering-analysis")

    df = load_top_words(args.run_id, args.column)
    shared = analyze(df)
    print_results(shared, df, args.column, args.run_id)


if __name__ == "__main__":
    main()
