"""
Small smoke test: load the already-trained clustering model from MLflow and
print which cluster each of the 5 synthetic job postings falls into.

The logged model is the full pipeline (median imputer + StandardScaler +
K-Means k=26) selected in notebook `3.2` / labeled in `4.0`. It maps a raw
feature matrix straight to a cluster id, so we only need to:

  1. turn the raw synthetic postings into a feature matrix via the inference
     data-prep pipeline (`prepare_data`, reusing persisted training artifacts);
  2. run them through `predict_clusters`, which reloads the persisted MLflow run
     (clustering.CLUSTERING_RUN_ID) and reapplies it.

No model is fitted here — we reuse the persisted run.

Run from anywhere:
    python tests/test_clustering_synthetic.py
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_CSV = os.path.join(REPO_ROOT, "data", "raw", "synthetic_postings.csv")


def main():
    sys.path.insert(0, REPO_ROOT)

    from src.data.run_pipeline import prepare_data
    from src.models.clustering import predict_clusters

    df = pd.read_csv(SYNTHETIC_CSV)
    print(f"Loaded {len(df):,} synthetic postings from {SYNTHETIC_CSV}")

    # Raw postings -> model-ready feature matrix (reuses training artifacts).
    feature_matrix = prepare_data(df, verbose=True)
    print(f"\nFeature matrix shape: {feature_matrix.shape}")

    # Feature matrix -> cluster id per posting (reloads the persisted MLflow run,
    # no fitting).
    clustered = predict_clusters(feature_matrix, verbose=True)

    titles = df["title"] if "title" in df.columns else pd.Series([""] * len(df))
    print("\nCluster assignments:")
    for i, (title, cluster) in enumerate(zip(titles, clustered["cluster"])):
        print(f"  Posting {i} | cluster {int(cluster):2d} | {title}")


if __name__ == "__main__":
    main()
