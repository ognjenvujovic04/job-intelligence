"""
Small smoke test: load the already-trained clustering model from MLflow and
print which cluster each of the 5 synthetic job postings falls into.

The logged model is the full pipeline (median imputer + StandardScaler +
K-Means k=26) selected in notebook `3.2` / labeled in `4.0`. It maps a raw
feature matrix straight to a cluster id, so we only need to:

  1. turn the raw synthetic postings into a feature matrix via the inference
     data-prep pipeline (`prepare_data`, reusing persisted training artifacts);
  2. load the model from MLflow and call predict.

No model is fitted here — we reuse the persisted run.

Run from anywhere:
    python tests/test_clustering_synthetic.py
"""

import os
import sys

import mlflow
import pandas as pd
from mlflow.models import Model

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO_ROOT, "src")
SRC_DATA = os.path.join(SRC, "data")
SYNTHETIC_CSV = os.path.join(REPO_ROOT, "data", "raw", "synthetic_postings.csv")

# Run that logged the final K-Means (k=26) pipeline in the clustering-analysis
# experiment. The fitted model lives in its own logged-model folder under this
# run, so `runs:/<RUN_ID>/model` resolves to the serialized pipeline.
RUN_ID = "7a7bdb80b722493991489242e2cd03bf"
MODEL_URI = f"runs:/{RUN_ID}/model"


def main():
    # Point MLflow at the repo-root tracking DB (absolute paths, cwd-independent).
    sys.path.insert(0, SRC)
    from models.tracking.config import configure_mlflow

    configure_mlflow(experiment_name="clustering-analysis")

    # Load the trained pipeline (imputer + scaler + K-Means). We load the raw
    # sklearn flavor rather than pyfunc so that the prediction is not subject to
    # strict dtype enforcement — `prepare_data` emits the one-hot domain columns
    # as int64 while the logged signature recorded them as bool, which the
    # scaler/K-Means handle interchangeably. The signature is still used (below)
    # to pick the right feature columns in the order the model expects.
    print(f"Loading clustering model from {MODEL_URI}")
    model = mlflow.sklearn.load_model(MODEL_URI)
    expected_cols = Model.load(MODEL_URI).get_input_schema().input_names()

    # Load the raw postings up front (absolute path), before changing cwd.
    df = pd.read_csv(SYNTHETIC_CSV)
    print(f"Loaded {len(df):,} synthetic postings from {SYNTHETIC_CSV}")

    # run_pipeline.py uses bare imports (e.g. `from feature_engineering import ...`)
    # and relative artifact paths ("../../data/precomputed/..."), so it must be
    # imported with src/data on sys.path and executed with src/data as cwd.
    sys.path.insert(0, SRC_DATA)
    os.chdir(SRC_DATA)
    from run_pipeline import prepare_data

    # Raw postings -> model-ready feature matrix (reuses training artifacts).
    feature_matrix = prepare_data(df, verbose=True)
    print(f"\nFeature matrix shape: {feature_matrix.shape}")

    # Select the model's feature columns in the expected order (drops job_id and
    # the leakage cols that aren't part of the model input), then predict.
    clusters = model.predict(feature_matrix[expected_cols])

    # Report: one line per posting, with its title for readability.
    titles = df["title"] if "title" in df.columns else pd.Series([""] * len(df))
    print("\nCluster assignments:")
    for i, (title, cluster) in enumerate(zip(titles, clusters)):
        print(f"  Posting {i} | cluster {int(cluster):2d} | {title}")


if __name__ == "__main__":
    main()
