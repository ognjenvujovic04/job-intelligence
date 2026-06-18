"""
Small smoke test: score the 5 synthetic job postings with the trained
anomaly-detection ensemble and print each posting's per-model flags + score.

The logged ensemble (IsolationForest + COPOD + Autoencoder + VAE, see notebook
`3.4`) was trained on v7 by `src.models.train.train_anomaly`. It maps a raw
feature matrix straight to a 0-4 anomaly score, so we only need to:

  1. turn the raw synthetic postings into a feature matrix via the inference
     data-prep pipeline (`prepare_data`, reusing persisted training artifacts);
  2. run them through `run_anomaly_detection`, which reloads the persisted
     MLflow run (`anomaly_detection.RUN_ID`) and reapplies it.

No model is fitted here — we reuse the persisted run.

Run from anywhere:
    python tests/test_anomaly_synthetic.py
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_CSV = os.path.join(REPO_ROOT, "data", "raw", "synthetic_postings.csv")


def main():
    sys.path.insert(0, REPO_ROOT)

    from src.data.run_pipeline import prepare_data
    from src.models.serve.anomaly_detection import run_anomaly_detection

    df = pd.read_csv(SYNTHETIC_CSV)
    print(f"Loaded {len(df):,} synthetic postings from {SYNTHETIC_CSV}")

    # Raw postings -> model-ready feature matrix (reuses training artifacts).
    feature_matrix = prepare_data(df, verbose=True)
    print(f"\nFeature matrix shape: {feature_matrix.shape}")

    # Feature matrix -> per-model flags + 0-4 anomaly score (reuses the
    # persisted MLflow run, no fitting).
    scored = run_anomaly_detection(feature_matrix, verbose=True)

    titles = df["title"] if "title" in df.columns else pd.Series([""] * len(df))
    print("\nAnomaly scores:")
    for i, (title, score) in enumerate(zip(titles, scored["anomaly_score"])):
        print(f"  Posting {i} | score {int(score)}/4 | {title}")


if __name__ == "__main__":
    main()
