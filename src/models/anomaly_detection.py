"""
Inference entry point for the anomaly-detection ensemble.

This is the model-side mirror of `src/data/run_pipeline.py::prepare_data`:
`prepare_data` turns raw postings into a feature matrix, and
`run_anomaly_detection` turns that feature matrix into a per-row anomaly score.
The two compose:

    run_anomaly_detection(prepare_data(raw_df))

The four detectors (IsolationForest, COPOD, Autoencoder, VAE) and their fitted
preprocessing were trained and logged by
`src/models/train/train_anomaly.py::train_and_log`. The anomaly score is the
number of models (0-4) that flag a row as an outlier. No model is fitted here —
we reload the persisted run and reapply it.

Run `python -m src.models.train.train_anomaly` first, then paste the printed
run id into `RUN_ID` below.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Run that logged the anomaly ensemble (bundle.joblib + autoencoder.pt + vae.pt
# under the "anomaly_artifacts/" artifact path), and the experiment it lives in.
# Both are pinned centrally in tracking/config.py (ANOMALY_RUN_ID /
# DEFAULT_ANOMALY_EXPERIMENT_NAME) and re-exported here; imported lazily in
# _load_artifacts so this module still runs as a script, where the repo root
# isn't on sys.path until __main__.
RUN_ID = None
EXPERIMENT_NAME = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Output column names appended to the returned DataFrame.
_FLAG_COLS = {
    "isolation_forest": "anomaly_isolation_forest",
    "copod": "anomaly_copod",
    "autoencoder": "anomaly_autoencoder",
    "vae": "anomaly_vae",
}
_SCORE_COL = "anomaly_score"

# Cache the loaded artifacts (models + preprocessing) so repeated calls don't
# re-download/rebuild them.
_ARTIFACTS = None


def _load_artifacts():
    """Load and reconstruct the exported ensemble from disk, caching it.

    Returns a dict with the fitted preprocessing, the two sklearn/pyod models,
    the two reconstructed PyTorch models (CPU, eval mode), and their thresholds.
    """
    global _ARTIFACTS, RUN_ID, EXPERIMENT_NAME
    if _ARTIFACTS is not None:
        return _ARTIFACTS

    import joblib
    import torch

    from src.models.tracking.config import (
        ANOMALY_RUN_ID,
        DEFAULT_ANOMALY_EXPERIMENT_NAME,
        served_model_path,
    )
    from src.models.train.train_anomaly import (
        AE_WEIGHTS_NAME,
        BUNDLE_NAME,
        VAE_WEIGHTS_NAME,
        Autoencoder,
        VAE,
    )

    # Re-exported as the module's public names (introspection / back-compat).
    RUN_ID = ANOMALY_RUN_ID
    EXPERIMENT_NAME = DEFAULT_ANOMALY_EXPERIMENT_NAME

    local_dir = served_model_path("anomaly")
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(
            f"Exported anomaly artifacts not found at '{local_dir}'. "
            "Run `python -m scripts.export_models` first."
        )
    logger.info("Loading anomaly artifacts from %s", local_dir)

    bundle = joblib.load(os.path.join(local_dir, BUNDLE_NAME))

    device = torch.device("cpu")
    ae_model = Autoencoder(bundle["input_dim"], latent_dim=bundle["latent_dim"])
    ae_model.load_state_dict(torch.load(os.path.join(local_dir, AE_WEIGHTS_NAME), map_location=device))
    ae_model.to(device).eval()

    vae_model = VAE(bundle["input_dim"], latent_dim=bundle["latent_dim"])
    vae_model.load_state_dict(torch.load(os.path.join(local_dir, VAE_WEIGHTS_NAME), map_location=device))
    vae_model.to(device).eval()

    _ARTIFACTS = {
        "feature_cols": bundle["feature_cols"],
        "medians": bundle["medians"],
        "scaler": bundle["scaler"],
        "iso_model": bundle["iso_model"],
        "copod_model": bundle["copod_model"],
        "ae_model": ae_model,
        "vae_model": vae_model,
        "ae_threshold": bundle["ae_threshold"],
        "vae_threshold": bundle["vae_threshold"],
        "l_samples": bundle.get("l_samples", 10),
        "device": device,
    }
    return _ARTIFACTS


def run_anomaly_detection(df, verbose=True):
    """
    Score a prepared feature matrix with the anomaly-detection ensemble.

    Parameters:
        df (pd.DataFrame): feature matrix produced by
            src.data.run_pipeline.prepare_data (must contain the trained feature
            columns; extra columns such as job_id are preserved). Missing
            normalized_salary is filled with the persisted train median.
        verbose (bool): emit INFO logging.

    Returns:
        pd.DataFrame: a copy of df with five appended columns — the four per-model
            binary flags ('anomaly_isolation_forest', 'anomaly_copod',
            'anomaly_autoencoder', 'anomaly_vae') and 'anomaly_score' (int 0-4,
            the number of models flagging the row).

    Raises:
        ValueError: when df is missing required feature columns.
        RuntimeError: when RUN_ID is unset or the run/artifacts are missing.
    """
    from src.models.train.train_anomaly import compute_flags, apply_preprocessing

    art = _load_artifacts()
    feature_cols = art["feature_cols"]

    if verbose:
        logger.info("=" * 60)
        logger.info("Scoring %s rows for anomalies", f"{len(df):,}")
        logger.info("=" * 60)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature column(s): "
            f"{missing}. Pass the output of prepare_data "
            "(src/data/run_pipeline.py)."
        )

    X_scaled = apply_preprocessing(df, feature_cols, art["medians"], art["scaler"])

    flags = compute_flags(
        art["iso_model"], art["copod_model"], art["ae_model"], art["vae_model"],
        art["ae_threshold"], art["vae_threshold"], X_scaled,
        device=art["device"], l_samples=art["l_samples"],
    )

    out = df.copy()
    for src_col, out_col in _FLAG_COLS.items():
        out[out_col] = flags[src_col].values
    out[_SCORE_COL] = flags["anomaly_score"].values

    if verbose:
        logger.info("Anomaly scoring complete")
        logger.info(
            "Score distribution (0-4):\n%s",
            out[_SCORE_COL].value_counts().sort_index().to_string(),
        )

    return out


if __name__ == "__main__":
    # Smoke test: score the v7 test split and report the 0-4 distribution
    # (should roughly match notebook 3.4: ~98.4% at score 0).
    import sys

    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
    )

    sys.path.insert(0, _REPO_ROOT)

    v7_test = os.path.join(_REPO_ROOT, "data", "processed", "v7", "feature_matrix_test.csv")
    df_test = pd.read_csv(v7_test)
    print(f"Loaded {len(df_test):,} rows from {v7_test}")

    result = run_anomaly_detection(df_test, verbose=True)

    print("\nAnomaly score distribution (0-4):")
    dist = result[_SCORE_COL].value_counts().sort_index()
    for score, n in dist.items():
        print(f"  {score} model(s): {n:>6,} ({n / len(result) * 100:.2f}%)")
    print(f"\nConsensus (4/4): {(result[_SCORE_COL] == 4).sum():,}")
    print(f">= 2 models    : {(result[_SCORE_COL] >= 2).sum():,}")
