"""
Inference entry point for the experience-level classifier.

This is the model-side mirror of `src/data/run_pipeline.py::prepare_data`:
`prepare_data` turns raw postings into a feature matrix, and
`run_classification` turns that feature matrix into predicted experience-level
labels. The two compose:

    run_classification(prepare_data(raw_df))

The model is the Optuna-tuned LightGBM logged to MLflow under run
`4098554e105542e8ae5e16454c7b21fe` (experiment "experience-level-classification").
No model is fitted here -- we reuse the persisted run.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# Run that logged the final Optuna-tuned LightGBM classifier. Under the MLflow
# 3.x logged-model layout the fitted model lives in its own logged-model folder,
# so we resolve this run to its logged-model id and load `models:/<model_id>`
# (the `runs:/<RUN_ID>/model` path does not resolve for this layout).
RUN_ID = "4098554e105542e8ae5e16454c7b21fe"
EXPERIMENT_NAME = "experience-level-classification"

# Ordinal -> human-readable experience level (from notebook 3.1).
EXPERIENCE_LABELS = {
    0: "Internship",
    1: "Entry level",
    2: "Associate",
    3: "Mid-Senior level",
    4: "Director",
    5: "Executive",
}

# Repo-root-anchored absolute path so this works regardless of cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FEATURE_COLUMNS_JSON = os.path.join(
    _REPO_ROOT, "data", "precomputed", "feature_columns.json"
)

# Columns present in the training feature matrix that are not model inputs.
_NON_FEATURE_COLS = ("job_id", "experience_level_ord")

# Cache the loaded model (≈130 MB) so repeated calls don't reload it.
_MODEL = None


def _load_feature_columns():
    """
    Return the model's input feature names, in training order.

    Read from the persisted `feature_columns.json` (the authoritative training
    schema) and drop the non-feature columns. Selecting by these original
    (spaced) names is required because LightGBM sanitizes its stored
    `feature_name_` (e.g. spaces -> underscores), so they no longer match the
    column names emitted by `prepare_data`.

    Raises:
        FileNotFoundError: when the persisted schema is missing (run the
            training pipeline first to generate it).
    """
    if not os.path.exists(FEATURE_COLUMNS_JSON):
        raise FileNotFoundError(
            f"Feature schema not found at '{FEATURE_COLUMNS_JSON}'. "
            "Run train_pipeline (src/data/run_pipeline.py) first to generate it."
        )
    with open(FEATURE_COLUMNS_JSON) as f:
        columns = json.load(f)
    return [c for c in columns if c not in _NON_FEATURE_COLS]


def _load_model():
    """
    Load the logged LightGBM classifier for RUN_ID, caching it module-wide.

    Resolves RUN_ID to its logged-model id via the MLflow client (rather than
    relying on a `runs:/.../model` artifact path, which does not exist under the
    MLflow 3.x logged-model layout), then loads `models:/<model_id>`.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import mlflow
    from mlflow import MlflowClient

    from src.models.tracking.config import configure_mlflow

    configure_mlflow(experiment_name=EXPERIMENT_NAME)

    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{EXPERIMENT_NAME}' not found. "
            "Has the classifier been trained and logged?"
        )

    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id='{RUN_ID}'",
    )
    if not logged_models:
        raise RuntimeError(
            f"No logged model found for run '{RUN_ID}' in experiment "
            f"'{EXPERIMENT_NAME}'."
        )

    model_uri = f"models:/{logged_models[0].model_id}"
    logger.info(f"Loading classifier from {model_uri}")
    _MODEL = mlflow.lightgbm.load_model(model_uri)
    return _MODEL


def run_classification(df, verbose=True):
    """
    Predict experience level for a prepared feature matrix.

    Parameters:
        df (pd.DataFrame): feature matrix produced by
            src.data.run_pipeline.prepare_data (must contain the model's
            feature columns; extra columns such as job_id are preserved).
        verbose (bool): emit INFO logging.

    Returns:
        pd.DataFrame: a copy of df with two appended columns:
            'predicted_experience_level_ord' (int 0-5) and
            'predicted_experience_level' (str label).

    Raises:
        ValueError: when df is missing required feature columns.
        FileNotFoundError: when the persisted feature schema is missing.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info(f"Classifying {len(df):,} rows")
        logger.info("=" * 60)

    features = _load_feature_columns()

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature column(s): "
            f"{missing}. Pass the output of prepare_data "
            "(src/data/run_pipeline.py)."
        )

    model = _load_model()

    # Predict on ordered values (numpy) so LightGBM matches features by position,
    # sidestepping the spaced-vs-sanitized feature-name mismatch.
    preds = model.predict(df[features].values)

    out = df.copy()
    out["predicted_experience_level_ord"] = preds.astype(int)
    out["predicted_experience_level"] = out["predicted_experience_level_ord"].map(
        EXPERIENCE_LABELS
    )

    if verbose:
        logger.info("Classification complete")
        logger.info(
            "Prediction distribution:\n"
            f"{out['predicted_experience_level'].value_counts()}"
        )

    return out


if __name__ == "__main__":
    # Smoke test: take the first 50 real postings with a known experience level,
    # run them through prepare_data -> run_classification, then report metrics
    # (true experience_level_ord vs predicted) via src/utils/evaluation.py.
    import sys

    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
    )

    N_ROWS = 50
    EXPERIENCE_COL = "formatted_experience_level"

    postings_csv = os.path.join(_REPO_ROOT, "data", "raw", "postings.csv")

    # Put the repo root on sys.path so the `src.*` package imports below resolve
    # when this file is run directly (python src/models/classification.py).
    sys.path.insert(0, _REPO_ROOT)

    # Keep the first N_ROWS rows that have a non-null experience level so we have
    # ground truth to score against.
    df_raw = pd.read_csv(postings_csv)
    df_raw = df_raw[df_raw[EXPERIENCE_COL].notna()].head(N_ROWS).reset_index(drop=True)
    print(f"Loaded {len(df_raw):,} postings with known experience level")

    # prepare_data is cwd-independent (repo-root-anchored paths + package imports),
    # so no chdir/path juggling is needed.
    from src.data.run_pipeline import prepare_data
    from src.utils.evaluation import compute_metrics

    feature_matrix = prepare_data(df_raw, verbose=True)

    result = run_classification(feature_matrix, verbose=True)

    # Score predictions against the true ordinal (both live in `result`, so they
    # stay aligned even if prepare_data dropped any rows). Guard against any rows
    # whose experience level failed to encode.
    scored = result.dropna(subset=["experience_level_ord"])
    y_true = scored["experience_level_ord"].astype(int)
    y_pred = scored["predicted_experience_level_ord"].astype(int)

    print(f"\nMetrics on {len(scored):,} rows:")
    metrics = compute_metrics(y_true, y_pred)
    for key in ("accuracy", "mae", "f1_macro", "f1_weighted"):
        print(f"  {key:12s}: {metrics[key]:.4f}")
