"""
Inference entry point for the salary regression model.

This is the model-side mirror of `src/data/run_pipeline.py::prepare_data`:
`prepare_data` turns raw postings into a feature matrix, and `predict_salary`
turns that feature matrix into a predicted normalized salary. The two compose:

    predict_salary(prepare_data(raw_df))

The model is the best salary regressor logged to MLflow under run
`SALARY_RUN_ID` (experiment "salary-regression"). No model is fitted here -- we
reload the persisted run and reapply it. The same loader backs the salary
imputation in `src/data/prepare_clustering_dataset.py`, so there is one
salary-inference path, not two.

Run `python -m src.models.salary_regression` for a small smoke test.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Run that logged the salary regressor. Pinned centrally in tracking/config.py
# (SALARY_RUN_ID) and re-exported here as the module's public name. Imported
# lazily in _load_model so this module still runs as a script
# (python src/models/salary_regression.py), where the repo root isn't on
# sys.path until __main__.
RUN_ID = None

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Target column, plus columns present in the feature matrix that are not model
# inputs (kept in sync with prepare_clustering_dataset._NON_FEATURE_COLS).
TARGET = "normalized_salary"
_NON_FEATURE_COLS = {TARGET, "job_id", "salary_is_missing"}

# Cache the loaded model so repeated calls don't reload it.
_MODEL = None


def _load_model():
    """
    Load the logged salary regressor for RUN_ID, caching it module-wide.

    Resolves RUN_ID to a registered model version via `search_model_versions`
    and loads `models:/<name>/<version>`; falls back to `runs:/<RUN_ID>/model`
    when the run was not registered. Loaded with the pyfunc flavor so the model
    is served regardless of the underlying framework.
    """
    global _MODEL, RUN_ID
    if _MODEL is not None:
        return _MODEL

    import mlflow
    from mlflow.tracking import MlflowClient

    from src.models.tracking.config import (
        DEFAULT_REGRESSION_EXPERIMENT_NAME,
        SALARY_RUN_ID,
        configure_mlflow,
    )

    RUN_ID = SALARY_RUN_ID
    if not RUN_ID:
        raise RuntimeError(
            "SALARY_RUN_ID is not set in tracking/config.py. Train and log a "
            "salary regressor, then paste its run id into SALARY_RUN_ID."
        )

    configure_mlflow(experiment_name=DEFAULT_REGRESSION_EXPERIMENT_NAME)

    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{RUN_ID}'")
    if versions:
        mv = versions[0]
        model_uri = f"models:/{mv.name}/{mv.version}"
    else:
        model_uri = f"runs:/{RUN_ID}/model"

    logger.info("Loading salary regression model from %s", model_uri)
    _MODEL = mlflow.pyfunc.load_model(model_uri)
    return _MODEL


def predict_salary(df, verbose=True):
    """
    Predict normalized salary for a prepared feature matrix.

    Parameters:
        df (pd.DataFrame): feature matrix produced by
            src.data.run_pipeline.prepare_data (must contain the model's
            feature columns; extra columns such as job_id are preserved).
        verbose (bool): emit INFO logging.

    Returns:
        pd.DataFrame: a copy of df with an appended 'predicted_salary' column.

    Raises:
        RuntimeError: when RUN_ID is unset or the run/artifacts are missing.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Predicting salary for %s rows", f"{len(df):,}")
        logger.info("=" * 60)

    model = _load_model()

    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_COLS]
    predictions = model.predict(df[feature_cols])

    out = df.copy()
    out["predicted_salary"] = predictions

    if verbose:
        logger.info("Salary prediction complete")
        logger.info(
            "Predicted salary summary:\n%s",
            out["predicted_salary"].describe().to_string(),
        )

    return out


if __name__ == "__main__":
    # Smoke test: take the first 50 real postings, run them through
    # prepare_data -> predict_salary, and print the predicted-salary summary.
    import sys

    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
    )

    N_ROWS = 50
    postings_csv = os.path.join(_REPO_ROOT, "data", "raw", "postings.csv")

    # Put the repo root on sys.path so the `src.*` package imports resolve when
    # this file is run directly (python src/models/salary_regression.py).
    sys.path.insert(0, _REPO_ROOT)

    from src.data.run_pipeline import prepare_data

    df_raw = pd.read_csv(postings_csv).head(N_ROWS).reset_index(drop=True)
    print(f"Loaded {len(df_raw):,} postings")

    feature_matrix = prepare_data(df_raw, verbose=True)
    result = predict_salary(feature_matrix, verbose=True)

    print(f"\nPredicted salary for {len(result):,} rows:")
    print(result["predicted_salary"].describe().to_string())
