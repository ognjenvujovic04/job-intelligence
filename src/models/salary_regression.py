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
    Load the exported salary regressor, caching it module-wide.

    Loads from the local ``models/salary`` folder (written by
    scripts/export_models.py) via the pyfunc flavor so the model is served
    regardless of the underlying framework -- no MLflow tracking DB is consulted
    on the serving path.
    """
    global _MODEL, RUN_ID
    if _MODEL is not None:
        return _MODEL

    import mlflow.pyfunc

    from src.models.tracking.config import SALARY_RUN_ID, served_model_path

    # Re-exported as the module's public name (introspection / back-compat).
    RUN_ID = SALARY_RUN_ID

    model_path = served_model_path("salary")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Exported salary regressor not found at '{model_path}'. "
            "Run `python -m scripts.export_models` first."
        )

    logger.info("Loading salary regression model from %s", model_path)
    _MODEL = mlflow.pyfunc.load_model(model_path)
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
