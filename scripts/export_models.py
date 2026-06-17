"""
Export the four served MLflow runs into a flat ``models/`` folder for serving.

This is the one-time promotion step that decouples the inference server from the
MLflow tracking DB. It reads the pinned RUN_IDs from ``src/models/tracking/config.py``
(the single source of "which run is served") and copies each run's model
artifacts into ``models/<track>/`` so the serving loaders can load them from a
local path -- no ``mlflow.db`` / ``mlruns/`` needed at run time.

Run it from the repo root, where ``mlflow.db`` and ``mlruns/`` live:

    python -m scripts.export_models                 # export all four tracks
    python -m scripts.export_models --models-dir ./models

Output layout (each folder is gitignored):

    models/classification/   MLmodel + LightGBM model         (mlflow.lightgbm)
    models/salary/           MLmodel + pyfunc model           (mlflow.pyfunc)
    models/clustering/       MLmodel + sklearn pipeline        (mlflow.sklearn)
    models/anomaly/          bundle.joblib + autoencoder.pt + vae.pt  (raw files)

After exporting, inspect ``models/salary/MLmodel`` to confirm the regressor's
underlying flavor -- the serving requirements must include whatever library it
needs (lightgbm / xgboost / catboost).
"""

import argparse
import os
import shutil
import sys

# Repo-root anchored so `python -m scripts.export_models` and a direct
# `python scripts/export_models.py` both resolve `import src...`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mlflow
from mlflow.tracking import MlflowClient

from src.models.tracking.config import (
    ANOMALY_RUN_ID,
    CLASSIFICATION_RUN_ID,
    CLUSTERING_RUN_ID,
    DEFAULT_ANOMALY_EXPERIMENT_NAME,
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_REGRESSION_EXPERIMENT_NAME,
    SALARY_RUN_ID,
    configure_mlflow,
)

DEFAULT_MODELS_DIR = os.path.join(_REPO_ROOT, "models")

# Mirrors src/models/train/train_anomaly.py: the anomaly run logs its ensemble
# under this artifact sub-path as raw files (not an MLflow flavor).
_ANOMALY_ARTIFACT_DIR = "anomaly_artifacts"


def _copy_into(src_local_path, dest):
    """Replace ``dest`` with a fresh copy of the downloaded artifact dir."""
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.copytree(src_local_path, dest)


def _export_classification(dest):
    """LightGBM classifier: resolve the MLflow 3.x logged-model id, then copy.

    Mirrors src/models/classification.py::_load_model -- the run has no
    ``runs:/.../model`` path under the logged-model layout, so we resolve the
    model_id via the client and download ``models:/<model_id>``.
    """
    configure_mlflow(experiment_name=DEFAULT_EXPERIMENT_NAME)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError(
            f"MLflow experiment '{DEFAULT_EXPERIMENT_NAME}' not found. "
            "Has the classifier been trained and logged?"
        )
    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"source_run_id='{CLASSIFICATION_RUN_ID}'",
    )
    if not logged_models:
        raise RuntimeError(
            f"No logged model found for run '{CLASSIFICATION_RUN_ID}' in "
            f"experiment '{DEFAULT_EXPERIMENT_NAME}'."
        )
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{logged_models[0].model_id}"
    )
    _copy_into(local, dest)


def _export_salary(dest):
    """Salary regressor (pyfunc). Mirrors salary_regression.py::_load_model:
    prefer a registered version, fall back to ``runs:/<id>/model``."""
    configure_mlflow(experiment_name=DEFAULT_REGRESSION_EXPERIMENT_NAME)
    client = MlflowClient()
    versions = client.search_model_versions(f"run_id='{SALARY_RUN_ID}'")
    if versions:
        mv = versions[0]
        uri = f"models:/{mv.name}/{mv.version}"
    else:
        uri = f"runs:/{SALARY_RUN_ID}/model"
    local = mlflow.artifacts.download_artifacts(artifact_uri=uri)
    _copy_into(local, dest)


def _export_clustering(dest):
    """K-Means pipeline (sklearn). Mirrors clustering.py::_load_clustering_model."""
    configure_mlflow(experiment_name="clustering-analysis")
    local = mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{CLUSTERING_RUN_ID}/model"
    )
    _copy_into(local, dest)


def _export_anomaly(dest):
    """Anomaly ensemble (raw joblib + PyTorch). Mirrors
    anomaly_detection.py::_load_artifacts -- downloads the whole artifact dir."""
    configure_mlflow(experiment_name=DEFAULT_ANOMALY_EXPERIMENT_NAME)
    local = mlflow.artifacts.download_artifacts(
        run_id=ANOMALY_RUN_ID, artifact_path=_ANOMALY_ARTIFACT_DIR
    )
    _copy_into(local, dest)


# (track name, run id, exporter). The run-id presence is checked up front so a
# single unset id is reported clearly rather than failing deep in mlflow.
_TRACKS = [
    ("classification", CLASSIFICATION_RUN_ID, _export_classification),
    ("salary", SALARY_RUN_ID, _export_salary),
    ("clustering", CLUSTERING_RUN_ID, _export_clustering),
    ("anomaly", ANOMALY_RUN_ID, _export_anomaly),
]


def export_models(models_dir, only=None):
    """Export each track's model into ``models_dir/<track>/``.

    Parameters:
        models_dir (str): destination root (created if missing).
        only (set[str] or None): if given, export just these track names.
    """
    os.makedirs(models_dir, exist_ok=True)
    for name, run_id, exporter in _TRACKS:
        if only and name not in only:
            continue
        if not run_id:
            raise RuntimeError(
                f"{name.upper()}_RUN_ID is not set in tracking/config.py -- "
                "train and log that track first."
            )
        dest = os.path.join(models_dir, name)
        print(f"[export] {name}: run {run_id} -> {dest}")
        exporter(dest)
        print(f"[export] {name}: done ({_summarize(dest)})")
    print(f"\nExported to {models_dir}")
    print(
        "Next: inspect models/salary/MLmodel to confirm the regressor flavor, "
        "then build the image."
    )


def _summarize(dest):
    """Short description of what landed in a track folder (for the log line)."""
    mlmodel = os.path.join(dest, "MLmodel")
    if os.path.exists(mlmodel):
        return "MLmodel + artifacts"
    return ", ".join(sorted(os.listdir(dest))) or "empty"


def main():
    parser = argparse.ArgumentParser(description="Export served models to a flat folder.")
    parser.add_argument(
        "--models-dir",
        default=DEFAULT_MODELS_DIR,
        help=f"Destination root (default: {DEFAULT_MODELS_DIR}).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[name for name, _, _ in _TRACKS],
        help="Export only these tracks (default: all).",
    )
    args = parser.parse_args()
    export_models(args.models_dir, only=set(args.only) if args.only else None)


if __name__ == "__main__":
    main()
