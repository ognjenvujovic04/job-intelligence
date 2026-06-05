import logging
import os

import mlflow

logger = logging.getLogger(__name__)

# Absolute path so the same DB is used regardless of working directory
# (running from notebooks/ vs. the project root both resolve to the same file)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_DB_PATH = os.path.join(_REPO_ROOT, "mlflow.db").replace("\\", "/")
DEFAULT_TRACKING_URI = f"sqlite:///{_DB_PATH}"
DEFAULT_EXPERIMENT_NAME = "experience-level-classification"
DEFAULT_TARGET = "experience_level_ord"

DEFAULT_REGRESSION_EXPERIMENT_NAME = "salary-regression"
DEFAULT_REGRESSION_TARGET = "normalized_salary"


def configure_mlflow(tracking_uri=None, experiment_name=None):
    """
    Set the MLflow tracking URI and active experiment.

    If the experiment does not exist yet it will be created automatically.

    Parameters:
        tracking_uri (str or None): File-system path or remote URI.
            Defaults to ``DEFAULT_TRACKING_URI``.
        experiment_name (str or None): Human-readable experiment name.
            Defaults to ``DEFAULT_EXPERIMENT_NAME``.

    Returns:
        str: The experiment ID for the active experiment.
    """
    uri = tracking_uri or DEFAULT_TRACKING_URI
    name = experiment_name or DEFAULT_EXPERIMENT_NAME

    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(name)

    experiment = mlflow.get_experiment_by_name(name)
    experiment_id = experiment.experiment_id

    logger.info(
        f"MLflow configured | uri={uri} | experiment='{name}' "
        f"(id={experiment_id})"
    )
    return experiment_id
