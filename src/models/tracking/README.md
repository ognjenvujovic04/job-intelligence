# MLflow Tracking

Wraps the training and evaluation pipeline with MLflow logging for the experience-level classification task.

## Package layout

```
src/models/tracking/
├── config.py           # configure_mlflow, DEFAULT_* constants
├── experiment.py       # run_experiment
├── logging_helpers.py  # internal _log_* helpers
├── pipeline.py         # run_default_experiment, _resolve_model_functions
├── runs.py             # compare_runs, get_best_run
└── __main__.py         # CLI entry point
```

## Importing

```python
# Preferred — import from the package
from src.models.tracking import (
    configure_mlflow,
    run_experiment,
    run_default_experiment,
    compare_runs,
    get_best_run,
)

# Legacy path still works (backward-compat shim)
from src.models.mlflow_tracking import run_experiment
```

## Key functions

### `configure_mlflow(tracking_uri, experiment_name)`
Sets the MLflow tracking URI and active experiment. Call once before training.

```python
configure_mlflow(
    tracking_uri="sqlite:///mlflow.db",
    experiment_name="experience-level-classification",
)
```

### `run_experiment(...)`
Core entry point. Trains, evaluates, and logs a single run. You supply the train/eval functions; it handles all MLflow lifecycle.

```python
from src.models.tracking import configure_mlflow, run_experiment
from src.models.train_lightgbm import train_lightgbm, get_feature_importance
from src.utils.evaluation import compute_metrics

configure_mlflow()

result = run_experiment(
    X_train=X_train, y_train=y_train,
    X_test=X_test,   y_test=y_test,
    train_fn=train_lightgbm,
    eval_fn=compute_metrics,
    feature_importance_fn=get_feature_importance,
    model_type="lightgbm",
    params={"num_leaves": 63, "learning_rate": 0.05},
    run_name="lgbm-baseline",
    tags={"author": "ov"},
)
# result = {"run_id": ..., "metrics": {...}, "model": ..., "params": {...}}
```

Supported `model_type` values: `lightgbm`, `xgboost`, `catboost`, `sklearn`, `tabnet`, `generic`.

### `run_default_experiment(model_type, params, ...)`
One-call convenience wrapper. Loads the processed CSVs, configures MLflow, and calls `run_experiment` automatically.

```python
from src.models.tracking import run_default_experiment

result = run_default_experiment(
    model_type="xgboost",
    params={"n_estimators": 300, "learning_rate": 0.05},
    run_name="xgb-v2",
    register_model_name="xgboost-exp-level-classifier",
)
```

### `compare_runs(experiment_name, metric_keys, top_n)`
Returns a DataFrame of recent runs with their metrics.

```python
from src.models.tracking import compare_runs

df = compare_runs(top_n=20, metric_keys=["accuracy", "f1_macro", "mae"])
print(df.sort_values("f1_macro", ascending=False))
```

### `get_best_run(metric, higher_is_better)`
Returns `{"run_id", "run_name", "model_type", "metric_value"}` for the top run.

```python
from src.models.tracking import get_best_run

best = get_best_run(metric="f1_macro")
print(best["run_id"], best["metric_value"])
```

## CLI

Run a single model:

```bash
python -m src.models.tracking --model-type lightgbm --run-name lgbm-v1
```

Run all supported models sequentially:

```bash
python -m src.models.tracking --all
```

Override hyperparameters:

```bash
python -m src.models.tracking --model-type xgboost \
    --learning-rate 0.03 --n-estimators 500 \
    --register xgboost-exp-level-classifier
```

Available flags:

| Flag | Description |
|---|---|
| `--model-type` | `lightgbm` \| `xgboost` \| `catboost` \| `sklearn_logreg` \| `sklearn_mlp` \| `tabnet` |
| `--run-name` | Display name in the MLflow UI |
| `--learning-rate` | Override `learning_rate` |
| `--num-leaves` | Override `num_leaves` (LightGBM) |
| `--n-estimators` | Override `n_estimators` |
| `--register` | Register model in the MLflow Model Registry under this name |
| `--all` | Train all model types in sequence |

## Viewing results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open `http://localhost:5000`.
