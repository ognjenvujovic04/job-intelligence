import logging

from .pipeline import run_default_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

SUPPORTED_MODELS = [
    "lightgbm", "xgboost", "catboost",
    "sklearn_logreg", "sklearn_mlp", "tabnet",
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a tracked classification experiment."
    )
    parser.add_argument(
        "--model-type", type=str, default="lightgbm",
        choices=SUPPORTED_MODELS,
        help="Which model to train (default: lightgbm).",
    )
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="Descriptive run name for the MLflow UI.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Override learning_rate hyperparameter.",
    )
    parser.add_argument(
        "--num-leaves", type=int, default=None,
        help="Override num_leaves hyperparameter.",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=None,
        help="Override n_estimators hyperparameter.",
    )
    parser.add_argument(
        "--register", type=str, default=None,
        help="Register the model under this name in the Model Registry.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all supported models sequentially.",
    )
    args = parser.parse_args()

    cli_params = {}
    if args.learning_rate is not None:
        cli_params["learning_rate"] = args.learning_rate
    if args.num_leaves is not None:
        cli_params["num_leaves"] = args.num_leaves
    if args.n_estimators is not None:
        cli_params["n_estimators"] = args.n_estimators

    if args.all:
        for mt in SUPPORTED_MODELS:
            print(f"\n{'='*60}")
            print(f"  Running: {mt}")
            print(f"{'='*60}")
            try:
                result = run_default_experiment(
                    model_type=mt,
                    params=cli_params or None,
                    run_name=args.run_name or mt,
                    tags={"source": "cli", "batch": "all_models"},
                )
                print(f"Run ID:      {result['run_id']}")
                print(f"Accuracy:    {result['metrics']['accuracy']:.4f}")
                print(f"F1 (macro):  {result['metrics']['f1_macro']:.4f}")
                print(f"MAE:         {result['metrics']['mae']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                logging.getLogger(__name__).exception(f"Failed to run {mt}")
    else:
        result = run_default_experiment(
            model_type=args.model_type,
            params=cli_params or None,
            run_name=args.run_name,
            register_model_name=args.register,
            tags={"source": "cli"},
        )
        print(f"\nRun ID: {result['run_id']}")
        print(f"Accuracy:    {result['metrics']['accuracy']:.4f}")
        print(f"F1 (macro):  {result['metrics']['f1_macro']:.4f}")
        print(f"MAE:         {result['metrics']['mae']:.4f}")
