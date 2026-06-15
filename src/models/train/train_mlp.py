"""
MLP Multiclass Classifier
================================
Trains a sklearn MLPClassifier to predict experience_level_ord (6 classes).

Designed to be imported from a notebook:
    from src.models.train.train_mlp import train_mlp, get_feature_importance

Unlike tree-based models, MLP cannot handle NaN values or
unscaled features. This module applies median imputation and
standard scaling internally via a Pipeline, so raw feature
matrices can be passed in directly.

Feature importance is computed via permutation importance
since MLP has no built-in split/gain importance.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Default tunable hyperparameters (safe to override via `params`)
DEFAULT_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "tanh",
    "solver": "adam",
    "alpha": 1e-2,
    "batch_size": 256,
    "learning_rate": "adaptive",
    "learning_rate_init": 3e-3,
    "max_iter": 300,
    "early_stopping": True,
    "validation_fraction": 0.13,
    "n_iter_no_change": 20,
}


# =========================================================
# TRAINING
# =========================================================

def train_mlp(X_train, y_train, val_size=0.15, params=None):
    """
    Train an MLP multiclass classifier with early stopping.

    Preprocessing (imputation + scaling) is wrapped inside a
    sklearn Pipeline so the same transforms apply at predict time.

    sklearn's built-in early_stopping splits a fraction of the
    training data internally to monitor validation loss. We first
    carve out an explicit validation set for post-training evaluation,
    then let the MLP handle its own early-stopping split from the
    remaining training data.

    Class imbalance is handled via sample_weight, computed using
    sklearn's 'balanced' strategy (equivalent to class_weight='balanced'
    in tree-based models).

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (integer-encoded).
        val_size (float): Fraction of training data held out for
            post-training validation reporting.
        params (dict, optional): Hyperparameter overrides merged on top of
            `DEFAULT_PARAMS` (shallow merge).

    Returns:
        sklearn.pipeline.Pipeline: Fitted pipeline (imputer + scaler + MLP).
    """
    X_trn, X_val, y_trn, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"MLP train/val split: train={len(X_trn)}, val={len(X_val)}"
    )

    # Compute sample weights to handle class imbalance
    sample_weights = compute_sample_weight("balanced", y_trn)

    # Preprocess before fitting so we can pass sample_weight
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_trn_processed = scaler.fit_transform(imputer.fit_transform(X_trn))
    X_val_processed = scaler.transform(imputer.transform(X_val))

    # Merge provided params on top of defaults (backward compatible)
    merged = {**DEFAULT_PARAMS, **(params or {})}

    mlp = MLPClassifier(
        random_state=RANDOM_STATE,
        verbose=True,
        **merged,
    )

    logger.info("Training MLP classifier...")
    logger.info(
        f"  Architecture: {mlp.hidden_layer_sizes} | "
        f"alpha={mlp.alpha} | batch_size={mlp.batch_size}"
    )

    # Suppress ConvergenceWarning if early stopping kicks in
    # before max_iter (that is expected, not an error)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        mlp.fit(X_trn_processed, y_trn, sample_weight=sample_weights)

    logger.info(
        f"Training finished at iteration {mlp.n_iter_} / {mlp.max_iter} | "
        f"best validation score: {mlp.best_validation_score_:.4f}"
    )

    # Report accuracy on our explicit held-out set
    val_acc = mlp.score(X_val_processed, y_val)
    logger.info(f"Held-out validation accuracy: {val_acc:.4f}")

    # Package into a pipeline for clean predict() calls
    fitted_pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("mlp", mlp),
    ])

    return fitted_pipeline


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(pipeline, X_test, y_test, feature_names, top_n=15):
    """
    Compute top N features by permutation importance.

    MLP has no built-in feature importance. Permutation importance
    measures the drop in accuracy when a single feature is randomly
    shuffled, giving a model-agnostic importance score.

    Note: this is slower than tree-based importance because it
    requires multiple re-evaluations. n_repeats controls the
    trade-off between speed and stability.

    Parameters:
        pipeline (Pipeline): Fitted MLP pipeline.
        X_test (pd.DataFrame): Test features (raw, before imputation/scaling).
        y_test (pd.Series): Test target.
        feature_names (list[str]): Feature column names.
        top_n (int): Number of top features to return.

    Returns:
        pd.Series: Mean permutation importances sorted descending, length top_n.
    """
    logger.info("Computing permutation importance (this may take a moment)...")

    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="accuracy",
    )

    importance = pd.Series(
        result.importances_mean,
        index=feature_names,
    ).sort_values(ascending=False)

    return importance.head(top_n)


# =========================================================
# STANDALONE EXECUTION
# =========================================================

if __name__ == "__main__":
    from sklearn.metrics import classification_report

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    TRAIN_PATH = "../../../data/processed/feature_matrix_train.csv"
    TEST_PATH = "../../../data/processed/feature_matrix_test.csv"
    TARGET = "experience_level_ord"

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    df_train = df_train.dropna(subset=[TARGET])
    df_test = df_test.dropna(subset=[TARGET])

    y_train = df_train[TARGET].astype(int)
    y_test = df_test[TARGET].astype(int)
    X_train = df_train.drop(columns=[TARGET])
    X_test = df_test.drop(columns=[TARGET])

    pipeline = train_mlp(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    imp = get_feature_importance(pipeline, X_test, y_test, X_train.columns, top_n=15)
    print("\n=== Top 15 Features (permutation importance) ===")
    for feat, val in imp.items():
        print(f"  {feat:<35s} {val:>8.4f}")