"""
CatBoost Multiclass Classifier
================================
Trains a CatBoost model to predict experience_level_ord (6 classes).

Designed to be imported from a notebook:
    from models.train_catboost import train_catboost, get_feature_importance

CatBoost handles NaN in numerical features natively, so no imputation
is needed. Boolean columns are cast to int to avoid dtype issues.
Class weights are computed manually (balanced) and passed via
class_weights, since CatBoost does not accept a 'balanced' string.
"""

import logging
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Default tunable hyperparameters (safe to override via `params`)
DEFAULT_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "MVS",
    "subsample": 0.8,
}

REGRESSOR_DEFAULT_PARAMS = {
    "iterations": 2000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "bootstrap_type": "MVS",
    "subsample": 0.8,
}


# =========================================================
# PREPROCESSING
# =========================================================

def _cast_bool_columns(X):
    """
    Cast boolean columns to int.

    CatBoost can be strict about dtypes and may reject bool columns
    depending on the version. Casting to int avoids this silently.

    Parameters:
        X (pd.DataFrame): Feature matrix (modified in place).

    Returns:
        pd.DataFrame
    """
    bool_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        X[bool_cols] = X[bool_cols].astype(int)
        logger.info(f"Cast {len(bool_cols)} bool columns to int")
    return X


# =========================================================
# CLASS WEIGHT COMPUTATION
# =========================================================

def _compute_class_weights(y):
    """
    Compute per-class weights inversely proportional to frequency.

    CatBoost expects a dict {class_label: weight}, not a sample-weight
    array like XGBoost.

    Parameters:
        y (pd.Series): Target labels.

    Returns:
        dict: {class_label: weight}.
    """
    class_counts = y.value_counts().sort_index()
    total = len(y)
    n_classes = y.nunique()

    return {
        cls: total / (n_classes * count)
        for cls, count in class_counts.items()
    }


# =========================================================
# TRAINING
# =========================================================

def train_catboost(X_train, y_train, val_size=0.15, params=None):
    """
    Train a CatBoost multiclass classifier with early stopping.

    A stratified validation split is carved from the training data
    to monitor MultiClass loss and stop when it stops improving.

    Boolean columns are automatically cast to int before training.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (integer-encoded).
        val_size (float): Fraction of training data held out for
            early-stopping validation.
        params (dict, optional): Hyperparameter overrides merged on top of
            `DEFAULT_PARAMS` (shallow merge).

    Returns:
        CatBoostClassifier: Fitted model.
    """
    X_train = _cast_bool_columns(X_train.copy())

    X_trn, X_val, y_trn, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"CatBoost train/val split: train={len(X_trn)}, val={len(X_val)}"
    )

    class_weights = _compute_class_weights(y_trn)
    # Merge provided params on top of defaults (backward compatible)
    merged_params = {**DEFAULT_PARAMS, **(params or {})}

    model = CatBoostClassifier(
        class_weights=class_weights,
        loss_function="MultiClass",
        eval_metric="MultiClass",
        auto_class_weights=None,
        random_seed=RANDOM_STATE,
        verbose=100,
        early_stopping_rounds=100,
        task_type="CPU",
        allow_writing_files=False,
        **merged_params,
    )

    train_pool = Pool(X_trn, label=y_trn)
    eval_pool = Pool(X_val, label=y_val)

    logger.info("Training CatBoost classifier...")
    model.fit(train_pool, eval_set=eval_pool)

    logger.info(f"Early stopping at iteration {model.get_best_iteration()}")
    return model


def train_catboost_regressor(X_train, y_train, val_size=0.15, params=None):
    """
    Train a CatBoost regression model with early stopping.

    Boolean columns are automatically cast to int before training.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Continuous target (e.g. normalized_salary).
        val_size (float): Fraction held out for early-stopping validation.
        params (dict, optional): Hyperparameter overrides merged on top of
            `REGRESSOR_DEFAULT_PARAMS` (shallow merge).

    Returns:
        CatBoostRegressor: Fitted model.
    """
    X_train = _cast_bool_columns(X_train.copy())

    X_trn, X_val, y_trn, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"CatBoost regressor train/val split: train={len(X_trn)}, val={len(X_val)}"
    )

    merged_params = {**REGRESSOR_DEFAULT_PARAMS, **(params or {})}

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="MAE",
        random_seed=RANDOM_STATE,
        verbose=100,
        early_stopping_rounds=100,
        task_type="CPU",
        allow_writing_files=False,
        **merged_params,
    )

    train_pool = Pool(X_trn, label=y_trn)
    eval_pool = Pool(X_val, label=y_val)

    logger.info("Training CatBoost regressor...")
    model.fit(train_pool, eval_set=eval_pool)

    logger.info(f"Early stopping at iteration {model.get_best_iteration()}")
    return model


# =========================================================
# PREDICTION HELPERS
# =========================================================

def predict(model, X_test):
    """
    Predict class labels, handling CatBoost's 2D output.

    CatBoost's predict() returns a 2D array for multiclass, so
    this flattens and casts to int for consistency with the other
    models' output format.

    Parameters:
        model (CatBoostClassifier): Fitted model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: 1D integer predictions.
    """
    X_test = _cast_bool_columns(X_test.copy())
    return model.predict(X_test).flatten().astype(int)


def predict_regressor(model, X_test):
    """
    Predict continuous values from a CatBoostRegressor, applying bool casting.

    Parameters:
        model (CatBoostRegressor): Fitted model.
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: 1D float predictions.
    """
    X_test = _cast_bool_columns(X_test.copy())
    return model.predict(X_test).flatten()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract top N features by PredictionValuesChange importance.

    CatBoost's default importance type measures how much each feature
    changes the prediction values on average. This is conceptually
    similar to gain-based importance in XGBoost.

    Parameters:
        model (CatBoostClassifier): Fitted CatBoost model.
        feature_names (list[str]): Feature column names.
        top_n (int): Number of top features to return.

    Returns:
        pd.Series: Feature importances sorted descending, length top_n.
    """
    importance = pd.Series(
        model.get_feature_importance(),
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

    TRAIN_PATH = "../../data/processed/feature_matrix_train.csv"
    TEST_PATH = "../../data/processed/feature_matrix_test.csv"
    TARGET = "experience_level_ord"

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    df_train = df_train.dropna(subset=[TARGET])
    df_test = df_test.dropna(subset=[TARGET])

    y_train = df_train[TARGET].astype(int)
    y_test = df_test[TARGET].astype(int)
    X_train = df_train.drop(columns=[TARGET])
    X_test = df_test.drop(columns=[TARGET])

    model = train_catboost(X_train, y_train)
    y_pred = predict(model, X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    imp = get_feature_importance(model, X_train.columns, top_n=15)
    print("\n=== Top 15 Features (PredictionValuesChange) ===")
    for feat, val in imp.items():
        print(f"  {feat:<35s} {val:>8.2f}")