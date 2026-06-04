"""
LightGBM Multiclass Classifier
================================
Trains a LightGBM model to predict experience_level_ord (6 classes).

Designed to be imported from a notebook:
    from models.train_lightgbm import train_lightgbm, get_feature_importance

LightGBM handles NaN natively, so no imputation is needed.
Unlike XGBoost, LightGBM supports class_weight='balanced' directly,
so no manual sample weight computation is required.
"""

import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Default tunable hyperparameters (safe to override via `params`)
DEFAULT_PARAMS = {
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "class_weight": "balanced",
    "n_estimators": 2000,
}

# =========================================================
# TRAINING
# =========================================================

def train_lightgbm(X_train, y_train, val_size=0.15, params=None):
    """
    Train a LightGBM multiclass classifier with early stopping.

    A stratified validation split is carved from the training data
    to monitor multi_logloss and stop when it stops improving.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (integer-encoded).
        val_size (float): Fraction of training data held out for
            early-stopping validation.
        params (dict, optional): Hyperparameter overrides merged on top of
            `DEFAULT_PARAMS`.

    Returns:
        lgb.LGBMClassifier: Fitted model.
    """
    num_classes = y_train.nunique()
    merged_params = {**DEFAULT_PARAMS, **(params or {})}

    X_trn, X_val, y_trn, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"LightGBM train/val split: train={len(X_trn)}, val={len(X_val)}"
    )

    # Structural parameters are kept explicit and internal
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        metric="multi_logloss",
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=-1,
        **merged_params,
    )

    logger.info("Training LightGBM classifier...")

    model.fit(
        X_trn,
        y_trn,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.log_evaluation(period=100),
            lgb.early_stopping(stopping_rounds=100),
        ],
    )

    logger.info(f"Early stopping at iteration {model.best_iteration_}")
    return model


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract top N features by split-based importance.

    LightGBM's default importance_type is 'split' (number of times
    a feature is used across all trees). This differs from XGBoost's
    default gain-based importance, so keep that in mind when comparing.

    Parameters:
        model (lgb.LGBMClassifier): Fitted LightGBM model.
        feature_names (list[str]): Feature column names.
        top_n (int): Number of top features to return.

    Returns:
        pd.Series: Feature importances sorted descending, length top_n.
    """
    importance = pd.Series(
        model.feature_importances_,
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

    model = train_lightgbm(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    imp = get_feature_importance(model, X_train.columns, top_n=15)
    print("\n=== Top 15 Features (split importance) ===")
    for feat, val in imp.items():
        print(f"  {feat:<35s} {val:>6d}")