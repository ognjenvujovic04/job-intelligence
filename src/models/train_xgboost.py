"""
XGBoost Multiclass Classifier
==============================
Trains an XGBoost model to predict experience_level_ord (6 classes).

Designed to be imported from a notebook:
    from models.train_xgboost import train_xgboost, get_feature_importance

XGBoost handles NaN natively, so no imputation is needed.
Balanced sample weights are computed manually since XGBoost does not
have a class_weight parameter like scikit-learn estimators.
"""

import logging
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# =========================================================
# CLASS WEIGHT COMPUTATION
# =========================================================

def _compute_sample_weights(y):
    """
    Compute per-sample weights inversely proportional to class frequency.

    Replicates sklearn's 'balanced' class_weight behaviour:
        weight_c = n_samples / (n_classes * count_c)

    Parameters:
        y (array-like): Target labels.

    Returns:
        np.ndarray: Per-sample weights.
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)

    weight_map = {
        c: n_samples / (n_classes * count)
        for c, count in zip(classes, counts)
    }
    return np.array([weight_map[label] for label in y])


# =========================================================
# TRAINING
# =========================================================

def train_xgboost(X_train, y_train, val_size=0.15):
    """
    Train an XGBoost multiclass classifier with early stopping.

    A stratified validation split is carved from the training data
    to monitor mlogloss and stop training when it stops improving.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (integer-encoded).
        val_size (float): Fraction of training data held out for
            early-stopping validation.

    Returns:
        xgb.XGBClassifier: Fitted model.
    """
    num_classes = y_train.nunique()

    X_trn, X_val, y_trn, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"XGBoost train/val split: train={len(X_trn)}, val={len(X_val)}"
    )

    sample_weights = _compute_sample_weights(y_trn)

    model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_classes,
        eval_metric="mlogloss",
        booster="gbtree",
        learning_rate=0.05,
        max_depth=0,
        max_leaves=63,
        min_child_weight=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_estimators=2000,
        early_stopping_rounds=100,
        random_state=RANDOM_STATE,
        verbosity=0,
        n_jobs=-1,
        tree_method="hist",
    )

    logger.info("Training XGBoost classifier...")

    model.fit(
        X_trn,
        y_trn,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    try:
        best_iter = model.best_iteration
        logger.info(f"Early stopping at iteration {best_iter}")
    except AttributeError:
        logger.info("No early stopping triggered, used all estimators.")

    return model


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(model, feature_names, top_n=15):
    """
    Extract top N features by gain-based importance.

    Parameters:
        model (xgb.XGBClassifier): Fitted XGBoost model.
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

    model = train_xgboost(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    imp = get_feature_importance(model, X_train.columns, top_n=15)
    print("\n=== Top 15 Features (gain) ===")
    for feat, val in imp.items():
        print(f"  {feat:<35s} {val:>10.4f}")