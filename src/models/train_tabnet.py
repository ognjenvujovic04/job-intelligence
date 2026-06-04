"""
TabNet Multiclass Classifier
================================
Trains a TabNet model to predict experience_level_ord (6 classes).

Designed to be imported from a notebook:
    from models.train_tabnet import train_tabnet, get_feature_importance

TabNet does NOT handle NaN natively, so median imputation is applied
before training. Features are also standardized since TabNet's
attention mechanism is sensitive to feature scale.

Requires: pytorch-tabnet
    pip install pytorch-tabnet
"""

import logging
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger(__name__)

RANDOM_STATE = 42

# Default tunable hyperparameters (safe to override via `params`)
DEFAULT_PARAMS = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 3,
    "gamma": 1.8,
    "lambda_sparse": 1e-2,
    "cat_idxs": [],
    "cat_dims": [],
    "cat_emb_dim": [],
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": {"lr": 2e-2, "weight_decay": 1e-5},
    "scheduler_fn": torch.optim.lr_scheduler.StepLR,
    "scheduler_params": {"step_size": 15, "gamma": 0.5},
    "mask_type": "entmax",
    # Fit-time params
    "max_epochs": 200,
    "patience": 30,
    "batch_size": 1024,
    "virtual_batch_size": 256,
    "drop_last": False,
}


# =========================================================
# PREPROCESSING HELPERS
# =========================================================

def _impute_and_scale(X_train, X_val, X_test=None):
    """
    Median-impute NaNs and standardize features.

    Fit on X_train only to avoid data leakage.
    Returns numpy arrays (TabNet expects ndarray input).

    Parameters:
        X_train (pd.DataFrame): Training features.
        X_val (pd.DataFrame): Validation features.
        X_test (pd.DataFrame | None): Test features (optional).

    Returns:
        tuple: (X_train_arr, X_val_arr, X_test_arr_or_None, scaler, medians)
    """
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)

    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train).astype(np.float32)
    X_val_arr = scaler.transform(X_val).astype(np.float32)

    X_test_arr = None
    if X_test is not None:
        X_test = X_test.fillna(medians)
        X_test_arr = scaler.transform(X_test).astype(np.float32)

    return X_train_arr, X_val_arr, X_test_arr, scaler, medians


# =========================================================
# TRAINING
# =========================================================

def train_tabnet(X_train, y_train, val_size=0.15, params=None):
    """
    Train a TabNet multiclass classifier with early stopping.

    A stratified validation split is carved from the training data
    to monitor validation logloss and stop when it stops improving.

    TabNet uses sequential attention to select features at each
    decision step, giving it built-in feature selection.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (integer-encoded).
        val_size (float): Fraction of training data held out for
            early-stopping validation.
        params (dict, optional): Hyperparameter overrides merged on top of
            `DEFAULT_PARAMS` (shallow merge).

    Returns:
        dict with keys:
            "model" : fitted TabNetClassifier
            "scaler": fitted StandardScaler (needed at inference)
            "medians": median values used for imputation
    """
    num_classes = y_train.nunique()

    X_trn, X_val, y_trn, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    logger.info(
        f"TabNet train/val split: train={len(X_trn)}, val={len(X_val)}"
    )

    # Impute and scale (fit on train split only)
    X_trn_arr, X_val_arr, _, scaler, medians = _impute_and_scale(
        X_trn, X_val
    )
    y_trn_arr = y_trn.values.astype(np.int64)
    y_val_arr = y_val.values.astype(np.int64)

    # Compute class weights to handle imbalance
    classes = np.unique(y_trn_arr)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_trn_arr,
    )
    sample_weights = np.array([weights[c] for c in y_trn_arr])

    # Merge provided params on top of defaults (backward compatible)
    merged = {**DEFAULT_PARAMS, **(params or {})}

    # Extract fit-time params
    fit_keys = ("max_epochs", "patience", "batch_size", "virtual_batch_size", "drop_last")
    fit_kwargs = {k: merged.pop(k) for k in fit_keys if k in merged}

    model = TabNetClassifier(
        seed=RANDOM_STATE,
        verbose=10,
        **merged,
    )

    logger.info("Training TabNet classifier...")

    model.fit(
        X_trn_arr,
        y_trn_arr,
        eval_set=[(X_val_arr, y_val_arr)],
        eval_name=["val"],
        eval_metric=["logloss"],
        weights=sample_weights,
        **fit_kwargs,
    )

    best_epoch = model.best_epoch or len(model.history["loss"]) - 1
    logger.info(f"Best epoch: {best_epoch}")

    return {
        "model": model,
        "scaler": scaler,
        "medians": medians,
    }


def predict_tabnet(bundle, X_test):
    """
    Generate predictions using a trained TabNet bundle.

    Applies the same imputation and scaling that was fit during training.

    Parameters:
        bundle (dict): Output of train_tabnet().
        X_test (pd.DataFrame): Test features.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model = bundle["model"]
    scaler = bundle["scaler"]
    medians = bundle["medians"]

    X = X_test.fillna(medians)
    X_arr = scaler.transform(X).astype(np.float32)
    return model.predict(X_arr)


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(bundle, feature_names, top_n=15):
    """
    Extract top N features by TabNet attention-based importance.

    TabNet computes feature importance from its learned attention
    masks, reflecting how often each feature was selected across
    decision steps. This is conceptually different from tree-based
    split/gain importance.

    Parameters:
        bundle (dict): Output of train_tabnet().
        feature_names (list[str]): Feature column names.
        top_n (int): Number of top features to return.

    Returns:
        pd.Series: Feature importances sorted descending, length top_n.
    """
    model = bundle["model"]
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

    bundle = train_tabnet(X_train, y_train)
    y_pred = predict_tabnet(bundle, X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    imp = get_feature_importance(bundle, X_train.columns, top_n=15)
    print("\n=== Top 15 Features (attention importance) ===")
    for feat, val in imp.items():
        print(f"  {feat:<35s} {val:>8.4f}")