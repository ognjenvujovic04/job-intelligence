"""
MLP Multiclass Classifier (Keras)
====================================
Trains a Keras MLP to predict experience_level_ord (6 classes).

Designed to be imported from a notebook:
    from models.train_mlp_keras import train_mlp, get_feature_importance

Preprocessing (median imputation + standard scaling) is applied
before training. The fitted imputer and scaler are returned
alongside the model so the same transforms can be reused at
predict time.

Feature importance is computed via permutation importance
since MLP has no built-in split/gain importance.
"""

import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_class_weight
from sklearn.base import BaseEstimator, ClassifierMixin

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class OrdinalMAE(keras.metrics.Metric):
    """Mean absolute error between predicted and true class indices.

    Unlike standard MAE on probabilities, this measures how many ordinal
    levels the model is off on average — e.g., predicting class 2 when
    true is 4 counts as an error of 2.
    """
    def __init__(self, name="ordinal_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self._sum = self.add_weight(name="sum", initializer="zeros")
        self._count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_cls = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)
        y_true_flat = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        abs_err = tf.abs(y_true_flat - y_pred_cls)
        self._sum.assign_add(tf.reduce_sum(abs_err))
        self._count.assign_add(tf.cast(tf.shape(y_true_flat)[0], tf.float32))

    def result(self):
        return self._sum / self._count

    def reset_state(self):
        self._sum.assign(0.0)
        self._count.assign(0.0)


# =========================================================
# TRAINING
# =========================================================

def train_mlp(X_train, y_train, val_size=0.15):
    """
    Train a Keras MLP multiclass classifier with early stopping.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (integer-encoded).
        val_size (float): Fraction held out for validation reporting.

    Returns:
        tuple: (model, imputer, scaler)
    """
    tf.random.set_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_trn, X_val, y_trn, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        stratify=y_train,
        random_state=RANDOM_STATE,
    )

    logger.info(f"Train/val split: train={len(X_trn)}, val={len(X_val)}")

    # --- Preprocessing ---
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_trn = scaler.fit_transform(imputer.fit_transform(X_trn))
    X_val = scaler.transform(imputer.transform(X_val))

    y_trn = np.asarray(y_trn)
    y_val = np.asarray(y_val)

    num_classes = len(np.unique(y_trn))
    input_dim = X_trn.shape[1]

    # --- Class weights ---
    classes = np.unique(y_trn)
    weights = compute_class_weight("balanced", classes=classes, y=y_trn)
    class_weight_dict = dict(zip(classes, weights))

    # =======================================================
    # MODEL DEFINITION
    # =======================================================
    model = keras.Sequential([

        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3),
                     input_shape=(input_dim,)),
        layers.BatchNormalization(),
        
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        
        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        
        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-3)),
        layers.BatchNormalization(),
        
        layers.Dense(num_classes, activation="softmax"),
        
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy", OrdinalMAE()],
    )

    model.summary(print_fn=logger.info)

    class _EpochLogger(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                logs = logs or {}
                logger.info(
                    f"Epoch {epoch+1:>3} | "
                    f"loss={logs.get('loss', 0):.4f} | "
                    f"val_loss={logs.get('val_loss', 0):.4f} | "
                    f"val_acc={logs.get('val_accuracy', 0):.4f} | "
                    f"val_mae={logs.get('val_ordinal_mae', 0):.3f} | "
                    f"lr={logs.get('learning_rate', 0):.2e}"
                )

    # --- Callbacks ---
    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=0,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=0,
        ),
        _EpochLogger(),
    ]

    # --- Fit ---
    history = model.fit(
        X_trn, y_trn,
        validation_data=(X_val, y_val),
        epochs=300,
        batch_size=256,
        class_weight=class_weight_dict,
        callbacks=cb,
        verbose=0,
    )

    val_loss, val_acc, val_mae = model.evaluate(X_val, y_val, verbose=0)
    logger.info(f"Validation — accuracy: {val_acc:.4f} | ordinal MAE: {val_mae:.3f}")

    return model, imputer, scaler


# =========================================================
# HELPERS
# =========================================================

def preprocess(X, imputer, scaler):
    """Apply fitted imputer + scaler to raw features."""
    return scaler.transform(imputer.transform(X))


def predict(model, X, imputer, scaler):
    """Predict classes from raw features."""
    X_proc = preprocess(X, imputer, scaler)
    return np.argmax(model.predict(X_proc, verbose=0), axis=1)


# =========================================================
# STANDALONE EXECUTION
# =========================================================

if __name__ == "__main__":
    from sklearn.metrics import (
        classification_report,
        f1_score,
        accuracy_score,
        cohen_kappa_score,
        mean_absolute_error,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    TRAIN_PATH = "data/processed/v1/feature_matrix_train.csv"
    TEST_PATH = "data/processed/v1/feature_matrix_test.csv"
    TARGET = "experience_level_ord"

    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    df_train = df_train.dropna(subset=[TARGET])
    df_test = df_test.dropna(subset=[TARGET])

    y_train = df_train[TARGET].astype(int)
    y_test = df_test[TARGET].astype(int)
    X_train = df_train.drop(columns=[TARGET])
    X_test = df_test.drop(columns=[TARGET])

    model, imputer, scaler = train_mlp(X_train, y_train)
    y_pred = predict(model, X_test, imputer, scaler)

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")

    print("\n=== Metrics ===")
    print(f"  Accuracy:          {acc:.4f}")
    print(f"  F1 (weighted):     {f1_weighted:.4f}")
    print(f"  F1 (macro):        {f1_macro:.4f}")
    print(f"  Ordinal MAE:       {mae:.4f}")
    print(f"  Quadratic Kappa:   {qwk:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))