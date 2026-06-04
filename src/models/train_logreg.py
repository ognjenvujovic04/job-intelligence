"""
Logistic Regression - Experience Level Classifier
====================================================
Builds a sklearn Pipeline (imputer + scaler + LogReg) for predicting
experience_level_ord from the feature matrix.

Evaluation is handled separately by src/utils/evaluation.py.

Usage from notebook:
    from models.train_logreg import build_logreg_pipeline, get_feature_importance
    pipeline = build_logreg_pipeline(X_train, y_train)
    y_pred = pipeline.predict(X_test)
"""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Initialize module-level logger
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# =========================================================
# PIPELINE CONSTRUCTION AND TRAINING
# =========================================================

def build_logreg_pipeline(
    X_train,
    y_train,
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=RANDOM_STATE,
):
    """
    Build and fit a Logistic Regression pipeline.

    The pipeline chains three steps:
        1. Median imputation - fills NaN values that LogReg cannot
           handle natively (~81% missing in apply-related features,
           ~63% in normalized_salary).
        2. Standard scaling - centers and scales features so that
           regularization penalizes coefficients fairly.
        3. Logistic Regression - multinomial classifier with L2
           regularization. class_weight='balanced' adjusts sample
           weights inversely proportional to class frequency, which
           helps with the heavy imbalance (classes 0, 4, 5 are under
           5% each).

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target labels.
        C (float): Inverse regularization strength.
        max_iter (int): Maximum solver iterations.
        solver (str): Optimization algorithm.
        class_weight (str or dict): Class weighting strategy.
        random_state (int): Random seed.

    Returns:
        sklearn.pipeline.Pipeline: Fitted pipeline.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver=solver,
            class_weight=class_weight,
            max_iter=max_iter,
            C=C,
            random_state=random_state,
            verbose=0,
        )),
    ])

    logger.info(
        f"Training Logistic Regression | C={C}, solver={solver}, "
        f"class_weight={class_weight}, max_iter={max_iter}"
    )
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    return pipeline


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def get_feature_importance(pipeline, feature_names, top_n=15):
    """
    Extract feature importance from a fitted LogReg pipeline.

    For multinomial logistic regression the coefficient matrix has
    shape (n_classes, n_features). The mean of absolute values across
    classes serves as a proxy for overall feature importance.

    Parameters:
        pipeline (sklearn.pipeline.Pipeline): Fitted pipeline with a
            'classifier' step containing coef_.
        feature_names (list): Feature column names matching the order
            used during training.
        top_n (int): Number of top features to return.

    Returns:
        pd.Series: Top features sorted by mean absolute coefficient,
            descending.
    """
    classifier = pipeline.named_steps["classifier"]
    coef_matrix = classifier.coef_  # (n_classes, n_features)

    mean_abs_coef = np.abs(coef_matrix).mean(axis=0)
    importance = pd.Series(
        mean_abs_coef,
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

    # Build and train the pipeline
    pipeline = build_logreg_pipeline(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    imp = get_feature_importance(pipeline, X_train.columns, top_n=15)
    print("\n=== Top 15 Features (mean abs coefficient) ===")
    for feat, val in imp.items():
        print(f"  {feat:<35s} {val:>10.4f}")