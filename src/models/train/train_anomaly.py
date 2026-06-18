"""
Anomaly Detection Trainer
=========================
Trains the four unsupervised anomaly detectors from notebook 3.4 and logs the
fitted weights + preprocessing to MLflow:

- **IsolationForest** (sklearn)
- **COPOD** (PyOD - copula-based)
- **Autoencoder** (PyTorch - reconstruction error)
- **VAE** (PyTorch - reconstruction-probability proxy)

Each model emits a binary outlier flag per row; the ensemble **anomaly score**
is the number of models that flag a row (0-4). This module owns the model
definitions, the shared preprocessing, and the per-model flag logic so that the
inference entry point (``src/models/serve/anomaly_detection.py``) can reuse them.

Unlike the gradient-boosting trainers this does not go through
``tracking.experiment.run_experiment`` (which assumes a single supervised model
with an eval_fn / confusion matrix). Anomaly detection is unsupervised and
bundles four models, so — like ``clustering.py`` — it logs everything manually
inside a single ``mlflow.start_run`` (the tracking here is "basic", mainly to
persist weights). The MLflow ``run_id`` it prints is what you paste into
``anomaly_detection.py``'s ``RUN_ID``.

Usage
-----
    python -m src.models.train.train_anomaly
"""

import argparse
import logging
import os
import sys
import tempfile
from time import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyod.models.copod import COPOD
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Allow running as `python -m src.models.train.train_anomaly` from repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from src.models.tracking.config import (  # noqa: E402
    DEFAULT_ANOMALY_EXPERIMENT_NAME,
    configure_mlflow,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config (mirrors notebook 3.4)
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
V7_TRAIN = os.path.join(_REPO_ROOT, "data", "processed", "v7", "feature_matrix_train.csv")
V7_TEST = os.path.join(_REPO_ROOT, "data", "processed", "v7", "feature_matrix_test.csv")

CONTAMINATION = 0.005  # expected fraction of anomalies
RANDOM_STATE = 42
LATENT_DIM = 16
N_ESTIMATORS = 200  # IsolationForest
EPOCHS = 50
BATCH_SIZE = 512
LR = 1e-3
WEIGHT_DECAY = 1e-5
VAE_BETA = 1.0  # KL weight (1.0 = standard VAE)
L_SAMPLES = 10  # stochastic latent draws for the VAE recon-probability proxy

# job_id is an identifier, not a model input.
NON_FEATURE_COLS = ["job_id"]
# Artifact layout under the MLflow run.
ARTIFACT_DIR = "anomaly_artifacts"
BUNDLE_NAME = "bundle.joblib"
AE_WEIGHTS_NAME = "autoencoder.pt"
VAE_WEIGHTS_NAME = "vae.pt"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_float_array(X):
    """Coerce a DataFrame/ndarray to a contiguous float32 numpy array."""
    if isinstance(X, pd.DataFrame):
        X = X.values
    return np.ascontiguousarray(X, dtype=np.float32)


# ===========================================================================
# PREPROCESSING (fit on train, reused at inference)
# ===========================================================================

def drop_biweekly(df: pd.DataFrame, name: str = "") -> pd.DataFrame:
    """Drop the near-empty ``pay_period_BIWEEKLY`` one-hot rows (train-time only).

    These few sparse rows destabilise the reconstruction models (see notebook
    section 3.2), so they are removed before fitting. At inference we score every
    row, so the entry point does not call this.
    """
    if "pay_period_BIWEEKLY" not in df.columns:
        return df.reset_index(drop=True)
    mask = df["pay_period_BIWEEKLY"] == True  # noqa: E712
    if name:
        logger.info("%s: dropping %d BIWEEKLY rows (%d -> %d)",
                    name, int(mask.sum()), len(df), len(df) - int(mask.sum()))
    return df[~mask].reset_index(drop=True)


def fit_preprocessing(X_train_raw: pd.DataFrame):
    """Fit the median-fill + StandardScaler on the training feature matrix.

    Returns
    -------
    (medians, scaler) : (pd.Series, StandardScaler)
        ``medians`` is indexed by feature name; ``scaler`` is fitted on the
        median-filled training matrix.
    """
    medians = X_train_raw.median()
    X_filled = X_train_raw.fillna(medians)
    scaler = StandardScaler()
    scaler.fit(X_filled)
    return medians, scaler


def apply_preprocessing(X_raw, feature_cols, medians, scaler) -> pd.DataFrame:
    """Reapply the persisted median-fill + scaler to a raw feature matrix.

    Selects ``feature_cols`` in training order, fills missing values with the
    persisted train medians (this is what fills missing ``normalized_salary`` at
    inference), then applies the fitted scaler. Returns a scaled DataFrame.
    """
    X = X_raw[feature_cols].copy()
    X = X.fillna(medians)
    scaled = scaler.transform(X)
    return pd.DataFrame(scaled, columns=feature_cols, index=X.index)


# ===========================================================================
# NEURAL NETWORK MODELS
# ===========================================================================

class Autoencoder(nn.Module):
    """Reconstruction autoencoder: input_dim -> 64 -> 32 -> latent -> 32 -> 64 -> input_dim."""

    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VAE(nn.Module):
    """Variational autoencoder with a 16-dim Gaussian latent posterior."""

    def __init__(self, input_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32, latent_dim)
        self.fc_logvar = nn.Linear(32, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


def vae_loss(recon, x, mu, logvar, beta=VAE_BETA):
    """Negative ELBO = reconstruction MSE + beta * KL(N(mu,sigma) || N(0,1))."""
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kld, recon_loss, kld


# ===========================================================================
# TRAINERS (no MLflow inside — they just fit and return)
# ===========================================================================

def train_isolation_forest(X_scaled, contamination=CONTAMINATION,
                           random_state=RANDOM_STATE, n_estimators=N_ESTIMATORS):
    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=n_estimators,
        n_jobs=-1,
    )
    model.fit(_as_float_array(X_scaled))
    return model


def train_copod(X_scaled, contamination=CONTAMINATION):
    model = COPOD(contamination=contamination)
    model.fit(_as_float_array(X_scaled))
    return model


def train_autoencoder(X_scaled, contamination=CONTAMINATION, epochs=EPOCHS,
                      batch_size=BATCH_SIZE, lr=LR, latent_dim=LATENT_DIM,
                      device=None):
    """Train the autoencoder; return (model, threshold).

    The threshold is the ``contamination`` upper percentile of the per-row train
    reconstruction errors.
    """
    device = device or get_device()
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_tensor = torch.FloatTensor(_as_float_array(X_scaled)).to(device)
    n_features = X_tensor.shape[1]

    loader = DataLoader(
        TensorDataset(X_tensor, X_tensor), batch_size=batch_size, shuffle=True
    )

    model = Autoencoder(n_features, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss(reduction="mean")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_x, _ in loader:
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  AE epoch %3d/%d  loss: %.6f",
                        epoch + 1, epochs, epoch_loss / len(X_tensor))

    train_scores = ae_recon_error(model, X_scaled, device=device)
    threshold = float(np.percentile(train_scores, 100 * (1 - contamination)))
    return model, threshold


def train_vae(X_scaled, contamination=CONTAMINATION, epochs=EPOCHS,
              batch_size=BATCH_SIZE, lr=LR, latent_dim=LATENT_DIM,
              beta=VAE_BETA, l_samples=L_SAMPLES, device=None):
    """Train the VAE; return (model, threshold)."""
    device = device or get_device()
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    X_tensor = torch.FloatTensor(_as_float_array(X_scaled)).to(device)
    n_features = X_tensor.shape[1]

    loader = DataLoader(
        TensorDataset(X_tensor, X_tensor), batch_size=batch_size, shuffle=True
    )

    model = VAE(n_features, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    for epoch in range(epochs):
        model.train()
        ep_total = 0.0
        for batch_x, _ in loader:
            recon, mu, logvar = model(batch_x)
            loss, _, _ = vae_loss(recon, batch_x, mu, logvar, beta=beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_total += loss.item() * batch_x.size(0)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("  VAE epoch %3d/%d  total: %.6f",
                        epoch + 1, epochs, ep_total / len(X_tensor))

    train_scores = vae_recon_error(model, X_scaled, device=device, l_samples=l_samples)
    threshold = float(np.percentile(train_scores, 100 * (1 - contamination)))
    return model, threshold


# ===========================================================================
# SCORING / FLAGS (shared by training eval and inference)
# ===========================================================================

def _model_device(model):
    return next(model.parameters()).device


def ae_recon_error(model, X_scaled, device=None):
    """Per-row mean squared reconstruction error for the autoencoder."""
    device = device or _model_device(model)
    X_tensor = torch.FloatTensor(_as_float_array(X_scaled)).to(device)
    model.eval()
    with torch.no_grad():
        recon = model(X_tensor)
        err = ((X_tensor - recon) ** 2).mean(dim=1).cpu().numpy()
    return err


def vae_recon_error(model, X_scaled, device=None, l_samples=L_SAMPLES):
    """Per-row VAE reconstruction error averaged over ``l_samples`` latent draws."""
    device = device or _model_device(model)
    X_tensor = torch.FloatTensor(_as_float_array(X_scaled)).to(device)
    model.eval()
    with torch.no_grad():
        err_sum = torch.zeros(X_tensor.size(0), device=device)
        for _ in range(l_samples):
            recon, _, _ = model(X_tensor)
            err_sum += ((X_tensor - recon) ** 2).mean(dim=1)
        err = (err_sum / l_samples).cpu().numpy()
    return err


def iso_flag(model, X_scaled):
    """1 where IsolationForest predicts an outlier (-1), else 0."""
    return (model.predict(_as_float_array(X_scaled)) == -1).astype(int)


def copod_flag(model, X_scaled):
    """COPOD already returns 1 = outlier, 0 = inlier."""
    return model.predict(_as_float_array(X_scaled)).astype(int)


def ae_flag(model, X_scaled, threshold, device=None):
    """1 where the autoencoder recon error >= the train-derived threshold."""
    return (ae_recon_error(model, X_scaled, device=device) >= threshold).astype(int)


def vae_flag(model, X_scaled, threshold, device=None, l_samples=L_SAMPLES):
    """1 where the VAE recon error >= the train-derived threshold."""
    return (vae_recon_error(model, X_scaled, device=device, l_samples=l_samples) >= threshold).astype(int)


def compute_flags(iso_model, copod_model, ae_model, vae_model, ae_threshold,
                  vae_threshold, X_scaled, device=None, l_samples=L_SAMPLES):
    """Run all four detectors and return a DataFrame of 0/1 flags + the 0-4 score.

    Columns: ``isolation_forest``, ``copod``, ``autoencoder``, ``vae`` (each 0/1)
    and ``anomaly_score`` (their row-wise sum).
    """
    flags = pd.DataFrame(
        {
            "isolation_forest": iso_flag(iso_model, X_scaled),
            "copod": copod_flag(copod_model, X_scaled),
            "autoencoder": ae_flag(ae_model, X_scaled, ae_threshold, device=device),
            "vae": vae_flag(vae_model, X_scaled, vae_threshold, device=device, l_samples=l_samples),
        },
        index=X_scaled.index if isinstance(X_scaled, pd.DataFrame) else None,
    )
    flags["anomaly_score"] = flags.sum(axis=1)
    return flags


# ===========================================================================
# ORCHESTRATION + MLFLOW LOGGING
# ===========================================================================

def train_and_log(
    train_path: str = V7_TRAIN,
    test_path: str = V7_TEST,
    contamination: float = CONTAMINATION,
    experiment_name: str = DEFAULT_ANOMALY_EXPERIMENT_NAME,
):
    """Fit the four detectors on v7 train and log weights + metrics to MLflow.

    Returns the MLflow ``run_id`` (paste it into ``anomaly_detection.py``).
    """
    import mlflow

    configure_mlflow(experiment_name=experiment_name)
    device = get_device()
    logger.info("Using device: %s", device)

    # --- Load + preprocess (fit on train only) ---
    df_train = drop_biweekly(pd.read_csv(train_path), "Train")
    df_test = drop_biweekly(pd.read_csv(test_path), "Test")

    feature_cols = [c for c in df_train.columns if c not in NON_FEATURE_COLS]
    X_train_raw = df_train[feature_cols]
    X_test_raw = df_test[feature_cols]

    medians, scaler = fit_preprocessing(X_train_raw)
    X_train_scaled = apply_preprocessing(X_train_raw, feature_cols, medians, scaler)
    X_test_scaled = apply_preprocessing(X_test_raw, feature_cols, medians, scaler)
    logger.info("Train: %s | Test: %s | %d features",
                X_train_scaled.shape, X_test_scaled.shape, len(feature_cols))

    # --- Train the four models on train ---
    t0 = time()
    logger.info("Training IsolationForest ...")
    iso_model = train_isolation_forest(X_train_scaled, contamination=contamination)
    logger.info("Training COPOD ...")
    copod_model = train_copod(X_train_scaled, contamination=contamination)
    logger.info("Training Autoencoder ...")
    ae_model, ae_threshold = train_autoencoder(X_train_scaled, contamination=contamination, device=device)
    logger.info("Training VAE ...")
    vae_model, vae_threshold = train_vae(X_train_scaled, contamination=contamination, device=device)
    train_time = time() - t0

    # --- Evaluate on test (for "basic" tracking) ---
    test_flags = compute_flags(
        iso_model, copod_model, ae_model, vae_model,
        ae_threshold, vae_threshold, X_test_scaled, device=device,
    )
    agreement_dist = test_flags["anomaly_score"].value_counts().sort_index()
    logger.info("Test agreement distribution:\n%s", agreement_dist.to_string())

    # --- Log one run: params, metrics, weights ---
    input_dim = len(feature_cols)
    with mlflow.start_run(run_name="anomaly-ensemble") as run:
        mlflow.set_tags({"task": "anomaly-detection", "models": "iforest+copod+ae+vae"})
        mlflow.log_params({
            "contamination": contamination,
            "random_state": RANDOM_STATE,
            "n_features": input_dim,
            "n_train": len(X_train_scaled),
            "latent_dim": LATENT_DIM,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "vae_beta": VAE_BETA,
            "l_samples": L_SAMPLES,
            "iforest_n_estimators": N_ESTIMATORS,
        })
        mlflow.log_metrics({
            "train_time_s": train_time,
            "ae_threshold": ae_threshold,
            "vae_threshold": vae_threshold,
            "test_flagged_isolation_forest": int(test_flags["isolation_forest"].sum()),
            "test_flagged_copod": int(test_flags["copod"].sum()),
            "test_flagged_autoencoder": int(test_flags["autoencoder"].sum()),
            "test_flagged_vae": int(test_flags["vae"].sum()),
            "test_consensus_4of4": int((test_flags["anomaly_score"] == 4).sum()),
            "test_at_least_2": int((test_flags["anomaly_score"] >= 2).sum()),
        })
        # Per-score-bucket counts (0..4) as individual metrics.
        for score in range(5):
            mlflow.log_metric(f"test_score_{score}", int((test_flags["anomaly_score"] == score).sum()))

        # --- Save weights + preprocessing as artifacts ---
        bundle = {
            "feature_cols": feature_cols,
            "medians": medians,
            "scaler": scaler,
            "iso_model": iso_model,
            "copod_model": copod_model,
            "ae_threshold": ae_threshold,
            "vae_threshold": vae_threshold,
            "contamination": contamination,
            "latent_dim": LATENT_DIM,
            "input_dim": input_dim,
            "l_samples": L_SAMPLES,
        }
        with tempfile.TemporaryDirectory() as tmp:
            bundle_path = os.path.join(tmp, BUNDLE_NAME)
            joblib.dump(bundle, bundle_path)
            mlflow.log_artifact(bundle_path, artifact_path=ARTIFACT_DIR)

            ae_path = os.path.join(tmp, AE_WEIGHTS_NAME)
            torch.save(ae_model.state_dict(), ae_path)
            mlflow.log_artifact(ae_path, artifact_path=ARTIFACT_DIR)

            vae_path = os.path.join(tmp, VAE_WEIGHTS_NAME)
            torch.save(vae_model.state_dict(), vae_path)
            mlflow.log_artifact(vae_path, artifact_path=ARTIFACT_DIR)

        run_id = run.info.run_id

    logger.info("Weights logged to run %s under '%s/'", run_id, ARTIFACT_DIR)
    print("\n" + "=" * 60)
    print(f"Anomaly ensemble logged.\n  RUN_ID = {run_id}")
    print("  Paste this into src/models/serve/anomaly_detection.py (RUN_ID).")
    print("=" * 60)
    return run_id


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Train the anomaly-detection ensemble and log weights to MLflow."
    )
    parser.add_argument("--train-path", default=V7_TRAIN, help="v7 train feature matrix CSV.")
    parser.add_argument("--test-path", default=V7_TEST, help="v7 test feature matrix CSV.")
    parser.add_argument("--contamination", type=float, default=CONTAMINATION,
                        help="Expected anomaly fraction (default: 0.005).")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    train_and_log(
        train_path=args.train_path,
        test_path=args.test_path,
        contamination=args.contamination,
    )
