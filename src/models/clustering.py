# src/clustering.py

import logging
import os
import time
import tempfile
import mlflow
import mlflow.sklearn
from mlflow.models import Model
from .tracking.config import CLUSTERING_RUN_ID, configure_mlflow
from .tracking.logging_helpers import _log_tags
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
import warnings

_EXTRA_STOPWORDS = {
    "experience", "work", "will", "team", "skills", "including",
    "ability", "job", "business", "management", "required", "company",
    "role","position","time","years",
    "may", "must", "requirements", "benefits", "opportunity", "opportunities",
    "employment", "support", "information", "ensure", "environment",
    "status", "working", "strong", "related", "new", "based", "duties",
    "employee", "employees", "project", "knowledge", "client", "product", 
    "customer", "application", "perform", "able", "projects", "clients",
    "need", "provide", "equal", "program", "member", "provided", "service"
}
_STOPWORDS = STOPWORDS | _EXTRA_STOPWORDS
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


K_RANGE = [26]
# HDBSCAN_MIN_CLUSTER_SIZES = [500, 750, 1000, 1250, 1500]


# =========================================================
# DATA PREPARATION
# =========================================================

def load_and_prepare(feature_path, cleaned_path, drop_high_missing=True):
    """Load feature matrix + cleaned df for interpretation."""
    df_feat = pd.read_csv(feature_path)
    df_clean = pd.read_csv(cleaned_path)

    # Use job_id to align df_clean to the exact rows (and order) in the feature matrix.
    if 'job_id' in df_feat.columns:
        job_ids = df_feat['job_id'].values
        df_clean = (
            df_clean.set_index('job_id')
            .loc[job_ids]
            .reset_index()
        )
        df_feat = df_feat.drop(columns=['job_id'])

    # Extract normalized salary from the feature matrix before scaling and
    # attach it to df_clean so it's available for profiling without distorting
    # the clustering features.
    if "normalized_salary" in df_feat.columns:
        df_clean = df_clean.copy()
        df_clean["normalized_salary"] = df_feat["normalized_salary"].values
    drop_cols = []
    if drop_high_missing:
        drop_cols += ["log_applies", "apply_rate"]
    drop_cols += ["state_salary_enc", "fips_salary_enc"]

    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # Bundle imputation + scaling into a single fitted preprocessor so it can be
    # serialized together with the clustering model (raw features -> clusters).
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_scaled = pd.DataFrame(
        preprocessor.fit_transform(df_feat),
        columns=df_feat.columns,
    )

    return X_scaled, df_clean, preprocessor, df_feat


# =========================================================
# CLUSTERING MODELS
# =========================================================

def get_model(name, **params):
    """Factory for clustering models."""
    models = {
        "kmeans": KMeans,
        "agglomerative": AgglomerativeClustering,
        "hdbscan": HDBSCAN,
    }
    return models[name](**params)


# =========================================================
# METRICS
# =========================================================

def compute_metrics(X, labels, model=None, sample_size=10_000):
    """Compute internal clustering metrics on a sample. Skips if only 1 cluster."""
    n_total = len(labels)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())

    if n_clusters < 2:
        return {"n_clusters": n_clusters, "n_noise": n_noise, "noise_ratio": n_noise / n_total}

    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    n = len(X_valid)
    actual_sample = min(sample_size, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=actual_sample, replace=False)
    X_s = X_valid.iloc[idx] if hasattr(X_valid, "iloc") else X_valid[idx]
    l_s = labels_valid[idx]

    print(f"  Computing metrics on {actual_sample:,} / {n:,} points ...")
    metrics = {
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "noise_ratio": n_noise / n_total,
        "silhouette": float(silhouette_score(X_s, l_s)),
        "davies_bouldin": float(davies_bouldin_score(X_s, l_s)),
        "calinski_harabasz": float(calinski_harabasz_score(X_s, l_s)),
    }
    if model is not None and hasattr(model, "inertia_"):
        metrics["inertia"] = float(model.inertia_)
    return metrics


# =========================================================
# INTERPRETATION / ARTIFACTS
# =========================================================

def plot_feature_distributions(X, labels, df_clean):
    """Box plots of key features per cluster. Returns figure."""
    key_features = ["log_views", "title_length", "desc_word_count",
                    "domain_similarity", "company_freq", "experience_level_ord"]
    key_features = [f for f in key_features if f in X.columns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, feat in zip(axes.ravel(), key_features):
        data = pd.DataFrame({"cluster": labels, feat: X[feat]})
        data.boxplot(column=feat, by="cluster", ax=ax)
        ax.set_title(feat)
    plt.suptitle("Feature Distributions per Cluster", fontsize=14)
    plt.tight_layout()
    return fig


def plot_word_clouds(labels, df_clean, column="title"):
    """Generate a word cloud per cluster for a text column. Returns figure."""
    unique_labels = sorted(set(labels) - {-1})

    fig, axes = plt.subplots(1, len(unique_labels), figsize=(6 * len(unique_labels), 5))
    if len(unique_labels) == 1:
        axes = [axes]

    for ax, cl in zip(axes, unique_labels):
        text = " ".join(df_clean.loc[labels == cl, column].dropna().astype(str))
        wc = WordCloud(width=600, height=400, background_color="white", stopwords=_STOPWORDS).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Cluster {cl}")
        ax.axis("off")

    plt.suptitle(f"Word Clouds: {column}", fontsize=14)
    plt.tight_layout()
    return fig


def get_cluster_top_words(labels, df_clean, column="title", n=15):
    """Return a DataFrame of top n words per cluster, excluding stopwords."""
    rows = []
    for cl in sorted(set(labels) - {-1}):
        text = " ".join(df_clean.loc[labels == cl, column].dropna().astype(str))
        tokens = re.findall(r"[a-zA-Z]+", text.lower())
        counts = Counter(t for t in tokens if t not in _STOPWORDS and len(t) > 1)
        for rank, (word, freq) in enumerate(counts.most_common(n), start=1):
            rows.append({"cluster": cl, "rank": rank, "word": word, "count": freq})
    return pd.DataFrame(rows)


def get_centroid_representatives(X, labels, df_clean, n=5):
    """Find the n rows closest to each cluster centroid."""
    results = {}
    for cl in sorted(set(labels) - {-1}):
        mask = labels == cl
        cluster_data = X[mask]
        centroid = cluster_data.mean(axis=0).values
        distances = np.linalg.norm(cluster_data.values - centroid, axis=1)
        closest_idx = cluster_data.index[np.argsort(distances)[:n]]
        results[cl] = df_clean.loc[closest_idx]
    return results


def log_centroid_representatives(representatives, output_dir):
    """Save representative rows to CSV in output_dir and return paths."""
    paths = []
    for cl, df_repr in representatives.items():
        path = os.path.join(output_dir, f"cluster_{cl}_representatives.csv")
        df_repr.to_csv(path, index=False)
        paths.append(path)
    return paths


# =========================================================
# CLUSTER COMPOSITION SUMMARY
# =========================================================

def cluster_profile(X, labels, df_clean):
    """Build a summary table: one row per cluster with mean features + top categories."""
    profiles = []
    for cl in sorted(set(labels) - {-1}):
        mask = labels == cl
        row = {"cluster": cl, "size": int(mask.sum())}

        row.update(X[mask].mean().to_dict())

        subset = df_clean[mask]
        if "formatted_work_type" in subset.columns:
            row["top_work_type"] = subset["formatted_work_type"].mode().iloc[0] \
                                   if not subset["formatted_work_type"].mode().empty else "N/A"
        if "formatted_experience_level" in subset.columns:
            row["top_experience"] = subset["formatted_experience_level"].mode().iloc[0] \
                                    if not subset["formatted_experience_level"].mode().empty else "N/A"
        profiles.append(row)

    return pd.DataFrame(profiles)


# =========================================================
# MAIN RUN
# =========================================================

def save_labels(labels, df_clean, output_path):
    """Save cluster labels aligned with job_id to a CSV file."""
    df_labels = pd.DataFrame({
        "job_id": df_clean["job_id"].values,
        "cluster": labels,
    })
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df_labels.to_csv(output_path, index=False)
    return df_labels


def run_clustering(
    model_name="kmeans",
    model_params=None,
    feature_path="data/processed/v6/feature_matrix_train.csv",
    cleaned_path="data/processed/cleaned_job_postings.csv",
    experiment_name="clustering-analysis",
    labels_output_path=None,
):
    if model_params is None:
        model_params = {"n_clusters": 5, "random_state": 42, "n_init": 10}

    mlflow.set_experiment(experiment_name)

    run_tag = f"{model_name}_{model_params}"
    print(f"\n{'='*60}")
    print(f"Run: {run_tag}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=run_tag):
        # 1. Prepare
        print("[1/4] Loading and preparing data ...")
        t0 = time.time()
        X, df_clean, preprocessor, X_raw = load_and_prepare(feature_path, cleaned_path)
        print(f"  X shape: {X.shape}  |  df_clean rows: {len(df_clean):,}  ({time.time()-t0:.1f}s)")

        # 2. Fit
        print(f"[2/4] Fitting {model_name} ...")
        t0 = time.time()
        model = get_model(model_name, **model_params)
        labels = model.fit_predict(X)
        sizes = np.bincount(labels[labels >= 0])
        print(f"  Done in {time.time()-t0:.1f}s  |  cluster sizes: {sizes.tolist()}")

        # 2a. Log the full pipeline (imputer + scaler + clustering model) so the
        # fitted scaler is persisted alongside the model and raw features can be
        # mapped to clusters at inference time.
        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        try:
            from mlflow.models import infer_signature
            signature = infer_signature(X_raw, labels)
            mlflow.sklearn.log_model(
                sk_model=full_pipeline,
                name="model",
                signature=signature,
                input_example=X_raw.head(5),
            )
            print("  pipeline (preprocessor + model): logged")
        except Exception as e:
            print(f"  WARNING: failed to log pipeline model: {e}")

        # 2b. Save labels
        with tempfile.TemporaryDirectory() as _label_tmp:
            _label_path = os.path.join(_label_tmp, "cluster_labels.csv")
            save_labels(labels, df_clean, _label_path)
            mlflow.log_artifact(_label_path)
        if labels_output_path is not None:
            save_labels(labels, df_clean, labels_output_path)
            print(f"  Labels saved to {labels_output_path}")

        # 3. Metrics
        print("[3/4] Computing metrics ...")
        t0 = time.time()
        metrics = compute_metrics(X.values, labels, model=model)
        print(f"  {metrics}  ({time.time()-t0:.1f}s)")
        _log_tags({}, run_tag, model_type=model_name, task="clustering")
        dataset_name = os.path.splitext(os.path.basename(feature_path))[0]
        dataset = mlflow.data.from_pandas(
            X.reset_index(drop=True), source=feature_path, name=dataset_name
        )
        mlflow.log_input(dataset, context="training")
        mlflow.log_params({
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            **model_params,
        })
        mlflow.log_metrics(metrics)

        # 4. Artifacts (MLflow only — no local files)
        n_clusters = metrics.get("n_clusters", 0)
        if n_clusters > 50:
            print(f"[4/4] Skipping artifacts ({n_clusters} clusters > 50).")
        else:
            print("[4/4] Generating artifacts ...")

            with tempfile.TemporaryDirectory() as _wc_tmp:
                title_png = os.path.join(_wc_tmp, "wordcloud_title.png")
                wc_title_fig = plot_word_clouds(labels, df_clean, "title")
                wc_title_fig.savefig(title_png, dpi=150, bbox_inches="tight")
                plt.close(wc_title_fig)
                mlflow.log_artifact(title_png)

                desc_png = os.path.join(_wc_tmp, "wordcloud_description.png")
                wc_desc_fig = plot_word_clouds(labels, df_clean, "description")
                wc_desc_fig.savefig(desc_png, dpi=150, bbox_inches="tight")
                plt.close(wc_desc_fig)
                mlflow.log_artifact(desc_png)

                img_t = Image.open(title_png)
                img_d = Image.open(desc_png)
                combined = Image.new(
                    "RGB",
                    (max(img_t.width, img_d.width), img_t.height + img_d.height),
                    "white",
                )
                combined.paste(img_t, (0, 0))
                combined.paste(img_d, (0, img_t.height))
                combined_png = os.path.join(_wc_tmp, "wordcloud_combined.png")
                combined.save(combined_png)
                mlflow.log_artifact(combined_png)

            print("  word clouds (title + description + combined): logged")

            with tempfile.TemporaryDirectory() as _wc_tmp:
                for col in ("title", "description"):
                    top_words = get_cluster_top_words(labels, df_clean, column=col)
                    p = os.path.join(_wc_tmp, f"top_words_{col}.csv")
                    top_words.to_csv(p, index=False)
                    mlflow.log_artifact(p)
            print("  top words (title + description): logged")

            feat_dist_fig = plot_feature_distributions(X, labels, df_clean)
            mlflow.log_figure(feat_dist_fig, "feature_distributions.png")
            plt.close(feat_dist_fig)
            print("  feature distributions: logged")

            representatives = get_centroid_representatives(X, labels, df_clean)
            with tempfile.TemporaryDirectory() as tmp_dir:
                profile = cluster_profile(X, labels, df_clean)
                profile_path = os.path.join(tmp_dir, "cluster_profiles.csv")
                profile.to_csv(profile_path, index=False)
                mlflow.log_artifact(profile_path)

                repr_paths = log_centroid_representatives(representatives, tmp_dir)
                for p in repr_paths:
                    mlflow.log_artifact(p)

            print(f"  profile + {len(representatives)} representative files: logged")

        print(f"\nRun complete. Metrics: {metrics}")
        return labels, metrics


# =========================================================
# INFERENCE
# =========================================================

# Cache the loaded pipeline + its expected input columns so repeated calls don't
# reload from MLflow.
_MODEL = None
_EXPECTED_COLS = None


def _load_clustering_model():
    """
    Load the logged K-Means pipeline for CLUSTERING_RUN_ID, caching it module-wide.

    The logged model is the full sklearn pipeline (median imputer + StandardScaler
    + K-Means k=26). We load the raw sklearn flavor rather than pyfunc so the
    prediction is not subject to strict dtype enforcement — `prepare_data` emits
    the one-hot domain columns as int64 while the logged signature recorded them
    as bool, which the scaler/K-Means handle interchangeably. The signature is
    still used (returned alongside the model) to select the right feature columns
    in the order the model expects.

    Returns:
        tuple: (fitted sklearn pipeline, list[str] expected input column names).
    """
    global _MODEL, _EXPECTED_COLS
    if _MODEL is not None:
        return _MODEL, _EXPECTED_COLS

    configure_mlflow(experiment_name="clustering-analysis")

    model_uri = f"runs:/{CLUSTERING_RUN_ID}/model"
    logger.info("Loading clustering model from %s", model_uri)
    _MODEL = mlflow.sklearn.load_model(model_uri)
    _EXPECTED_COLS = Model.load(model_uri).get_input_schema().input_names()
    return _MODEL, _EXPECTED_COLS


def predict_clusters(df, verbose=True):
    """
    Assign a cluster id to each row of a prepared feature matrix.

    Parameters:
        df (pd.DataFrame): feature matrix produced by
            src.data.run_pipeline.prepare_data (must contain the model's
            feature columns; extra columns such as job_id are preserved).
        verbose (bool): emit INFO logging.

    Returns:
        pd.DataFrame: a copy of df with an appended 'cluster' column (int).

    Raises:
        ValueError: when df is missing required feature columns.
        RuntimeError: when the run/artifacts are missing.
    """
    if verbose:
        logger.info("=" * 60)
        logger.info("Clustering %s rows", f"{len(df):,}")
        logger.info("=" * 60)

    model, expected_cols = _load_clustering_model()

    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required feature column(s): "
            f"{missing}. Pass the output of prepare_data "
            "(src/data/run_pipeline.py)."
        )

    # Select the model's feature columns in the expected order, then predict.
    clusters = model.predict(df[expected_cols])

    out = df.copy()
    out["cluster"] = clusters.astype(int)

    if verbose:
        logger.info("Clustering complete")
        logger.info(
            "Cluster distribution:\n%s",
            out["cluster"].value_counts().sort_index().to_string(),
        )

    return out


# =========================================================
# SWEEP PLOTS
# =========================================================

def plot_kmeans_sweep(k_range, inertias, silhouette_scores, experiment_name="clustering-analysis"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(list(k_range), inertias, "o-", color="steelblue", linewidth=2)
    axes[0].set_title("Elbow Method (Inertia)")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(list(k_range), silhouette_scores, "o-", color="darkorange", linewidth=2)
    axes[1].set_title("Silhouette Score vs k")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/kmeans_sweep.png", dpi=150)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="kmeans_sweep_summary"):
        mlflow.log_figure(fig, "kmeans_sweep.png")

    plt.show()
    plt.close(fig)


def plot_hdbscan_sweep(min_cluster_sizes, n_clusters_list, noise_ratios, experiment_name="clustering-analysis"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(min_cluster_sizes, n_clusters_list, "o-", color="steelblue", linewidth=2)
    axes[0].set_title("Number of Clusters vs min_cluster_size")
    axes[0].set_xlabel("min_cluster_size")
    axes[0].set_ylabel("Number of Clusters")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(min_cluster_sizes, noise_ratios, "o-", color="darkorange", linewidth=2)
    axes[1].set_title("Noise Ratio vs min_cluster_size")
    axes[1].set_xlabel("min_cluster_size")
    axes[1].set_ylabel("Noise Ratio")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("artifacts", exist_ok=True)
    plt.savefig("artifacts/hdbscan_sweep.png", dpi=150)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="hdbscan_sweep_summary"):
        mlflow.log_figure(fig, "hdbscan_sweep.png")

    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    EXPERIMENT = "clustering-analysis"
    FEATURE_PATH = "data/processed/v6/feature_matrix.csv"
    CLEANED_PATH = "data/processed/cleaned_job_postings.csv"

    # KMeans sweep — collect inertia + silhouette across k values
    inertias = []
    silhouette_scores = []
    for k in K_RANGE:
        _, metrics = run_clustering(
            "kmeans",
            {"n_clusters": k, "random_state": 42, "n_init": 10},
            feature_path=FEATURE_PATH,
            cleaned_path=CLEANED_PATH,
            experiment_name=EXPERIMENT,
        )
        inertias.append(metrics.get("inertia", float("nan")))
        silhouette_scores.append(metrics.get("silhouette", float("nan")))

    # plot_kmeans_sweep(K_RANGE, inertias, silhouette_scores, EXPERIMENT)

    # HDBSCAN sweep — collect n_clusters + noise_ratio across min_cluster_size values
    # hdbscan_n_clusters = []
    # hdbscan_noise_ratios = []
    # for min_cluster_size in HDBSCAN_MIN_CLUSTER_SIZES:
    #     _, metrics = run_clustering(
    #         "hdbscan",
    #         {"min_cluster_size": min_cluster_size, "min_samples": 10},
    #         feature_path=FEATURE_PATH,
    #         cleaned_path=CLEANED_PATH,
    #         experiment_name=EXPERIMENT,
    #     )
    #     hdbscan_n_clusters.append(metrics.get("n_clusters", 0))
    #     hdbscan_noise_ratios.append(metrics.get("noise_ratio", float("nan")))

    # plot_hdbscan_sweep(HDBSCAN_MIN_CLUSTER_SIZES, hdbscan_n_clusters, hdbscan_noise_ratios, EXPERIMENT)
