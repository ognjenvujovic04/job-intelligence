# src/clustering.py

import time
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from wordcloud import WordCloud
import warnings
import os
warnings.filterwarnings("ignore")


# =========================================================
# DATA PREPARATION
# =========================================================

def load_and_prepare(feature_path, cleaned_path, drop_high_missing=True):
    """Load feature matrix + cleaned df for interpretation."""
    df_feat = pd.read_csv(feature_path)
    df_clean = pd.read_csv(cleaned_path)

    # Use job_id to align df_clean to the exact rows (and order) in the feature matrix.
    # The feature matrix is a subset of cleaned_job_postings (80% split + noise removal).
    if 'job_id' in df_feat.columns:
        job_ids = df_feat['job_id'].values
        df_clean = (
            df_clean.set_index('job_id')
            .loc[job_ids]
            .reset_index()
        )
        df_feat = df_feat.drop(columns=['job_id'])

    # Drop columns with too much missing data for clustering
    drop_cols = []
    if drop_high_missing:
        drop_cols += ["log_applies", "apply_rate"]
    # Drop target-encoded cols (leak supervised target)
    drop_cols += ["state_salary_enc", "fips_salary_enc"]

    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # Impute remaining missing (log_views ~1.3%, experience_level_ord ~23.8%)
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(
        imputer.fit_transform(df_feat),
        columns=df_feat.columns
    )

    # Scale continuous features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns
    )

    return X_scaled, df_clean


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

def compute_metrics(X, labels, sample_size=10_000):
    """Compute internal clustering metrics on a sample. Skips if only 1 cluster."""
    n_clusters = len(set(labels) - {-1})  # exclude noise label
    if n_clusters < 2:
        return {"n_clusters": n_clusters}

    # Exclude noise points
    mask = labels != -1
    X_valid = X[mask]
    labels_valid = labels[mask]

    # Sample once; reuse for all three metrics so scores are comparable
    n = len(X_valid)
    actual_sample = min(sample_size, n)
    rng = np.random.default_rng(42)
    idx = rng.choice(n, size=actual_sample, replace=False)
    X_s = X_valid.iloc[idx] if hasattr(X_valid, "iloc") else X_valid[idx]
    l_s = labels_valid[idx]

    print(f"  Computing metrics on {actual_sample:,} / {n:,} points ...")
    return {
        "n_clusters": n_clusters,
        "n_noise": int((labels == -1).sum()),
        "silhouette": float(silhouette_score(X_s, l_s)),
        "davies_bouldin": float(davies_bouldin_score(X_s, l_s)),
        "calinski_harabasz": float(calinski_harabasz_score(X_s, l_s)),
    }


# =========================================================
# INTERPRETATION / ARTIFACTS
# =========================================================

def plot_feature_distributions(X, labels, df_clean, output_dir="artifacts"):
    """Box plots of key features per cluster."""
    os.makedirs(output_dir, exist_ok=True)
    
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
    path = f"{output_dir}/feature_distributions.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_word_clouds(labels, df_clean, column="title", output_dir="artifacts"):
    """Generate a word cloud per cluster for a text column."""
    os.makedirs(output_dir, exist_ok=True)
    unique_labels = sorted(set(labels) - {-1})
    
    fig, axes = plt.subplots(1, len(unique_labels), figsize=(6 * len(unique_labels), 5))
    if len(unique_labels) == 1:
        axes = [axes]

    for ax, cl in zip(axes, unique_labels):
        text = " ".join(df_clean.loc[labels == cl, column].dropna().astype(str))
        wc = WordCloud(width=600, height=400, background_color="white").generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"Cluster {cl}")
        ax.axis("off")

    plt.suptitle(f"Word Clouds: {column}", fontsize=14)
    plt.tight_layout()
    path = f"{output_dir}/wordcloud_{column}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def get_centroid_representatives(X, labels, df_clean, n=5):
    """Find the n rows closest to each cluster centroid."""
    results = {}
    for cl in sorted(set(labels) - {-1}):
        mask = labels == cl
        cluster_data = X[mask]
        centroid = cluster_data.mean(axis=0).values
        distances = np.linalg.norm(cluster_data.values - centroid, axis=1)
        closest_idx = cluster_data.index[np.argsort(distances)[:n]]
        results[cl] = df_clean.loc[closest_idx, 
                      ["title", "company_name", "location", 
                       "formatted_work_type", "formatted_experience_level"]]
    return results


def log_centroid_representatives(representatives, output_dir="artifacts"):
    """Save representative rows to CSV for MLflow logging."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for cl, df_repr in representatives.items():
        path = f"{output_dir}/cluster_{cl}_representatives.csv"
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
        
        # Mean of numeric features
        row.update(X[mask].mean().to_dict())
        
        # Top title keyword, work type, experience level from original data
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

def run_clustering(
    model_name="kmeans",
    model_params=None,
    feature_path="data/processed/v5/feature_matrix_train.csv",
    cleaned_path="data/processed/cleaned_job_postings.csv",
    experiment_name="clustering-analysis",
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
        X, df_clean = load_and_prepare(feature_path, cleaned_path)
        print(f"  X shape: {X.shape}  |  df_clean rows: {len(df_clean):,}  ({time.time()-t0:.1f}s)")

        # 2. Fit
        print(f"[2/4] Fitting {model_name} ...")
        t0 = time.time()
        model = get_model(model_name, **model_params)
        labels = model.fit_predict(X)
        sizes = np.bincount(labels[labels >= 0])
        print(f"  Done in {time.time()-t0:.1f}s  |  cluster sizes: {sizes.tolist()}")

        # 3. Metrics
        print("[3/4] Computing metrics ...")
        t0 = time.time()
        metrics = compute_metrics(X.values, labels)
        print(f"  {metrics}  ({time.time()-t0:.1f}s)")
        mlflow.log_params({
            "model_name": model_name,
            "dataset": os.path.splitext(os.path.basename(feature_path))[0],
            **model_params,
        })
        mlflow.log_metrics(metrics)

        # 4. Artifacts
        n_clusters = metrics.get("n_clusters", 0)
        if n_clusters > 50:
            print(f"[4/4] Skipping artifacts ({n_clusters} clusters > 50).")
        else:
            print("[4/4] Generating artifacts ...")
            wc_title = plot_word_clouds(labels, df_clean, "title")
            print(f"  word cloud (title): {wc_title}")
            wc_desc = plot_word_clouds(labels, df_clean, "description")
            print(f"  word cloud (description): {wc_desc}")
            feat_dist = plot_feature_distributions(X, labels, df_clean)
            print(f"  feature distributions: {feat_dist}")

            representatives = get_centroid_representatives(X, labels, df_clean)
            repr_paths = log_centroid_representatives(representatives)
            print(f"  representatives: {len(repr_paths)} files")

            profile = cluster_profile(X, labels, df_clean)
            profile.to_csv("artifacts/cluster_profiles.csv", index=False)

            mlflow.log_artifact(wc_title)
            mlflow.log_artifact(wc_desc)
            mlflow.log_artifact(feat_dist)
            mlflow.log_artifact("artifacts/cluster_profiles.csv")
            for p in repr_paths:
                mlflow.log_artifact(p)

        print(f"\nRun complete. Metrics: {metrics}")
        return labels, metrics
    

if __name__ == "__main__":
    # Try different K values
    for k in range(8, 40, 2):
        run_clustering("kmeans", {"n_clusters": k, "random_state": 42, "n_init": 10})

    # Sweep min_cluster_size for HDBSCAN
    for min_cluster_size in [500, 750, 1000, 1250, 1500]:
        run_clustering("hdbscan", {"min_cluster_size": min_cluster_size, "min_samples": 10})
