"""
Domain classification for job postings using zero-shot cosine similarity
against hand-crafted domain prototypes.

Model: 0xnbk/nbk-ats-domain-v1-en (domain-aware BERT)

Input:  data/processed/cleaned_job_postings.csv
Output: data/precomputed/domain_probabilities.csv
        - one column per domain with cosine similarity scores
        - no argmax/threshold applied here; downstream code decides
"""

import os
import json
import logging

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Repo-root-anchored absolute path so the prototype-embedding cache resolves
# regardless of cwd (src/data is two levels below the repo root). This path is
# read on the inference path via compute_domain_sim_df -> load_prototype_embeddings.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

REPO_ID = "0xnbk/nbk-ats-domain-v1-en"
INPUT_PATH = "../../data/precomputed/summarization/text_rank_summaries.csv"
OUTPUT_PATH = "../../data/precomputed/summarization/text_rank_domain_probabilities.csv"
PROTOTYPE_EMB_PATH = os.path.join(
    _REPO_ROOT, "data", "precomputed", "domain_prototype_embeddings.npz"
)
BATCH_SIZE = 64
MAX_LENGTH = 8192

DOMAIN_PROTOTYPES = {
    "Technology":               "Software engineer Python machine learning cloud AWS DevOps",
    "Healthcare":               "Registered nurse patient care clinical hospital medical records",
    "Finance":                  "Financial analyst accounting budget investment banking CFA",
    "Education":                "Teacher curriculum classroom students learning outcomes",
    "Legal":                    "Attorney contract compliance litigation law firm paralegal",
    "Sales/Marketing":          "Sales revenue B2B lead generation marketing campaigns CRM",
    "Human Resources":          "Recruitment onboarding HR payroll employee relations talent",
    "Manufacturing/Operations": "Production line quality control supply chain lean manufacturing",
    "Design":                   "UX UI designer Figma Adobe creative visual branding",
    "Retail/Hospitality":       "Customer service retail store cashier hotel hospitality",
    "Construction/Real Estate": "Construction site manager civil engineer real estate property",
    "Government/Nonprofit":     "Public policy government nonprofit grant administration civic",
    "Media/Entertainment":      "Content creator video production journalism broadcasting media",
}


# ---------------------------------------------------------------------------
# Model download + patching
# ---------------------------------------------------------------------------
def download_and_patch_model(repo_id: str) -> str:
    """Download the HF model and apply compatibility patches."""
    local_path = snapshot_download(repo_id)
    log.info("Model downloaded to: %s", local_path)

    # Patch configuration_bert.py  (transformers.onnx removal)
    config_bert = os.path.join(local_path, "configuration_bert.py")
    if os.path.exists(config_bert):
        with open(config_bert, "r") as f:
            src = f.read()
        if "from transformers.onnx import OnnxConfig" in src:
            src = src.replace(
                "from transformers.onnx import OnnxConfig",
                "try:\n    from transformers.onnx import OnnxConfig\n"
                "except ImportError:\n    OnnxConfig = None",
            )
            with open(config_bert, "w") as f:
                f.write(src)
            log.info("Patched configuration_bert.py")

    # Patch modeling_bert.py  (pytorch_utils import path)
    modeling_bert = os.path.join(local_path, "modeling_bert.py")
    if os.path.exists(modeling_bert):
        with open(modeling_bert, "r") as f:
            src = f.read()
        old_import = "from transformers.pytorch_utils import find_pruneable_heads_and_indices"
        if old_import in src:
            src = src.replace(
                old_import,
                "try:\n"
                "    from transformers.pytorch_utils import find_pruneable_heads_and_indices\n"
                "except ImportError:\n"
                "    from transformers.utils import find_pruneable_heads_and_indices",
            )
            with open(modeling_bert, "w") as f:
                f.write(src)
            log.info("Patched modeling_bert.py")

    # Patch config.json  (feed_forward_type)
    config_json = os.path.join(local_path, "config.json")
    with open(config_json, "r") as f:
        config = json.load(f)
    if config.get("feed_forward_type") == "geglu":
        config["feed_forward_type"] = "original"
        with open(config_json, "w") as f:
            json.dump(config, f, indent=2)
        log.info("Patched config.json: feed_forward_type -> original")

    return local_path


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def encode(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int = BATCH_SIZE,
    max_length: int = MAX_LENGTH,
) -> np.ndarray:
    """Mean-pooled embeddings with batched inference."""
    all_embeddings = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding", leave=False):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)

        mask = inputs["attention_mask"].unsqueeze(-1).float()
        embs = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        all_embeddings.append(embs.numpy())

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Classification (vectorized over all rows at once)
# ---------------------------------------------------------------------------
def compute_domain_similarities(
    description_embeddings: np.ndarray,
    prototype_embeddings: np.ndarray,
) -> np.ndarray:
    """
    Cosine similarity of every description against every prototype.

    Returns: (n_descriptions, n_domains) array of similarity scores.
    """
    # L2-normalize both matrices
    desc_norms = np.linalg.norm(description_embeddings, axis=1, keepdims=True)
    desc_norms = np.where(desc_norms == 0, 1e-9, desc_norms)
    desc_unit = description_embeddings / desc_norms

    proto_norms = np.linalg.norm(prototype_embeddings, axis=1, keepdims=True)
    proto_norms = np.where(proto_norms == 0, 1e-9, proto_norms)
    proto_unit = prototype_embeddings / proto_norms

    # (N, D) @ (D, K) -> (N, K)
    return desc_unit @ proto_unit.T


# ---------------------------------------------------------------------------
# Model loading + prototype embeddings (persisted for reuse)
# ---------------------------------------------------------------------------
# Module-level cache for the loaded (tokenizer, model), keyed by repo_id, so the
# BERT model is downloaded/patched/instantiated once per process instead of on
# every inference request (compute_domain_sim_df -> load_model runs per
# prepare_data call). Mirrors the _MODEL caching in src/models/classification.py.
_MODEL_CACHE = {}


def load_model(repo_id: str = REPO_ID, force_reload: bool = False):
    """Download/patch the model and return (tokenizer, model) in eval mode.

    The result is cached module-wide per repo_id, so repeated calls (e.g. one per
    inference request) reuse a single in-memory model. Pass force_reload=True to
    bypass the cache and reload.
    """
    if not force_reload and repo_id in _MODEL_CACHE:
        return _MODEL_CACHE[repo_id]

    local_path = download_and_patch_model(repo_id)
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModel.from_pretrained(local_path)
    model.eval()
    log.info("Model loaded")
    _MODEL_CACHE[repo_id] = (tokenizer, model)
    return tokenizer, model


def warmup():
    """Eagerly load the domain model and ensure prototype embeddings exist.

    For a serving layer that wants the BERT model resident before the first
    request rather than paying the load cost on it. Safe to call repeatedly --
    load_model caches the model module-wide.
    """
    ensure_prototype_embeddings()
    load_model()


def get_prototype_embeddings(tokenizer, model, recompute=False):
    """
    Return prototype embeddings (and domain names), persisting them to disk.

    The DOMAIN_PROTOTYPES are fixed, so their embeddings are computed once at
    training time and saved to PROTOTYPE_EMB_PATH. Subsequent runs (e.g. the
    inference pipeline) load them instead of re-encoding with the BERT model.

    Returns:
        (np.ndarray, list[str]): (K, D) embeddings and the K domain names.
    """
    domain_names = list(DOMAIN_PROTOTYPES.keys())

    if not recompute and os.path.exists(PROTOTYPE_EMB_PATH):
        data = np.load(PROTOTYPE_EMB_PATH, allow_pickle=True)
        log.info("Loaded prototype embeddings from %s", PROTOTYPE_EMB_PATH)
        return data["embeddings"], data["domain_names"].tolist()

    prototype_texts = list(DOMAIN_PROTOTYPES.values())
    prototype_embs = encode(prototype_texts, tokenizer, model)
    log.info("Prototype matrix: %s", prototype_embs.shape)

    os.makedirs(os.path.dirname(PROTOTYPE_EMB_PATH), exist_ok=True)
    np.savez(
        PROTOTYPE_EMB_PATH,
        embeddings=prototype_embs,
        domain_names=np.array(domain_names, dtype=object),
    )
    log.info("Saved prototype embeddings to %s", PROTOTYPE_EMB_PATH)
    return prototype_embs, domain_names


def ensure_prototype_embeddings():
    """
    Make sure prototype embeddings exist on disk, computing them if needed.

    Cheap no-op when PROTOTYPE_EMB_PATH already exists. Otherwise loads the
    model and encodes the 13 fixed prototypes once. Useful when the full
    domain recomputation is skipped (e.g. the similarities CSV already exists)
    but the inference pipeline still needs the persisted prototype embeddings.
    """
    if os.path.exists(PROTOTYPE_EMB_PATH):
        return
    log.info("Prototype embeddings missing; generating them")
    tokenizer, model = load_model()
    get_prototype_embeddings(tokenizer, model, recompute=True)


def load_prototype_embeddings():
    """
    Load persisted prototype embeddings for inference.

    Raises:
        FileNotFoundError: when the embeddings are missing (run train_pipeline
            first to generate them).
    """
    if not os.path.exists(PROTOTYPE_EMB_PATH):
        raise FileNotFoundError(
            f"Prototype embeddings not found at '{PROTOTYPE_EMB_PATH}'. "
            "Run train_pipeline first to generate the precomputed artifacts."
        )
    data = np.load(PROTOTYPE_EMB_PATH, allow_pickle=True)
    return data["embeddings"], data["domain_names"].tolist()


# ---------------------------------------------------------------------------
# Inference: domain similarities for an in-memory dataframe
# ---------------------------------------------------------------------------
def compute_domain_sim_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute domain similarity scores for new data, returned in memory.

    Loads the BERT model and the persisted prototype embeddings, encodes the
    descriptions in ``df``, and returns a dataframe of [job_id, domain_sim_*]
    without writing to disk. Mirrors the columns produced by ``main``.

    Parameters:
        df (pd.DataFrame): Cleaned dataframe with 'job_id' and 'description'.

    Returns:
        pd.DataFrame
    """
    prototype_embs, domain_names = load_prototype_embeddings()

    tokenizer, model = load_model()

    descriptions = df["description"].fillna("").astype(str).tolist()
    log.info("Encoding %d descriptions (batch_size=%d) ...", len(descriptions), BATCH_SIZE)
    desc_embs = encode(descriptions, tokenizer, model, batch_size=BATCH_SIZE)

    similarities = compute_domain_similarities(desc_embs, prototype_embs)

    sim_df = pd.DataFrame(
        np.round(similarities, 4),
        columns=[f"domain_sim_{name}" for name in domain_names],
    )
    return pd.concat([df[["job_id"]].reset_index(drop=True), sim_df], axis=1)


# ---------------------------------------------------------------------------
# Main (training): compute + save domain similarities for the full dataset
# ---------------------------------------------------------------------------
def main():
    # ---- Load data ----
    log.info("Loading dataset from %s", INPUT_PATH)
    df = pd.read_csv(INPUT_PATH)
    log.info("Loaded %d rows", len(df))

    descriptions = df["description"].fillna("").astype(str).tolist()

    # ---- Model ----
    tokenizer, model = load_model()

    # ---- Prototype embeddings (computed once and persisted) ----
    prototype_embs, domain_names = get_prototype_embeddings(
        tokenizer, model, recompute=True
    )

    # ---- Encode all descriptions (batched) ----
    log.info("Encoding %d descriptions (batch_size=%d) ...", len(descriptions), BATCH_SIZE)
    desc_embs = encode(descriptions, tokenizer, model, batch_size=BATCH_SIZE)
    log.info("Description matrix: %s", desc_embs.shape)

    # ---- Cosine similarities (vectorized) ----
    similarities = compute_domain_similarities(desc_embs, prototype_embs)

    # ---- Build output dataframe ----
    sim_df = pd.DataFrame(
        np.round(similarities, 4),
        columns=[f"domain_sim_{name}" for name in domain_names],
    )
    result = pd.concat([df[["job_id"]], sim_df], axis=1)

    # ---- Save ----
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d rows x %d cols to %s", *result.shape, OUTPUT_PATH)


if __name__ == "__main__":
    main()