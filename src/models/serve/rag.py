"""
Retrieval-Augmented Generation over the job postings, as a single query -> answer call.

This is the reusable home for the RAG "calculation" prototyped in notebook 3.5.
It loads a prebuilt FAISS index + metadata, retrieves the top-k most similar
postings for a free-text query, and asks a local Ollama LLM to answer using only
those postings as context:

    result = answer_query("remote ML jobs that need python and sql")
    # -> {"query", "answer", "model", "retrieved": [{rank, job_id, score, document}, ...]}

Unlike the prediction tracks, RAG does **not** follow the ``raw df -> df`` contract
of ``src.models.serve.inference`` -- it takes a query string and returns the LLM
answer plus the retrieved job metadata -- so the API service imports this module
directly rather than going through that facade.

The embedding model, retrieval depth, and document cap are fixed module constants.
The Ollama base URL and model are environment-overridable (``OLLAMA_BASE_URL`` /
``OLLAMA_MODEL``) so the Dockerized API can point at an Ollama running on the host
and use whichever model is pulled there.

Heavy deps (``faiss``, ``sentence_transformers``, ``torch``, ``requests``) are
imported inside the functions so importing this module stays cheap, mirroring
``src.models.serve.summarize``. The embedding model and the index are cached
module-wide, so a long-running process loads each once on first use.
"""

import logging
import os

logger = logging.getLogger(__name__)


class OllamaUnreachableError(RuntimeError):
    """Raised when the Ollama server can't be reached for generation.

    A domain error so the API layer can translate it into a clean 503 instead of
    leaking a ``requests.ConnectionError`` traceback as a generic 500.
    """


# Repo-root-anchored absolute paths so this module works regardless of cwd
# (src/models/serve is three levels below the repo root).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RAG_DIR = os.path.join(_REPO_ROOT, "data", "precomputed", "rag")
METADATA_FILE = os.path.join(_RAG_DIR, "job_metadata.csv")
FAISS_INDEX = os.path.join(_RAG_DIR, "faiss_index.bin")

ID_COL = "job_id"
DOC_COL = "document"

# Asymmetric embedding model: queries are prefixed "search_query:" and the
# indexed documents were embedded with a "search_document:" prefix (see notebook 3.5).
EMBED_MODEL = "nomic-ai/nomic-embed-text-v1"

# Ollama: both the base URL (host + port) and the model are overridable so the
# container can point at the host's Ollama (e.g. http://host.docker.internal:11434)
# and use whichever model is pulled there (e.g. OLLAMA_MODEL=llama3.2:3b). The
# values below are the defaults when the env vars are unset.
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "mistral"

TOP_K = 5
MAX_DOC_CHARS = 1500

# Module-level caches: the SentenceTransformer and the (index, metadata) pair are
# loaded once on first use and kept resident across requests, mirroring
# src/models/serve/summarize.py's _T5_CACHE.
_EMBED_CACHE = {}
_INDEX_CACHE = {}


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
def _ollama_base_url():
    """Ollama base URL, overridable via the OLLAMA_BASE_URL env var."""
    return os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL)


def _ollama_model():
    """Ollama model name, overridable via the OLLAMA_MODEL env var."""
    return os.environ.get("OLLAMA_MODEL", OLLAMA_MODEL)


def _load_embed_model():
    """Return the cached SentenceTransformer, loading it on first use."""
    cached = _EMBED_CACHE.get(EMBED_MODEL)
    if cached is not None:
        return cached

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading embedding model %s on %s", EMBED_MODEL, device)
    model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device=device)

    _EMBED_CACHE[EMBED_MODEL] = model
    return model


def _load_index():
    """Return the cached ``(faiss_index, metadata_df)``, loading on first use."""
    cached = _INDEX_CACHE.get("loaded")
    if cached is not None:
        return cached

    import faiss
    import pandas as pd

    metadata = pd.read_csv(METADATA_FILE)
    index = faiss.read_index(FAISS_INDEX)
    if len(metadata) != index.ntotal:
        raise ValueError(
            f"Metadata rows ({len(metadata)}) != index vectors ({index.ntotal}); "
            "the RAG artifacts are out of sync."
        )
    logger.info("Loaded FAISS index: %d vectors (dim=%d)", index.ntotal, index.d)

    _INDEX_CACHE["loaded"] = (index, metadata)
    return index, metadata


# ---------------------------------------------------------------------------
# Core RAG steps (ported from notebook 3.5)
# ---------------------------------------------------------------------------
def retrieve(query, top_k=TOP_K):
    """Encode the query and return the top-k most similar postings.

    Returns a list of ``{rank, job_id, score, document}`` dicts (documents
    truncated to ``MAX_DOC_CHARS``).
    """
    import numpy as np

    model = _load_embed_model()
    index, metadata = _load_index()

    query_embedding = model.encode(
        [f"search_query: {query}"],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    scores, indices = index.search(query_embedding, k=top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        row = metadata.iloc[idx]
        results.append(
            {
                "rank": rank,
                "job_id": int(row[ID_COL]),
                "score": float(score),
                "document": str(row[DOC_COL])[:MAX_DOC_CHARS],
            }
        )
    return results


def build_prompt(query, retrieved_docs):
    """Build the (system, user) prompt that grounds the answer in the retrieved postings."""
    context_block = "\n\n".join(
        f'--- Job {doc["rank"]} (score: {doc["score"]:.3f}) ---\n{doc["document"]}'
        for doc in retrieved_docs
    )

    system_prompt = (
        "You are a helpful job market assistant. "
        "Answer the user's question using ONLY the job postings provided below. "
        "If the answer cannot be found in the provided postings, say so clearly. "
        "Be concise and specific. Reference job details when possible."
    )

    user_prompt = (
        f"Here are the most relevant job postings:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Question: {query}"
    )

    return system_prompt, user_prompt


def generate(system_prompt, user_prompt):
    """Send the prompt to Ollama (/api/chat, streaming disabled) and return the answer text.

    Raises ``OllamaUnreachableError`` if the server can't be contacted (DNS/connect/
    timeout) so the caller can surface a clean error rather than a raw traceback.
    """
    import requests

    base_url = _ollama_base_url()
    payload = {
        "model": _ollama_model(),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }

    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=120)
    except (requests.ConnectionError, requests.Timeout) as exc:
        raise OllamaUnreachableError(
            f"Could not reach Ollama at {base_url}. Is it running and reachable "
            "from here? (In Docker, point OLLAMA_BASE_URL at the host, e.g. "
            "http://host.docker.internal:11434.)"
        ) from exc
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def answer_query(query, top_k=TOP_K):
    """Full RAG pipeline: retrieve postings, build the prompt, generate an answer.

    Returns ``{"query", "answer", "model", "retrieved": [...]}`` -- the Ollama
    answer alongside the metadata of the jobs retrieved for the query.
    """
    docs = retrieve(query, top_k=top_k)
    system_prompt, user_prompt = build_prompt(query, docs)
    answer = generate(system_prompt, user_prompt)
    return {
        "query": query,
        "answer": answer,
        "model": _ollama_model(),
        "retrieved": docs,
    }


# ---------------------------------------------------------------------------
# Health / warmup
# ---------------------------------------------------------------------------
def check_ollama():
    """Return True if the Ollama server is reachable, False otherwise."""
    import requests

    try:
        resp = requests.get(f"{_ollama_base_url()}/api/tags", timeout=5)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        return False


def warmup():
    """Eagerly load the embedding model + FAISS index so the first /rag is fast.

    Ollama is external and is intentionally not contacted here -- it does not need
    to be up at warmup, only when a query is actually answered.
    """
    _load_embed_model()
    _load_index()
    return True


if __name__ == "__main__":
    # Retrieval + (if Ollama is up) generation smoke test over a sample query.
    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
    )

    sample = "remote machine learning engineer with python"
    print(f"Query: {sample}\n")

    for doc in retrieve(sample):
        print(f'  Rank {doc["rank"]} | score {doc["score"]:.4f} | job_id {doc["job_id"]}')

    if check_ollama():
        print("\nGenerating answer via Ollama...\n")
        result = answer_query(sample)
        print(result["answer"])
    else:
        print(f"\nOllama not reachable at {_ollama_base_url()}; skipping generation.")
