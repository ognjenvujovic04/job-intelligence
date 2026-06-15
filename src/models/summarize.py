"""
Job-description summarization (three variants), each a DataFrame -> DataFrame call.

This is the reusable home for the summarization "calculation" that used to live
inline in notebook 3.3 and in the one-off Colab scripts (T5 base + DistilBART).
Every summarizer takes a DataFrame with at least ``job_id`` and ``description``
columns and returns a copy with a generated ``summary`` column:

    out = textrank_summarize_df(df)        # extractive, statistical (sumy)
    out = distilbart_summarize_df(df)      # abstractive, DistilBART-CNN
    out = t5_summarize_df(df)              # abstractive, T5 base

The notebook itself no longer runs these models -- it loads the precomputed CSVs
under ``data/precomputed/summarization/``. This module documents (and can
regenerate) those artifacts; run it as ``python -m src.models.summarize``.

The abstractive functions mirror the GPU Colab scripts that produced the
precomputed CSVs (same models, prompts, length/beam settings and short-text
passthrough), but are **device-aware**: they use CUDA + fp16 when a GPU is
available and fall back to CPU + fp32 otherwise (fp16 generation is not
supported on CPU), so the same code runs locally without a GPU.

Heavy deps (``sumy``, ``torch``, ``transformers``) are imported inside the
functions so importing this module stays cheap, mirroring
``classification._load_model``.
"""

import logging
import os

logger = logging.getLogger(__name__)

# Repo-root-anchored absolute paths so this module works regardless of cwd
# (src/models is two levels below the repo root).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SUMMARY_DIR = os.path.join(_REPO_ROOT, "data", "precomputed", "summarization")

# Model identities match the Colab scripts / notebook section-5/6 titles.
DISTILBART_MODEL = "sshleifer/distilbart-cnn-12-6"
T5_MODEL = "t5-base"

# Texts shorter than this (after stripping) are kept verbatim -- too short to
# summarize. Matches the Colab scripts' guard.
MIN_CHARS_TO_SUMMARIZE = 50

DEFAULT_TEXTRANK_SENTENCES = 3

# Domain-aware task prefix from the T5 Colab script: T5 is text-to-text, so the
# prefix defines *how* it summarizes -- here, while preserving the industry domain.
T5_TASK_PREFIX = (
    "summarize this job posting while clearly stating the industry domain "
    "such as healthcare, finance, technology, retail, education, or manufacturing: "
)


# ---------------------------------------------------------------------------
# Extractive: TextRank (sumy)
# ---------------------------------------------------------------------------
def _textrank_summarize(text, sentence_count):
    """
    Extractive TextRank summary of a single string.

    Falls back to the original text when it is too short to summarize or when
    the summarizer raises. Ported from notebook 3.3.
    """
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    if not isinstance(text, str) or len(text.strip()) < MIN_CHARS_TO_SUMMARIZE:
        return text if isinstance(text, str) else ""

    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary_sentences = summarizer(parser.document, sentence_count)
        summary = " ".join(str(s) for s in summary_sentences)
        return summary if summary.strip() else text
    except Exception:
        return text


def textrank_summarize_df(
    df, text_col="description", sentence_count=DEFAULT_TEXTRANK_SENTENCES
):
    """
    Add an extractive TextRank ``summary`` column to ``df``.

    Parameters:
        df (pd.DataFrame): must contain ``job_id`` and ``text_col``.
        text_col (str): source text column (default ``description``).
        sentence_count (int): number of sentences the extractive summary keeps.

    Returns:
        pd.DataFrame: copy of ``df[["job_id", text_col]]`` with a ``summary`` column.
    """
    out = df[["job_id", text_col]].copy()
    logger.info("TextRank summarizing %d rows (sentence_count=%d)", len(out), sentence_count)
    out["summary"] = out[text_col].apply(
        lambda t: _textrank_summarize(t, sentence_count)
    )
    return out


# ---------------------------------------------------------------------------
# Abstractive helpers
# ---------------------------------------------------------------------------
def _device_and_dtype():
    """
    Pick (torch device, dtype) for generation.

    GPU -> (cuda, float16) as in the Colab scripts; CPU -> (cpu, float32),
    because fp16 generation is unsupported / extremely slow on CPU.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    return torch.device("cpu"), torch.float32


# ---------------------------------------------------------------------------
# Abstractive: DistilBART-CNN
# ---------------------------------------------------------------------------
def distilbart_summarize_df(
    df,
    text_col="description",
    model_name=DISTILBART_MODEL,
    batch_size=32,
    max_input_chars=1024,
    max_summary_tokens=142,
    min_summary_tokens=40,
):
    """
    Add an abstractive DistilBART-CNN ``summary`` column to ``df``.

    Mirrors the Colab DistilBART script: a ``transformers`` summarization
    pipeline, descriptions truncated to ``max_input_chars`` characters, greedy
    decoding (``do_sample=False``), and short texts kept verbatim. fp16 on GPU,
    fp32 on CPU. The Colab default batch size was 128 (GPU); 32 is a safer local
    default.

    Returns:
        pd.DataFrame: copy of ``df[["job_id", text_col]]`` with a ``summary`` column.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

    out = df[["job_id", text_col]].copy()
    texts = out[text_col].fillna("").astype(str).tolist()

    device, dtype = _device_and_dtype()
    logger.info("Loading %s on %s (%s)", model_name, device.type, dtype)
    # Preload the PyTorch model/tokenizer and pin framework="pt" so the pipeline
    # never tries to resolve the TF flavor -- transformers' framework inference
    # otherwise imports the TF model when tensorflow is installed, which fails
    # under Keras 3.
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        framework="pt",
        device=0 if device.type == "cuda" else -1,
    )

    summaries = [None] * len(texts)
    # Long-enough texts go through the model; short ones are kept verbatim.
    to_idx, to_txt = [], []
    for i, t in enumerate(texts):
        truncated = t[:max_input_chars] if len(t) > MIN_CHARS_TO_SUMMARIZE else t
        if len(truncated.strip()) < MIN_CHARS_TO_SUMMARIZE:
            summaries[i] = truncated
        else:
            to_idx.append(i)
            to_txt.append(truncated)

    for start in range(0, len(to_txt), batch_size):
        chunk_idx = to_idx[start : start + batch_size]
        chunk_txt = to_txt[start : start + batch_size]
        try:
            results = summarizer(
                chunk_txt,
                max_length=max_summary_tokens,
                min_length=min_summary_tokens,
                do_sample=False,
                truncation=True,
                batch_size=batch_size,
            )
            for i, r in zip(chunk_idx, results):
                summaries[i] = r["summary_text"]
        except Exception as e:  # fall back to the (truncated) input on failure
            logger.warning("DistilBART batch at %d failed: %s", start, e)
            for i in chunk_idx:
                summaries[i] = texts[i][:max_input_chars]
        logger.info("DistilBART summarized %d/%d", min(start + batch_size, len(to_txt)), len(to_txt))

    out["summary"] = summaries
    return out


# ---------------------------------------------------------------------------
# Abstractive: T5 base
# ---------------------------------------------------------------------------
def t5_summarize_df(
    df,
    text_col="description",
    model_name=T5_MODEL,
    batch_size=16,
    max_input_tokens=480,
    max_summary_tokens=150,
    min_summary_tokens=40,
):
    """
    Add an abstractive T5 ``summary`` column to ``df``.

    Mirrors the Colab T5 script: ``T5Tokenizer(legacy=True)`` +
    ``T5ForConditionalGeneration``, the domain-aware ``T5_TASK_PREFIX``, beam
    search (``num_beams=4``, ``length_penalty=1.0``, ``no_repeat_ngram_size=3``),
    and short texts kept verbatim. T5 base has a 512-token window, so inputs are
    truncated to ``max_input_tokens`` (+ the prefix tokens). fp16 on GPU, fp32 on
    CPU. The Colab default batch size was 32 (GPU); 16 is a safer local default.

    Returns:
        pd.DataFrame: copy of ``df[["job_id", text_col]]`` with a ``summary`` column.
    """
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer

    out = df[["job_id", text_col]].copy()
    texts = out[text_col].fillna("").astype(str).tolist()

    device, dtype = _device_and_dtype()
    logger.info("Loading %s on %s (%s)", model_name, device.type, dtype)
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=True)
    model = T5ForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype
    ).to(device)
    model.eval()

    prefix_len = len(tokenizer.encode(T5_TASK_PREFIX, add_special_tokens=False))

    summaries = [None] * len(texts)
    to_idx, to_txt = [], []
    for i, t in enumerate(texts):
        if len(t.strip()) < MIN_CHARS_TO_SUMMARIZE:
            summaries[i] = t
        else:
            to_idx.append(i)
            to_txt.append(t)

    for start in range(0, len(to_txt), batch_size):
        chunk_idx = to_idx[start : start + batch_size]
        chunk_txt = to_txt[start : start + batch_size]
        try:
            inputs = tokenizer(
                [T5_TASK_PREFIX + t for t in chunk_txt],
                max_length=max_input_tokens + prefix_len,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_summary_tokens,
                    min_length=min_summary_tokens,
                    num_beams=4,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            for i, s in zip(chunk_idx, decoded):
                summaries[i] = s
        except Exception as e:  # fall back to a truncated input on failure
            logger.warning("T5 batch at %d failed: %s", start, e)
            for i in chunk_idx:
                summaries[i] = texts[i][:500]
        logger.info("T5 summarized %d/%d", min(start + batch_size, len(to_txt)), len(to_txt))

    out["summary"] = summaries
    return out


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------
_SUMMARIZERS = {
    "textrank": textrank_summarize_df,
    "distilbart": distilbart_summarize_df,
    "t5": t5_summarize_df,
}


def summarize(df, method, **kwargs):
    """
    Dispatch to one of the three summarizers by name.

    Parameters:
        df (pd.DataFrame): input postings.
        method (str): one of ``"textrank"``, ``"distilbart"``, ``"t5"``.
        **kwargs: forwarded to the chosen summarizer.

    Returns:
        pd.DataFrame: ``df`` with a ``summary`` column.
    """
    if method not in _SUMMARIZERS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {sorted(_SUMMARIZERS)}."
        )
    return _SUMMARIZERS[method](df, **kwargs)


if __name__ == "__main__":
    # Regenerate the three precomputed summary CSVs from the 2k sample exported
    # by notebook 3.3 (job_descriptions_2k.csv). Documents how the artifacts in
    # data/precomputed/summarization/ were produced. The abstractive models are
    # heavy on CPU; a GPU/Colab is strongly recommended.
    import pandas as pd

    logging.basicConfig(
        level=logging.INFO, format="[%(levelname)s] %(asctime)s - %(message)s"
    )

    input_csv = os.path.join(_REPO_ROOT, "data/processed/descriptions/job_descriptions_2k.csv")
    df_in = pd.read_csv(input_csv)
    print(f"Loaded {len(df_in):,} postings from {input_csv}")

    os.makedirs(_SUMMARY_DIR, exist_ok=True)

    # TextRank stores the summary in a `description` column (the schema the
    # notebook's TextRank loader expects); the abstractive variants keep both
    # the original `description` and the generated `summary`.
    textrank = textrank_summarize_df(df_in)
    textrank[["job_id", "summary"]].rename(columns={"summary": "description"}).to_csv(
        os.path.join(_SUMMARY_DIR, "text_rank_summaries.csv"), index=False
    )

    distilbart_summarize_df(df_in).to_csv(
        os.path.join(_SUMMARY_DIR, "bart_job_summaries.csv"), index=False
    )
    t5_summarize_df(df_in).to_csv(
        os.path.join(_SUMMARY_DIR, "t5_job_summaries.csv"), index=False
    )

    print(f"Wrote summary CSVs to {_SUMMARY_DIR}")
