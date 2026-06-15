"""
Small smoke test: take one job description and run it through all three
summarizers (TextRank, DistilBART, T5), printing the original vs each summary.

This exercises `src/models/summarize.py` end-to-end. The two abstractive models
(DistilBART-CNN, T5 base) are downloaded from HuggingFace on first run and are
slow on CPU, so this is a manual smoke test rather than a fast unit test.

Run from anywhere:
    python tests/test_summarize.py
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SYNTHETIC_CSV = os.path.join(REPO_ROOT, "data", "raw", "synthetic_postings.csv")


def main():
    # Put the repo root on sys.path so the `src.*` package imports resolve.
    sys.path.insert(0, REPO_ROOT)
    from src.models.summarize import (
        distilbart_summarize_df,
        t5_summarize_df,
        textrank_summarize_df,
    )

    # Take the first synthetic posting as our single test description.
    df = pd.read_csv(SYNTHETIC_CSV)
    one = df[["job_id", "description"]].head(1).reset_index(drop=True)
    original = one["description"].iloc[0]
    print(f"Loaded posting job_id={one['job_id'].iloc[0]} from {SYNTHETIC_CSV}")

    print("\n" + "=" * 70)
    print("ORIGINAL")
    print("=" * 70)
    print(f"({len(original)} chars)\n{original}")

    # Each summarizer takes a DataFrame and returns one with a `summary` column.
    summarizers = {
        "TextRank (extractive)": textrank_summarize_df,
        "DistilBART (abstractive)": distilbart_summarize_df,
        "T5 base (abstractive)": t5_summarize_df,
    }

    for name, summarize_fn in summarizers.items():
        summary = summarize_fn(one)["summary"].iloc[0]
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(f"({len(summary)} chars)\n{summary}")


if __name__ == "__main__":
    main()
