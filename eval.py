"""
eval.py — Evaluation script for the Fashion Retrieval System

Runs the 5 required evaluation queries from the assignment prompt,
prints results, and saves a structured evaluation report.
"""

import os
import json
import shutil
from retriever import FashionRetriever


# ── The 5 evaluation queries from the assignment ───────────────────────
EVAL_QUERIES = [
    "A person wearing a red floral dress standing in a sunny garden",
    "A man in a navy blue suit with a striped tie at a formal event",
    "A woman in athletic wear jogging in an urban park",
    "A child wearing a yellow raincoat splashing in puddles",
    "A couple dressed in matching white outfits on a beach at sunset",
]

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "eval_results"
)


def run_evaluation():
    """Run all evaluation queries and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    retriever = FashionRetriever(top_k=5)

    all_results = {}

    for qi, query in enumerate(EVAL_QUERIES, 1):
        print(f"\n{'=' * 70}")
        print(f"  Query {qi}: {query}")
        print(f"{'=' * 70}")

        results = retriever.search(query)
        all_results[f"query_{qi}"] = {
            "query": query,
            "results": results,
        }

        # Print results
        for r in results:
            print(
                f"  #{r['rank']:>2}  "
                f"ITM: {r['itm_score']:.4f}  "
                f"Recall: {r['recall_score']:.4f}  "
                f"File: {r['filename']}"
            )

        # Copy top-k images to results directory for easy inspection
        query_dir = os.path.join(RESULTS_DIR, f"query_{qi}")
        os.makedirs(query_dir, exist_ok=True)

        for r in results:
            src = r["path"]
            dst = os.path.join(query_dir, f"rank{r['rank']}_{r['filename']}")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"  [warn] Could not copy {src}: {e}")

    # Save structured JSON report
    report_path = os.path.join(RESULTS_DIR, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n[eval] Full report saved to {report_path}")
    print(f"[eval] Top-k images copied to {RESULTS_DIR}/query_*/")


if __name__ == "__main__":
    run_evaluation()
