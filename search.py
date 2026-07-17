"""
search.py — Interactive CLI for Fashion Image Search

Usage:
    python search.py "a person wearing a red dress in a garden"
    python search.py                          # interactive mode
"""

import sys
import os
from retriever import FashionRetriever


def print_results(results: list[dict], query: str):
    """Pretty-print search results."""
    print(f"\n{'-' * 60}")
    print(f"  Query: {query}")
    print(f"{'-' * 60}")
    for r in results:
        print(
            f"  #{r['rank']:>2}  "
            f"ITM: {r['itm_score']:.4f}  "
            f"Recall: {r['recall_score']:.4f}  "
            f"File: {r['filename']}"
        )
    print(f"{'-' * 60}\n")


def main():
    retriever = FashionRetriever()

    # If a query is passed as a CLI argument, run it and exit
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        results = retriever.search(query)
        print_results(results, query)
        return

    # Otherwise, interactive mode
    print("\n[*] Fashion Image Search - Interactive Mode")
    print("   Type your query and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        results = retriever.search(query)
        print_results(results, query)


if __name__ == "__main__":
    main()
