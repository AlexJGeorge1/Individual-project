#!/usr/bin/env python3

"""Minimal CLI placeholder for rag-qa.
This file currently provides a stub interface for future wiring.
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="rag-qa CLI (placeholder)")
    parser.add_argument("--ingest", type=str, help="Path to corpus directory", required=False)
    parser.add_argument("--build-index", type=str, help="Path to index output directory", required=False)
    parser.add_argument("--ask", type=str, help="Question to ask", required=False)
    parser.add_argument("--k", type=int, default=5, help="Top-K results")
    _ = parser.parse_args()
    print("rag-qa scaffold is set up. Implement features inside the ragqa/ package.")


if __name__ == "__main__":
    main()
