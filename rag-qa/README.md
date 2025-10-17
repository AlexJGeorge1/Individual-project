# rag-qa

Local Retrieval-Augmented Question Answering (RAG-QA) project scaffold.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Quick Start

1. Place documents in `data/corpus/` (PDFs or text files).
2. Run ingestion and indexing via the minimal CLI:

```bash
python rag_qa_minimal.py --ingest data/corpus --build-index data/index
```

3. Ask a question:

```bash
python rag_qa_minimal.py --ask "What is in these docs?" --k 5
```

## Project Structure

```text
rag-qa/
├─ rag_qa_minimal.py           # CLI entry point
├─ ragqa/
│  ├─ __init__.py
│  ├─ ingest.py                # Document parsing, chunking
│  ├─ index_sparse.py          # BM25 indexing
│  ├─ index_dense.py           # Embeddings + FAISS
│  ├─ retrieve.py              # Hybrid retrieval + fusion
│  ├─ rerank.py                # Cross-encoder reranker
│  ├─ qa_extractive.py         # RoBERTa SQuAD2 QA
│  ├─ qa_generative.py         # Llama 3.2 via Ollama
│  ├─ eval.py                  # EM/F1 evaluation
│  └─ utils.py                 # Shared utilities
├─ tests/
│  ├─ __init__.py
│  ├─ test_chunking.py
│  └─ test_retrieval.py
├─ data/
│  ├─ corpus/.gitkeep
│  ├─ index/.gitkeep
│  ├─ gold_qa.csv
│  └─ mini_corpus/.gitkeep
├─ requirements.txt
├─ README.md
├─ .gitignore
└─ LICENSE
```

This repository currently provides a skeleton; module files are minimal placeholders to help you wire up the rest.
