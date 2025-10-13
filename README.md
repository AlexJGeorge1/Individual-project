# Individual-project
Local Retrieval-Augmented Question Answering (RAG) system that reads your documents and answers questions using open-source NLP models (BM25, FAISS, MiniLM, and RoBERTa). Runs entirely offline — no APIs required.

# 🧠 RAG-QA: Retrieval-Augmented Question Answering System

**RAG-QA** is a local, open-source NLP system that can read a folder of documents (PDF, TXT, MD) and answer natural-language questions about them.  
It combines **information retrieval** and **extractive question answering** — similar to ChatGPT, but limited to your own data and completely offline.

---

## 🚀 Overview

When you ask a question, RAG-QA:
1. **Indexes** your documents using both keyword search (BM25) and semantic search (MiniLM + FAISS).
2. **Retrieves** the most relevant text passages.
3. Optionally **reranks** them using a cross-encoder model.
4. **Extracts** the best answer with a QA model (`deepset/roberta-base-squad2`).
5. Returns the answer, confidence score, and original source passage.

No cloud APIs, no internet dependency — all models run locally.

---

## 🧩 Tech Stack

| Component | Technology | Purpose |
|------------|-------------|----------|
| **Language** | Python 3.10+ | Core implementation |
| **NLP Models** | Hugging Face Transformers, Sentence-Transformers | Embeddings, reranking, QA |
| **Retrieval** | BM25 (rank-bm25), FAISS | Sparse + dense search |
| **Parsing** | PyPDF | PDF text extraction |
| **Evaluation** | NumPy, scikit-learn | EM/F1 metrics |
| **Optional UI** | Streamlit / Gradio | Web interface for questions |

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/rag-qa.git
cd rag-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
