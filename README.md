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

## 🧹 Tech Stack

| Component       | Technology                                       | Purpose                     |
| --------------- | ------------------------------------------------ | --------------------------- |
| **Language**    | Python 3.10+                                     | Core implementation         |
| **NLP Models**  | Hugging Face Transformers, Sentence-Transformers | Embeddings, reranking, QA   |
| **Retrieval**   | BM25 (rank-bm25), FAISS                          | Sparse + dense search       |
| **Parsing**     | PyPDF                                            | PDF text extraction         |
| **Evaluation**  | NumPy, scikit-learn                              | EM/F1 metrics               |
| **Optional UI** | Streamlit / Gradio                               | Web interface for questions |

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/rag-qa.git
cd rag-qa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🧠 Usage

### 1. Build the index

```bash
python rag_qa_minimal.py index --data_dir ./corpus
```

### 2. Ask a question

```bash
python rag_qa_minimal.py ask --question "What is the submission deadline?" --rerank
```

### 3. (Optional) Evaluate

Create a CSV file `gold_qa.csv` with columns `question,answer`:

```bash
python rag_qa_minimal.py eval --qa_file ./gold_qa.csv --rerank
```

---

## 📊 Evaluation

* **Exact Match (EM)** and **F1** scores to assess answer quality.
* **Ablation tests** comparing sparse vs dense retrieval.
* **Human review** for answer correctness and relevance.

---

## 🎯 Learning Outcomes

This project demonstrates:

* Real-world **Natural Language Processing (NLP)** workflow
* Integration of **Information Retrieval** and **Question Answering**
* **Evaluation** and analysis of AI performance
* **Ethical, transparent AI** — no cloud models or private data exposure

Designed as part of a **Final Year Computer Science with AI project.**

---

## 📜 License

Released under the **MIT License** – free to use, modify, and learn from.

---

## 👨‍💻 Author

Developed by **Alex George**
For the **Individual Project Module**,
*MEng, BSc (Hons) Computer Science with Artificial Intelligence.*
