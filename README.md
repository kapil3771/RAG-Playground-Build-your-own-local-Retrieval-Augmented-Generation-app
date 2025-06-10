# 🔥 RAG Playground — Build Your Own Local Retrieval-Augmented Generation App

A fully local **Retrieval-Augmented Generation (RAG)** app:

✅ Ollama + Gemma 2B
✅ Sentence Transformers for local embeddings
✅ FAISS vector store for fast retrieval
✅ LangChain for orchestration

🚀 **Runs 100% locally — no API keys required!**
💻 **Private & Fast** — Works on your laptop.

---

## ✨ Features

* 💻 Run on your local machine — no cloud required
* 🧠 Augment your LLM with your own data (example: YouTube transcript)
* 🔍 Fast semantic search using FAISS
* 🤖 Local embedding model (`all-MiniLM-L6-v2`)
* 🗂 Modular and extensible codebase
* ⚡️ Simple run script (`run.sh`)
* 🗘️ Ready to deploy to GitHub

---

## 🛠 Tech Stack

* [LangChain](https://github.com/langchain-ai/langchain)
* [Ollama](https://ollama.com/) (with `gemma:2b`)
* [Sentence Transformers](https://www.sbert.net/)
* [FAISS](https://github.com/facebookresearch/faiss) vector store
* Python 3.10+

---

## 🚀 Installation

1️⃣ **Clone the repo**:

```bash
git clone https://github.com/kapil3771/rag-playground.git
cd rag-playground
```

2️⃣ **Create virtual environment**:

```bash
python3.10 -m venv genai-env
source genai-env/bin/activate
```

3️⃣ **Install dependencies**:

```bash
pip install -r requirements.txt
```

4️⃣ **Run Ollama server**:

```bash
ollama serve
```

👉 In **another terminal** (same env or not):

```bash
ollama pull gemma:2b
```

5️⃣ **Place your transcript.txt file**:

```
rag-playground/
├── transcript.txt  ← your input data
```

---

## 🏃️ Usage

**Run the RAG pipeline**:

```bash
bash run.sh
```

Or manually:

```bash
python rag.py
```

---

## 📂 Project Structure

```
rag-playground/
├── embeddings/                # Local embedding model wrapper
├── faiss_index_folder/        # Saved FAISS index (auto-created)
├── transcript.txt             # Input transcript (you provide)
├── rag.py                     # Main RAG pipeline
├── run.sh                     # Run script
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚖️ License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Kapil Pravin Marathe**
GitHub → [kapil3771](https://github.com/kapil3771)

---

## 🙏 Credits

* [LangChain](https://github.com/langchain-ai/langchain)
* [Ollama](https://ollama.com/)
* [FAISS (Facebook AI Research)](https://github.com/facebookresearch/faiss)
* [Sentence Transformers (SBERT)](https://www.sbert.net/)
