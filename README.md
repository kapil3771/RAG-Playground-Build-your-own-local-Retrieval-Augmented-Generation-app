RAG Pipeline with Ollama + Local Embeddings + FAISS

Retrieve & Augment Pipeline (RAG) with:
	•	Local LLM: Ollama (gemma:2b)
	•	Local Embedding Model: sentence-transformers/all-MiniLM-L6-v2
	•	Vector Store: FAISS (with persistence)
	•	Context: YouTube Transcript or Manual Transcript
	•	Framework: LangChain + Ollama + FAISS + Sentence Transformers

⸻

Project Structure

rag-playground/
├── rag.py              # Main RAG pipeline
├── embeddings/
│   └── local_embedding_model.py
├── transcript.txt      # Your transcript to process
├── faiss_index_folder/ # FAISS index (auto created)
├── run.sh              # Run script
├── requirements.txt    # Python dependencies
├── README.md
└── LICENSE


⸻

Setup Instructions

1. Clone the repo & enter folder

git clone https://github.com/yourusername/rag-playground.git
cd rag-playground

2. Setup Python environment

python3 -m venv genai-env
source genai-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

3. Install Ollama

brew install --cask ollama
ollama serve  # Run this in a separate terminal tab/window

4. Pull required model

ollama pull gemma:2b

5. Provide transcript

Place your transcript in:

transcript.txt

(Plain text file)

6. Run the pipeline

./run.sh

Or to force clean the FAISS index:

./run.sh --clean


⸻

Example Questions

questions = [
    "What is nuclear fusion?",
    "What temperature is needed for fusion?",
    "What are the challenges in achieving fusion?",
    "What does the sun use for its energy?",
    "Is plasma mentioned in the context?"
]


⸻

Features
	•	Works fully offline
	•	No OpenAI API keys required
	•	Ollama serves LLM locally
	•	SentenceTransformers for embeddings
	•	FAISS persistence – avoids recomputing embeddings
	•	Modular and clean code

⸻

Notes
	•	You can replace gemma:2b with any other Ollama model (e.g. mistral, llama3, qwen, etc).
	•	You can use any sentence-transformers embedding model.
	•	The pipeline is modular and extensible.

⸻

License

MIT License
See LICENSE file.

⸻

Contribution

Pull requests welcome! If you like this project, please ⭐ the repo.

⸻

Enjoy building your RAG applications!