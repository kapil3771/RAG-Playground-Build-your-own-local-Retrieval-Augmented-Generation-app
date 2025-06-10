# rag.py

from langchain_ollama import ChatOllama
from embeddings.local_embedding_model import LocalEmbeddingModel
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os

# === CONFIG ===
INDEX_DIR = "faiss_index_folder"
USE_EXISTING_INDEX = os.path.exists(INDEX_DIR)
TRANSCRIPT_PATH = "transcript.txt"

# === LLM ===
llm = ChatOllama(model="gemma:2b", temperature=0.0)

# === Embedding ===
embeddings = LocalEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Vector store ===
if USE_EXISTING_INDEX:
    print(f"Loading existing FAISS index from {INDEX_DIR}...")
    vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
else:
    print("Building new FAISS index...")
    with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
        transcript = f.read()
    print("Loaded transcript from file.")

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Build index
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)
    print(f"Saved FAISS index to {INDEX_DIR}.")

# === Retriever ===
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

# === Prompt ===
prompt = PromptTemplate(
    template="""
    You are a helpful and precise assistant.

    You will be given CONTEXT extracted from a video transcript.
    Answer the QUESTION using ONLY the provided CONTEXT.

    - If the CONTEXT does not contain enough information to answer the QUESTION, say clearly: "I don't know based on the context provided."
    - Do NOT make up any facts not present in the CONTEXT.
    - Respond in a concise and informative manner.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """,
    input_variables=['context', 'question']
)

# === Format retrieved docs ===
def format_docs(retrieved_docs):
    if not retrieved_docs:
        return "No relevant context found."

    context_text = ""
    for i, doc in enumerate(retrieved_docs):
        context_text += f"\n--- Chunk {i+1} ---\n"
        context_text += doc.page_content.strip() + "\n"

    return context_text

# === Chain ===
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

# === Run batch questions ===
questions = [
    "What is nuclear fusion?",
    "What temperature is needed for fusion?",
    "What are the challenges in achieving fusion?",
    "What does the sun use for its energy?",
    "Is plasma mentioned in the context?"
]

for question in questions:
    print(f"\n=== Question: {question} ===")
    result = main_chain.invoke(question)
    print(result)