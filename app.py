# app.py (PRO version 🚀)

import streamlit as st
from langchain_ollama import ChatOllama
from embeddings.local_embedding_model import LocalEmbeddingModel
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import os
import shutil

# CONFIG
INDEX_DIR = "faiss_index_folder"
UPLOAD_FOLDER = "uploaded_transcripts"
TRANSCRIPT_PATH = "transcript.txt"

# Ensure folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load LLM
llm = ChatOllama(model="gemma:2b", temperature=0.0)

# Load embedding model
embeddings = LocalEmbeddingModel(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Functions
def load_faiss_index():
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def build_faiss_index_from_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(INDEX_DIR)
    return vector_store

# Load or build index
if os.path.exists(INDEX_DIR):
    vector_store = load_faiss_index()
else:
    vector_store = build_faiss_index_from_file(TRANSCRIPT_PATH)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

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

def format_docs(retrieved_docs):
    if not retrieved_docs:
        return "No relevant context found."

    context_text = ""
    for i, doc in enumerate(retrieved_docs):
        context_text += f"\n--- Chunk {i+1} ---\n"
        context_text += doc.page_content.strip() + "\n"

    return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | llm | parser

# --- Streamlit UI ---

st.set_page_config(page_title="🧠 RAG Playground PRO", page_icon="🧠", layout="wide")
st.title("🧠 RAG Playground PRO")
st.markdown("**Local Retrieval-Augmented Generation with Gemma 2b + Sentence Transformers + FAISS**")

# Sidebar
st.sidebar.header("Settings")
st.sidebar.markdown(f"**LLM:** gemma:2b")
st.sidebar.markdown(f"**Embedding model:** all-MiniLM-L6-v2")
st.sidebar.markdown(f"**FAISS index:** {INDEX_DIR}")

# FAISS index size info
index_size = len(vector_store.index_to_docstore_id)
st.sidebar.markdown(f"**Index size:** {index_size} chunks")

# Upload file
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("Upload new transcript (.txt)", type="txt")

if uploaded_file is not None:
    uploaded_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(uploaded_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.sidebar.success(f"Uploaded {uploaded_file.name}")

    if st.sidebar.button("Rebuild index from uploaded file 🚀"):
        with st.spinner("Rebuilding FAISS index..."):
            vector_store = build_faiss_index_from_file(uploaded_path)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        st.sidebar.success("FAISS index rebuilt successfully!")
        index_size = len(vector_store.index_to_docstore_id)
        st.sidebar.markdown(f"**Index size:** {index_size} chunks")

# Clear index
if st.sidebar.button("Clear FAISS index ❌"):
    shutil.rmtree(INDEX_DIR)
    st.sidebar.success("FAISS index cleared. Will rebuild on next run.")

# Main app
st.markdown("---")
question = st.text_input("Ask a question about the transcript:")

if st.button("Run RAG pipeline 🚀"):
    if question.strip() == "":
        st.warning("Please enter a question.")
    else:
        with st.spinner("Running..."):
            result = main_chain.invoke(question)
            retrieved_docs = retriever.invoke(question)

        st.markdown("### Retrieved Context")
        for i, doc in enumerate(retrieved_docs):
            with st.expander(f"Chunk {i+1}"):
                st.write(doc.page_content)

        st.markdown("### Answer")
        st.success(result)

st.markdown("---")
st.markdown("Built by **Kapil Pravin Marathe** · [GitHub](https://github.com/kapil3771)")