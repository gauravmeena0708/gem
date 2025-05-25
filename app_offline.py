import streamlit as st
import os
from typing import List

# --- Re-import/Re-initialize necessary components ---
#from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.api.types import EmbeddingFunction

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db_offline"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "gemma:7b"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
collection_name = "pdf_knowledge_base_offline"

# --- Define Chroma-compatible wrapper for Ollama embeddings ---
class ChromaCompatibleOllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str, base_url: str):
        self.ollama = OllamaEmbeddings(model=model, base_url=base_url)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.ollama.embed_documents(input)

    def embed_query(self, text: str) -> List[float]:
        return self.ollama.embed_query(text)


# Initialize Embedding Model for app
embedding_model = ChromaCompatibleOllamaEmbeddingFunction(
    model=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

# Initialize Ollama LLM
llm = OllamaLLM(
    model=OLLAMA_LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

# Re-initialize ChromaDB client and collection
chroma_client_retrieval = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection_retrieval = chroma_client_retrieval.get_collection(
    name=collection_name,
    embedding_function=embedding_model
)
vectorstore = Chroma(
    client=chroma_client_retrieval,
    collection_name=collection_name,
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # Retrieve top 5 relevant chunks

# Define the prompt template for Gemma
prompt_template = """
You are a helpful assistant specialized in answering questions based on the provided documents.
Answer the question as truthfully as possible using only the context provided.
If the answer is not found in the context, politely state that you don't have enough information.
Do not make up answers.

Context:
{context}

Question: {question}

Answer:
"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template)

# Create the RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# --- Streamlit App ---
st.set_page_config(page_title="Offline PDF Assistant with Gemma", layout="centered")
st.title("📚 Ask Your PDFs (Powered by Local Gemma)")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your PDFs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        with st.chat_message("assistant"):
            try:
                result = qa_chain.invoke({"query": prompt})
                response = result["result"]
                source_documents = result.get("source_documents", [])

                st.markdown(response)

                if source_documents:
                    st.subheader("Sources:")
                    for doc in source_documents:
                        st.write(f"- {doc.metadata.get('source', 'Unknown Source')}, Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("Please ensure the Ollama server is running and the specified Gemma model is downloaded.")
                response = "I apologize, but I encountered an error while processing your request. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.markdown("💡 This application uses local Gemma via Ollama and your PDF documents as a knowledge base.")
