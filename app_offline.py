import streamlit as st
import os

# --- Re-import/Re-initialize necessary components ---
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama # For the LLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma

import chromadb

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db_offline"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "gemma:7b" # Or "gemma:7b" if you downloaded that
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
collection_name = "pdf_knowledge_base_offline"

# Initialize Embedding Model for app
embedding_model = OllamaEmbeddings(
    model=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

# Initialize Ollama LLM
# Make sure your Ollama server is running and the model is downloaded (`ollama run gemma:2b`)
llm = Ollama(
    model=OLLAMA_LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

# Re-initialize ChromaDB client and collection
chroma_client_retrieval = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection_retrieval = chroma_client_retrieval.get_collection(
    name=collection_name,
    embedding_function=embedding_model # Use the same embedding function
)
vectorstore = Chroma(
    client=chroma_client_retrieval,
    collection_name=collection_name,
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 relevant chunks

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