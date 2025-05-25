import sys
from typing import List
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
COLLECTION_NAME = "pdf_knowledge_base_offline"

# --- Chroma-compatible wrapper for Ollama embeddings ---
class ChromaCompatibleOllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model: str, base_url: str):
        self.ollama = OllamaEmbeddings(model=model, base_url=base_url)

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self.ollama.embed_documents(input)

    def embed_query(self, text: str) -> List[float]:
        return self.ollama.embed_query(text)

# --- Initialize Embeddings and LLM ---
embedding_model = ChromaCompatibleOllamaEmbeddingFunction(
    model=OLLAMA_EMBEDDING_MODEL,
    base_url=OLLAMA_BASE_URL
)

llm = OllamaLLM(
    model=OLLAMA_LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
)

# --- Connect to Chroma Vector DB ---
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_model
)
vectorstore = Chroma(
    client=chroma_client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Prompt Template ---
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

# --- Create RetrievalQA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

# --- Run CLI ---
def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py \"Your question here\"")
        sys.exit(1)

    question = sys.argv[1]
    try:
        result = qa_chain.invoke({"query": question})
        answer = result["result"]
        sources = result.get("source_documents", [])

        print("\nAnswer:\n", answer)

        if sources:
            print("\nSources:")
            for doc in sources:
                print(f"- {doc.metadata.get('source', 'Unknown Source')} (Chunk ID: {doc.metadata.get('chunk_id', 'N/A')})")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure the Ollama server is running and the Gemma model is downloaded.")

if __name__ == "__main__":
    main()
