import os
os.environ['TESSERACT_TEMP_DIR'] = '/home/aizceq/Downloads/gem/temp_tesseract_files'
os.environ['TESSDATA_PREFIX'] = '/home/aizceq/Downloads/tessdata_best-main'

import pytesseract
import chromadb
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.utils import embedding_functions

pytesseract.pytesseract.tesseract_cmd = '/snap/bin/tesseract'


PDF_FOLDER_PATH = "./data/" 
CHROMA_DB_PATH = "./chroma_db_offline" 
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text" 

class ChromaCompatibleOllama(EmbeddingFunction[Documents]):
    def __init__(self, model: str = OLLAMA_EMBEDDING_MODEL, base_url: str = OLLAMA_BASE_URL): 
        self.ollama = OllamaEmbeddings(model=model, base_url=base_url) 
    def __call__(self, input: Documents) -> Embeddings:
        return self.ollama.embed_documents(input)


embedding_model = ChromaCompatibleOllama(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

all_chunks = []
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)

for filename in os.listdir(PDF_FOLDER_PATH):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(PDF_FOLDER_PATH, filename)
        print(f"Processing {filepath}...")

        elements = partition_pdf(
            filepath,
            strategy="auto",
        )
        full_text = "\n\n".join([str(el) for el in elements if el.text])
        chunks = text_splitter.split_text(full_text)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {"source": filename, "chunk_id": i}
            })

print(f"Total chunks created: {len(all_chunks)}")

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection_name = "pdf_knowledge_base_offline"
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_model
)


documents = [chunk["text"] for chunk in all_chunks]
metadatas = [chunk["metadata"] for chunk in all_chunks]
ids = [f"{m['source']}_{m['chunk_id']}" for m in metadatas]

print(f"Adding {len(documents)} documents to ChromaDB...")
collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
print("Vector database populated!")