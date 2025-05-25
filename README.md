# PDF Question Answering with Local Gemma

This project allows you to chat with your PDF documents using a locally running Large Language Model (LLM), Gemma, via Ollama. It consists of two main scripts:

1.  `ingest_pdfs.py`: This script processes PDF files from a specified folder, extracts text, splits it into manageable chunks, generates embeddings using a local embedding model, and stores them in a ChromaDB vector database.
2.  `app_offline.py`: This script runs a Streamlit web application that allows you to ask questions about the documents processed by `ingest_pdfs.py`. It retrieves relevant chunks from ChromaDB and uses the local Gemma LLM to generate answers.

---
##  Prerequisites

Before you begin, ensure you have the following installed and configured:

1.  **Python**: Version 3.8 or higher. You mentioned you've already created an environment named "gem" with Python.
2.  **Ollama**:
    * Download and install Ollama from [https://ollama.com/](https://ollama.com/).
    * Pull the necessary models using the following commands in your terminal:
        ```bash
        ollama pull gemma:7b
        ollama pull nomic-embed-text
        ```
    * Ensure the Ollama server is running (you can usually start it by just running `ollama serve` in a terminal or by opening the Ollama application).
3.  **Tesseract OCR**:
    * For Windows, download and install Tesseract OCR from the official UB Mannheim page: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
    * **Important**: During installation, make sure to select the option to add Tesseract to your system's PATH.
    * You will also need the Tesseract language data. The `ingest_pdfs.py` script references `tessdata_best-main`. You can download "best" quality English language data (`eng.traineddata`) from [https://github.com/tesseract-ocr/tessdata_best](https://github.com/tesseract-ocr/tessdata_best) and place it in your Tesseract installation's `tessdata` directory (e.g., `C:\Program Files\Tesseract-OCR\tessdata`).
4.  **Git**: For cloning the repository (which you've already done).

---
## Installation

1.  **Clone the Repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Activate your Python Environment**:
    You mentioned you created an environment named `gem`. Activate it. If you used `conda`:
    ```bash
    conda activate gem
    ```
    If you used `venv`:
    ```bash
    .\gem\Scripts\activate
    ```

3.  **Install Python Dependencies**:
    Create a `requirements.txt` file in the root of your project directory with the following content:

    ```txt
    streamlit
    langchain-community
    langchain
    chromadb
    ollama
    langchain-ollama
    unstructured
    pytesseract
    # Add pdfminer.six if unstructured[pdf] needs it explicitly on your system
    # pdfminer.six
    # Add other dependencies if unstructured requires them (e.g., for specific image processing)
    # Pillow
    # opencv-python
    ```

    Then, install these dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `unstructured` might have additional dependencies for PDF processing depending on the complexity of your PDFs (e.g., if they contain many images). If you encounter issues, you might need to install `unstructured[pdf]` or specific libraries like `pdfminer.six`, `Pillow`, or `opencv-python`.*

4.  **Configure Tesseract Path in `ingest_pdfs.py` (if needed)**:
    The script `ingest_pdfs.py` has the following line:
    ```python
    pytesseract.pytesseract.tesseract_cmd = '/snap/bin/tesseract' #
    ```
    You will need to change this to the path where Tesseract OCR is installed on your Windows system. Typically, it's something like:
    ```python
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```
    Also, ensure the environment variables for `TESSERACT_TEMP_DIR` and `TESSDATA_PREFIX` are correctly set for your system, or modify the script to point to the correct locations. For Windows, you might not need to set `TESSERACT_TEMP_DIR` explicitly. `TESSDATA_PREFIX` should point to the directory containing the `tessdata` folder.
    ```python
    # Example for Windows, adjust as necessary
    os.environ['TESSDATA_PREFIX'] = r'C:\Program Files\Tesseract-OCR' #
    # You might need to create a temp directory for tesseract files
    # os.environ['TESSERACT_TEMP_DIR'] = r'C:\path\to\your\temp_tesseract_files' #
    ```

---
## Usage

### 1. Prepare Your PDF Documents

* Create a folder named `data` in the root of your project directory.
* Place all the PDF files you want to process into this `data` folder. The `ingest_pdfs.py` script looks for PDFs in `./data/`.

### 2. Ingest PDF Documents

Run the `ingest_pdfs.py` script to process your PDFs and populate the vector database. Make sure your Ollama server is running.

```bash
python ingest_pdfs.py
