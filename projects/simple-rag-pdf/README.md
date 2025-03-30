# Vietnamese PDF Question-Answering System

This project implements a Question-Answering system for Vietnamese PDF documents using LangChain, FAISS vector database, and Llama 2 LLM.

## Features

- PDF document loading and processing using Unstructured
- Text chunking with optimized overlap for context preservation
- Embedding generation using BAAI/bge-m3 model
- Vector storage with FAISS for efficient similarity search
- LLM-powered question answering using Llama 2 (13B parameters)
- 4-bit quantization for efficient model loading
- Interactive Q&A interface in Vietnamese

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Hugging Face API token
- Unstructured API key

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/vietnamese-pdf-qa.git
cd vietnamese-pdf-qa
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Install system dependencies:
```bash
apt-get update && apt-get install -y poppler-utils tesseract-ocr
```

## Usage

1. Add your PDF file to the project directory

2. Update the `PDF_PATH` variable in `pdf-qa-system.py` to point to your PDF file

3. Run the script:
```bash
python pdf-qa-system.py
```

4. The script will:
   - Prompt for your API keys if not set as environment variables
   - Process the PDF and create embeddings
   - Load the LLM model
   - Run sample questions
   - Start an interactive Q&A session

## Configuration

You can modify the following parameters in the script:
- `PDF_PATH`: Path to your PDF file
- `VECTOR_DB_PATH`: Location to save the vector database
- `MODEL_ID`: Hugging Face model ID for the LLM
- `EMBEDDING_MODEL`: Model used for text embeddings
