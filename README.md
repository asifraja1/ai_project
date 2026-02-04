# RAG AI Project

A Retrieval-Augmented Generation (RAG) system capable of ingesting content from Wikipedia and answering questions via a specific API entry point. Built with LangChain, FastAPI, and OpenAI.

## Features

- **Data Ingestion**: Scrape and process Wikipedia articles (`ingest.py`).
- **Vector Store**: Uses ChromaDB to store embeddings (generated via `text-embedding-3-small` or similar default OpenAI embeddings).
- **RAG Chain**: Retrieves relevant context to answer user queries (`rag_chain.py`).
- **API**: FastAPI providing a `/ask` endpoint (`main.py`).

## Prerequisites

- Python 3.9+
- OpenAI API Key

## Installation

1.  **Clone the repository** (if not already done):
    ```bash
    git clone <repo-url>
    cd ai_project
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Environment Setup**:
    Create a `.env` file in the root directory and add your OpenAI API key:
    ```env
    OPENAI_API_KEY=sk-...
    ```

## Usage

### 1. Ingest Data

Run the ingestion script to scrape a website (e.g., Wikipedia) and build the vector database.

```bash
python ingest.py
```
*Follow the prompt to enter a URL.*

### 2. Run the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### 3. Query the System

You can use the Swagger UI at `http://localhost:8000/docs` or send a POST request:

```bash
curl -X POST "http://localhost:8000/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the summary of the article?"}'

     ## Submission Note
Project submitted for evaluation.
```
     ## Submission Note
Project submitted for evaluation.
```
