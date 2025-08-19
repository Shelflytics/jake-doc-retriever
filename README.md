# Policy Navigator Backend

This repository contains the backend service for the Policy Navigator chatbot. It's a minimal, self-contained RAG (Retrieval-Augmented Generation) application built with FastAPI.

The service indexes a local corpus of text documents and uses a cloud-based LLM (Google's Gemini) to answer questions based on the retrieved context.

## Quick Start

Follow these steps to get the API server running locally.

### 1. Setup Environment

Create and activate a Python virtual environment. This example uses PowerShell on Windows.

```powershell
# Create the virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\Activate.ps1

# Install required packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example `.env.example` file to a new file named `.env` and fill in your Google AI Studio API key.

```powershell
copy .env.example .env
```

Then, open the `.env` file and set your `GOOGLE_API_KEY`.

### 3. Build the Search Index

Run the indexer script once to process your documents in `data/docs/` and create the FAISS vector index.

```powershell
python index.py
```

This will create `faiss.index` and `metadata.json` inside the `data/index/` directory.

### 4. Run the API Server

Start the FastAPI server using Uvicorn.

```powershell
uvicorn app.main:app --reload
```

The server will be running at `http://127.0.0.1:8000`. The `--reload` flag enables hot-reloading for development.

### 5. Test the API

You can send a request to the `/chat` endpoint using `curl` or any API client.

```powershell
curl -X POST "http://localhost:8000/chat" -H "Content-Type: application/json" -d '{"query":"Who can approve discounts?","k":3}'
```

## Helper Scripts

### Testing the Index

To test the retrieval component directly without running the API, use the `test_index.py` script.

```powershell
python scripts/test_index.py "your query here"
```

This will show you the top matching text chunks and their relevance scores (higher is better).

### Rendering PDFs

To convert the source `.txt` files into PDFs for a frontend to display, run the PDF renderer script.

```powershell
python scripts/render_txt_to_pdf.py
```

This reads from `data/docs/` and writes PDFs to `data/docs_rendered/`.
