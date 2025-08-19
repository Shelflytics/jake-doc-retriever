from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer
import requests
import uvicorn
from typing import Any

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "data/index")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AI_MODEL = os.getenv("AI_MODEL")

app = FastAPI()

class ChatRequest(BaseModel):
    query: str
    k: int = 3

@app.on_event("startup")
def load_index():
    global index, metadata, embed_model
    idx_path = Path(INDEX_DIR) / "faiss.index"
    meta_path = Path(INDEX_DIR) / "metadata.json"
    if not idx_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Index not found in {INDEX_DIR}. Run index.py first.")
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "r", encoding="utf8") as f:
        metadata = json.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Loaded index with {len(metadata)} chunks")

@app.post("/chat")
def chat(req: ChatRequest):
    q = req.query
    k = req.k if req.k and req.k > 0 else 3

    qvec = embed_model.encode([q], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qvec)
    D, I = index.search(qvec, k)
    results = []
    snippets = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = metadata[idx]
        results.append({
            "source": m.get("source"),
            "chunk_id": int(idx),
            "score": float(score),
            "start": int(m.get("start", 0)),
            "end": int(m.get("end", 0)),
        })
        # sanitize/truncate snippet
        text = m.get("text", "")
        if len(text) > 800:
            text = text[:800] + "..."
        snippets.append(f"[{m.get('source')}] {text}")

    # build prompt
    system = "You are a helpful internal policy assistant. Use only the provided context to answer. If answer not present, say 'I don't know - consult the policies.'"
    context = "\n\n".join([f"[{i+1}] {s}" for i, s in enumerate(snippets)])
    user_message = f"Context:\n{context}\n\nQuestion: {q}\n\nAnswer concisely and cite sources like [filename]."

    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API key not configured on server")

    # Call Google AI Studio (Generative Language) REST API
    try:
        # Corrected URL and payload structure for Gemini
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{AI_MODEL}:generateContent?key={GOOGLE_API_KEY}"
        
        # The Gemini API expects a 'contents' list with a specific structure
        # We can simulate a multi-turn chat history if needed, but for RAG, a single prompt is fine.
        # The system prompt can be part of the user message for simpler models.
        full_prompt = f"{system}\n\n{user_message}"
        
        payload = {
            "contents": [{
                "parts": [{"text": full_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 512,
            }
        }
        
        resp = requests.post(url, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        # Corrected response parsing for the generateContent endpoint
        if "candidates" in data and data["candidates"] and "content" in data["candidates"][0] and "parts" in data["candidates"][0]["content"]:
            answer = data["candidates"][0]["content"]["parts"][0].get("text", "").strip()
        else:
            answer = "Could not parse a valid answer from the model's response."

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"API request failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")


    # sanitize response to native python types before returning
    response_content = {"answer": answer, "sources": results}
    response_content = _make_serializable(response_content)
    return response_content


def _make_serializable(obj: Any) -> Any:
    """Recursively convert objects (NumPy scalars/arrays, etc.) to native Python types.
    Uses .item() when available (NumPy scalars) and falls back to str() for unknown types.
    """
    # handle dict
    if isinstance(obj, dict):
        return { _make_serializable(k): _make_serializable(v) for k, v in obj.items() }
    # handle list/tuple
    if isinstance(obj, (list, tuple)):
        return [ _make_serializable(v) for v in obj ]
    # primitive python types
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    # numpy scalars/arrays often expose .item() to convert to python native
    try:
        if hasattr(obj, "item") and callable(getattr(obj, "item")):
            return obj.item()
    except Exception:
        pass
    # fallback to str for anything else (safe for JSON encoding)
    try:
        return str(obj)
    except Exception:
        return None


    # end _make_serializable

    
    
    



if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8001
    print("--- Policy Navigator Backend ---")
    print(f"Starting server on http://{host}:{port}")
    print(f"Interactive API docs (Swagger): http://127.0.0.1:{port}/docs")
    print(f"Alternative API docs (ReDoc): http://127.0.0.1:{port}/redoc")
    print("---------------------------------")
    uvicorn.run(app, host=host, port=port)
