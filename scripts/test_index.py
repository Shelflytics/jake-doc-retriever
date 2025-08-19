"""
A simple script to test the FAISS index with a command-line query.
"""
import argparse
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def test_index(query: str, index_dir: str = "data/index", top_k: int = 3):
    """
    Loads the index and performs a search for the given query.
    """
    index_path = Path(index_dir)
    idx_file = index_path / "faiss.index"
    meta_file = index_path / "metadata.json"

    if not idx_file.exists() or not meta_file.exists():
        print(f"Error: Index not found in '{index_path.resolve()}'.")
        print("Please run 'python index.py' first to build the index.")
        return

    # 1. Load model, index, and metadata
    print("Loading model and index...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index(str(idx_file))
    with open(meta_file, "r", encoding="utf8") as f:
        metadata = json.load(f)
    print("...done.")

    # 2. Embed the query
    print(f"\nSearching for: '{query}'")
    qvec = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qvec)

    # 3. Search the index
    distances, indices = index.search(qvec, top_k)

    # 4. Print the results
    print(f"\nTop {top_k} results:")
    print("-" * 20)
    for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0:
            continue
        
        meta = metadata[idx]
        source = meta.get("source", "Unknown")
        text = meta.get("text", "").strip().replace("\n", " ")
        
        print(f"Result {i+1} (Score: {score:.4f})")
        print(f"  Source: {source}")
        print(f"  Text:   \"{text[:250]}...\"")
        print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the FAISS index with a query.")
    parser.add_argument("query", type=str, help="The search query to test.")
    parser.add_argument("--index_dir", type=str, default="data/index", help="Directory containing the FAISS index and metadata.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to retrieve.")
    
    args = parser.parse_args()
    
    test_index(query=args.query, index_dir=args.index_dir, top_k=args.top_k)
