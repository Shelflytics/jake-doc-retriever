"""index.py
One-off script to build FAISS index and metadata from text files.
"""
import argparse
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def chunk_text(text, chunk_size=800, overlap=200):
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append((start, end, chunk))
        if end >= length:
            break
        start = max(0, end - overlap)
    return chunks


def build_index(docs_dir, out_dir, model_name, chunk_size, overlap, batch_size=64):
    docs_path = Path(docs_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(model_name)

    metadata = []
    texts = []

    for txt in sorted(docs_path.glob("*.txt")):
        raw = txt.read_text(encoding="utf-8")
        for start, end, chunk in chunk_text(raw, chunk_size=chunk_size, overlap=overlap):
            metadata.append({
                "source": txt.name,
                "start": start,
                "end": end,
                "text": chunk,
            })
            texts.append(chunk)

    if len(texts) == 0:
        print("No documents found to index.")
        return

    print(f"Embedding {len(texts)} chunks using {model_name}...")
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        vectors.append(emb)
    vectors = np.vstack(vectors).astype("float32")

    # normalize for cosine
    faiss.normalize_L2(vectors)
    d = vectors.shape[1]
    # Use METRIC_INNER_PRODUCT for cosine similarity
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vectors)

    faiss.write_index(index, str(out_path / "faiss.index"))
    with open(out_path / "metadata.json", "w", encoding="utf8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved index and metadata to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", default="data/docs")
    parser.add_argument("--out-dir", default="data/index")
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()
    build_index(args.docs_dir, args.out_dir, args.model, args.chunk_size, args.overlap)
