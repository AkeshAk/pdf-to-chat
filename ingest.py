import os
import json
import fitz  # pymupdf
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

PDF_DIR = "pdfs"
VECTOR_DIR = "vectorstore"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "all-MiniLM-L6-v2"


def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc)
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk:
            chunks.append(chunk)
    return chunks


def ingest():
    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []

    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            print(f"Processing: {filename}")
            chunks = extract_chunks(os.path.join(PDF_DIR, filename))
            for chunk in chunks:
                all_chunks.append({"text": chunk, "source": filename})

    if not all_chunks:
        print("No PDFs found in ./pdfs folder.")
        return

    texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings, dtype="float32"))

    faiss.write_index(index, os.path.join(VECTOR_DIR, "index.faiss"))
    with open(os.path.join(VECTOR_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)

    print(f"\nDone! Indexed {len(all_chunks)} chunks from {PDF_DIR}/")


if __name__ == "__main__":
    ingest()
