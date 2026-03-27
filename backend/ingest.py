import os
import json
import uuid
import fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

load_dotenv()

PDF_DIR = "../pdfs"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = os.environ["COLLECTION_NAME"]
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
BATCH_SIZE = 100

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)
model = SentenceTransformer(EMBED_MODEL)


def extract_chunks(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join(page.get_text() for page in doc)
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = " ".join(words[i: i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ensure_collection(vector_size):
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION_NAME}")


def ingest():
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print("No PDFs found.")
        return

    print(f"Found {len(pdf_files)} PDFs. Starting ingestion...")

    sample = model.encode(["test"])
    ensure_collection(sample.shape[1])

    all_points = []
    for i, filename in enumerate(pdf_files):
        print(f"[{i+1}/{len(pdf_files)}] Processing: {filename}")
        chunks = extract_chunks(os.path.join(PDF_DIR, filename))
        texts = [c for c in chunks]
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)

        for chunk, embedding in zip(texts, embeddings):
            all_points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={"text": chunk, "source": filename}
            ))

        if len(all_points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=all_points)
            all_points = []

    if all_points:
        client.upsert(collection_name=COLLECTION_NAME, points=all_points)

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"\nDone! Total vectors in Qdrant: {count}")


if __name__ == "__main__":
    ingest()
