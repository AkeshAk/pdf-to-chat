import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import numpy as np

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
COLLECTION_NAME = os.environ["COLLECTION_NAME"]
TOP_K = 5

groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"]
)
embed_model = SentenceTransformer(EMBED_MODEL)


class ChatRequest(BaseModel):
    question: str


def retrieve(query: str):
    embedding = embed_model.encode([query])[0].tolist()
    results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=TOP_K
    )
    return [r.payload for r in results]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    chunks = retrieve(req.question)
    context = "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)

    prompt = f"""You are a helpful assistant for Tamil documents. 
- If the user asks in Tamil, answer in Tamil.
- If the user asks in English, answer in English.
- Answer using ONLY the context below. Be specific and detailed.
- If the answer is not in the context, say "I don't know" (or "தெரியவில்லை" if asked in Tamil).

Context:
{context}

Question: {req.question}
Answer:"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return {"answer": response.choices[0].message.content}
