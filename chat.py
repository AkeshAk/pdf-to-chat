import json
import os
import faiss
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

VECTOR_DIR = "vectorstore"
EMBED_MODEL = "all-MiniLM-L6-v2"
GROQ_MODEL = "llama-3.1-8b-instant"
TOP_K = 5

client = Groq(api_key=os.environ["GROQ_API_KEY"])


def load_store():
    index = faiss.read_index(f"{VECTOR_DIR}/index.faiss")
    with open(f"{VECTOR_DIR}/chunks.json") as f:
        chunks = json.load(f)
    return index, chunks


def retrieve(query, model, index, chunks):
    embedding = model.encode([query])
    _, indices = index.search(np.array(embedding, dtype="float32"), TOP_K)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


def ask(query, model, index, chunks):
    results = retrieve(query, model, index, chunks)
    context = "\n\n".join(f"[{r['source']}]\n{r['text']}" for r in results)
    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below. Be specific and detailed. If the answer is not in the context, say "I don't know".

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def main():
    print("Loading vector store...")
    index, chunks = load_store()
    embed_model = SentenceTransformer(EMBED_MODEL)
    print(f"Ready! Chatting with {GROQ_MODEL}. Type 'exit' to quit.\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        answer = ask(query, embed_model, index, chunks)
        print(f"\nBot: {answer} \n")


if __name__ == "__main__":
    main()
