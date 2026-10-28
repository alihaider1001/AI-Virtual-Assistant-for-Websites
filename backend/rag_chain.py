from functools import lru_cache

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from .config import get_settings
from .utils import load_local_index

@lru_cache(maxsize=1)
def get_embedder():
    settings = get_settings()
    return SentenceTransformer(settings.embedding_model)


def generate_with_ollama(prompt: str) -> str:
    settings = get_settings()
    response = requests.post(
        f"{settings.ollama_url.rstrip('/')}/api/generate",
        json={
            "model": settings.generation_model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 256},
        },
        timeout=300,
    )
    if not response.ok:
        detail = response.text.strip().replace("\n", " ")[:500]
        raise RuntimeError(
            f"Ollama returned HTTP {response.status_code}: {detail}"
        )
    payload = response.json()
    answer = payload.get("response", "").strip()
    if not answer:
        raise RuntimeError("Ollama returned an empty response")
    return answer

# Function to answer a query
def retrieve_records(query: str):
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    settings = get_settings()
    embeddings, records = load_local_index()
    query_embedding = get_embedder().encode(query, normalize_embeddings=True)
    scores = embeddings @ np.asarray(query_embedding, dtype=np.float32)
    selected = np.argsort(scores)[-settings.top_k :][::-1]
    return [records[index] for index in selected]


def build_context(records: list[dict]) -> str:
    return "\n\n".join(
        f"Source: {record['source']}\n{record['text']}"
        for record in records
    )


def answer_query(query: str):
    selected_records = retrieve_records(query)
    context = build_context(selected_records)
    prompt = (
        "Answer the question using only the context below. "
        "If the context does not contain the answer, say you do not know.\n\n"
        f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    generated = generate_with_ollama(prompt)
    sources = [
        {
            "source": record.get("source", ""),
            "title": record.get("title", ""),
        }
        for record in selected_records
    ]
    return {"answer": generated, "sources": sources}
