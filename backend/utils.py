import hashlib
import json
from typing import Iterable
from pathlib import Path

from .config import get_settings
from langchain_community.document_loaders import WebBaseLoader, WikipediaLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer

# 1. Load documents from web URLs
def load_web_docs(urls: Iterable[str]):
    loader = WebBaseLoader(urls)
    return loader.load()

# 2. Load documents from Wikipedia topics
def load_wikipedia_docs(topics: Iterable[str], lang: str = "en"):
    documents = []
    for topic in topics:
        documents.extend(WikipediaLoader(query=topic, lang=lang).load())
    return documents

# 3. Chunk documents
def chunk_docs(docs, chunk_size: int = 800, chunk_overlap: int = 100):
    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be greater than chunk_overlap")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

# 4. Embed documents (OpenAI)
def embed_docs(docs):
    settings = get_settings()
    texts = [doc.page_content for doc in docs]
    if not texts:
        return []
    embedder = SentenceTransformer(settings.embedding_model)
    embeddings = embedder.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32).tolist()

# 5. Save a local vector index
def upsert_to_local(embeddings, docs):
    settings = get_settings()
    if len(embeddings) != len(docs):
        raise ValueError("embeddings and documents must have the same length")
    if not embeddings:
        raise ValueError("cannot create an index from zero documents")

    settings.index_directory.mkdir(parents=True, exist_ok=True)
    records = []
    for embedding, doc in zip(embeddings, docs):
        source = doc.metadata.get("source", "unknown")
        vector_id = hashlib.sha256(
            f"{source}\n{doc.page_content}".encode("utf-8")
        ).hexdigest()
        records.append(
            {
                "id": vector_id,
                "text": doc.page_content,
                "source": source,
                "title": doc.metadata.get("title", ""),
            }
        )

    embedding_path = settings.index_directory / "embeddings.npy"
    records_path = settings.index_directory / "records.json"
    with embedding_path.open("wb") as file:
        np.save(file, np.asarray(embeddings, dtype=np.float32))
    records_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def load_local_index() -> tuple[np.ndarray, list[dict]]:
    settings = get_settings()
    embedding_path = settings.index_directory / "embeddings.npy"
    records_path = settings.index_directory / "records.json"
    if not embedding_path.exists() or not records_path.exists():
        raise RuntimeError(
            "The local index is empty. Run `python -m backend.ingest` first."
        )
    embeddings = np.load(embedding_path)
    records = json.loads(records_path.read_text(encoding="utf-8"))
    if len(embeddings) != len(records):
        raise RuntimeError("The local index files contain different document counts")
    return embeddings, records
