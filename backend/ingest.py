from .utils import load_web_docs, load_wikipedia_docs, chunk_docs, embed_docs, upsert_to_local

if __name__ == "__main__":
    # Example sources
    urls = [
        "https://help.shopify.com/en/manual/orders",
        "https://www.notion.so/help"
    ]
    wiki_topics = ["Shopify", "Notion (software)"]

    print("Loading website documents...", flush=True)
    web_docs = load_web_docs(urls)
    print(f"Loaded {len(web_docs)} website documents.", flush=True)

    print("Loading Wikipedia documents...", flush=True)
    wiki_docs = load_wikipedia_docs(wiki_topics)
    print(f"Loaded {len(wiki_docs)} Wikipedia documents.", flush=True)
    all_docs = web_docs + wiki_docs

    print("Splitting documents into chunks...", flush=True)
    chunks = chunk_docs(all_docs)
    print(f"Created {len(chunks)} chunks.", flush=True)

    print("Creating local embeddings; the first run downloads the embedding model...", flush=True)
    embeddings = embed_docs(chunks)

    print("Saving the local vector index...", flush=True)
    upsert_to_local(embeddings, chunks)

    print(f"Indexed {len(chunks)} chunks locally.")