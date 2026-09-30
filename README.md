## Website Assistant

This project indexes website and Wikipedia content locally, then answers questions with local embeddings and Ollama through a FastAPI service. The Streamlit frontend provides the chat interface. No OpenAI, Pinecone, or paid subscription is required.

### Setup

Use Python 3.10 or newer and install the dependencies into the project environment:

```powershell
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. Install and run Ollama, then download the configured model with `ollama pull qwen2.5:1.5b`. The embedding model downloads from Hugging Face during the first ingestion. These downloads are free, but require local disk space and memory.

### Run

From the project root, run ingestion once after configuring `.env`:

```powershell
python -m backend.ingest
```

Start the API:

```powershell
uvicorn backend.main:app --reload --port 8000
```

Ollama must be running separately. Verify it with:

```powershell
ollama list
```

Start the frontend in another terminal:

```powershell
streamlit run frontend/app.py
```

The API exposes `GET /health` and `POST /ask`. The frontend uses `BACKEND_URL`, which defaults to `http://localhost:8000`.

### Configuration

| Variable | Purpose |
| --- | --- |
| `EMBEDDING_MODEL` | Local embedding model |
| `GENERATION_MODEL` | Ollama model, default `llama3.2:3b` |
| `OLLAMA_URL` | Ollama API URL, default `http://localhost:11434` |
| `INDEX_DIRECTORY` | Local vector index directory |
| `TOP_K` | Number of retrieved chunks |
| `ALLOWED_ORIGINS` | Comma-separated browser origins |
| `BACKEND_URL` | Frontend API URL |
