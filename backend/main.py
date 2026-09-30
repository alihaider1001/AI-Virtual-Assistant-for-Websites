from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import requests

from .config import get_settings
from .rag_chain import answer_query
from .utils import load_local_index

app = FastAPI()
settings = get_settings()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/status")
async def status():
    index_ready = False
    chunk_count = 0
    embedding_dimensions = 0
    try:
        embeddings, records = load_local_index()
        index_ready = True
        chunk_count = len(records)
        embedding_dimensions = int(embeddings.shape[1])
    except (RuntimeError, OSError, ValueError):
        pass

    ollama_online = False
    ollama_models = []
    try:
        response = requests.get(
            f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=2
        )
        response.raise_for_status()
        ollama_online = True
        ollama_models = [
            model.get("name", "") for model in response.json().get("models", [])
        ]
    except requests.RequestException:
        pass

    return {
        "index_ready": index_ready,
        "chunk_count": chunk_count,
        "embedding_dimensions": embedding_dimensions,
        "embedding_model": settings.embedding_model,
        "generation_model": settings.generation_model,
        "ollama_online": ollama_online,
        "ollama_models": ollama_models,
        "generation_model_available": settings.generation_model in ollama_models,
        "top_k": settings.top_k,
    }

@app.post("/ask")
async def ask(request: AskRequest):
    try:
        return await run_in_threadpool(answer_query, request.query)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="The assistant is temporarily unavailable.") from error
