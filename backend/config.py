import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("USER_AGENT", "website-assistant/0.1")


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    generation_model: str
    ollama_url: str
    index_directory: Path
    top_k: int
    allowed_origins: list[str]


def get_settings(validate_external: bool = True) -> Settings:
    settings = Settings(
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        generation_model=os.getenv("GENERATION_MODEL", "llama3.2:3b"),
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        index_directory=Path(os.getenv("INDEX_DIRECTORY", "data/index")),
        top_k=int(os.getenv("TOP_K", "4")),
        allowed_origins=[
            origin.strip()
            for origin in os.getenv(
                "ALLOWED_ORIGINS", "http://localhost:8501"
            ).split(",")
            if origin.strip()
        ],
    )

    return settings