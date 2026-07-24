from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is optional until requirements are installed.
    pass

try:
    from pydantic import BaseModel, ConfigDict, Field
except Exception:  # pragma: no cover - keeps the app importable before deps are installed.
    ConfigDict = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[misc,assignment]

    def Field(default: Any = None, **_: Any) -> Any:  # type: ignore[override]
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class AppSettings(BaseModel):  # type: ignore[misc]
    if ConfigDict is not None:
        model_config = ConfigDict(arbitrary_types_allowed=True)

    app_name: str = "Enterprise Local RAG"
    data_dir: Path = Field(default_factory=lambda: Path(os.getenv("RAG_DATA_DIR", "data")))
    upload_dir: Path = Field(default_factory=lambda: Path(os.getenv("RAG_UPLOAD_DIR", "data/uploads")))
    groq_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    hf_token: Optional[str] = Field(default_factory=lambda: os.getenv("HF_TOKEN"))
    ollama_host: str = Field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = Field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3"))
    ollama_layer1_model: str = Field(
        default_factory=lambda: os.getenv("OLLAMA_LAYER1_MODEL", os.getenv("OLLAMA_MODEL", "llama3"))
    )
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_batch_size: int = Field(default_factory=lambda: _int_env("EMBEDDING_BATCH_SIZE", 16))
    reranker_model: str = Field(
        default_factory=lambda: os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    top_k: int = Field(default_factory=lambda: _int_env("RAG_TOP_K", 20))
    hybrid_alpha: float = Field(default_factory=lambda: _float_env("RAG_HYBRID_ALPHA", 0.5))
    search_hint_boost: float = Field(default_factory=lambda: _float_env("RAG_SEARCH_HINT_BOOST", 1.5))
    rerank_cutoff: float = Field(default_factory=lambda: _float_env("RAG_RERANK_CUTOFF", -2.0))
    parent_chunk_tokens: int = Field(default_factory=lambda: _int_env("RAG_PARENT_CHUNK_TOKENS", 1400))
    child_chunk_tokens: int = Field(default_factory=lambda: _int_env("RAG_CHILD_CHUNK_TOKENS", 280))
    child_overlap_tokens: int = Field(default_factory=lambda: _int_env("RAG_CHILD_OVERLAP_TOKENS", 50))
    request_timeout_seconds: int = Field(default_factory=lambda: _int_env("RAG_TIMEOUT_SECONDS", 45))
    layer1_timeout_seconds: int = Field(default_factory=lambda: _int_env("LAYER1_TIMEOUT_SECONDS", 15))
    layer2_timeout_seconds: int = Field(default_factory=lambda: _int_env("LAYER2_TIMEOUT_SECONDS", 30))
    max_context_parent_chunks: int = Field(default_factory=lambda: _int_env("RAG_MAX_CONTEXT_PARENT_CHUNKS", 6))
    max_context_characters: int = Field(default_factory=lambda: _int_env("RAG_MAX_CONTEXT_CHARACTERS", 9000))
    enable_chroma: bool = Field(default_factory=lambda: os.getenv("RAG_ENABLE_CHROMA", "false").lower() == "true")
    enable_reranker: bool = Field(default_factory=lambda: os.getenv("RAG_ENABLE_RERANKER", "false").lower() == "true")

    if ConfigDict is None:
        class Config:
            arbitrary_types_allowed = True

    def ensure_dirs(self) -> "AppSettings":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "chroma").mkdir(parents=True, exist_ok=True)
        return self

    def with_overrides(self, **kwargs: Any) -> "AppSettings":
        data = self.model_dump() if hasattr(self, "model_dump") else dict(self.__dict__)
        data.update({key: value for key, value in kwargs.items() if value is not None})
        return AppSettings(**data).ensure_dirs()


def get_settings() -> AppSettings:
    return AppSettings().ensure_dirs()
