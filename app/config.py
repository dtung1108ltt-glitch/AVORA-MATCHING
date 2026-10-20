from __future__ import annotations

import os
from pydantic import ConfigDict, Field, TypeAdapter
from pydantic_settings import BaseSettings


def _to_async_database_url(url: str) -> str:
    """Chuyển đổi chuỗi kết nối PostgreSQL đồng bộ sang driver bất đồng bộ asyncpg."""
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if scheme.lower().startswith("postgres"):
            return f"postgresql+asyncpg://{rest}"
    return url


def _to_sync_database_url(url: str) -> str:
    """Đảm bảo chuỗi kết nối đồng bộ chuẩn cho psycopg2 khi cần tương thích ngược."""
    if "://" in url:
        scheme, rest = url.split("://", 1)
        if scheme.lower().startswith("postgres"):
            return f"postgresql+psycopg2://{rest}"
    return url


class AppSettings(BaseSettings):
    """Cấu hình hệ thống AVORA Matching chuẩn Pydantic V2."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=False,
    )

    database_url: str = Field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://avora:avora@localhost:5432/avora_matching"
        )
    )
    async_database_url: str = Field(
        default_factory=lambda: os.getenv(
            "ASYNC_DATABASE_URL",
            _to_async_database_url(
                os.getenv(
                    "DATABASE_URL",
                    "postgresql+asyncpg://avora:avora@localhost:5432/avora_matching",
                )
            ),
        )
    )
    qdrant_host: str = Field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    qdrant_http_port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_HTTP_PORT", "6333")))
    qdrant_grpc_port: int = Field(default_factory=lambda: int(os.getenv("QDRANT_GRPC_PORT", "6334")))
    qdrant_prefer_grpc: bool = Field(
        default_factory=lambda: os.getenv("QDRANT_PREFER_GRPC", "true").lower() in ("1", "true", "yes")
    )
    qdrant_collection_jobs: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_JOBS", "avora_job_descriptions")
    )
    qdrant_collection_candidates: str = Field(
        default_factory=lambda: os.getenv("QDRANT_COLLECTION_CANDIDATES", "avora_candidates")
    )
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    )
    onnx_model_path: str | None = Field(
        default_factory=lambda: os.getenv("ONNX_MODEL_PATH", None)
    )
    onnx_quantization: str = Field(
        default_factory=lambda: os.getenv("ONNX_QUANTIZATION", "INT8")
    )
    dynamic_batch_size: int = Field(
        default_factory=lambda: int(os.getenv("DYNAMIC_BATCH_SIZE", "32"))
    )
    embedding_cache_max_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_CACHE_MAX_SIZE", "16384"))
    )
    vector_dim: int = Field(default_factory=lambda: int(os.getenv("VECTOR_DIM", "1024")))
    top_k_jobs: int = Field(default_factory=lambda: int(os.getenv("TOP_K_JOBS", "10")))
    top_k_candidates: int = Field(default_factory=lambda: int(os.getenv("TOP_K_CANDIDATES", "20")))


AppSettingsAdapter: TypeAdapter[AppSettings] = TypeAdapter(AppSettings)
settings = AppSettings()

# Các hằng số cấp module giữ tương thích ngược 100% với toàn bộ codebase
DATABASE_URL: str = _to_sync_database_url(settings.database_url)
SYNC_DATABASE_URL: str = DATABASE_URL
ASYNC_DATABASE_URL: str = _to_async_database_url(settings.async_database_url)
QDRANT_HOST: str = settings.qdrant_host
QDRANT_HTTP_PORT: int = settings.qdrant_http_port
QDRANT_GRPC_PORT: int = settings.qdrant_grpc_port
QDRANT_PREFER_GRPC: bool = settings.qdrant_prefer_grpc
QDRANT_COLLECTION_JOBS: str = settings.qdrant_collection_jobs
QDRANT_JOB_COLLECTION: str = settings.qdrant_collection_jobs
QDRANT_COLLECTION_CANDIDATES: str = settings.qdrant_collection_candidates
QDRANT_CANDIDATE_COLLECTION: str = settings.qdrant_collection_candidates
EMBEDDING_MODEL_NAME: str = settings.embedding_model_name
ONNX_MODEL_PATH: str | None = settings.onnx_model_path
ONNX_QUANTIZATION: str = settings.onnx_quantization
DYNAMIC_BATCH_SIZE: int = settings.dynamic_batch_size
EMBEDDING_CACHE_MAX_SIZE: int = settings.embedding_cache_max_size
VECTOR_DIM: int = settings.vector_dim
TOP_K_JOBS: int = settings.top_k_jobs
TOP_K_CANDIDATES: int = settings.top_k_candidates
