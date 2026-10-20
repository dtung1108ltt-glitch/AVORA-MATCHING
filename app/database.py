from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    Range,
    VectorParams,
)
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import (
    ASYNC_DATABASE_URL,
    DATABASE_URL,
    QDRANT_CANDIDATE_COLLECTION,
    QDRANT_GRPC_PORT,
    QDRANT_HOST,
    QDRANT_HTTP_PORT,
    QDRANT_JOB_COLLECTION,
    QDRANT_PREFER_GRPC,
    SYNC_DATABASE_URL,
    VECTOR_DIM,
)

# ---------------------------------------------------------------------------
# 1. SQLAlchemy 2.0 Async Engine (postgresql+asyncpg://) & Sync Fallback
# ---------------------------------------------------------------------------

async_engine: AsyncEngine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# Giữ engine đồng bộ phục vụ tương thích ngược và các công cụ quản trị
engine = create_engine(SYNC_DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(AsyncAttrs, DeclarativeBase):
    """Lớp cha chuẩn SQLAlchemy 2.0 Async cho tất cả model ORM."""
    pass


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency bất đồng bộ chuẩn SQLAlchemy 2.0 (`postgresql+asyncpg://`)."""
    async with AsyncSessionLocal() as session:
        yield session


def get_db() -> Generator[Session, None, None]:
    """Dependency của FastAPI: mở session, trả cho endpoint dùng, rồi luôn đóng lại."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_async_db() -> None:
    """Bật extension pgvector và tạo bảng bất đồng bộ qua asyncpg."""
    async with async_engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.run_sync(Base.metadata.create_all)


def init_db() -> None:
    """Bật extension pgvector và tạo toàn bộ bảng nếu chưa tồn tại."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# 2. AsyncQdrantClient (gRPC port 6334) & Filterable HNSW Payload Indexing
# ---------------------------------------------------------------------------

_qdrant_client: AsyncQdrantClient | None = None

#: Danh sách các trường lọc cứng (Hard Constraints) được đánh chỉ mục Payload trong Qdrant
HARD_CONSTRAINT_PAYLOAD_FIELDS: dict[str, PayloadSchemaType] = {
    "location": PayloadSchemaType.KEYWORD,
    "salary_min": PayloadSchemaType.FLOAT,
    "salary_max": PayloadSchemaType.FLOAT,
    "experience_years": PayloadSchemaType.FLOAT,
    "career_domain": PayloadSchemaType.KEYWORD,
    "nhom_khuyet_tat": PayloadSchemaType.KEYWORD,
    "work_mode": PayloadSchemaType.KEYWORD,
}


def get_async_qdrant_client() -> AsyncQdrantClient:
    """Khởi tạo và trả về `AsyncQdrantClient(prefer_grpc=True)` qua cổng gRPC 6334."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = AsyncQdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_HTTP_PORT,
            grpc_port=QDRANT_GRPC_PORT,
            prefer_grpc=QDRANT_PREFER_GRPC,
        )
    return _qdrant_client


async def init_qdrant_collections(
    client: AsyncQdrantClient | None = None,
    *,
    job_collection: str = QDRANT_JOB_COLLECTION,
    candidate_collection: str = QDRANT_CANDIDATE_COLLECTION,
    vector_dim: int = VECTOR_DIM,
) -> None:
    """Khởi tạo các collection Qdrant cùng Filterable HNSW và Payload Indexing.

    Các trường lọc cứng (`location`, `salary_min`, `salary_max`, `experience_years`,
    `career_domain`, `nhom_khuyet_tat`, `work_mode`) được đánh chỉ mục ngay tại tầng
    lưu trữ để thực hiện Pre-filtering trước khi đưa vào `src/core/matching_engine.py`.
    """
    qclient = client or get_async_qdrant_client()

    for collection_name in (job_collection, candidate_collection):
        exists = await qclient.collection_exists(collection_name)
        if not exists:
            await qclient.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_dim,
                    distance=Distance.COSINE,
                    on_disk=False,
                ),
                hnsw_config=HnswConfigDiff(
                    m=16,
                    ef_construct=128,
                    full_scan_threshold=10_000,
                    on_disk=False,
                ),
            )

        for field_name, field_schema in HARD_CONSTRAINT_PAYLOAD_FIELDS.items():
            await qclient.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=field_schema,
                wait=True,
            )


def build_qdrant_hard_constraint_filter(
    *,
    location: str | None = None,
    min_salary: float | None = None,
    max_salary: float | None = None,
    min_experience_years: float | None = None,
    career_domain: str | None = None,
    nhom_khuyet_tat: str | list[str] | None = None,
    work_mode: str | None = None,
) -> Filter | None:
    """Xây dựng bộ lọc Pre-filtering (Filterable HNSW) cho Qdrant theo các trường lọc cứng."""
    must_conditions: list[Any] = []

    if location:
        must_conditions.append(
            FieldCondition(key="location", match=MatchValue(value=location))
        )
    if career_domain:
        must_conditions.append(
            FieldCondition(key="career_domain", match=MatchValue(value=career_domain))
        )
    if work_mode:
        must_conditions.append(
            FieldCondition(key="work_mode", match=MatchValue(value=work_mode))
        )
    if nhom_khuyet_tat:
        if isinstance(nhom_khuyet_tat, str):
            must_conditions.append(
                FieldCondition(
                    key="nhom_khuyet_tat",
                    match=MatchValue(value=nhom_khuyet_tat),
                )
            )
        else:
            must_conditions.append(
                FieldCondition(
                    key="nhom_khuyet_tat",
                    match=MatchAny(any=list(nhom_khuyet_tat)),
                )
            )
    if min_salary is not None:
        must_conditions.append(
            FieldCondition(key="salary_max", range=Range(gte=float(min_salary)))
        )
    if max_salary is not None:
        must_conditions.append(
            FieldCondition(key="salary_min", range=Range(lte=float(max_salary)))
        )
    if min_experience_years is not None:
        must_conditions.append(
            FieldCondition(
                key="experience_years",
                range=Range(lte=float(min_experience_years)),
            )
        )

    if not must_conditions:
        return None
    return Filter(must=must_conditions)
