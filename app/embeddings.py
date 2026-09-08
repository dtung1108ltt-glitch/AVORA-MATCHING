from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import EMBEDDING_MODEL_NAME, VECTOR_DIM


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load model đúng một lần trong mỗi process API.
    """
    print(
        f"Loading embedding model: {EMBEDDING_MODEL_NAME}...",
        flush=True,
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    actual_dimension = model.get_sentence_embedding_dimension()

    if actual_dimension != VECTOR_DIM:
        raise RuntimeError(
            "Kích thước embedding không khớp: "
            f"model trả về {actual_dimension}, "
            f"nhưng VECTOR_DIM={VECTOR_DIM}"
        )

    print(
        "Embedding model loaded successfully.",
        flush=True,
    )
    print(
        f"Embedding dimension: {actual_dimension}",
        flush=True,
    )

    return model


def normalize_text(text: str) -> str:
    """
    Chuẩn hóa giống nhau cho mô tả database và query.
    """
    normalized = " ".join(
        str(text or "").split()
    )

    if not normalized:
        raise ValueError("Text không được để trống")

    return normalized


def get_embedding(text: str) -> list[float]:
    """
    Dùng cùng một pipeline cho:
    - mô tả công việc trong database
    - query của nhà tuyển dụng
    """
    normalized_text = normalize_text(text)
    model = get_embedding_model()

    vector = model.encode(
        normalized_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    vector = np.asarray(
        vector,
        dtype=np.float32,
    ).reshape(-1)

    if vector.shape[0] != VECTOR_DIM:
        raise RuntimeError(
            "Kích thước vector không khớp: "
            f"model trả về {vector.shape[0]}, "
            f"nhưng VECTOR_DIM={VECTOR_DIM}"
        )

    return vector.tolist()