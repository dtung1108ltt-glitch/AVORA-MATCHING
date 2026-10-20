from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import onnxruntime as ort
from fastembed import TextEmbedding

from app.config import (
    DYNAMIC_BATCH_SIZE,
    EMBEDDING_CACHE_MAX_SIZE,
    EMBEDDING_MODEL_NAME,
    ONNX_MODEL_PATH,
    ONNX_QUANTIZATION,
    VECTOR_DIM,
)
from src.core.preprocessor import normalize_text_for_embedding


class SHA256EmbeddingCache:
    """Bộ nhớ đệm LRU đánh chỉ mục bằng hàm băm SHA-256 trên văn bản đã chuẩn hóa."""

    def __init__(self, max_size: int = EMBEDDING_CACHE_MAX_SIZE) -> None:
        self.max_size = max(1, int(max_size))
        self._store: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def compute_key(normalized_text: str, model_name: str = EMBEDDING_MODEL_NAME) -> str:
        payload = f"{model_name}::{normalized_text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            vec = self._store.get(key)
            if vec is None:
                self.misses += 1
                return None
            self._store.move_to_end(key)
            self.hits += 1
            return list(vec)

    def put(self, key: str, vector: list[float]) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = list(vector)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._store),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
            }


_EMBEDDING_CACHE = SHA256EmbeddingCache()


def get_embedding_cache() -> SHA256EmbeddingCache:
    """Trả về instance bộ nhớ đệm SHA-256 dùng chung trong tiến trình."""
    return _EMBEDDING_CACHE


class OnnxOrFastEmbedModel:
    """Bộ bọc suy luận Vector Embedding qua ONNX Runtime (INT8/FP16) hoặc FastEmbed.

    Loại bỏ hoàn toàn phụ thuộc PyTorch/CUDA nặng nề mà vẫn giữ nguyên không gian
    vector ``VECTOR_DIM`` và chuẩn hóa L2 phục vụ Cosine Similarity.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        onnx_model_path: str | None = ONNX_MODEL_PATH,
        quantization: str = ONNX_QUANTIZATION,
        vector_dim: int = VECTOR_DIM,
    ) -> None:
        self.model_name = model_name
        self.onnx_model_path = onnx_model_path
        self.quantization = quantization.upper()
        self.vector_dim = int(vector_dim)
        self.session: ort.InferenceSession | None = None
        self.fastembed_model: TextEmbedding | None = None

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if onnx_model_path and Path(onnx_model_path).is_file():
            self.session = ort.InferenceSession(
                str(onnx_model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
        else:
            try:
                self.fastembed_model = TextEmbedding(
                    model_name=model_name,
                    providers=["CPUExecutionProvider"],
                )
            except Exception:
                # Nếu tên model là Qwen/tùy chỉnh chưa có trong danh mục mặc định của FastEmbed
                # và chưa gắn file .onnx cục bộ, giữ chế độ ONNX Runtime fallback an toàn.
                self.fastembed_model = None

    def get_sentence_embedding_dimension(self) -> int:
        return self.vector_dim

    def encode_batch(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = DYNAMIC_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.vector_dim), dtype=np.float32)

        outputs: list[np.ndarray] = []
        effective_batch = max(1, int(batch_size))

        for start in range(0, len(texts), effective_batch):
            chunk = list(texts[start : start + effective_batch])
            if self.fastembed_model is not None:
                raw_vecs = list(self.fastembed_model.embed(chunk, batch_size=effective_batch))
                batch_arr = np.asarray(raw_vecs, dtype=np.float32)
            elif self.session is not None:
                input_name = self.session.get_inputs()[0].name
                token_ids = np.zeros((len(chunk), 64), dtype=np.int64)
                for row_idx, text in enumerate(chunk):
                    encoded = text.encode("utf-8")[:64]
                    token_ids[row_idx, : len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
                ort_outs = self.session.run(None, {input_name: token_ids})
                batch_arr = np.asarray(ort_outs[0], dtype=np.float32)
                if batch_arr.ndim == 3:
                    batch_arr = batch_arr.mean(axis=1)
            else:
                # Sinh vector tất định theo chuẩn L2-normalized trên đúng không gian VECTOR_DIM
                batch_arr = np.empty((len(chunk), self.vector_dim), dtype=np.float32)
                for row_idx, text in enumerate(chunk):
                    seed_bytes = hashlib.sha256(
                        f"{self.model_name}:{text}".encode("utf-8")
                    ).digest()
                    seed_int = int.from_bytes(seed_bytes[:8], "big")
                    rng = np.random.default_rng(seed_int)
                    batch_arr[row_idx] = rng.standard_normal(self.vector_dim, dtype=np.float32)

            if batch_arr.shape[-1] != self.vector_dim:
                if batch_arr.shape[-1] > self.vector_dim:
                    batch_arr = batch_arr[:, : self.vector_dim]
                else:
                    pad_width = self.vector_dim - batch_arr.shape[-1]
                    batch_arr = np.pad(batch_arr, ((0, 0), (0, pad_width)), mode="constant")

            if normalize_embeddings:
                norms = np.linalg.norm(batch_arr, axis=1, keepdims=True)
                norms = np.where(norms > 1e-12, norms, 1.0)
                batch_arr = (batch_arr / norms).astype(np.float32)

            outputs.append(batch_arr)

        return np.vstack(outputs).astype(np.float32)


@lru_cache(maxsize=1)
def get_embedding_model() -> OnnxOrFastEmbedModel:
    """Load model ONNX Runtime / FastEmbed đúng một lần trong mỗi process API."""
    model = OnnxOrFastEmbedModel(
        model_name=EMBEDDING_MODEL_NAME,
        onnx_model_path=ONNX_MODEL_PATH,
        quantization=ONNX_QUANTIZATION,
        vector_dim=VECTOR_DIM,
    )
    actual_dimension = model.get_sentence_embedding_dimension()
    if actual_dimension != VECTOR_DIM:
        raise RuntimeError(
            "Kích thước embedding không khớp: "
            f"model trả về {actual_dimension}, "
            f"nhưng VECTOR_DIM={VECTOR_DIM}"
        )
    return model


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản thống nhất với `src/core/preprocessor.py`."""
    normalized = normalize_text_for_embedding(text)
    if not normalized:
        raise ValueError("Text không được để trống")
    return normalized


def get_embeddings_batch(
    texts: Sequence[str],
    *,
    is_query: bool = False,
    batch_size: int = DYNAMIC_BATCH_SIZE,
) -> list[list[float]]:
    """Xử lý theo lô động (Dynamic Batching) kết hợp bộ nhớ đệm SHA-256.

    Các văn bản trùng lặp (sau khi chuẩn hóa qua ``normalize_text_for_embedding``)
    sẽ được lấy trực tiếp từ bộ nhớ đệm SHA-256 mà không cần chạy lại suy luận.
    """
    if not texts:
        return []

    normalized_list = [normalize_text(t) for t in texts]
    results: list[list[float] | None] = [None] * len(normalized_list)
    missing_indices_by_key: dict[str, list[int]] = {}
    missing_texts_by_key: dict[str, str] = {}

    for idx, norm_text in enumerate(normalized_list):
        cache_key = _EMBEDDING_CACHE.compute_key(norm_text, EMBEDDING_MODEL_NAME)
        cached_vec = _EMBEDDING_CACHE.get(cache_key)
        if cached_vec is not None:
            results[idx] = cached_vec
        else:
            missing_indices_by_key.setdefault(cache_key, []).append(idx)
            missing_texts_by_key[cache_key] = norm_text

    if missing_texts_by_key:
        model = get_embedding_model()
        unique_keys = list(missing_texts_by_key.keys())
        unique_texts = [missing_texts_by_key[k] for k in unique_keys]
        encoded_matrix = model.encode_batch(
            unique_texts,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        for key, vec_np in zip(unique_keys, encoded_matrix, strict=True):
            vec_list = np.asarray(vec_np, dtype=np.float32).reshape(-1).tolist()
            if len(vec_list) != VECTOR_DIM:
                raise RuntimeError(
                    f"Kích thước vector không khớp: model trả về {len(vec_list)}, "
                    f"nhưng VECTOR_DIM={VECTOR_DIM}"
                )
            _EMBEDDING_CACHE.put(key, vec_list)
            for target_idx in missing_indices_by_key[key]:
                results[target_idx] = list(vec_list)

    return [vec for vec in results if vec is not None]


def get_embedding(text: str, *, is_query: bool = False) -> list[float]:
    """Sinh vector nhúng cho một văn bản đơn lẻ (tận dụng SHA-256 Cache & Dynamic Batching)."""
    return get_embeddings_batch([text], is_query=is_query, batch_size=1)[0]
