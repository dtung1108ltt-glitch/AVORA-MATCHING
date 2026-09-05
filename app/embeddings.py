from functools import lru_cache                # cache model để chỉ load 1 lần duy nhất
from sentence_transformers import SentenceTransformer   # thư viện wrapper để dùng model embedding

from app.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)                              # đảm bảo model chỉ load vào RAM 1 lần cho cả app
def get_embedding_model() -> SentenceTransformer:
    """Load model Qwen3-Embedding-0.6B, dùng chung cho cả luồng index job và luồng search."""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)   # tải model từ HuggingFace (hoặc cache local)


def get_embedding(text: str, is_query: bool = False) -> list[float]:
    """
    Chuyển 1 đoạn text thành vector embedding.

    - is_query=False: dùng khi encode "mô tả cv" (job) lúc lưu vào DB.
    - is_query=True : dùng khi encode "yêu cầu tuyển dụng" của nhà tuyển dụng lúc search,
                       Qwen3-Embedding hỗ trợ prompt riêng cho query giúp tăng độ chính xác.
    """
    model = get_embedding_model()                   # lấy model đã cache sẵn

    if is_query:
        # prompt_name="query" là prompt đặc biệt Qwen3-Embedding định nghĩa sẵn cho câu truy vấn
        vector = model.encode(text, prompt_name="query", normalize_embeddings=True)
    else:
        # Với văn bản tài liệu (job description) thì encode bình thường, không cần prompt riêng
        vector = model.encode(text, normalize_embeddings=True)

    return vector.tolist()                          # convert numpy array -> list[float] để lưu vào pgvector
