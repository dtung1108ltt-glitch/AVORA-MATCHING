import os   # dùng để đọc biến môi trường do docker-compose truyền vào

# Chuỗi kết nối tới PostgreSQL, mặc định trỏ vào service "db" trong docker-compose
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://avora:avora@localhost:5432/avora_matching"
)

# Tên model embedding dùng chung cho cả 2 luồng: indexing job và encode câu query
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")

# Số chiều vector mà Qwen3-Embedding-0.6B sinh ra, phải khớp với cột Vector() trong models.py
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

# Số lượng công việc giống nhất lấy ra để "bầu chọn" nhóm khuyết tật phù hợp nhất
TOP_K_JOBS = int(os.getenv("TOP_K_JOBS", "10"))

# Số lượng ứng viên tối đa trả về trong 1 lần match
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "20"))
