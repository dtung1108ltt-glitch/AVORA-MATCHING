# ===========================================================================
# Stage 1: Builder - Cài đặt dependencies siêu tốc bằng Astral uv
# ===========================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Tích hợp trình quản lý gói Astral uv từ official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Cài các thư viện hệ thống tối thiểu phục vụ biên dịch C-extensions nếu cần
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng tối đa Docker layer caching
COPY requirements.txt .

# Cài đặt toàn bộ gói vào môi trường hệ thống bằng uv (không dùng PyTorch/CUDA nặng nề)
RUN uv pip install --system --no-cache -r requirements.txt

# ===========================================================================
# Stage 2: Runtime - Image thực thi tinh gọn cho Production
# ===========================================================================
FROM python:3.12-slim AS runtime

WORKDIR /code

# Tích hợp uv vào runtime để hỗ trợ quản lý hoặc chạy script khi cần
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy các gói Python đã cài đặt và binaries từ tầng builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy toàn bộ mã nguồn ứng dụng, lõi thuật toán và scripts vào container
COPY src ./src
COPY app ./app
COPY scripts ./scripts

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh chạy server khi container khởi động
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
