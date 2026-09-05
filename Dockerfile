# Dùng Python 3.12 bản slim để image nhẹ
FROM python:3.12-slim

# Thư mục làm việc bên trong container
WORKDIR /code

# Cài các thư viện hệ thống cần để build psycopg2 và torch chạy được
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements trước để tận dụng cache layer của Docker
COPY requirements.txt .

# Cài torch bản CPU trước (nhẹ hơn nhiều so với bản GPU mặc định)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Cài các thư viện còn lại (tăng timeout phòng khi mạng chậm khi tải các gói lớn)
RUN pip install --no-cache-dir --default-timeout=180 -r requirements.txt

# Copy toàn bộ code ứng dụng vào container
COPY app ./app
COPY scripts ./scripts

# Mở cổng 8000 cho FastAPI
EXPOSE 8000

# Lệnh chạy server khi container start
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
