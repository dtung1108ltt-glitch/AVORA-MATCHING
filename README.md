# AVORA Matching API

AVORA Matching là backend FastAPI dùng semantic embedding để xác định nhóm khuyết tật phù hợp với một mô tả công việc, sau đó lọc danh sách ứng viên thuộc nhóm đã chọn.

> Lưu ý: dữ liệu được tạo bởi các script trong project là dữ liệu mô phỏng phục vụ phát triển và kiểm thử. Kết quả matching không thay thế đánh giá trực tiếp về năng lực, điều kiện tiếp cận hoặc sự phù hợp của từng ứng viên.

## 1. Workflow hệ thống

```text
job_descriptions.csv
├── Nhóm khuyết tật
└── Mô tả công việc
        │
        ▼
ingest_job_descriptions.py
        │
        ▼
POST /job-descriptions
        │
        ▼
Mô tả công việc -> Embedding
        │
        ▼
PostgreSQL + pgvector

candidates.csv
├── Họ và tên
└── Nhóm khuyết tật
        │
        ▼
ingest_candidates.py
        │
        ▼
POST /candidates
        │
        ▼
PostgreSQL

Query của nhà tuyển dụng
        │
        ▼
POST /match
        │
        ▼
Query -> Embedding
        │
        ▼
Cosine similarity với embedding mô tả công việc
        │
        ▼
Tổng hợp điểm theo nhóm khuyết tật
        │
        ▼
Lọc ứng viên theo nhóm
        │
        ▼
Trả nhóm, mô tả gần nhất và danh sách ứng viên
```

### Quy tắc embedding

Query và mô tả trong database đều là mô tả công việc. Vì vậy, cả hai phải dùng:

- Cùng model embedding
- Cùng cách chuẩn hóa văn bản
- Cùng hàm `get_embedding(text)`
- Cùng `normalize_embeddings=True`
- Không thêm tên nhóm khuyết tật vào nội dung được embedding

Nếu thay model hoặc thay cách tạo embedding, phải xóa vector cũ và ingest lại toàn bộ `job_descriptions.csv`.

## 2. Cấu trúc project

```text
AVORA-MATCHING-main/
│
├── docker-compose.yml
├── Dockerfile
├── README.md
├── requirements.txt
│
├── app/
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── __init__.py
│
└── scripts/
    ├── generate_job_description_database.py
    ├── generate_mock_candidates.py
    ├── ingest_job_descriptions.py
    ├── ingest_candidates.py
    └── data/
        ├── job_descriptions.csv
        └── candidates.csv
```

## 3. Yêu cầu môi trường

Cài đặt:

- Docker Desktop
- Docker Compose
- Python 3.10 trở lên nếu chạy script từ Windows
- Kết nối Internet trong lần đầu tải model Qwen

Kiểm tra:

```bat
docker --version
docker compose version
python --version
```

## 4. Cấu hình chính

`docker-compose.yml` cần truyền các biến môi trường sau cho service `web`:

```yaml
environment:
  DATABASE_URL: postgresql://avora:avora@db:5432/avora_matching
  EMBEDDING_MODEL_NAME: Qwen/Qwen3-Embedding-0.6B
  VECTOR_DIM: "1024"
```

Cấu hình `app/config.py` tối thiểu:

```python
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://avora:avora@localhost:5432/avora_matching",
)

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "Qwen/Qwen3-Embedding-0.6B",
)

VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))
```

## 5. Cài thư viện cho script chạy trên Windows

Nếu chạy các script ingest từ Windows host, tạo virtual environment:

```bat
cd C:\Users\Bi\Downloads\AVORA-MATCHING-main
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install requests
```

Các script generate chỉ dùng thư viện chuẩn Python. Hai script ingest cần `requests`.

Nếu muốn cài toàn bộ dependency project trên máy host:

```bat
pip install -r requirements.txt
```

## 6. Khởi tạo lại database sau khi đổi schema

Project cũ có schema `candidates` khác với schema hiện tại. `Base.metadata.create_all()` không tự sửa bảng cũ.

Trong môi trường phát triển, chạy một lần:

```bat
docker compose down -v
docker compose up --build -d
```

Cảnh báo: `docker compose down -v` xóa toàn bộ dữ liệu PostgreSQL trong volume hiện tại.

Các lần chạy sau, nếu không đổi schema và không muốn xóa dữ liệu:

```bat
docker compose up -d
```

## 7. Kiểm tra container

Kiểm tra trạng thái:

```bat
docker compose ps
```

Xem log API:

```bat
docker compose logs -f web
```

Xem log database:

```bat
docker compose logs -f db
```

Dừng theo dõi log bằng `Ctrl + C`.

Mở Swagger UI:

```text
http://localhost:8000/docs
```

Kiểm tra API:

```text
http://localhost:8000/health
```

Kết quả ban đầu:

```json
{
  "status": "healthy",
  "job_descriptions": 0,
  "candidates": 0
}
```

## 8. Tạo database mô tả công việc

Chạy:

```bat
python scripts\generate_job_description_database.py --count 100
```

File đầu ra:

```text
scripts\data\job_descriptions.csv
```

CSV có đúng hai cột:

```csv
Nhóm khuyết tật,Mô tả công việc
Khiếm thính,"Nhập và đối chiếu dữ liệu trên Excel, trao đổi chủ yếu bằng văn bản."
Khuyết tật vận động,"Phát triển API bằng Python, làm việc với SQL và có thể làm từ xa."
```

Một mô tả có thể xuất hiện nhiều dòng nếu được gắn với nhiều nhóm khuyết tật.

## 9. Tạo mock database ứng viên

Tạo 3.000 ứng viên:

```bat
python scripts\generate_mock_candidates.py --count 300
```

File đầu ra:

```text
scripts\data\candidates.csv
```

CSV có đúng hai cột:

```csv
Họ và tên,Nhóm khuyết tật
Nguyễn Minh An,Khiếm thính
Trần Ngọc Linh,Khuyết tật vận động
```

## 10. Kiểm tra dữ liệu trước khi ingest

Kiểm tra mô tả công việc:

```bat
python scripts\ingest_job_descriptions.py --dry-run
```

Kiểm tra ứng viên:

```bat
python scripts\ingest_candidates.py --dry-run
```

`--dry-run` chỉ kiểm tra CSV và in payload mẫu. Dữ liệu chưa được gửi lên API.

## 11. Ingest mô tả công việc và tạo vector

Chạy:

```bat
python scripts\ingest_job_descriptions.py --timeout 300
```

Request đầu tiên có thể chậm vì container cần tải và load model embedding.

Mỗi dòng được gửi dưới dạng:

```json
{
  "nhom_khuyet_tat": "Khiếm thính",
  "mo_ta_cong_viec": "Nhập và đối chiếu dữ liệu trên Excel..."
}
```

Backend chỉ embedding trường `mo_ta_cong_viec`.

## 12. Ingest ứng viên

Chạy:

```bat
python scripts\ingest_candidates.py
```

Mỗi dòng được gửi dưới dạng:

```json
{
  "ho_ten": "Nguyễn Minh An",
  "nhom_khuyet_tat": "Khiếm thính"
}
```

Ứng viên không được tạo embedding.

## 13. Kiểm tra số lượng sau ingest

Mở:

```text
http://localhost:8000/health
```

Ví dụ:

```json
{
  "status": "healthy",
  "job_descriptions": 275,
  "candidates": 3000
}
```

Số `job_descriptions` có thể lớn hơn số `--count`, vì một mô tả có thể thuộc nhiều nhóm.

## 14. Test matching bằng Swagger

Mở:

```text
http://localhost:8000/docs
```

Chọn `POST /match`, bấm **Try it out**, nhập:

```json
{
  "query": "Cần người nhập và kiểm tra dữ liệu trên Excel, làm việc tại bàn, trao đổi qua tin nhắn và không cần nghe điện thoại",
  "top_k_descriptions": 20,
  "top_k_candidates": 20
}
```

Response mẫu:

```json
{
  "matched_groups": [
    {
      "nhom_khuyet_tat": "Khiếm thính",
      "score": 0.82
    }
  ],
  "matched_descriptions": [
    {
      "id": 1,
      "nhom_khuyet_tat": "Khiếm thính",
      "mo_ta_cong_viec": "Nhập và đối chiếu dữ liệu trên Excel...",
      "similarity": 0.88
    }
  ],
  "candidates": [
    {
      "id": 10,
      "ho_ten": "Nguyễn Minh An",
      "nhom_khuyet_tat": "Khiếm thính"
    }
  ]
}
```

## 15. Test bằng PowerShell

```powershell
$body = @{
    query = "Cần người nhập và kiểm tra dữ liệu trên Excel, làm việc tại bàn và trao đổi qua tin nhắn"
    top_k_descriptions = 20
    top_k_candidates = 20
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://localhost:8000/match" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## 16. Test bằng curl trên một dòng

```bat
curl -X POST "http://localhost:8000/match" -H "Content-Type: application/json" -d "{\"query\":\"Cần người nhập và kiểm tra dữ liệu trên Excel, trao đổi qua tin nhắn\",\"top_k_descriptions\":20,\"top_k_candidates\":20}"
```

## 17. Kiểm tra score embedding

### Test cùng một văn bản

Nếu cùng một đoạn văn được encode bằng cùng pipeline:

```text
get_embedding(text) với get_embedding(text)
```

cosine similarity phải gần `1.0`.

### Test mô tả gần nghĩa

Điều quan trọng không chỉ là score tuyệt đối. Cần kiểm tra thứ tự:

```text
Score cùng văn bản
    >
Score văn bản gần nghĩa
    >
Score văn bản không liên quan
```

Ví dụ hợp lý:

```text
Cùng văn bản:       1.00
Gần nghĩa:          0.62
Không liên quan:    0.25
```

Các số trên chỉ là ví dụ. Ngưỡng thực tế phải được hiệu chỉnh bằng bộ query kiểm thử.

### Khi score cùng văn bản chỉ khoảng 0.5

Kiểm tra theo thứ tự:

1. Query và database có dùng cùng `get_embedding(text)` hay không.
2. Có đang dùng vector cũ không.
3. API container có được restart sau khi sửa code không.
4. Model có đúng `Qwen/Qwen3-Embedding-0.6B` không.
5. `VECTOR_DIM` có bằng `1024` không.
6. Có đang đọc cosine distance thay vì similarity không.
7. Có tính similarity bằng `1 - cosine_distance` không.

Sau khi đổi pipeline embedding, reset và ingest lại:

```bat
docker compose down -v
docker compose up --build -d
python scripts\ingest_job_descriptions.py --timeout 300
python scripts\ingest_candidates.py
```

## 18. Các lỗi thường gặp

### Lỗi `404` khi ingest job descriptions

```text
Endpoint POST /job-descriptions chưa tồn tại
```

Kiểm tra:

- `app/main.py` đã được thay bằng phiên bản mới
- Container `web` đã restart
- Swagger có endpoint `/job-descriptions`

Chạy:

```bat
docker compose restart web
```

### Lỗi `422 Unprocessable Entity`

Schema backend không khớp với payload.

`JobDescriptionCreate` phải nhận:

```text
nhom_khuyet_tat
mo_ta_cong_viec
```

`CandidateCreate` phải nhận:

```text
ho_ten
nhom_khuyet_tat
```

### Lỗi database vẫn yêu cầu cột cũ

Ví dụ database vẫn yêu cầu `mo_ta` hoặc `embedding` trong bảng candidates.

Nguyên nhân: volume PostgreSQL đang giữ schema cũ.

Cách xử lý trong môi trường test:

```bat
docker compose down -v
docker compose up --build -d
```

### Lỗi không kết nối được API

Kiểm tra:

```bat
docker compose ps
docker compose logs web
docker compose logs db
```

Đảm bảo truy cập được:

```text
http://localhost:8000/health
```

### Lỗi tải model hoặc request đầu tiên bị timeout

Chạy ingest với timeout dài hơn:

```bat
python scripts\ingest_job_descriptions.py --timeout 600
```

Nên mount Hugging Face cache trong `docker-compose.yml` để không tải lại model mỗi khi tạo container mới.

### Sửa code nhưng API chưa thay đổi

Do Uvicorn không chạy với `--reload`, cần restart service:

```bat
docker compose restart web
```

Nếu thay `Dockerfile` hoặc `requirements.txt`:

```bat
docker compose up --build -d
```

## 19. Lệnh chạy nhanh từ đầu

```bat
cd C:\Users\Bi\Downloads\AVORA-MATCHING-main

python scripts\generate_job_description_database.py --count 100
python scripts\generate_mock_candidates.py --count 300

python scripts\ingest_job_descriptions.py --dry-run
python scripts\ingest_candidates.py --dry-run

docker compose down -v
docker compose up --build -d

python scripts\ingest_job_descriptions.py --timeout 300
python scripts\ingest_candidates.py
```

Sau đó mở:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

## 20. Dừng hệ thống

Dừng container nhưng giữ dữ liệu:

```bat
docker compose down
```

Dừng và xóa toàn bộ database:

```bat
docker compose down -v
```

Không dùng `-v` nếu muốn giữ dữ liệu đã ingest.

## 21. Endpoint reference

### `GET /health`

Kiểm tra API, database và số bản ghi.

### `POST /job-descriptions`

Thêm mô tả công việc, tạo embedding và lưu vector.

### `GET /job-descriptions`

Liệt kê mô tả công việc. Hỗ trợ `skip`, `limit` và `nhom_khuyet_tat`.

### `POST /candidates`

Thêm ứng viên, không tạo embedding.

### `GET /candidates`

Liệt kê ứng viên. Hỗ trợ `skip`, `limit` và `nhom_khuyet_tat`.

### `POST /match`

Embedding query, tìm mô tả gần nhất, tổng hợp nhóm và trả danh sách ứng viên.

## 22. Giới hạn hiện tại

- Dữ liệu generate là dữ liệu mô phỏng.
- Một mô tả thuộc nhiều nhóm đang được lưu thành nhiều bản ghi và có thể lặp vector.
- Ngưỡng score cần được hiệu chỉnh bằng bộ test thực tế.
- Matching theo nhóm không đủ để kết luận một ứng viên cụ thể phù hợp với công việc.
- Hệ thống chưa kiểm tra kỹ năng, kinh nghiệm, địa điểm, yêu cầu tiếp cận hoặc điều kiện bắt buộc của từng ứng viên.
