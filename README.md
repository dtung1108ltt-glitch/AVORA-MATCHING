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
AVORA-MATCHING/
│
├── docker-compose.yml
├── Dockerfile
├── README.md
├── document-tech.md
├── requirements.txt                # FastAPI + Postgres + embedding (dùng cho Docker)
├── requirements-dashboard.txt      # Streamlit cho dashboard local (tùy chọn)
├── dashboard.py                    # Dashboard Streamlit -> localhost:8501
│
├── app/                            # API FastAPI -> localhost:8000
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── __init__.py
│
├── src/                            # Engine Gale-Shapley (stdlib thuần)
│   ├── core/
│   │   ├── preprocessor.py
│   │   ├── matching_engine.py
│   │   └── postprocessor.py
│   └── evaluation/
│       ├── metrics.py
│       └── benchmark.py
│
├── tests/                          # pytest / unittest
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

## 23. Core Matching Engine - Ghép cặp ổn định hai phía (Gale-Shapley)

Phần trên (mục 1-22) mô tả API lọc ứng viên *một chiều* theo nhóm khuyết tật.
Mục này bổ sung **engine ghép cặp ổn định hai phía** (Two-Sided Stable Matching)
nằm trong `src/`, giải quyết đúng bản chất bài toán thị trường lao động: cả ứng
viên và nhà tuyển dụng đều có nguyện vọng riêng.

### 23.1. Vì sao cần matching ổn định?

Cách lọc theo điểm (ranking) hiện tại có thể tạo ra *blocking pair*: một ứng
viên và một vị trí đều thích nhau hơn lựa chọn hiện tại, nhưng hệ thống không
ghép họ với nhau. Matching ổn định loại bỏ hoàn toàn tình huống đó.

### 23.2. Pipeline 4 giai đoạn

```text
GIAI ĐOẠN 1 & 2 - TIỀN XỬ LÝ  (src/core/preprocessor.py)
─────────────────────────────────────────────────────────────
 Ma trận E[c][j]  ┐
 Ma trận C[c][j]  ├─► Chuẩn hóa ─► Hard Constraints ─► Truncated
 Quota q_j        ┘   (thưa/đặc)   (điểm=0, không      Preference Lists
                                    khả thi, < sàn)     (L, M)
        │
        ▼
 PreferenceProfile { candidate_preferences, job_preferences, quotas }
        │
        ▼
GIAI ĐOẠN 2 - RANK LOOKUP TABLE  (src/core/matching_engine.py)
─────────────────────────────────────────────────────────────
 job_rank[j][c]  ─► O(1)      candidate_rank[c][j] ─► O(1)
        │
        ▼
GIAI ĐOẠN 3 - GALE-SHAPLEY MANY-TO-ONE DEFERRED ACCEPTANCE
─────────────────────────────────────────────────────────────
 Candidate-Proposing (mặc định)  ──► kết quả C-optimal
 Employer-Proposing              ──► kết quả J-optimal
 Min-Heap mỗi vị trí: đỉnh = ứng viên kém nhất  ─► loại O(log q_j)
        │
        ▼
 MatchingResult { matches, job_assignments, unmatched_* }
        │
        ▼
GIAI ĐOẠN 4 - HẬU XỬ LÝ TOP-K  (src/core/postprocessor.py)
─────────────────────────────────────────────────────────────
 Normalized Surplus Score        ──► generate_top_k()
 Iterative Clearance Rounds      ──► iterative_clearance_rounds()
        │
        ▼
 ĐÁNH GIÁ  (src/evaluation/metrics.py)
 Blocking pairs = 0%  |  Điểm E, C trung bình  |  Fill rate
```

### 23.3. Cấu trúc thư mục bổ sung

```text
AVORA-MATCHING/
│
├── conftest.py                       # đưa repo root vào sys.path cho pytest
│
├── src/
│   ├── core/
│   │   ├── __init__.py               # tái xuất API công khai
│   │   ├── preprocessor.py           # Giai đoạn 1 & 2
│   │   ├── matching_engine.py        # Giai đoạn 3 (Gale-Shapley + Rank Table)
│   │   └── postprocessor.py          # Giai đoạn 4 (Top-K)
│   └── evaluation/
│       ├── __init__.py
│       ├── metrics.py                # blocking pairs, satisfaction, fill rate
│       └── benchmark.py              # benchmark vs baseline + CLI
│
└── tests/
    ├── __init__.py
    └── test_matching.py              # 31 test (unittest + pytest)
```

### 23.4. Cơ sở lý thuyết

| Chủ đề | Nguồn | Áp dụng trong code |
| --- | --- | --- |
| Thuật toán Deferred Acceptance (1962) | Gale, D., & Shapley, L. S. *College Admissions and the Stability of Marriage*. American Mathematical Monthly. | `GaleShapleyMatching._solve_candidate_proposing` |
| Many-to-One / Hospital-Residents | Roth, A. E. (1984). *The Evolution of the Labor Market for Medical Interns*. | `_solve_employer_proposing`, `quotas` |
| Tính C-optimal / J-optimal | Gale & Shapley (1962); Roth & Sotomayor (1990). | tham số `optimal="candidate"` / `"employer"` |
| Strategy-proofness phía proposer | Dubins, L. E., & Freedman, D. A. (1981). *Machiavelli and the Gale-Shapley Algorithm*. | lựa chọn candidate-proposing làm mặc định |
| Rank lookup table | Knuth, D. E. (1997). *Stable Marriage and Its Relation to Other Combinatorial Problems*. | `_build_rank_tables`, `job_rank`, `candidate_rank` |
| Min-Heap eviction | Cormen et al., *Introduction to Algorithms* (heap) | `heapq.heapreplace`, `held[job]` |

### 23.5. Dữ liệu đầu vào E, C, Quota

Cả hai ma trận đều đánh chỉ số `[candidate][job]` với giá trị > 0:

- `E[c][j]` - **Employer Preference Score**: mức độ phù hợp kỹ năng của ứng viên `c` với vị trí `j`.
- `C[c][j]` - **Candidate Preference Score**: mức độ yêu thích vị trí `j` của ứng viên `c`.
- `q_j` - **Quota**: hạn mức tuyển dụng của vị trí `j` (Many-to-One).

Nhận cả ma trận **thưa** (`dict[c][j]`) và **đặc** (`list[list[float]]`):

```python
from src.core import Preprocessor, PreprocessConfig

profile = Preprocessor(PreprocessConfig(max_candidate_preferences=40)).build_profile(
    e_scores={0: {0: 0.9, 1: 0.2}},   # E[candidate][job]
    c_scores={0: {0: 0.4, 1: 0.8}},   # C[candidate][job]
    quotas={0: 1, 1: 3},              # q_j
)
```

### 23.6. Hard Constraints

`apply_hard_constraints` loại vĩnh viễn cặp `(c, j)` nếu:

- `(c, j)` không nằm trong `feasible_pairs` (ví dụ khoảng cách địa lý, mức lương
  sàn, sai nhóm kỹ năng bắt buộc).
- `E[c][j] <= 0` hoặc `C[c][j] <= 0`.
- Điểm dưới ngưỡng `min_e_score` / `min_c_score`.

Cặp bị loại không xuất hiện trong bất kỳ danh sách ưu tiên nào, nên **không bao
giờ** trở thành blocking pair.

### 23.7. Truncated Preference Lists

Với thị trường lớn, mỗi ứng viên chỉ giữ `L` công việc tốt nhất
(`PreprocessConfig.max_candidate_preferences`, khuyến nghị 30-50) và tùy chọn
mỗi vị trí giữ `M` ứng viên tốt nhất (`max_employer_preferences`). Cách này đưa
độ phức tạp từ `O(N * M)` về `O(N * L)`.

### 23.8. Sử dụng engine

```python
from src.core import GaleShapleyMatching, Preprocessor, PreprocessConfig
from src.evaluation.metrics import evaluate

profile = Preprocessor(PreprocessConfig(max_candidate_preferences=40)).build_profile(
    e_scores=..., c_scores=..., quotas=...
)

# C-optimal (khuyến nghị): ứng viên chủ động, strategy-proof phía ứng viên.
result = GaleShapleyMatching(profile, optimal="candidate").solve()

result.matches              # {"c1": "j1", "c2": "j2"}
result.job_assignments      # {"j1": ["c1"], "j2": ["c2"]}
result.unmatched_candidates # [...]
result.unmatched_jobs       # [...]
result.proposals            # số lời đề nghị (đo chi phí)

# J-optimal: nhà tuyển dụng chủ động.
j_optimal = GaleShapleyMatching(profile, optimal="employer").solve()

report = evaluate(profile, result)
assert report.blocking.is_stable          # blocking pair = 0
print(report.satisfaction.average_e)      # điểm E trung bình
print(report.satisfaction.average_c)      # điểm C trung bình
print(report.fill_rate.overall)           # tỷ lệ lấp đầy slot
```

### 23.9. Top-K đề xuất dự phòng

```python
from src.core import generate_top_k, iterative_clearance_rounds

# Cách 1: Normalized Surplus Score (nhanh, một lượt).
recs = generate_top_k(profile, result, k=3)
# {"c1": [Recommendation(j1, pos=1, primary=True), Recommendation(j3, pos=2), ...]}

# Cách 2: Iterative Clearance Rounds (chạy lại DA nhiều vòng).
recs = iterative_clearance_rounds(profile, k=3, optimal="candidate")
```

Với mỗi ứng viên, gợi ý số 1 (nếu đã ghép) là vị trí chính thức
(`is_primary=True`), các gợi ý sau là phương án dự phòng không trùng lặp.

### 23.10. Độ phức tạp

| Thành phần | Độ phức tạp |
| --- | --- |
| Tiền xử lý (lọc + cắt) | `O(N * M)` khi dựng cột, `O(N * L log L)` khi sắp xếp |
| Rank Lookup Table | `O(N * L)` không gian, tra cứu `O(1)` |
| Gale-Shapley DA | `O(N * L)` lời đề nghị, mỗi vòng `O(log q_j)` cho heap |
| Kiểm tra blocking pair | `O(N * L)` |
| Top-K (surplus) | `O(N * L log L)` |
| Top-K (clearance) | `O(K * N * L)` |

Trong đó `N` = số ứng viên, `M` = số vị trí, `L` = giới hạn nguyện vọng,
`q_j` = quota, `K` = số gợi ý.

### 23.11. Chạy test

```bat
python -m pytest tests -q
:: hoặc dùng unittest thuần (không cần pytest)
python -m unittest discover -s tests -v
```

Bộ test bao phủ: ổn định trên thị trường ngẫu nhiên (cả hai chế độ), C-optimal vs
J-optimal, Hard Constraints, truncation, thừa cầu/thừa cung, phá hòa tất định,
quota > 1, Top-K, Iterative Clearance, các chỉ số đánh giá và adapter CSV.

### 23.12. Chạy benchmark và so sánh baseline

```bat
python -m src.evaluation.benchmark --candidates 2000 --jobs 400 --avg-quota 5 --density 0.4 --seed 0
python -m src.evaluation.benchmark --candidates 2000 --jobs 400 --json
```

Kết quả thực đo (Python 3.14, `N=2000` ứng viên, `M=400` vị trí, quota TB=5,
density=0.4, `L=40`):

| Cấu hình | Cặp ghép | Coverage | Số phép tính | Blocking pairs | Tỷ lệ blocking | Điểm E TB | Điểm C TB | Fill rate | Thời gian (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GaleShapley (C-optimal) | 1983 | 99.15% | 16448 | 0 | 0.00% | 0.9250 | 0.9531 | 100.00% | 39.35 |
| GaleShapley (J-optimal) | 1983 | 99.15% | 10097 | 0 | 0.00% | 0.9818 | 0.9050 | 100.00% | 36.53 |
| Greedy Baseline | 1974 | 98.70% | 80000 | 1521 | 1.90% | 0.9346 | 0.9468 | 99.55% | 146.00 |

**Đọc bảng:**

- **Blocking pairs = 0** ở cả hai chế độ Gale-Shapley: đúng chuẩn ổn định; trong
  khi baseline tham lam để lại **1521 blocking pair (1.90%)** và bỏ sót 9 cặp ghép.
- **C-optimal** cho điểm `C` trung bình cao nhất (0.9531) và **J-optimal** cho
  điểm `E` trung bình cao nhất (0.9818) - minh họa đúng tính đối ngẫu của DA.
- Số phép tính của Gale-Shapley (10k-16k) **nhỏ hơn ~5-8 lần** baseline (80k) nhờ
  Rank Table `O(1)` và dừng sớm khi mọi nguyện vọng được xử lý.
- Baseline nhanh về hằng số nhưng chậm hơn ở quy mô này do phải sắp xếp toàn bộ
  `N * M = 800.000` cặp, đồng thời chất lượng thấp hơn.

### 23.13. Tích hợp với dữ liệu hiện có (adapter CSV)

Engine độc lập với API. Để chạy thử trên chính dữ liệu của repo:

```python
from src.core import Preprocessor, PreprocessConfig, load_market_from_csv, GaleShapleyMatching
from src.evaluation.metrics import evaluate

e_scores, c_scores, quotas, feasible = load_market_from_csv(
    "scripts/data/candidates.csv",
    "scripts/data/job_descriptions.csv",
    default_quota=2,
)

profile = Preprocessor(PreprocessConfig(max_candidate_preferences=30)).build_profile(
    e_scores, c_scores, quotas, feasible_pairs=feasible
)
result = GaleShapleyMatching(profile, optimal="candidate").solve()
print(evaluate(profile, result).blocking.count)  # 0
```

`load_market_from_csv` áp Hard Constraint **trùng nhóm khuyết tật** và sinh điểm
`E`, `C` mô phỏng tất định để benchmark được ngay. Khi lên production, thay điểm
mô phỏng bằng:

- `E[c][j]` - điểm tương đồng giữa embedding kỹ năng ứng viên và embedding mô tả
  công việc (tái sử dụng `app/embeddings.py`).
- `C[c][j]` - tín hiệu sở thích/mức độ quan tâm của ứng viên với vị trí.

### 23.14. Lưu ý khi tích hợp

- Engine dùng **stdlib thuần** (`heapq`, `collections`, `dataclasses`, `hashlib`),
  không thêm dependency mới; `numpy` là tùy chọn.
- Matching sinh ra chỉ là gợi ý xếp hạng, **không thay thế** đánh giá về năng lực,
  điều kiện tiếp cận và sự phù hợp của từng ứng viên.
- `optimal="candidate"` được chọn làm mặc định vì bảo vệ ứng viên
  (strategy-proof), phù hợp định hướng của AVORA.
- Khi quota hoặc danh sách ưu tiên thay đổi, cần dựng lại `PreferenceProfile`
  trước khi gọi `solve()`.

### 23.15. Cơ chế Direct Track, filled_slots và Xử lý Biên (HTTP 422)

- **Direct Track (`direct_track=True`)**: Áp dụng cho các ứng viên đã được doanh nghiệp tuyển thẳng vào một vị trí cụ thể (`hired_job_id`). Ứng viên này được rút khỏi matching pool của batch hiện tại và trừ 1 slot trong quota khả dụng của `hired_job_id`.
- **Chỗ đã lấp ngoài hệ thống (`filled_slots`)**: Đại diện cho số lượng chỉ tiêu doanh nghiệp đã tuyển từ **bên ngoài hệ thống** (external hires) trước đợt matching.
- **Quy tắc phân định tránh đếm hai lần**: Khi đồng thời có ứng viên Direct Track và khai báo `filled_slots`, hai số này sẽ được cộng dồn. `filled_slots` **chỉ tính chỗ đã lấp ngoài hệ thống, KHÔNG tính ứng viên Direct Track có trong input** để tránh bị trừ hai lần quota.
- **Kiểm soát tính toàn vẹn dữ liệu (HTTP 422 Unprocessable Entity)**: API `/matching/stable` và engine kiểm tra nghiêm ngặt 3 trường hợp biên sau thay vì chạy tiếp trong im lặng:
  1. `direct_track=True` nhưng thiếu hoặc rỗng `hired_job_id` -> Trả về lỗi 422 (chặn nguy cơ ứng viên bị rút khỏi pool mà không trừ quota vị trí nào).
  2. `hired_job_id` trỏ tới vị trí không tồn tại trong danh sách việc làm -> Trả về lỗi 422 kèm danh sách job hợp lệ.
  3. Tuyển trực tiếp nhiều hơn quota hoặc `direct_track_count + filled_slots > quota` -> Trả về lỗi 422 (ngăn chặn tình trạng quota bị âm thầm ép về 0).
- **Lớp kiểm tra tính ổn định vét cạn độc lập (`tests/test_exhaustive_stability_checker.py`)**: Kiểm thử song song độc lập hoàn toàn với engine từ nguyên lý toán học gốc Gale-Shapley Many-to-One, bổ sung cho lớp kiểm tra của dự án.


## 24. Dashboard Streamlit - giao diện thử nghiệm engine Gale-Shapley

Repo hiện có **hai giao diện web độc lập**, không phụ thuộc nhau:

| Giao diện | Framework | Entrypoint | Cổng | Dùng cho |
| --- | --- | --- | --- | --- |
| API matching theo embedding | **FastAPI + Uvicorn** | `app/main.py` | `8000` | Query vector -> nhóm khuyết tật -> danh sách ứng viên |
| Dashboard engine hai phía | **Streamlit** | `dashboard.py` | `8501` | Ma trận E/C + quota -> matching ổn định + chỉ số |

Dashboard dùng chung package `src/` nhưng **không cần** PostgreSQL, không cần
`torch`/`sentence-transformers`, không cần Docker.

### 24.1. Cài thư viện

```bash
pip install -r requirements-dashboard.txt

# Hoặc chỉ cài đúng gói cần thiết:
pip install streamlit
```

`streamlit` cố tình **không** khai báo trong `requirements.txt` để image Docker của
service `web` không phình thêm; xem `requirements-dashboard.txt`.

### 24.2. Khởi chạy và truy cập

```bash
# Chạy tại thư mục gốc repo (nơi có dashboard.py và thư mục src/)
streamlit run dashboard.py

# Cố định cổng + cho máy khác trong LAN truy cập
streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
```

Trình duyệt mở tự động tại **http://localhost:8501**. Nếu cổng 8501 đang bận, đổi
`--server.port 8502`.

### 24.3. Bốn nguồn dữ liệu trên dashboard

1. **Thị trường mô phỏng** - sinh N ứng viên x M vị trí tất định theo `seed`.
2. **CSV của repo** - đọc `scripts/data/candidates.csv` + `job_descriptions.csv`
   qua `load_market_from_csv` (Hard Constraint: trùng nhóm khuyết tật).
3. **Dán ma trận E/C** - dán trực tiếp hai ma trận:

```text
,J1,J2,J3
C1,0.9,0.4,0.2
C2,0.8,,0.3        <- ô trống = cặp không khả thi
```

4. **Upload file ma trận** - cùng định dạng trên, dạng `.csv` / `.tsv` / `.txt`;
   file quota tùy chọn, mỗi dòng `J1, 2` hoặc `J1=2`.

### 24.4. Chức năng

- Chạy Gale-Shapley Many-to-One ở chế độ **C-optimal** hoặc **J-optimal**, tùy chỉnh
  `L`, `M`, ngưỡng sàn E/C, seed phá hòa, Top-K và trọng số điểm danh dư.
- Khối chỉ số: **Blocking Pairs** (kèm banner ổn định), **Fill Rate**, Coverage,
  điểm E/C trung bình, số lời đề nghị và thời gian chạy.
- Bảng phân bổ ứng viên - công việc (kèm hạng nguyện vọng hai phía lấy từ Rank
  Lookup Table), tình trạng từng vị trí, Top-K gợi ý dự phòng, danh sách không ghép được.
- Tải về `matching_assignments.csv`, `job_fill_rate.csv`,
  `top_k_recommendations.csv` và `matching_report.json`.

### 24.5. Test dashboard

`tests/test_dashboard.py` dùng `streamlit.testing.v1.AppTest` để **chạy thật**
dashboard: mở trang, bấm nút chạy, rồi khẳng định `Blocking pairs = 0` và các bảng
kết quả đã render.

```bash
pytest tests -q          # 38 passed, 2 subtests passed
```

Nếu môi trường chưa cài `streamlit`, file test này bị **skip** thay vì fail nên bộ
test của engine không bị ảnh hưởng.

