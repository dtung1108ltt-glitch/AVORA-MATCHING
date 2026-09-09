# TÀI LIỆU KỸ THUẬT AVORA MATCHING

## 1. Giới thiệu

AVORA Matching là hệ thống backend sử dụng **FastAPI**, **Qwen3 Embedding**, **PostgreSQL** và **pgvector** để xử lý bài toán sau:

1. Lưu các mô tả công việc mẫu cùng nhãn nhóm khuyết tật.
2. Chuyển mô tả công việc mẫu thành vector embedding.
3. Nhận một mô tả công việc mới từ người dùng.
4. Chuyển query thành vector bằng cùng pipeline embedding.
5. Tính cosine similarity giữa query và các mô tả mẫu.
6. Tổng hợp điểm theo nhóm khuyết tật.
7. Lọc danh sách ứng viên thuộc các nhóm được chọn.
8. Trả về nhóm, mô tả đối chiếu, ID và họ tên ứng viên.

> Dữ liệu do các script `generate_*` tạo là dữ liệu mô phỏng phục vụ phát triển và kiểm thử. Kết quả matching không phải kết luận chuyên môn về khả năng làm việc của một cá nhân.

---

## 2. Workflow tổng thể

```text
GIAI ĐOẠN A: TẠO DỮ LIỆU MÔ TẢ CÔNG VIỆC

generate_job_description_database.py
        ↓
job_descriptions.csv
├── Nhóm khuyết tật
└── Mô tả công việc


GIAI ĐOẠN B: INGEST VÀ TẠO EMBEDDING

job_descriptions.csv
        ↓
ingest_job_descriptions.py
        ↓
POST /job-descriptions
        ↓
get_embedding(mo_ta_cong_viec)
        ↓
Vector 1024 chiều
        ↓
PostgreSQL + pgvector


GIAI ĐOẠN C: TẠO VÀ INGEST ỨNG VIÊN

generate_mock_candidates.py
        ↓
candidates.csv
├── Họ và tên
└── Nhóm khuyết tật
        ↓
ingest_candidates.py
        ↓
POST /candidates
        ↓
Bảng candidates


GIAI ĐOẠN D: MATCHING

Query mô tả công việc
        ↓
POST /match
        ↓
get_embedding(query)
        ↓
Cosine similarity với JobDescription.embedding
        ↓
Lấy Top-K mô tả gần nhất
        ↓
Tổng hợp score theo nhóm khuyết tật
        ↓
Chọn nhóm đạt ngưỡng
        ↓
Lọc Candidate.nhom_khuyet_tat
        ↓
Trả nhóm + mô tả đối chiếu + ứng viên
```

### 2.1. Quy tắc embedding

Mô tả trong database và query đều là mô tả công việc. Hai phía phải dùng cùng pipeline:

```text
Văn bản
→ normalize_text(text)
→ get_embedding(text)
→ SentenceTransformer.encode(..., normalize_embeddings=True)
→ Vector float32, 1024 chiều
```

Không nối `nhom_khuyet_tat` vào nội dung embedding. Nhóm khuyết tật chỉ là nhãn metadata.

---

## 3. Cấu trúc project

```text
AVORA-MATCHING-main/
│
├── docker-compose.yml
├── Dockerfile
├── README.md
├── document-tech.md
├── requirements.txt
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── database.py
│   ├── embeddings.py
│   ├── main.py
│   ├── models.py
│   ├── schemas.py
│   └── __pycache__/
│
└── scripts/
    ├── convert_xlsx_to_csv.py
    ├── generate_job_description_database.py
    ├── generate_mock_candidates.py
    ├── ingest_job_descriptions.py
    ├── ingest_candidates.py
    └── data/
        ├── job_descriptions.csv
        └── candidates.csv
```

---

# 4. Chức năng của các file cấp gốc

## 4.1. `docker-compose.yml`

### Chức năng

Điều phối toàn bộ môi trường Docker của project.

### Các service

#### `db`

Khởi động PostgreSQL 16 có sẵn pgvector.

```yaml
image: pgvector/pgvector:pg16
```

Nhiệm vụ:

- Tạo database `avora_matching`.
- Tạo user và password PostgreSQL.
- Mở cổng `5432` để debug từ máy host.
- Lưu dữ liệu trong volume `pgdata`.
- Thực hiện healthcheck bằng `pg_isready`.

#### `web`

Khởi động backend FastAPI.

Nhiệm vụ:

- Build image từ `Dockerfile`.
- Nhận `DATABASE_URL` từ environment.
- Nhận tên model embedding và số chiều vector.
- Chờ `db` healthy trước khi chạy.
- Mở API tại cổng `8000`.
- Mount `app/` và `scripts/` vào container.

### Các volume

#### `pgdata`

Lưu bền vững:

- Schema database.
- Dữ liệu mô tả công việc.
- Vector embedding.
- Danh sách ứng viên.

#### `huggingface_cache`

Nếu được cấu hình, volume này lưu model Qwen đã tải để tránh tải lại khi tạo container mới.

### Lưu ý

Không dùng lệnh sau nếu muốn giữ dữ liệu:

```bat
docker compose down -v
```

---

## 4.2. `Dockerfile`

### Chức năng

Tạo Docker image chạy FastAPI và model embedding.

### Các bước chính

1. Chọn base image Python 3.12 slim.
2. Đặt thư mục làm việc, ví dụ `/code`.
3. Cài các package hệ thống cần thiết.
4. Cài PyTorch CPU.
5. Cài dependency từ `requirements.txt`.
6. Copy mã nguồn `app/` và `scripts/`.
7. Mở cổng `8000`.
8. Chạy Uvicorn:

```text
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Lưu ý

Nếu sửa `Dockerfile` hoặc `requirements.txt`, cần build lại:

```bat
docker compose up --build -d
```

---

## 4.3. `requirements.txt`

### Chức năng

Khai báo dependency Python của project.

### Các thư viện chính

- `fastapi`: xây dựng API.
- `uvicorn`: chạy FastAPI.
- `sqlalchemy`: ORM và truy vấn database.
- `psycopg2-binary`: PostgreSQL driver.
- `pgvector`: kiểu dữ liệu vector và phép đo khoảng cách.
- `sentence-transformers`: load model và tạo embedding.
- `transformers`: hỗ trợ kiến trúc Qwen.
- `numpy`: xử lý vector.
- `requests`: gửi HTTP request từ script ingest.
- `pandas`: xử lý dữ liệu bảng nếu cần.
- `openpyxl`: đọc file Excel `.xlsx`.

### Lưu ý

Nếu PyTorch CPU đã được cài riêng trong Dockerfile thì không nên khai báo một cấu hình `torch` khác trong `requirements.txt`.

---

## 4.4. `README.md`

### Chức năng

Tài liệu dành cho người cài đặt và vận hành project.

Nội dung thường gồm:

- Giới thiệu project.
- Workflow tổng quát.
- Cách cài đặt.
- Cách generate dữ liệu.
- Cách ingest dữ liệu.
- Cách chạy Docker.
- Cách test API.
- Cách chạy lại project.
- Các lỗi thường gặp.

---

## 4.5. `document-tech.md`

### Chức năng

Tài liệu kỹ thuật chi tiết của project.

Nội dung gồm:

- Kiến trúc hệ thống.
- Chức năng tất cả file.
- Mô tả biến, class và hàm.
- Cấu trúc bảng và schema.
- Luồng endpoint.
- Cách tính similarity và group score.
- Quan hệ giữa các module.

---

# 5. Các file trong thư mục `app`

## 5.1. `app/__init__.py`

### Chức năng

Đánh dấu thư mục `app` là Python package.

Cho phép import theo dạng:

```python
from app.database import get_db
from app.embeddings import get_embedding
from app.models import Candidate
```

File có thể để trống.

---

## 5.2. `app/config.py`

### Chức năng

Quản lý các biến cấu hình dùng chung.

### Các biến chính

#### `DATABASE_URL`

Chuỗi kết nối PostgreSQL.

Ví dụ trong Docker:

```text
postgresql://avora:avora@db:5432/avora_matching
```

#### `EMBEDDING_MODEL_NAME`

Tên model embedding:

```text
Qwen/Qwen3-Embedding-0.6B
```

#### `VECTOR_DIM`

Số chiều vector:

```text
1024
```

#### Các biến tùy chọn

Có thể khai báo thêm:

```text
TOP_K_DESCRIPTIONS
TOP_K_CANDIDATES
MIN_GROUP_SCORE
GROUP_SCORE_MARGIN
```

### Quan hệ với module khác

```text
config.py
├── database.py dùng DATABASE_URL
├── embeddings.py dùng EMBEDDING_MODEL_NAME và VECTOR_DIM
├── models.py dùng VECTOR_DIM
└── main.py có thể dùng ngưỡng và giá trị Top-K
```

---

## 5.3. `app/database.py`

### Chức năng

Tạo kết nối SQLAlchemy, quản lý session và khởi tạo extension pgvector.

### Thành phần `engine`

Tạo SQLAlchemy Engine từ `DATABASE_URL`.

Có thể cấu hình:

```python
pool_pre_ping=True
```

để kiểm tra connection trước khi giao cho request.

### Thành phần `SessionLocal`

Factory tạo session database.

Mỗi request FastAPI sử dụng một session riêng.

### Class `Base`

Lớp cơ sở của các model SQLAlchemy:

```python
class Base(DeclarativeBase):
    pass
```

### Hàm `get_db()`

#### Mục đích

Cấp database session cho endpoint thông qua `Depends(get_db)`.

#### Workflow

```text
Tạo SessionLocal
→ yield session cho endpoint
→ endpoint hoàn thành hoặc phát sinh lỗi
→ đóng session trong finally
```

### Hàm `init_db()`

#### Mục đích

Khởi tạo extension và các bảng.

#### Workflow

```text
Kết nối PostgreSQL
→ CREATE EXTENSION IF NOT EXISTS vector
→ Base.metadata.create_all(bind=engine)
```

### Hàm `wait_for_database()` nếu có

#### Mục đích

Chờ PostgreSQL sẵn sàng khi API và database khởi động gần như đồng thời.

#### Workflow

```text
SELECT 1
→ nếu lỗi thì chờ
→ thử lại đến giới hạn
→ thành công hoặc báo RuntimeError
```

### Lưu ý

`Base.metadata.create_all()` không tự sửa cấu trúc bảng đã tồn tại. Khi thay schema trong môi trường test, có thể cần xóa volume database và tạo lại.

---

## 5.4. `app/embeddings.py`

### Chức năng

Chuyển văn bản thành vector embedding bằng Qwen3 Embedding.

### Biến cache model

Model được cache để không load lại trong mỗi request.

### Hàm `get_embedding_model()`

#### Mục đích

Load và trả về đối tượng `SentenceTransformer`.

#### Workflow

```text
Kiểm tra cache
→ nếu chưa có model thì load model
→ đặt device="cpu" nếu chạy CPU
→ lưu model trong cache
→ trả model
```

#### Lợi ích

- Giảm thời gian xử lý các request tiếp theo.
- Tránh tải model nhiều lần.
- Giảm sử dụng RAM.

### Hàm `normalize_text(text)`

#### Mục đích

Chuẩn hóa văn bản trước khi embedding.

#### Xử lý

- Chuyển đầu vào thành chuỗi.
- Xóa khoảng trắng thừa.
- Từ chối nội dung rỗng.

### Hàm `get_embedding(text)`

#### Mục đích

Tạo vector cho mô tả công việc hoặc query.

#### Workflow

```text
normalize_text(text)
→ get_embedding_model()
→ model.encode()
→ normalize_embeddings=True
→ convert_to_numpy=True
→ ép float32
→ reshape thành vector một chiều
→ kiểm tra VECTOR_DIM
→ trả list[float]
```

#### Quy tắc sử dụng

```python
get_embedding(payload.mo_ta_cong_viec)
get_embedding(payload.query)
```

Hai phía phải dùng cùng hàm và cùng tiền xử lý.

---

## 5.5. `app/models.py`

### Chức năng

Định nghĩa cấu trúc bảng PostgreSQL bằng SQLAlchemy ORM.

### Class `JobDescription`

Đại diện cho bảng:

```text
job_descriptions
```

#### Trường `id`

Khóa chính tự tăng của bản ghi mô tả.

#### Trường `nhom_khuyet_tat`

Nhãn nhóm khuyết tật gắn với mô tả.

#### Trường `mo_ta_cong_viec`

Văn bản mô tả công việc được dùng để tạo embedding.

#### Trường `embedding`

Vector có số chiều bằng `VECTOR_DIM`.

#### Trường `created_at`

Thời điểm tạo bản ghi.

#### Index nhóm

Index trên `nhom_khuyet_tat` giúp truy vấn và lọc theo nhóm nhanh hơn.

### Class `Candidate`

Đại diện cho bảng:

```text
candidates
```

#### Trường `id`

Khóa chính tự tăng của ứng viên.

#### Trường `ho_ten`

Họ và tên ứng viên.

#### Trường `nhom_khuyet_tat`

Nhóm khuyết tật của ứng viên.

#### Trường `created_at`

Thời điểm tạo bản ghi.

### Quy tắc

Candidate không có embedding trong workflow hiện tại. Candidate chỉ được lọc sau khi hệ thống xác định nhóm phù hợp.

---

## 5.6. `app/schemas.py`

### Chức năng

Định nghĩa cấu trúc dữ liệu đầu vào và đầu ra bằng Pydantic.

### Hằng `ALLOWED_DISABILITY_GROUPS`

Danh mục nhóm khuyết tật được backend chấp nhận.

### Hàm `normalize_required_text(value)`

#### Mục đích

- Xóa khoảng trắng thừa.
- Kiểm tra dữ liệu không rỗng.
- Trả chuỗi đã chuẩn hóa.

### Hàm `validate_disability_group(value)`

#### Mục đích

- Chuẩn hóa tên nhóm.
- Kiểm tra nhóm có nằm trong danh mục cho phép.
- Phát sinh lỗi validation nếu không hợp lệ.

### Class `JobDescriptionCreate`

Payload cho:

```http
POST /job-descriptions
```

Các trường:

```text
nhom_khuyet_tat
mo_ta_cong_viec
```

Validator kiểm tra nhóm và độ dài mô tả.

### Class `JobDescriptionResponse`

Response của endpoint mô tả công việc.

Không trả vector embedding để tránh response quá lớn.

### Class `CandidateCreate`

Payload cho:

```http
POST /candidates
```

Các trường:

```text
ho_ten
nhom_khuyet_tat
```

### Class `CandidateResponse`

Thông tin ứng viên trả về:

```text
id hoặc candidate_id
ho_ten
nhom_khuyet_tat
```

Nên dùng `candidate_id` để tránh nhầm với ID mô tả.

### Class `MatchQuery`

Payload cho:

```http
POST /match
```

Các trường:

#### `query`

Mô tả công việc cần phân tích.

#### `top_k_descriptions`

Số mô tả gần query được lấy để tính điểm nhóm.

#### `top_k_candidates`

Số ứng viên tối đa được trả về.

### Class `MatchedGroup`

Thông tin nhóm được chọn:

```text
nhom_khuyet_tat
score
```

### Class `MatchedDescription`

Thông tin mô tả được tìm thấy:

```text
id hoặc job_description_id
nhom_khuyet_tat
mo_ta_cong_viec
similarity
```

Nên dùng `job_description_id` để phân biệt với candidate ID.

### Class `MatchResponse`

Response tổng của `/match`:

```text
matched_groups
matched_descriptions
candidates
```

---

## 5.7. `app/main.py`

### Chức năng

Khởi tạo FastAPI và định nghĩa toàn bộ endpoint.

### Đối tượng `app`

Khởi tạo ứng dụng FastAPI với title, version và description.

### Hằng `GROUP_SCORE_MARGIN`

Khoảng chênh tối đa so với nhóm tốt nhất để một nhóm khác cũng được chọn.

Ví dụ:

```text
Nhóm tốt nhất: 0.90
Margin: 0.03

Nhóm 0.88: được chọn
Nhóm 0.84: không được chọn
```

### Hằng `MIN_GROUP_SCORE`

Ngưỡng score tối thiểu của nhóm.

### Hàm `on_startup()`

#### Mục đích

Chạy `init_db()` khi FastAPI khởi động.

### Hàm `root()`

#### Endpoint

```http
GET /
```

#### Kết quả

Trả tên service, trạng thái và đường dẫn Swagger.

### Hàm `health(db)`

#### Endpoint

```http
GET /health
```

#### Workflow

```text
Đếm JobDescription
→ đếm Candidate
→ trả trạng thái và số lượng
```

### Hàm `create_job_description(payload, db)`

#### Endpoint

```http
POST /job-descriptions
```

#### Workflow

```text
Nhận JobDescriptionCreate
→ get_embedding(mo_ta_cong_viec)
→ tạo JobDescription
→ db.add()
→ db.commit()
→ db.refresh()
→ trả JobDescriptionResponse
```

Nhóm khuyết tật không được đưa vào embedding.

### Hàm `list_job_descriptions(skip, limit, nhom_khuyet_tat, db)`

#### Endpoint

```http
GET /job-descriptions
```

#### Chức năng

- Phân trang bằng `skip` và `limit`.
- Lọc theo nhóm nếu có.
- Sắp xếp theo ID.

### Hàm `create_candidate(payload, db)`

#### Endpoint

```http
POST /candidates
```

#### Workflow

```text
Nhận CandidateCreate
→ tạo Candidate
→ lưu database
→ trả CandidateResponse
```

Không tạo embedding.

### Hàm `list_candidates(skip, limit, nhom_khuyet_tat, db)`

#### Endpoint

```http
GET /candidates
```

#### Chức năng

- Phân trang.
- Lọc theo nhóm.
- Sắp xếp theo ID.

### Hàm `aggregate_group_scores(rows)`

#### Mục đích

Tổng hợp similarity của các mô tả theo nhóm khuyết tật.

#### Công thức

```text
group_score
= 0.6 × score cao nhất
+ 0.4 × trung bình tối đa 3 score cao nhất
```

#### Workflow

```text
Gom score theo nhóm
→ sắp xếp score từng nhóm giảm dần
→ lấy tối đa 3 score đầu
→ tính max và average
→ tính group_score
→ sắp xếp nhóm giảm dần
```

### Hàm `match_candidates(payload, db)`

#### Endpoint

```http
POST /match
```

#### Workflow chi tiết

```text
Kiểm tra bảng job_descriptions có dữ liệu
→ get_embedding(query)
→ tạo biểu thức cosine_distance
→ similarity = 1 - cosine_distance
→ lấy Top-K JobDescription
→ aggregate_group_scores()
→ lấy best_score
→ chọn nhóm đạt MIN_GROUP_SCORE và GROUP_SCORE_MARGIN
→ truy vấn Candidate theo selected_groups
→ trả MatchResponse
```

### Xử lý lỗi

- Database chưa có mô tả: HTTP 409.
- Không tạo được embedding: HTTP 500.
- Lỗi truy vấn vector: HTTP 500.
- Lỗi insert database: rollback và trả HTTP phù hợp.

---

## 5.8. `app/__pycache__/`

### Chức năng

Chứa bytecode `.pyc` do Python tự tạo.

Không cần sửa và không nên commit lên Git.

Nên thêm vào `.gitignore`:

```gitignore
__pycache__/
*.pyc
```

---

# 6. Các file trong thư mục `scripts`

## 6.1. `scripts/generate_job_description_database.py`

### Chức năng

Tạo mock data gồm đúng hai cột:

```text
Nhóm khuyết tật | Mô tả công việc
```

File output:

```text
scripts/data/job_descriptions.csv
```

### Hằng và dữ liệu mẫu

#### `DEFAULT_COUNT`

Số mô tả duy nhất mặc định.

#### `DEFAULT_SEED`

Random seed giúp tái lập dữ liệu.

#### `DEFAULT_OUTPUT`

Đường dẫn output mặc định.

#### `JOB_TEMPLATES`

Danh sách template gồm:

- `tasks`: nhiệm vụ.
- `conditions`: điều kiện thực hiện.
- `requirements`: kỹ năng/yêu cầu.
- `groups`: các nhóm được gắn nhãn.

#### `INTRO_VARIANTS`

Các câu mở đầu khác nhau.

#### `CONDITION_VARIANTS`

Các biến thể mở đầu phần điều kiện.

#### `REQUIREMENT_VARIANTS`

Các biến thể mở đầu phần yêu cầu.

#### `ENDING_VARIANTS`

Các câu kết thúc mô tả.

### Hàm `normalize_text(text)`

Chuẩn hóa Unicode, HTML, khoảng trắng và dấu câu.

### Hàm `create_description(template, rng)`

Ghép template và biến thể câu thành một mô tả hoàn chỉnh.

### Hàm `generate_rows(unique_descriptions, rng)`

#### Workflow

```text
Chọn template ngẫu nhiên
→ tạo mô tả
→ kiểm tra mô tả chưa tồn tại
→ thêm vào tập seen
→ với mỗi group trong template, tạo một dòng
→ trộn danh sách dòng
```

Một mô tả có thể tạo nhiều dòng nếu có nhiều nhóm.

### Hàm `validate_rows(rows, expected_descriptions)`

Kiểm tra:

- Có dữ liệu.
- Đúng số mô tả duy nhất.
- Nhóm hợp lệ.
- Mô tả đủ độ dài.
- Không trùng cặp nhóm và mô tả.

### Hàm `write_csv(path, rows)`

Ghi CSV với header:

```text
Nhóm khuyết tật
Mô tả công việc
```

Sử dụng encoding `utf-8-sig`.

### Hàm `parse_args()`

Nhận tham số:

```text
--count
--seed
--output
```

### Hàm `main()`

Điều phối:

```text
Đọc arguments
→ kiểm tra count
→ tạo random generator
→ generate_rows()
→ validate_rows()
→ write_csv()
→ in thống kê
```

---

## 6.2. `scripts/generate_mock_candidates.py`

### Chức năng

Tạo mock data ứng viên gồm đúng hai cột:

```text
Họ và tên | Nhóm khuyết tật
```

File output:

```text
scripts/data/candidates.csv
```

### Hằng và dữ liệu mẫu

#### `DISABILITY_GROUPS`

Danh sách nhóm khuyết tật.

#### `FAMILY_NAMES`

Danh sách họ.

#### `MIDDLE_NAMES`

Danh sách tên đệm.

#### `GIVEN_NAMES`

Danh sách tên.

### Hàm `normalize_text(value)`

Chuẩn hóa Unicode và khoảng trắng.

### Hàm `build_name_pool()`

Tạo tập tên bằng tích Descartes:

```text
Họ × Tên đệm × Tên
```

Dùng `set` để loại tên trùng.

### Hàm `allocate_group_counts(total)`

Chia ứng viên gần đều cho tất cả nhóm.

Độ chênh giữa nhóm nhiều nhất và ít nhất tối đa là một.

### Hàm `generate_candidates(count, rng)`

#### Workflow

```text
Tạo name pool
→ chọn count tên không trùng
→ tính số lượng từng nhóm
→ tạo danh sách nhóm
→ shuffle nhóm
→ zip tên và nhóm
→ shuffle kết quả
```

### Hàm `validate_candidates(rows, expected_count)`

Kiểm tra:

- Đúng số dòng.
- Không thiếu họ tên.
- Không trùng tên.
- Nhóm hợp lệ.
- Các nhóm có dữ liệu.
- Phân bố gần cân bằng.

### Hàm `write_csv(path, rows)`

Ghi CSV UTF-8 BOM với hai cột.

### Hàm `parse_args()`

Nhận:

```text
--count
--seed
--output
```

### Hàm `main()`

Sinh ứng viên, kiểm tra, ghi CSV và in thống kê theo nhóm.

---

## 6.3. `scripts/ingest_job_descriptions.py`

### Chức năng

Đọc CSV mô tả công việc và gửi từng dòng tới FastAPI.

### Input

```text
scripts/data/job_descriptions.csv
```

### Endpoint

```http
POST /job-descriptions
```

### Payload

```json
{
  "nhom_khuyet_tat": "Khiếm thính",
  "mo_ta_cong_viec": "Nhập và đối chiếu dữ liệu trên Excel..."
}
```

### Hằng chính

- `DEFAULT_CSV_PATH`: file CSV mặc định.
- `DEFAULT_API_URL`: endpoint mặc định.
- `DEFAULT_TIMEOUT`: timeout request.
- `CSV_GROUP_COLUMN`: tên cột nhóm.
- `CSV_DESCRIPTION_COLUMN`: tên cột mô tả.
- `ALLOWED_GROUPS`: danh mục nhóm hợp lệ.

### Hàm `normalize_text(value)`

Chuẩn hóa Unicode và khoảng trắng.

### Hàm `create_session()`

Tạo `requests.Session` và cấu hình HTTP retry.

Không nên retry HTTP 500 vì đây thường là lỗi code, model hoặc database.

### Hàm `read_job_descriptions(csv_path, min_description_length)`

#### Kiểm tra

- File tồn tại.
- Đường dẫn là file.
- Có đúng hai cột.
- Nhóm hợp lệ.
- Mô tả không rỗng.
- Mô tả đủ độ dài.
- Cặp nhóm và mô tả không trùng.

#### Output

Danh sách payload:

```python
{
    "nhom_khuyet_tat": group,
    "mo_ta_cong_viec": description,
}
```

và danh sách cảnh báo.

### Hàm `api_base_url(api_url)`

Loại phần `/job-descriptions` để lấy base URL.

### Hàm `check_api(session, api_url, timeout)`

Kiểm tra `/openapi.json` hoặc `/docs` trước khi ingest.

### Hàm `extract_error(response)`

Đọc lỗi từ response JSON; nếu không phải JSON thì dùng response text.

### Hàm `ingest(...)`

#### Workflow

```text
Nếu dry-run thì in payload mẫu
→ tạo session
→ check_api()
→ lặp từng payload
→ POST /job-descriptions
→ cập nhật thống kê
→ in tiến độ
→ dừng nếu đạt max_errors
→ đóng session
```

### Hàm `print_summary(...)`

In số cặp hợp lệ, số mô tả duy nhất, số cảnh báo, thành công và thất bại.

### Hàm `parse_args()`

Nhận:

```text
--csv
--api-url
--timeout
--progress-every
--delay
--min-description-length
--max-errors
--dry-run
```

### Hàm `main()`

Điều phối toàn bộ tiến trình ingest.

---

## 6.4. `scripts/ingest_candidates.py`

### Chức năng

Đọc CSV ứng viên và gửi từng ứng viên tới FastAPI.

### Input

```text
scripts/data/candidates.csv
```

### Endpoint

```http
POST /candidates
```

### Payload

```json
{
  "ho_ten": "Nguyễn Minh An",
  "nhom_khuyet_tat": "Khiếm thính"
}
```

### Hằng chính

- `DEFAULT_API_URL`: endpoint mặc định.
- `DEFAULT_CSV_PATH`: file CSV mặc định.
- `CSV_NAME_COLUMN`: tên cột họ tên.
- `CSV_GROUP_COLUMN`: tên cột nhóm.
- `ALLOWED_GROUPS`: danh mục nhóm hợp lệ.

### Hàm `normalize_text(value)`

Chuẩn hóa Unicode và khoảng trắng.

### Hàm `create_http_session()`

Tạo HTTP session có retry cho lỗi tạm thời.

### Hàm `read_candidates(csv_path)`

Kiểm tra:

- File tồn tại.
- Đúng hai cột.
- Họ tên không rỗng.
- Nhóm không rỗng.
- Nhóm hợp lệ.
- Không trùng cặp tên và nhóm.

Trả danh sách payload và cảnh báo.

### Hàm `check_api(session, api_url, timeout)`

Kiểm tra FastAPI trước khi gửi dữ liệu.

### Hàm `extract_error(response)`

Đọc thông báo lỗi từ response.

### Hàm `ingest_candidates(...)`

#### Workflow

```text
Nếu dry-run thì in payload mẫu
→ tạo session
→ kiểm tra API
→ POST từng ứng viên
→ đếm thành công/thất bại
→ thống kê theo nhóm
→ in tiến độ
→ đóng session
```

### Hàm `print_summary(stats, warning_count, dry_run)`

In kết quả ingest.

### Hàm `parse_args()`

Nhận:

```text
--csv
--api-url
--timeout
--progress-every
--delay
--dry-run
```

### Hàm `main()`

Đọc CSV, kiểm tra, gọi ingest và trả exit code.

---

## 6.5. `scripts/convert_xlsx_to_csv.py`

### Chức năng

Chuyển file Excel `.xlsx` sang CSV UTF-8.

### Workflow

```text
Input XLSX
→ pandas.read_excel()
→ DataFrame
→ DataFrame.to_csv(encoding="utf-8-sig")
→ Output CSV
```

### Hàm chuyển đổi

Tùy phiên bản source, hàm chuyển đổi thường:

1. Nhận đường dẫn input và output.
2. Kiểm tra file Excel tồn tại.
3. Đọc Excel bằng pandas/openpyxl.
4. Ghi CSV bằng UTF-8 BOM.

### Hàm `main()`

Đọc tham số dòng lệnh và gọi hàm chuyển đổi.

### Lưu ý

Script chỉ chuyển định dạng. Script không:

- Tự gắn nhóm khuyết tật.
- Tự chuẩn hóa logic phù hợp.
- Tạo embedding.
- Gửi dữ liệu lên API.

---

# 7. Các file dữ liệu

## 7.1. `scripts/data/job_descriptions.csv`

### Chức năng

Nguồn dữ liệu mô tả công việc mẫu.

### Cột

```text
Nhóm khuyết tật
Mô tả công việc
```

### Quan hệ dữ liệu

Một mô tả có thể xuất hiện nhiều lần với các nhóm khác nhau.

### Luồng sử dụng

```text
job_descriptions.csv
→ ingest_job_descriptions.py
→ POST /job-descriptions
→ get_embedding(mo_ta_cong_viec)
→ job_descriptions table
```

---

## 7.2. `scripts/data/candidates.csv`

### Chức năng

Nguồn dữ liệu ứng viên.

### Cột

```text
Họ và tên
Nhóm khuyết tật
```

### Luồng sử dụng

```text
candidates.csv
→ ingest_candidates.py
→ POST /candidates
→ candidates table
```

---

# 8. Luồng dữ liệu giữa các file

```text
generate_job_description_database.py
        ↓ ghi
job_descriptions.csv
        ↓ đọc
ingest_job_descriptions.py
        ↓ HTTP POST
main.py:create_job_description()
        ↓ gọi
embeddings.py:get_embedding()
        ↓ tạo
Vector 1024 chiều
        ↓ ORM
models.py:JobDescription
        ↓ session
database.py
        ↓
PostgreSQL + pgvector
```

```text
generate_mock_candidates.py
        ↓ ghi
candidates.csv
        ↓ đọc
ingest_candidates.py
        ↓ HTTP POST
main.py:create_candidate()
        ↓ ORM
models.py:Candidate
        ↓ session
database.py
        ↓
PostgreSQL
```

```text
Frontend hoặc Swagger
        ↓ POST /match
main.py:match_candidates()
        ↓
embeddings.py:get_embedding(query)
        ↓
pgvector cosine distance
        ↓
main.py:aggregate_group_scores()
        ↓
Candidate filter
        ↓
schemas.py:MatchResponse
        ↓
JSON response
```

---

# 9. Cấu trúc database

## Bảng `job_descriptions`

```text
id                  INTEGER, PRIMARY KEY
nhom_khuyet_tat     TEXT, NOT NULL
mo_ta_cong_viec     TEXT, NOT NULL
embedding           VECTOR(1024), NOT NULL
created_at          TIMESTAMP WITH TIME ZONE
```

## Bảng `candidates`

```text
id                  INTEGER, PRIMARY KEY
ho_ten              TEXT, NOT NULL
nhom_khuyet_tat     TEXT, NOT NULL
created_at          TIMESTAMP WITH TIME ZONE
```

## Khuyến nghị chống trùng

Nên thêm unique constraint cho:

```text
job_descriptions:
(nhom_khuyet_tat, mo_ta_cong_viec)

candidates:
(ho_ten, nhom_khuyet_tat)
```

---

# 10. Endpoint API

## `GET /`

Trả thông tin cơ bản của service.

## `GET /health`

Trả trạng thái và số bản ghi.

## `POST /job-descriptions`

Nhận nhóm và mô tả, tạo embedding rồi lưu database.

## `GET /job-descriptions`

Liệt kê mô tả, hỗ trợ phân trang và lọc nhóm.

## `POST /candidates`

Nhận họ tên và nhóm rồi lưu ứng viên.

## `GET /candidates`

Liệt kê ứng viên, hỗ trợ phân trang và lọc nhóm.

## `POST /match`

Nhận query, tìm mô tả gần nhất, tính nhóm và trả ứng viên.

---

# 11. Cách tính score

## 11.1. Cosine similarity

```text
cosine_similarity = 1 - cosine_distance
```

Miền toán học:

```text
[-1, 1]
```

Ý nghĩa:

- `1`: cùng hướng, rất giống.
- `0`: ít liên quan.
- `-1`: ngược hướng.

Score không phải xác suất.

## 11.2. Group score

```text
group_score
= 0.6 × max_similarity
+ 0.4 × average(top_3_similarities)
```

Một nhóm được chọn khi:

```text
group_score >= MIN_GROUP_SCORE
```

và:

```text
best_score - group_score <= GROUP_SCORE_MARGIN
```

---

# 12. Ý nghĩa ID trong response

## ID trong `matched_descriptions`

Là ID của bản ghi mô tả công việc.

Nên đặt tên:

```text
job_description_id
```

## ID trong `candidates`

Là ID của ứng viên.

Nên đặt tên:

```text
candidate_id
```

Ứng viên nên được hiển thị với:

```text
candidate_id
ho_ten
nhom_khuyet_tat
```

---

# 13. Hướng dẫn chạy lần đầu

```bat
cd C:\Users\Bi\Downloads\AVORA-MATCHING-main

python scripts\generate_job_description_database.py --count 100
python scripts\generate_mock_candidates.py --count 3000

python scripts\ingest_job_descriptions.py --dry-run
python scripts\ingest_candidates.py --dry-run

docker compose down -v
docker compose up --build -d

python scripts\ingest_job_descriptions.py --timeout 300
python scripts\ingest_candidates.py
```

Kiểm tra:

```text
http://localhost:8000/health
http://localhost:8000/docs
```

---

# 14. Chạy lại project trong các lần sau

## Khởi động

```bat
cd C:\Users\Bi\Downloads\AVORA-MATCHING-main
docker compose up -d
docker compose ps
```

## Tắt nhưng giữ dữ liệu

```bat
docker compose down
```

## Sửa code trong `app/`

```bat
docker compose restart web
```

## Sửa Dockerfile hoặc dependency

```bat
docker compose up --build -d
```

## Đổi model hoặc pipeline embedding

```bat
docker compose down -v
docker compose up --build -d
python scripts\ingest_job_descriptions.py --timeout 300
python scripts\ingest_candidates.py
```

---

# 15. Lỗi thường gặp

## Model load lại liên tục

Nguyên nhân thường do `get_embedding_model()` phát sinh lỗi trước khi được cache.

Test:

```bat
docker compose exec web python -c "from app.embeddings import get_embedding; v=get_embedding('Nhập dữ liệu trên Excel'); print(len(v))"
```

Kết quả đúng:

```text
1024
```

## HTTP 422

Schema không khớp payload.

## HTTP 500

Lỗi embedding hoặc database. Xem log:

```bat
docker compose logs web --tail 200
```

## Database giữ schema cũ

```bat
docker compose down -v
docker compose up --build -d
```

## Dữ liệu bị ingest trùng

Không chạy lại toàn bộ ingest trên database đã có dữ liệu nếu chưa có unique constraint hoặc kiểm tra trùng.

## Cảnh báo Compose version obsolete

Xóa dòng:

```yaml
version: "3.9"
```

---

# 16. Giới hạn và hướng cải tiến

1. Dữ liệu generate hiện là synthetic mock data.
2. Một mô tả thuộc nhiều nhóm tạo nhiều bản ghi có vector giống nhau.
3. Cần chống ingest trùng.
4. Cần bộ query chuẩn để hiệu chỉnh ngưỡng score.
5. Matching hiện chỉ lọc ứng viên theo nhóm, chưa theo kỹ năng và kinh nghiệm.
6. Nên tách bảng mô tả và bảng quan hệ nhóm để không lưu lặp vector.
7. Nên đổi response ID thành `job_description_id` và `candidate_id`.
8. Khi dữ liệu lớn, cân nhắc HNSW index cho cột vector.
9. Nên dùng migration tool như Alembic khi chuyển sang môi trường production.
10. Không nên sử dụng kết quả như quyết định tự động cuối cùng đối với ứng viên.
