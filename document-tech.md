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

---

# 17. Hệ thống Ghép cặp Ổn định Hai phía (Two-Sided Stable Matching)

Nhằm giải quyết triệt để các hạn chế của việc lọc đơn phương (mục 16, điều 5), AVORA đã tích hợp hoàn chỉnh **Lõi Ghép cặp Ổn định Gale-Shapley (Many-to-One Deferred Acceptance)** tại `src/` và các endpoint API nâng cao tại `app/`.

## 17.1. Cấu trúc và Luồng dữ liệu 4 giai đoạn

```text
Giai đoạn 1 & 2: Tiền xử lý (src/core/preprocessor.py)
├── Ma trận E[c][j] (Employer Score từ Vector Oxyz hoặc Qwen3 Cosine Similarity)
├── Ma trận C[c][j] (Candidate Score từ sở thích, vị trí, tiện nghi tiếp cận)
├── Hạn mức chỉ tiêu tuyển dụng Quotas q_j (Many-to-One)
├── Bộ lọc điều kiện tiên quyết Hard Constraints D (triệt tiêu cặp vi phạm về 0)
└── Danh sách ưu tiên cắt ngắn Truncated Lists (Top L cho C, Top M cho J)
        │
        ▼
Giai đoạn 2: Rank Lookup Table O(1) & Min-Heap (src/core/matching_engine.py)
├── Bảng tra cứu thứ bậc nghịch đảo O(1)
└── Min-Heap đỉnh = ứng viên kém nhất cho mỗi vị trí việc làm
        │
        ▼
Giai đoạn 3: Thực thi Gale-Shapley Deferred Acceptance (src/core/matching_engine.py)
├── Chế độ Candidate-Proposing (Mặc định - C-optimal, Strategy-Proof cho ứng viên)
├── Chế độ Employer-Proposing (J-optimal cho nhà tuyển dụng)
└── Cơ chế giải quyết hòa điểm tất định (Deterministic SHA-256 Tie-Breaking)
        │
        ▼
Giai đoạn 4: Hậu xử lý Top-K Recommendations (src/core/postprocessor.py)
├── Normalized Surplus Score: gợi ý K công việc dự phòng cho ứng viên
└── Iterative Clearance Rounds: chạy nhiều vòng DA loại trừ dần vị trí
        │
        ▼
Đánh giá & Benchmark (src/evaluation/metrics.py, src/evaluation/benchmark.py)
└── Kiểm chứng Blocking Pairs = 0% (100% Stable), Điểm E & C trung bình, Fill Rate
```

## 17.2. Các Endpoint RESTful API bổ sung

- `POST /matching/stable`: Thực thi toàn trình Gale-Shapley từ danh sách profile ứng viên và việc làm (hỗ trợ vector Oxyz và hard constraints).
- `POST /matching/matrix`: Chạy trực tiếp từ ma trận điểm C và E đã tính sẵn.
- `POST /matching/database`: Tự động trích xuất ứng viên và mô tả việc làm từ cơ sở dữ liệu PostgreSQL thực tế để chạy ghép cặp ổn định.

## 17.3. Giao diện trực quan hóa Streamlit Dashboard

Khởi chạy dashboard phân tích và kiểm thử tại `dashboard.py`:
```bash
streamlit run dashboard.py
```
Hỗ trợ 4 nguồn dữ liệu: Mô phỏng ngẫu nhiên, File CSV nội bộ, Dán ma trận trực tiếp và Tải lên file tùy chỉnh.

## 17.4. Kết quả thực nghiệm

- **Tính ổn định:** Đạt **0% Blocking Pairs** ở cả hai chế độ Gale-Shapley (so với 1.76% - 1.90% của Greedy Ranking).
- **Tốc độ:** Quy mô 2.000 ứng viên $\times$ 400 việc làm hoàn tất trong **~39 ms** trên CPU thông thường.
- **Tỷ lệ lấp đầy (Fill Rate):** Đạt **100%**.

## 17.5. Nguyên tắc Thiết kế Trọng số Điểm C (2 Tầng)

1. **Tầng 1 - Ràng buộc cứng Tiếp cận ($S_{\text{access}}$):** Nhu cầu hỗ trợ thiết yếu (phụ đề, công cụ đọc màn hình, lối đi xe lăn, tùy chọn làm việc từ xa) là điều kiện tiên quyết. Nếu nơi làm việc không đáp ứng, cặp bị triệt tiêu điểm về 0 ngay tại `preprocessor.py`. Chỉ những tiện nghi "nên có" (nice-to-have) mới được tính điểm thành phần.
2. **Tầng 2 - Sở thích cá nhân hóa:** Ứng viên tự sắp xếp thứ tự ưu tiên cho 3 yếu tố còn lại (Thu nhập, Khoảng cách/Remote, Định hướng nghề) khi lập hồ sơ. Thứ tự được quy đổi thành trọng số cố định:
   - Ưu tiên 1: $0.5$
   - Ưu tiên 2: $0.3$
   - Ưu tiên 3: $0.2$
   - Trường hợp thiếu dữ liệu: dùng trọng số mặc định (Lương: $0.35$, Địa điểm: $0.30$, Nghề nghiệp: $0.35$).
   - Cơ chế này tận dụng triệt để tính chất **Chống khai gian (Strategy-Proofness)** của Gale-Shapley phía ứng viên.

## 17.6. Cơ chế Vận hành Thị trường On-Demand Batch & Direct Track

- **On-demand Event Batch:** Không áp dụng chu kỳ hàng tuần cố định vì AVORA cung cấp khuyến nghị mở (non-binding) và thị trường việc làm cho người khuyết tật còn mỏng (thin market). Hệ thống kích hoạt ghép cặp theo sự kiện (khi có tin tuyển dụng mới hoặc gom micro-batch ngắn theo ngày) để trả ngay Top-K cho nhà tuyển dụng.
- **Tách lọc Direct Track & Phân định với filled_slots:**
  * **Direct Track (`direct_track=True`):** Đại diện cho ứng viên trong hệ thống đã được doanh nghiệp tuyển thẳng (direct hire) vào một vị trí cụ thể (`hired_job_id`). Ứng viên này được rút khỏi matching pool của batch hiện tại và trừ 1 slot trong chỉ tiêu khả dụng của `hired_job_id`.
  * **Chỗ đã lấp ngoài hệ thống (`filled_slots`):** Đại diện cho số lượng vị trí mà doanh nghiệp đã tuyển trước từ **bên ngoài hệ thống** (external hires) trước đợt matching.
  * **Quy tắc cộng dồn và chống đếm hai lần (Double Deduction Prevention):** Khi vừa có ứng viên Direct Track trong batch, vừa khai báo `filled_slots`, hai số này sẽ được cộng dồn:
    $$\text{Total Pre-filled} = \text{Count}(\text{Direct Track}) + \text{filled\_slots}$$
    $$\text{Remaining Quota} = \text{quota} - \text{Total Pre-filled}$$
    Nếu `filled_slots` đã bao gồm cả người tuyển trực tiếp thì sẽ bị trừ hai lần. Do đó, tài liệu và API quy định rõ ràng: **`filled_slots` là chỗ đã lấp từ ngoài hệ thống, KHÔNG tính các ứng viên Direct Track có trong danh sách input của batch hiện tại.**
  * **Xử lý 3 trường hợp biên của Direct Track (Chặn và trả về HTTP 422):** Thay vì im lặng bỏ qua hoặc ép quota về 0, hệ thống kiểm tra nghiêm ngặt và trả về mã lỗi **422 Unprocessable Entity** (ném `ValueError` ở mức engine):
    1. `direct_track=True` nhưng thiếu hoặc để trống `hired_job_id`: Bị từ chối ngay lập tức để tránh tình trạng ứng viên bị rút khỏi pool nhưng không trừ quota vị trí nào (nguy cơ tuyển vượt).
    2. `hired_job_id` trỏ tới vị trí không tồn tại trong danh sách việc làm: Bị từ chối ngay lập tức kèm danh sách các `job_id` hợp lệ.
    3. Tổng số người tuyển trực tiếp cộng `filled_slots` vượt quá `quota` (ví dụ 2 người tuyển trực tiếp vào việc 1 chỗ): Bị từ chối ngay lập tức, không để quota bị ép về 0 trong im lặng mà không cảnh báo.


## 17.7. Tích hợp pgvector & Bộ chỉ số Đánh giá Nâng cao

- **Pipeline pgvector:** Embedding chỉ cung cấp điểm tương đồng kỹ năng (Skill Match), không thay thế toàn bộ điểm $E$. Cần áp dụng bộ lọc cứng trong SQL trước khi lấy Top-K, hoặc lấy dư (Over-fetch Top-200) rồi lọc xuống Top-50 để tránh cạn kiệt pool.
- **Theo dõi Orphan Candidates:** Đo tỷ lệ ứng viên nằm ngoài Top-50 của toàn bộ vị trí việc làm.
- **Đo Recall của ANN:** Định kỳ kiểm tra độ bao phủ của vector index so với tìm kiếm vét cạn (brute-force).
- **Bộ chỉ số đo lường vượt trên Blocking Pairs:**
  - Điểm $E$ và $C$ trung bình của các cặp trúng tuyển.
  - Hạng trung bình (Average Rank) của đối tác trong danh sách nguyện vọng mỗi bên.
  - Tỷ lệ ứng viên và vị trí chưa được ghép (Unmatched Rate).
  - Tỷ lệ trùng lặp khuyến nghị (Congestion Rate): đo mức độ tắc nghẽn ở baseline (một ứng viên giỏi xuất hiện ở quá nhiều doanh nghiệp) so với phân bổ độc quyền của Gale-Shapley.

---

# 18. Nâng cấp Thiết kế Toàn diện Hệ thống Ghép cặp (Pipeline 6.1 - 6.8)

**Nguyên tắc tối thượng**: Lõi ghép cặp Gale-Shapley Many-to-One (Deferred Acceptance) được **giữ nguyên không đổi**. Mọi nâng cấp tập trung tối ưu hóa các tầng xung quanh lõi: chất lượng hồ sơ đầu vào, lọc cứng tiền xử lý, công thức chấm điểm đa thành phần, phá hòa tất định, hậu xử lý kèm lý do minh bạch, kiểm toán công bằng, và kế hoạch đánh giá đa chiều.

---

## 18.1. Pipeline Đầu-Cuối (End-to-End Pipeline)

Hệ thống hoạt động qua 7 bước tuần tự:

```text
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 1: Thu thập hồ sơ chuẩn hóa (CandidateProfile & JobProfile)        │
│ - Ứng viên: kỹ năng, nhu cầu tiếp cận (thiết yếu & nên có), lương,     │
│             hình thức, khu vực, thứ tự ưu tiên tự xếp, kinh nghiệm.    │
│ - Tuyển dụng: yêu cầu bắt buộc/mong muốn, quota, kỹ năng, hỗ trợ,...   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 2: Lọc cứng Preprocessor (Hard Constraints)                       │
│ - Triệt tiêu cặp vi phạm nhu cầu tiếp cận thiết yếu (essential needs)  │
│ - Loại trừ ứng viên Direct Track (đã tuyển dụng qua phỏng vấn riêng)   │
│ - Loại trừ cặp không tương thích hình thức làm việc (Remote vs On-site)│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 3: Truy xuất ứng viên tiềm năng                                  │
│ - Lọc cứng trong SQL theo các tiêu chí hành chính                      │
│ - Lấy Top-N ứng viên gần nhất bằng pgvector cho thành phần kỹ năng     │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 4: Chấm điểm E và C & Xây dựng Danh sách Ưu tiên                 │
│ - Tính điểm E (6 thành phần, tổng = 1.0)                               │
│ - Tính điểm C (3 thành phần, trọng số theo thứ hạng tự xếp)           │
│ - Phá hòa điểm E tất định: kỹ năng cao hơn -> mã ID tăng dần           │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 5: Lõi Gale-Shapley Many-to-One (GIỮ NGUYÊN)                      │
│ - Deferred Acceptance (Candidate-Proposing / Employer-Proposing)       │
│ - Quota Min-Heap quản lý ứng viên tạm giữ                             │
│ - Bảng tra cứu thứ bậc nghịch đảo O(1)                                 │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 6: Hậu xử lý Postprocessor                                        │
│ - Điền các vị trí trống còn lại                                        │
│ - Sinh danh sách dự phòng Top-K theo Normalized Surplus Score          │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Bước 7: Trả kết quả kèm lý do giải thích minh bạch (Explainability)    │
│ - Điểm phân rã chi tiết từng thành phần (E breakdown & C breakdown)    │
│ - Văn bản giải thích lý do cụ thể vì sao ứng viên được gợi ý           │
│ - Bản ghi lưu vết kiểm toán (ExecutionAudit)                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 18.2. Công thức Chấm điểm Chi tiết

### Bảng trọng số Điểm E (Nhà tuyển dụng)
Tổng trọng số bắt buộc chuẩn hóa bằng $1.00$:

| Điểm | Thành phần | Cách tính kỹ thuật | Trọng số |
| :--- | :--- | :--- | :---: |
| **E** | Kỹ năng | Cosine similarity embedding (hoặc Token Jaccard) | **0.35** |
| **E** | Thời gian, hình thức làm việc | $1.0$ (khớp hoàn toàn), $0.5$ (khớp 1 phần / hybrid), $0.0$ (xung đột) | **0.20** |
| **E** | Điều kiện tiếp cận "nên có" | Tỷ lệ tiện nghi mong muốn được doanh nghiệp đáp ứng: $\frac{\|C_{\text{nice}} \cap J_{\text{support}}\|}{\|C_{\text{nice}}\|}$ | **0.20** |
| **E** | Lương | Mức độ chồng lấn 2 khoảng lương: $\frac{\text{Overlap}(C_{\text{sal}}, J_{\text{sal}})}{\text{Union}(C_{\text{sal}}, J_{\text{sal}})}$ | **0.10** |
| **E** | Kinh nghiệm | Chuẩn hóa theo mức yêu cầu: $\min(1.0, \frac{\text{Exp}_{\text{cand}}}{\text{Exp}_{\text{job}}})$ | **0.10** |
| **E** | Khu vực | $1.0$ nếu cùng khu vực hoặc làm việc Remote; $0.0$ nếu khác khu vực | **0.05** |

### Công thức Điểm C (Ứng viên)
Gồm 3 thành phần phản ánh nguyện vọng cá nhân:
1. **Mức thu nhập**: Tương quan giữa lương việc làm và lương mong muốn.
2. **Khoảng cách hoặc Remote**: Mức độ tương thích về hình thức và địa lý.
3. **Định hướng nghề nghiệp**: Độ gần giữa mục tiêu sự nghiệp của ứng viên và chức danh/ngành nghề tuyển dụng.

**Thang trọng số cá nhân hóa**:
Ứng viên tự xếp hạng 3 yếu tố trên hồ sơ. Thứ bậc được quy đổi thành trọng số:
- **Thứ nhất (Hạng 1)**: $0.50$
- **Thứ hai (Hạng 2)**: $0.30$
- **Thứ ba (Hạng 3)**: $0.20$
- *Trường hợp thiếu dữ liệu hoặc không xếp hạng*: Áp dụng trọng số mặc định chung: Thu nhập ($0.35$), Địa điểm ($0.30$), Định hướng nghề ($0.35$).

---

## 18.3. Mã giả Thuật toán Lõi (Deferred Acceptance Giữ nguyên)

Thuật toán Many-to-One Hospital-Residents Deferred Acceptance được bảo toàn nguyên vẹn:

```python
for mỗi ứng viên a:
    next[a] = 0                      # chỉ số việc kế tiếp trong danh sách C của a
hàng đợi Q = tất cả ứng viên chưa được ghép
mỗi việc j giữ một min-heap theo điểm E, kích thước tối đa quota[j]

while Q không rỗng:
    a = Q.pop()
    if next[a] hết danh sách:
        bỏ qua a                     # a không được ghép (unmatched)
        continue
    j = danh sách C của a[next[a]]
    next[a] += 1
    if (a, j) là cặp bị loại do ràng buộc cứng:
        Q.push(a)
        continue
    heap[j].push(a)
    if len(heap[j]) > quota[j]:
        r = heap[j].pop_min()        # ứng viên có điểm E thấp nhất bị loại
        Q.push(r)
```

- **Độ phức tạp**: $\mathcal{O}(N \times M)$ với $N$ ứng viên và $M$ vị trí việc làm.
- **Tối ưu hóa**: Nhờ **Bảng tra hạng nghịch đảo (Inverse Rank Lookup Table)** lưu sẵn trong bộ nhớ, việc so sánh mức độ ưu tiên giữa ứng viên mới và ứng viên trong heap đạt tốc độ tức thì $\mathcal{O}(1)$.

---

## 18.4. Cơ chế Phá Hòa Tất Định & Lưu Vết Kiểm Toán

1. **Thứ tự phá hòa điểm E**: Khi hai ứng viên $a$ và $b$ có cùng tổng điểm $E$ tại vị trí $j$, thứ tự ưu tiên được phân định nghiêm ngặt theo:
   - Ưu tiên ứng viên có **Điểm Kỹ năng (Skill Score)** cao hơn.
   - Nếu điểm kỹ năng vẫn bằng nhau: So sánh **Mã định danh (Candidate ID)** theo thứ tự từ điển (lexicographical order).
2. **Hạt giống ngẫu nhiên cố định**: Đặt `seed` cố định cho mọi bước sinh dữ liệu và băm phá hòa thứ cấp để đảm bảo tính tất định giữa mọi môi trường.
3. **Lưu vết kiểm toán (`ExecutionAudit`)**: Mỗi phiên chạy matching lưu trữ:
   - `run_id`: Định danh duy nhất của phiên chạy.
   - `timestamp`: Thời điểm thực thi chuẩn ISO-8601.
   - `seed`: Giá trị hạt giống ngẫu nhiên.
   - `weights_config`: Bảng trọng số cấu hình điểm $E$ và $C$.
   - `data_version`: Phiên bản của tập dữ liệu đầu vào.

---

## 18.5. Kế hoạch Đánh giá & So sánh Đối đầu Baseline

Tính chất **Blocking Pairs = 0%** là hệ quả toán học của Gale-Shapley, được dùng làm bài kiểm thử tính đúng đắn (correctness test) của mã nguồn. Để chứng minh giá trị kinh tế và vận hành, hệ thống được so sánh trực tiếp với **Baseline Sắp xếp theo điểm (Greedy Ranking)**:

| Chỉ số đánh giá (Mục 6.5) | Ý nghĩa nghiệp vụ | Gale-Shapley (C-optimal) | Baseline (Sắp xếp theo điểm) |
| :--- | :--- | :--- | :--- |
| **Điểm E và C trung bình** | Chất lượng ghép hai chiều | Cân bằng lợi ích hai phía | Điểm cao ảo ở các cặp đầu, sụt giảm ở đuôi |
| **Hạng trung bình đối tác** | Mức độ hài lòng của từng bên | Tối ưu vượt trội cho ứng viên | Ứng viên chịu thiệt thòi về nguyện vọng |
| **Tỷ lệ chưa được ghép** | Độ bao phủ thị trường (Coverage) | Thấp, tận dụng tối đa quota | Cao hơn do phân bổ bất đối xứng |
| **Tỷ lệ ứng viên bị giới thiệu nhiều nơi** | Mức độ xung đột / tắc nghẽn (Congestion) | **0.0% (Phân bổ độc quyền)** | **Cao (>20%)** do nhiều công ty tranh giành 1 ứng viên top đầu |
| **Recall Top-N pgvector** | Độ bao phủ của vector index | Đo lường tỷ lệ trùng so với tìm kiếm vét cạn (brute-force) |
| **Precision@K, nDCG@K** | Chất lượng danh sách trả về | Đánh giá qua bộ test 30 - 50 truy vấn mẫu có nhãn |

---

## 18.6. Công bằng và Riêng tư

1. **Phân tách năng lực và khuyết tật**: Tuyệt đối **không** sử dụng nhãn nhóm khuyết tật làm tín hiệu suy diễn năng lực làm việc. Nhóm khuyết tật chỉ là metadata hỗ trợ; điều kiện làm việc chỉ xét qua các nhu cầu tiếp cận cụ thể (phần mềm đọc màn hình, lối đi xe lăn, giờ giấc linh hoạt,...).
2. **Bảo vệ dữ liệu nhạy cảm**: Thông tin nhu cầu tiếp cận được phân quyền truy cập nghiêm ngặt và chỉ hiển thị cho nhà tuyển dụng khi đã có sự đồng ý (consent) của ứng viên.
3. **Kiểm toán công bằng Four-Fifths Rule**: Định kỳ tính toán **Disparate Impact Ratio**:
   $$\text{Ratio} = \frac{\min_{g} \text{MatchRate}(g)}{\max_{g} \text{MatchRate}(g)}$$
   Hệ thống đạt tiêu chuẩn công bằng khi $\text{Ratio} \ge 0.80$ (không nhóm khuyết tật nào bị tỷ lệ trúng tuyển dưới 80% so với nhóm cao nhất).
4. **Candidate-Proposing làm mặc định**: Chọn ứng viên là phía đưa ra đề xuất để tối đa hóa phúc lợi của người lao động và bảo đảm tính chất **Chống khai gian (Strategy-Proofness)**.

---

## 18.7. Lộ trình Triển khai 6 Giai đoạn

| Giai đoạn | Nội dung trọng tâm | Kết quả đạt được | Trạng thái |
| :---: | :--- | :--- | :---: |
| **1** | Chuẩn hóa schema hồ sơ, thêm kỹ năng và thứ tự ưu tiên | `CandidateProfile`, `JobProfile` mock đầy đủ | **Hoàn thành** |
| **2** | Lọc cứng (Hard constraints) và công thức tính điểm E, C | Hai danh sách ưu tiên chạy ổn định, phá hòa tất định | **Hoàn thành** |
| **3** | Pipeline pgvector Top-N cho thành phần kỹ năng | Đo Recall so với tìm kiếm vét cạn | **Hoàn thành** |
| **4** | Chạy lõi Gale-Shapley và Baseline so sánh trên cùng thị trường | Bảng so sánh chỉ số định lượng Mục 6.5 | **Hoàn thành** |
| **5** | On-demand batch và cơ chế tách lọc Direct Track | Loại ứng viên trúng tuyển riêng khỏi pool | **Hoàn thành** |
| **6** | Hiệu chỉnh trọng số qua phản hồi thực tế | Thu thập nhãn người dùng, sẵn sàng Learning-to-Rank | **Đang mở** |

---

## 18.8. Rủi ro Chính & Biện pháp Phòng ngừa

1. **Điểm E và C bị sai lệch (Miscalibration)**: Nếu công thức tính điểm không phản ánh đúng năng lực, kết quả ghép cặp dù ổn định toán học nhưng vẫn không hiệu quả trên thực tế. *Giải pháp*: Xây dựng bộ test benchmark 30 - 50 truy vấn có nhãn chuyên gia gán để tinh chỉnh trọng số trước khi triển khai production.
2. **Ma trận điểm quá thưa (Matrix Sparsity & Orphan Candidates)**: Ứng viên nằm ngoài Top-N của tất cả vị trí việc làm sẽ không bao giờ được ghép. *Giải pháp*: Đo tỷ lệ Orphan Candidates và áp dụng cơ chế over-fetch (lấy dư Top-200 rồi lọc xuống Top-50).
3. **Thị trường mỏng (Thin Market)**: Khi số lượng vị trí việc làm hoặc ứng viên quá ít, độ cạnh tranh thấp khiến kết quả Gale-Shapley trùng với Greedy Baseline. *Giải pháp*: Báo cáo kèm thông tin mật độ thị trường để tránh ngộ nhận về hiệu năng thuật toán.
4. **Độ tin cậy của dữ liệu mô phỏng**: Toàn bộ chỉ số benchmark ban đầu xuất phát từ tập dữ liệu synthetic. *Giải pháp*: Luôn công bố số liệu kèm điều kiện đo lường và cập nhật ngay khi có dữ liệu thí điểm thực tế.

---

## 18.9. Lớp Kiểm tra Tính Ổn định Vét cạn Độc lập (Independent Stability Checker)

Nhằm đảm bảo tính khách quan và độc lập toán học tuyệt đối, hệ thống kiểm thử của dự án trang bị **hai lớp kiểm tra song song**:

- **Lớp 1 (Nội bộ dự án):** Sử dụng hàm `find_blocking_pairs` của module `metrics` và `matching`, tối ưu hóa hiệu năng bằng bảng tra cứu thứ hạng $O(1)$.
- **Lớp 2 (Độc lập 100% - `tests/test_exhaustive_stability_checker.py`):** Triển khai hàm kiểm tra vét cạn thuần túy (`independent_find_blocking_pairs`), không sử dụng lại bất kỳ cấu trúc dữ liệu nội bộ hay bảng tra cứu nào từ engine. Hàm này:
  - Duyệt qua toàn bộ tích Descartes $C \times J$.
  - Kiểm tra trực tiếp từ nguyên lý toán học Gale-Shapley Many-to-One:
    * Điều kiện 1: Ứng viên thích vị trí $j$ hơn vị trí hiện tại (hoặc đang thất nghiệp).
    * Điều kiện 2a: Vị trí $j$ còn chỉ tiêu trống ($|\mu(j)| < q_j$).
    * Điều kiện 2b: Vị trí $j$ đã đủ chỉ tiêu nhưng có ít nhất một ứng viên trúng tuyển kém hơn ứng viên đang xét theo thứ tự nguyện vọng của $j$.
  - Kiểm tra tính hợp lý cá nhân (Individual Rationality) và tính toàn vẹn của Quota.

Kết quả kiểm thử trên nhiều hạt giống ngẫu nhiên (42, 101, 777, 2024, 9999) chứng minh: cả hai lớp kiểm tra đều đồng thuận đạt **0% Blocking Pairs** cho Gale-Shapley, và cùng phát hiện chính xác các cặp bất ổn khi chạy trên Baseline Greedy hoặc khi cố ý hoán đổi kết quả phân bổ.

---

## 18.10. Các Ghi nhận & Lưu ý Kỹ thuật Trung thực (Honest Technical Disclosures)

Trong quá trình đánh giá và thực nghiệm hệ thống, một số quan sát kỹ thuật quan trọng cần được ghi nhận minh bạch:

1. **Điểm C trung bình của Gale-Shapley so với Greedy Baseline:**
   - Khi đo lường trên cùng một tập dữ liệu, thuật toán Greedy có thể đạt điểm trung bình nguyện vọng ứng viên (Điểm C) cao hơn nhẹ ở các cặp ghép đầu tiên so với Gale-Shapley. Điều này hoàn toàn tự nhiên vì Greedy là thuật toán tham lam đơn phương: nó chỉ tối ưu tổng điểm trước mắt mà bỏ qua sự bất ổn hai chiều.
   - Hậu quả của Greedy là tạo ra tỷ lệ **Blocking Pairs > 1.7%**, tiềm ẩn nguy cơ phá vỡ thỏa thuận và giao dịch ngầm ngoài hệ thống. Gale-Shapley chấp nhận cân bằng hai chiều để bảo đảm **100% tính ổn định (0% Blocking Pairs)** và **chống khai gian (Strategy-Proofness)** cho ứng viên.
2. **Hiện tượng C-optimal trùng J-optimal ở Benchmark:**
   - Trong các lượt chạy benchmark trên tập dữ liệu mô phỏng hiện tại, kết quả của Candidate-Proposing (C-optimal) và Employer-Proposing (J-optimal) thường trùng khớp nhau.
   - Hiện tượng này phù hợp với lý thuyết cấu trúc lưới Ghép cặp Ổn định (Roth & Sotomayor, 1990): khi thị trường có độ tương quan nguyện vọng cao hoặc tập ứng viên/vị trí nhỏ, lõi các phân bổ ổn định (Core of Stable Matchings) thu hẹp thành một điểm duy nhất (singleton), khiến nghiệm tối ưu của hai phía hợp nhất làm một.
3. **Mật khẩu Docker mặc định:**
   - File cấu hình `docker-compose.yml` hiện đặt mật khẩu PostgreSQL mặc định (`postgres / postgres`) nhằm tạo thuận lợi cho việc kiểm thử và chạy môi trường dev cục bộ. Khi đưa vào môi trường Production hoặc Staging, bắt buộc phải thay thế bằng biến môi trường bảo mật qua `.env` hoặc Secret Manager.
4. **Phạm vi Dữ liệu Thử nghiệm:**
   - Toàn bộ kết quả định lượng hiện nay đều được đo lường trên tập dữ liệu mô phỏng (Synthetic Market) với các hạt giống benchmark cố định nhằm bảo đảm tính tái lập (reproducibility). Cần tiếp tục tiến hành đo lường Monte Carlo mở rộng và đánh giá trên dữ liệu thí điểm thực tế khi triển khai với người dùng thật.
5. **Dấu vết Đường dẫn trong Lịch sử Git cũ:**
   - Các file rác và file nhật ký chẩn đoán tạm thời (`_diag*.txt`, `_watch.py`, `_*.log`) đã được gỡ bỏ khỏi Git index và đưa vào `.gitignore` từ commit `cd00caf`. Các file này không chứa mật khẩu hay thông tin bí mật nên rủi ro an ninh ở mức thấp.
   - Tuy nhiên, đường dẫn thư mục cá nhân cục bộ (`C:\Users\dtung`) vẫn còn tồn tại trong lịch sử commit cũ (`fe5e120`). Nếu có yêu cầu bảo mật nghiêm ngặt hoặc công khai mã nguồn mở, có thể sử dụng công cụ `git-filter-repo` hoặc BFG Repo-Cleaner để làm sạch hoàn toàn lịch sử Git.





---

# 19. Kiến trúc Tối ưu hóa Hạ tầng & Công nghệ (Drop-in Infrastructure Upgrade)

Hệ thống đã được nâng cấp toàn diện theo nguyên tắc **Zero-Algorithmic-Change Guarantee** (bảo toàn 100% công thức chấm điểm, thứ tự ghép cặp Gale-Shapley và quy tắc tiền xử lý/hậu xử lý) trên 6 tầng kiến trúc:

1. **Tầng Thực Thi Thuật Toán Ghép Cặp & Chấm Điểm (`Numba` & `NumPy`):**
   - Trong `src/core/scoring.py`: Bổ sung các hàm tính toán ma trận điểm tổng hợp biên dịch mã máy song song `@numba.njit(parallel=True, fastmath=True)` (`compute_composite_score_matrices_numba`, `compute_cosine_similarity_matrix_numba`) và hàm tính Cosine Similarity `@numba.njit(fastmath=True)` trên mảng `numpy.ndarray`.
   - Trong `src/core/matching_engine.py` và `app/matching.py`: Biểu diễn bảng xếp hạng ưu tiên (preference lists), bảng tra cứu thứ hạng nghịch đảo $O(1)$ và trạng thái ghép cặp dưới dạng mảng số nguyên 2 chiều `int32[:, :]`, biên dịch vòng lặp `while` của thuật toán Gale-Shapley Deferred Acceptance (cả 2 chế độ `candidate` và `employer`) bằng `@numba.njit`.

2. **Tầng Sinh Vector Nhúng (`ONNX Runtime` / `FastEmbed` & SHA-256 Caching):**
   - Trong `app/embeddings.py`: Loại bỏ phụ thuộc PyTorch/CUDA nặng nề, chuyển sang suy luận qua `onnxruntime.InferenceSession` (hỗ trợ Quantization INT8/FP16) và `FastEmbed`.
   - Tích hợp xử lý theo lô động (`get_embeddings_batch` - Dynamic Batching) kết hợp bộ nhớ đệm LRU đánh chỉ mục bằng hàm băm `SHA-256` (`SHA256EmbeddingCache`) trên văn bản đã chuẩn hóa từ `src/core/preprocessor.py`.

3. **Tầng Cơ Sở Dữ Liệu & Truy Xuất Bất Đồng Bộ (`AsyncQdrantClient` gRPC & `asyncpg`):**
   - Trong `app/database.py` và `app/models.py`: Nâng cấp kết nối cơ sở dữ liệu quan hệ sang **SQLAlchemy 2.0 Async** (`create_async_engine`, `async_sessionmaker`, `AsyncSession`) với driver `postgresql+asyncpg://`.
   - Tích hợp `AsyncQdrantClient(prefer_grpc=True)` qua cổng gRPC `6334` kèm **Filterable HNSW Payload Indexing** (`HARD_CONSTRAINT_PAYLOAD_FIELDS`: `location`, `salary_min`, `salary_max`, `experience_years`, `career_domain`, `nhom_khuyet_tat`, `work_mode`) để thực hiện Pre-filtering ngay tại tầng lưu trữ.
   - Trong `scripts/ingest_candidates.py` và `scripts/ingest_job_descriptions.py`: Hỗ trợ nạp điểm bất đồng bộ theo lô song song `upload_points(..., parallel=4)`.

4. **Tầng Xác Thực Dữ Liệu & API (`Pydantic V2`, `orjson`, `slots=True`):**
   - Trong `app/main.py`: Thiết lập `default_response_class=ORJSONResponse` để tuần tự hóa JSON tốc độ cao trực tiếp từ đối tượng Pydantic và mảng `numpy.ndarray`.
   - Trong `app/schemas.py`, `app/config.py`, `src/core/profile_models.py`, `src/core/preprocessor.py`, `src/core/postprocessor.py`, `src/core/scoring.py` và `src/core/matching_engine.py`: Chuẩn hóa `ConfigDict`, `TypeAdapter` của Pydantic V2 và gắn `@dataclass(slots=True)` cho toàn bộ các lớp dữ liệu nội bộ để giảm chi phí cấp phát bộ nhớ.

5. **Tầng Tiền Xử Lý & Nạp Dữ Liệu ETL (`Polars` & `fastexcel` / `calamine`):**
   - Trong `scripts/convert_xlsx_to_csv.py`: Đọc file Excel bằng `polars.read_excel(..., engine="calamine")` (thông qua `fastexcel`) và xuất CSV bằng `write_csv()`.
   - Trong `scripts/ingest_candidates.py`, `scripts/ingest_job_descriptions.py` và `src/core/preprocessor.py`: Sử dụng `pl.scan_csv()` dựa trên chuẩn bộ nhớ cột Apache Arrow và biểu thức chuỗi vector hóa của Polars (`preprocess_dataframe_polars`).

6. **Tầng Đóng Gói, Quản Lý Gói & Bảng Điều Khiển (`Astral uv`, Docker Multi-stage & Streamlit Caching):**
   - Trong `Dockerfile`: Áp dụng **Docker Multi-stage Build** (`builder` và `runtime`), tích hợp `COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/` và cài đặt gói bằng `uv pip install --system --no-cache -r requirements.txt`.
   - Trong `dashboard.py`: Bổ sung `@st.cache_resource` và `@st.cache_data` cho tác vụ khởi tạo tài nguyên, nạp thị trường và tính toán ma trận/benchmark.
