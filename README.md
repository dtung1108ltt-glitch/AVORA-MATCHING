<<<<<<< HEAD
AVORA-MATCHING — High-Performance Two-Sided Stable Job-Candidate Matching EngineAVORA-MATCHING là hệ thống ghép cặp việc làm (Job Description) và ứng viên (Candidate) hai phía dựa trên lý thuyết trò chơi và tìm kiếm ngữ nghĩa vector (Semantic Vector Search)[cite: 1, 2]. Hệ thống kết hợp mô hình chấm điểm đa chiều có ràng buộc cứng (Hard Constraints) với thuật toán ghép cặp ổn định nhiều-một Gale-Shapley Many-to-One Deferred Acceptance, được tối ưu hóa toàn diện bằng ngăn xếp công nghệ hiệu năng cao dựa trên Rust và LLVM JIT.   1. Điểm Nhấn Kiến Trúc & Thuật Toán Cốt LõiHệ thống được thiết kế theo nguyên tắc Zero-Algorithmic-Change Guarantee (bảo toàn 100% độ chính xác toán học và tính ổn định của thuật toán trong khi tối đa hóa tốc độ thực thi phần cứng):   Mô hình Chấm điểm Tương hợp Hai phía (src/core/scoring.py): Tính toán đồng thời ma trận điểm từ góc nhìn Nhà tuyển dụng $\text{Score}_J(j, c)$ và góc nhìn Ứng viên $\text{Score}_C(c, j)$ dựa trên cấu hình trọng số EmployerWeightsConfig (@dataclass(slots=True)) cùng bộ lọc điều kiện tiên quyết (Hard Constraints)[cite: 1, 2]. Toàn bộ phép tính Cosine Similarity và tổng hợp điểm ma trận 4 tầng được biên dịch sang mã máy đa luồng bằng @numba.njit(parallel=True, fastmath=True) sử dụng numba.prange trên mảng numpy.ndarray 2 chiều.Bộ máy Ghép cặp Ổn định Gale-Shapley (src/core/matching_engine.py & app/matching.py): Giải bài toán ghép cặp ổn định nhiều-một (Many-to-One Deferred Acceptance) hỗ trợ cả hai chế độ Candidate-Proposing (C-Optimal) và Employer-Proposing (J-Optimal)[cite: 1, 2]. Bảng xếp hạng ưu tiên và bảng tra cứu hạng độ phức tạp $O(1)$ (_build_rank_tables) được biểu diễn dưới dạng mảng số nguyên 2 chiều int32[:, :] và biên dịch JIT qua @numba.njit, đảm bảo kết quả ghép cặp đạt trạng thái ổn định tuyệt đối (0 cặp chặn / blocking pairs) trong thời gian dưới 1 mili-giây.   Sinh Vector Nhúng Tiếng Việt & Bộ nhớ đệm Băm (app/embeddings.py): Sử dụng mô hình ngôn ngữ AITeamVN/Vietnamese_Embedding (1024 chiều) để biểu diễn ngữ nghĩa hồ sơ ứng viên và mô tả công việc[cite: 1, 2]. Tầng suy luận được tăng tốc bằng ONNX Runtime (onnxruntime.InferenceSession hỗ trợ lượng tử hóa INT8/FP16) và FastEmbed, kết hợp xử lý theo lô động (get_embeddings_batch) và bộ nhớ đệm băm SHA-256 (SHA256EmbeddingCache) trên văn bản đã chuẩn hóa.   Truy xuất Vector & Cơ sở dữ liệu Bất đồng bộ (app/database.py & app/models.py): Giao tiếp với cơ sở dữ liệu vector Qdrant qua giao thức nhị phân gRPC (cổng 6334) bằng AsyncQdrantClient(prefer_grpc=True)[cite: 1, 2]. Hệ thống thiết lập chỉ mục Filterable HNSW Payload Indexing cho các trường lọc cứng (industry, job_field, desired_locations, location, experience_years, min_experience_years, expected_salary, salary_max, education_level, min_education_level) để sơ tuyển ngay tại tầng lưu trữ, đồng thời kết nối cơ sở dữ liệu quan hệ qua SQLAlchemy 2.0 Async (postgresql+asyncpg://)[cite: 1, 2].Đường ống ETL Đa luồng (src/core/preprocessor.py & scripts/): Tiền xử lý văn bản, tách kỹ năng, chuẩn hóa từ đồng nghĩa và chuẩn hóa học vấn/kinh nghiệm bằng biểu thức vector hóa chuỗi của Polars (preprocess_dataframe_polars), đọc luồng dữ liệu CSV bằng Apache Arrow LazyFrame (pl.scan_csv) và chuyển đổi tệp Excel siêu tốc qua fastexcel (engine="calamine")[cite: 1, 2].2. Ngăn Xếp Công Nghệ 6 Tầng (Tech Stack)Phân hệ kiến trúcTệp mã nguồn chínhCông nghệ cốt lõiVai trò trong hệ thống1. Tầng Lõi Thuật Toán & Chấm Điểmsrc/core/matching_engine.py, src/core/scoring.py, app/matching.py[cite: 1, 2]Numba (@njit), NumPy (int32[:, :], ndarray)   Biên dịch JIT đa luồng cho ma trận điểm hai phía và vòng lặp Gale-Shapley   2. Tầng Sinh Vector Nhúngapp/embeddings.py[cite: 1, 2]ONNX Runtime, FastEmbed, SHA-256 Cache (AITeamVN/Vietnamese_Embedding)   Suy luận vector 1024 chiều theo lô động và lưu đệm băm chống tính toán lặp   3. Tầng Cơ Sở Dữ Liệu & I/Oapp/database.py, app/models.py[cite: 1, 2]Qdrant (gRPC + Filterable HNSW), SQLAlchemy 2.0 Async (asyncpg)   Lưu trữ vector, lọc cứng Pre-filtering trên HNSW và truy vấn quan hệ bất đồng bộ   4. Tầng Xác Thực & Dịch Vụ APIapp/main.py, app/schemas.py, app/config.py, src/core/profile_models.py, src/core/postprocessor.py[cite: 1, 2]FastAPI, Pydantic V2 (pydantic-core), orjson, @dataclass(slots=True)[cite: 1, 2]Xác thực dữ liệu bằng nhân Rust, tối ưu RAM cho đối tượng hồ sơ và tuần tự hóa JSON trực tiếp từ mảng NumPy   5. Tầng Tiền Xử Lý & ETL Pipelinesrc/core/preprocessor.py, scripts/convert_xlsx_to_csv.py, scripts/ingest_candidates.py, scripts/ingest_job_descriptions.py[cite: 1, 2]Polars (LazyFrame), Apache Arrow, fastexcel (calamine)   Chuyển đổi Excel sang CSV, đọc luồng dữ liệu lớn và chuẩn hóa văn bản/kỹ năng đa luồng   6. Tầng Đóng Gói & Giao DiệnDockerfile, docker-compose.yml, dashboard.py, requirements.txt, requirements-dashboard.txt[cite: 1, 2]Astral uv, Docker Multi-stage Build, Streamlit Caching   Cài đặt phụ thuộc siêu tốc, đóng gói container tối giản và trực quan hóa kết quả ghép cặp   3. Cấu Trúc Thư Mục Dự ÁnPlaintextAVORA-MATCHING-main/
├── app/                                      # Phân hệ ứng dụng, API & kết nối Cơ sở dữ liệu
│   ├── __init__.py
│   ├── config.py                             # Cấu hình hệ thống & biến môi trường (Pydantic V2)
│   ├── database.py                           # AsyncQdrantClient (gRPC 6334) & Filterable HNSW Indexing
│   ├── embeddings.py                         # ONNX Runtime / FastEmbed + SHA256EmbeddingCache (1024-d)
│   ├── main.py                               # Điểm vào FastAPI với ORJSONResponse
│   ├── matching.py                           # Dịch vụ ghép cặp tầng ứng dụng (Numba JIT Gale-Shapley)
│   ├── models.py                             # Mô hình ORM với SQLAlchemy 2.0 Async (asyncpg)
│   └── schemas.py                            # Lược đồ xác thực dữ liệu Pydantic V2 (ConfigDict, TypeAdapter)
├── src/                                      # Phân hệ lõi thuật toán & đánh giá thực nghiệm
│   ├── __init__.py
│   ├── core/                                 # Lõi thuật toán tiền xử lý, chấm điểm & ghép cặp ổn định
│   │   ├── __init__.py
│   │   ├── matching_engine.py                # Thuật toán Gale-Shapley Many-to-One (@numba.njit + Rank Tables)
│   │   ├── postprocessor.py                  # Hậu xử lý kết quả, trích xuất MatchPairDetail & MatchingReport
│   │   ├── preprocessor.py                   # Làm sạch dữ liệu & vector hóa chuỗi với Polars LazyFrame
│   │   ├── profile_models.py                 # Cấu trúc dữ liệu nội bộ (@dataclass(slots=True))
│   │   └── scoring.py                        # Chấm điểm hai phía Score_J, Score_C & lọc cứng (@numba.njit)
│   └── evaluation/                           # Đo lường hiệu năng & kiểm định chất lượng ghép cặp
│       ├── __init__.py
│       ├── benchmark.py                      # Kịch bản đo tốc độ thực thi & khả năng mở rộng
│       └── metrics.py                        # Các chỉ số đánh giá độ ổn định, mức độ hài lòng & Fill Rate
├── scripts/                                  # Kịch bản ETL, tạo dữ liệu giả lập & nạp dữ liệu hàng loạt
│   ├── data/
│   │   ├── candidates.csv                    # Tập dữ liệu hồ sơ ứng viên thực nghiệm (N = 300)
│   │   └── job_descriptions.csv              # Tập dữ liệu mô tả công việc thực nghiệm (M = 275)
│   ├── convert_xlsx_to_csv.py                # Chuyển đổi Excel sang CSV bằng Polars + fastexcel (calamine)
│   ├── generate_mock_candidates.py           # Sinh dữ liệu ứng viên giả lập phục vụ kiểm thử tải
│   ├── generate_mock_data.py                 # Sinh dữ liệu thị trường lao động tổng hợp
│   ├── ingest_candidates.py                  # Nạp bất đồng bộ hồ sơ ứng viên vào Qdrant (parallel=4)
│   └── ingest_job_descriptions.py            # Nạp bất đồng bộ JD vào Qdrant (parallel=4)
├── tests/                                    # Bộ kiểm thử đơn vị, hồi quy & kiểm chứng tính ổn định
│   ├── __init__.py
│   ├── test_app_stable_matching.py           # Kiểm thử ghép cặp ổn định ở tầng ứng dụng/API
│   ├── test_dashboard.py                     # Kiểm thử các luồng xử lý trên bảng điều khiển
│   ├── test_enhanced_matching.py             # Kiểm thử ghép cặp nâng cao & ràng buộc phức hợp
│   ├── test_exhaustive_stability_checker.py  # Kiểm tra vét cạn tính ổn định (đảm bảo 0 blocking pairs)
│   ├── test_matching.py                      # Kiểm thử lõi thuật toán Gale-Shapley C-Optimal & J-Optimal
│   └── test_regression_fixes.py              # Kiểm thử chống lỗi hồi quy toàn hệ thống
├── conftest.py                               # Cấu hình fixtures dùng chung cho Pytest
├── dashboard.py                              # Giao diện bảng điều khiển trực quan hóa (tích hợp Caching)
├── document-tech.md                          # Tài liệu đặc tả kỹ thuật & kiến trúc chi tiết
├── Dockerfile                                # Cấu hình Docker Multi-stage Build tích hợp Astral uv
├── docker-compose.yml                        # Khởi chạy toàn bộ dịch vụ & Qdrant (HTTP 6333 / gRPC 6334)
├── requirements.txt                          # Danh sách thư viện cốt lõi của hệ thống
├── requirements-dashboard.txt                # Danh sách thư viện phục vụ bảng điều khiển
└── README.md                                 # Tài liệu hướng dẫn dự án
(Cấu trúc thư mục và chức năng các tệp được tổng hợp từ kho mã nguồn AVORA-MATCHING-main và báo cáo kiến trúc hệ thống[cite: 1, 2]).4. Kết Quả Thực Nghiệm & Kiểm Chứng Chất Lượng (Benchmarks)4.1. Bảng So Sánh Hiệu Năng Trước và Sau Tối Ưu HóaTác vụ / Phân hệ đo lườngTrước tối ưu (Baseline)Sau tối ưu (Upgraded)Hệ số tăng tốcCông nghệ cốt lõi quyết địnhChuyển đổi Excel $\rightarrow$ CSV (3.000 dòng) (convert_xlsx_to_csv.py)   130,292 ms6,697 ms19,45xpolars.read_excel(engine="calamine") + fastexcel   Đọc & chuẩn hóa CSV Market ($300 \times 275$) (candidates.csv & job_descriptions.csv)   45,800 ms4,754 ms9,63xpl.scan_csv() (Apache Arrow LazyFrame)   Sinh Vector Embedding theo lô (275 JD, Cache Warm) (app/embeddings.py)   26.654,006 ms6,505 ms4.097,46xONNX Runtime / FastEmbed + SHA256EmbeddingCache   Chấm điểm ma trận Composite 4-tầng ($200 \times 50$) (src/core/scoring.py)   244,669 ms115,171 ms2,12x@numba.njit(parallel=True, fastmath=True) + numpy.ndarray   Giải ghép cặp ổn định C-Optimal (CSV $300 \times 275$) (src/core/matching_engine.py)   1,853 ms0,375 ms4,94x@numba.njit Gale-Shapley + mảng int32[:, :]   Giải ghép cặp ổn định J-Optimal (CSV $300 \times 275$) (src/core/matching_engine.py)   7,188 ms0,598 ms12,03x@numba.njit Gale-Shapley + mảng int32[:, :]   Giải ghép cặp ổn định C-Optimal (Composite $200 \times 50$) (src/core/matching_engine.py)   2,142 ms0,275 ms7,80x@numba.njit Gale-Shapley + @dataclass(slots=True)   Giải ghép cặp ổn định J-Optimal (Synthetic $2000 \times 400$) (src/core/matching_engine.py)   102,400 ms7,320 ms13,99x@numba.njit Gale-Shapley + Vectorized Rank TablesTổng thời gian chạy toàn bộ 93 bài kiểm thử (pytest tests/)[cite: 1, 2]27,820 s19,130 s1,45xToàn bộ 6 tầng hạ tầng mới4.2. Kiểm Chứng Độ Chính Xác Trên Golden Dataset ($N = 300, M = 275$)Trên tập dữ liệu thực tế gồm 300 ứng viên (scripts/data/candidates.csv) và 275 công việc (scripts/data/job_descriptions.csv) với 11.808 cặp khả thi:   Chế độ Candidate-Proposing (C-Optimal): Ghép thành công 257 cặp ổn định, 0 cặp chặn (blocking pairs), chỉ số hài lòng nhà tuyển dụng $E = 0.5474$, chỉ số hài lòng ứng viên $C = 0.9749$, tỷ lệ lấp đầy $\text{Fill Rate} = 0.1869$ với đúng 578 lượt đề xuất (proposals).Chế độ Employer-Proposing (J-Optimal): Ghép thành công 257 cặp ổn định, 0 cặp chặn (blocking pairs) với đúng 945 lượt đề xuất (proposals).Chế độ Composite Profile ($200 \times 50$): Ghép thành công 170 cặp ổn định tuyệt đối, khớp 100% giữa lõi thuật toán và phản hồi qua FastAPI.5. Hướng Dẫn Cài Đặt & Khởi Chạy5.1. Khởi chạy nhanh bằng Docker Compose (Khuyên dùng)Dự án đã được cấu hình sẵn Docker Multi-stage Build tích hợp trình quản lý gói Astral uv trong Dockerfile và dịch vụ Qdrant mở sẵn cổng gRPC 6334 trong docker-compose.yml[cite: 1, 2]:Bash# Xây dựng image và khởi chạy toàn bộ hệ thống (FastAPI + Qdrant + Dashboard)
docker-compose up --build -d
5.2. Cài đặt trên môi trường Local với uv (hoặc pip)Bash# 1. Cài đặt Astral uv (nếu chưa có) và đồng bộ thư viện siêu tốc
pip install uv
uv pip install --system -r requirements.txt
uv pip install --system -r requirements-dashboard.txt
(Danh sách phụ thuộc được tách riêng trong requirements.txt và requirements-dashboard.txt[cite: 1, 2]).5.3. Thực thi Pipeline Dữ liệu (ETL & Ingestion)Bash# Chuyển đổi dữ liệu từ Excel sang CSV (sử dụng Polars + fastexcel/calamine)
python scripts/convert_xlsx_to_csv.py

# Nạp bất đồng bộ dữ liệu Ứng viên và Công việc vào Qdrant qua gRPC (parallel=4)
python scripts/ingest_candidates.py
python scripts/ingest_job_descriptions.py
(Sử dụng bộ kịch bản trong thư mục scripts/ và dữ liệu tại scripts/data/[cite: 1, 2]).5.4. Khởi chạy API Server & Bảng điều khiển (Dashboard)Bash# Khởi chạy FastAPI Server (tích hợp ORJSONResponse)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Khởi chạy Bảng điều khiển phân tích & trực quan hóa kết quả ghép cặp
streamlit run dashboard.py
(Điểm vào dịch vụ đặt tại app/main.py và giao diện trực quan hóa đặt tại dashboard.py[cite: 1, 2]).6. Kiểm Thử & Đo Lường (Testing & Benchmarking)Hệ thống cung cấp bộ kiểm thử tự động toàn diện trong thư mục tests/ (93 bài kiểm thử chính và 48 subtests) cùng công cụ đo lường tại src/evaluation/:   Bash# Chạy toàn bộ bộ kiểm thử (Unit Tests, Stability Checker & Regression Tests)
pytest tests/ -v

# Kiểm tra riêng bộ kiểm định vét cạn tính ổn định Gale-Shapley
pytest tests/test_exhaustive_stability_checker.py -v

# Chạy kịch bản Benchmark đo lường hiệu năng thực tế
python -m src.evaluation.benchmark
7. Tài Liệu Kỹ Thuật Tham KhảoChi tiết về công thức toán học của hàm chấm điểm hai phía, chứng minh tính ổn định của thuật toán Gale-Shapley Many-to-One và kiến trúc 6 tầng công nghệ được trình bày đầy đủ trong tệp document-tech.md[cite: 1, 2].
=======
# AVORA Matching — High-Performance Two-Sided Stable Matching & Semantic Vector Search Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI ORJSONResponse](https://img.shields.io/badge/FastAPI-0.115%2B%20%7C%20ORJSONResponse-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Numba LLVM JIT](https://img.shields.io/badge/Numba-LLVM%20JIT%20%40njit-00A3E0?logo=llvm&logoColor=white)](https://numba.pydata.org/)
[![Qdrant gRPC + HNSW](https://img.shields.io/badge/Qdrant-gRPC%206334%20%2B%20Filterable%20HNSW-DC244C)](https://qdrant.tech/)
[![Polars Apache Arrow](https://img.shields.io/badge/Polars-Apache%20Arrow%20%2B%20Calamine-CD792C)](https://pola.rs/)
[![ONNX Runtime & FastEmbed](https://img.shields.io/badge/ONNX%20Runtime-INT8%2FFP16%20%2B%20FastEmbed-005CED)](https://onnxruntime.ai/)
[![Pytest Suite](https://img.shields.io/badge/Pytest-93%20passed%2C%2048%20subtests%20passed-2EA44F?logo=pytest&logoColor=white)](#8-hướng-dẫn-kiểm-thử--đánh-giá-hiệu-năng-testing--benchmarking)

---

## 1. Tổng quan Hệ thống (System Overview)

**AVORA Matching** là nền tảng Backend & Core Engine giải bài toán **Ghép cặp Việc làm – Ứng viên hai phía (Two-Sided Job–Candidate Matching Market)** chuyên biệt cho thị trường lao động hòa nhập (hỗ trợ người khuyết tật), kết hợp chặt chẽ giữa:

1. **Tìm kiếm Ngữ nghĩa Không gian Vector (Semantic Vector Search – 1024 chiều):** Biểu diễn mô tả công việc, kỹ năng chuyên môn và định hướng nghề nghiệp thành các vector dày (dense embeddings) chuẩn hóa $L_2$, kết hợp truy vấn lân cận xấp xỉ **Filterable HNSW** trên **Qdrant** (qua giao thức **gRPC port `6334`**) và **PostgreSQL 16 + `pgvector`** (qua driver bất đồng bộ **`asyncpg`**).
2. **Chấm điểm Tương hợp Đa chiều Hai phía (Two-Sided Composite Scoring):** Đánh giá đồng thời mức độ phù hợp của ứng viên đối với vị trí tuyển dụng $\text{Score}_J(j, c) \equiv E[c][j]$ (6 thành phần có trọng số) và mức độ hài lòng của ứng viên đối với công việc $\text{Score}_C(c, j) \equiv C[c][j]$ (3 thành phần theo thang ưu tiên cá nhân hóa `0.5 / 0.3 / 0.2`), đi kèm bộ lọc điều kiện tiên quyết **Hard Constraints** và cơ chế tuyển thẳng **Direct Track**.
3. **Lý thuyết Trò chơi & Ghép cặp Ổn định (Gale–Shapley Many-to-One Deferred Acceptance):** Giải bài toán phân bổ nhiều-một (*Hospital–Residents / College Admissions Problem*) với hạn mức tuyển dụng $q_j \ge 1$ ở cả hai chế độ đối ngẫu: **Candidate-Proposing (`C-Optimal`)** và **Employer-Proposing (`J-Optimal`)**, tiếp nối bởi tầng hậu xử lý gợi ý việc làm dự phòng **Top-$K$ Recommendations** (*Normalized Surplus Score* & *Iterative Clearance Rounds*).

### Nguyên tắc Thiết kế Cốt lõi: Zero-Algorithmic-Change Guarantee

Toàn bộ kiến trúc hệ thống đã được nâng cấp toàn diện qua **6 tầng công nghệ hiệu năng cao** (Polars/Arrow ETL, ONNX Runtime/FastEmbed + SHA-256 LRU Cache, Qdrant gRPC Filterable HNSW + SQLAlchemy 2.0 Async, Numba LLVM `@njit` Scoring & Gale–Shapley Engine, Pydantic V2 `@dataclass(slots=True)` + FastAPI `ORJSONResponse`, và Multi-stage Docker với Astral `uv`) dưới cam kết bất biến toán học nghiêm ngặt (**Zero-Algorithmic-Change Guarantee**):

* **Bảo toàn 100% công thức toán học và logic nghiệp vụ:** Không cắt giảm bước tính toán, không thay đổi trọng số, giữ nguyên quy tắc phá hòa tất định (*Deterministic Tie-Breaking* bằng `blake2b` 64-bit và thứ bậc điểm kỹ năng).
* **Độ ổn định tuyệt đối (`0` Blocking Pairs):** Mọi tập kết quả ghép cặp từ `GaleShapleyMatching` đều đạt tỷ lệ cặp chặn bằng **0.00%**, được kiểm chứng chéo bởi bộ kiểm tra vét cạn độc lập (`independent_find_blocking_pairs`).
* **Đồng nhất bit-level với Golden Dataset:** Khớp tuyệt đối **19/19 tiêu chí kiểm định** trước và sau nâng cấp, đồng thời đạt mức tăng tốc từ **2.12x đến 4,097.46x** tùy phân hệ.

---

## 2. Kiến trúc Luồng xử lý & Chi tiết 6 Tầng Công nghệ

### 2.1. Sơ đồ Kiến trúc Luồng Dữ liệu Toàn trình (End-to-End Pipeline Architecture)

```mermaid
flowchart TD
    subgraph T1["TẦNG 1: HIGH-SPEED COLUMNAR ETL (Polars + Apache Arrow + Calamine)"]
        A1["Dữ liệu thô (.xlsx / .csv)<br/>scripts/data/candidates.csv<br/>scripts/data/job_descriptions.csv"]
        A2["scripts/convert_xlsx_to_csv.py<br/>pl.read_excel(engine='calamine')"]
        A3["src/core/preprocessor.py<br/>pl.scan_csv() & preprocess_dataframe_polars()<br/>load_market_from_csv()"]
        A1 --> A2 --> A3
    end

    subgraph T2["TẦNG 2: LIGHTWEIGHT EMBEDDING & SHA-256 CACHE (ONNX Runtime / FastEmbed)"]
        B1["normalize_text_for_embedding() / normalize_text()<br/>Chuẩn hóa Unicode NFC & khoảng trắng"]
        B2["SHA256EmbeddingCache (LRU Thread-Safe)<br/>Khóa băm: SHA256(model_name :: normalized_text)"]
        B3["OnnxOrFastEmbedModel & get_embeddings_batch()<br/>ONNX Runtime (INT8/FP16) / FastEmbed 1024-dim L2-norm"]
        B1 --> B2 --> B3
    end

    subgraph T3["TẦNG 3: VECTOR DB & ASYNC RELATIONAL STORAGE (Qdrant gRPC + PostgreSQL asyncpg)"]
        C1["AsyncQdrantClient (prefer_grpc=True, Port 6334)<br/>init_qdrant_collections() & HARD_CONSTRAINT_PAYLOAD_FIELDS<br/>build_qdrant_hard_constraint_filter()"]
        C2["SQLAlchemy 2.0 AsyncEngine (postgresql+asyncpg://)<br/>AsyncSessionLocal & pgvector Vector(1024)<br/>Models: JobDescription, Candidate"]
    end

    subgraph T4["TẦNG 4: NUMBA LLVM JIT SCORING & GALE-SHAPLEY ENGINE (src/core/ & app/matching.py)"]
        D1["Hard Constraints & Direct Track Filter<br/>check_hard_constraints() / apply_hard_constraints()"]
        D2["Numba Parallel Composite Scoring (@njit parallel=True, fastmath=True)<br/>_cosine_similarity_numba()<br/>compute_cosine_similarity_matrix_numba()<br/>compute_composite_score_matrices_numba()"]
        D3["O(1) Rank Lookup Tables trên mảng int32[:, :]<br/>_build_rank_tables(): _cand_prefs_arr, _job_prefs_arr,<br/>_cand_rank_arr, _job_rank_arr"]
        D4["Numba JIT Gale-Shapley Deferred Acceptance (@njit cache=True)<br/>_gs_candidate_proposing_numba() (C-Optimal)<br/>_gs_employer_proposing_numba() (J-Optimal)<br/>_app_gs_candidate_proposing_numba() / _app_gs_employer_proposing_numba()"]
        D1 --> D2 --> D3 --> D4
    end

    subgraph T5["TẦNG 5 & 6: POSTPROCESSING, SERIALIZATION & PRESENTATION"]
        E1["src/core/postprocessor.py & src/evaluation/metrics.py<br/>generate_top_k() / iterative_clearance_rounds()<br/>evaluate() -> EvaluationReport (0 Blocking Pairs)"]
        E2["FastAPI (default_response_class=ORJSONResponse)<br/>Pydantic V2 ConfigDict & TypeAdapter<br/>Endpoints: /match, /matching/stable, /matching/matrix, /matching/database"]
        E3["Streamlit Interactive Dashboard (dashboard.py)<br/>@st.cache_resource & @st.cache_data (Port 8501)"]
        D4 --> E1 --> E2
        E1 --> E3
    end

    A3 --> B1
    A3 --> D1
    B3 --> C1
    B3 --> C2
    C1 --> D1
    C2 --> D1
```

### 2.2. Bảng Tổng hợp 6 Tầng Công nghệ & Các Thành phần Mã nguồn Trọng tâm

| Tầng | Lĩnh vực Nâng cấp | Công nghệ Trước Nâng cấp | Công nghệ Sau Nâng cấp | Module / Class / Hàm Trọng tâm trong Codebase |
| :--- | :--- | :--- | :--- | :--- |
| **Tầng 1** | **Đọc/Ghi & Tiền xử lý Dữ liệu (Data I/O & ETL)** | `pandas.read_excel(openpyxl)`, vòng lặp `csv.DictReader` tuần tự | **Polars (`LazyFrame` / `pl.scan_csv`)** trên chuẩn bộ nhớ cột **Apache Arrow** + Rust **`calamine` (`fastexcel`)** | • `scripts/convert_xlsx_to_csv.py`: `convert()` dùng `pl.read_excel(..., engine="calamine")` và `df.write_csv(..., include_bom=True)`.<br/>• `src/core/preprocessor.py`: `preprocess_dataframe_polars()`, `load_market_from_csv()`, `normalize_text_for_embedding()`, `SKILL_SYNONYM_MAP`, `EDUCATION_LEVEL_MAP`.<br/>• `scripts/ingest_candidates.py` & `scripts/ingest_job_descriptions.py`: `read_candidates()`, `read_job_descriptions()`. |
| **Tầng 2** | **Suy luận Vector Ngữ nghĩa & Lưu đệm (Embedding Inference & Caching)** | `SentenceTransformer` (PyTorch/CUDA nặng nề), biên dịch lẻ từng chuỗi | **ONNX Runtime (`INT8`/`FP16`)** / **FastEmbed (`TextEmbedding`)** + **Dynamic Batching** + **SHA-256 LRU Cache** | • `app/embeddings.py`: `OnnxOrFastEmbedModel` (tích hợp `ort.InferenceSession` với `ORT_ENABLE_ALL` và `TextEmbedding`), `SHA256EmbeddingCache` (`compute_key`, `get`, `put`, `stats`), `get_embedding_model()`, `get_embeddings_batch()`, `get_embedding()`, `normalize_text()`.<br/>• `app/config.py`: `EMBEDDING_MODEL_NAME` (hỗ trợ `Qwen/Qwen3-Embedding-0.6B` & `AITeamVN/Vietnamese_Embedding` 1024 chiều), `DYNAMIC_BATCH_SIZE=32`, `EMBEDDING_CACHE_MAX_SIZE=16384`. |
| **Tầng 3** | **Truy vấn Vector & Cơ sở dữ liệu Bất đồng bộ (Vector & Relational DB)** | Truy vấn tuần tự đồng bộ `psycopg2`, lọc điều kiện cứng bằng vòng lặp Python | **`AsyncQdrantClient` (gRPC `6334`)** + **Filterable HNSW Payload Indexing** + **SQLAlchemy 2.0 `asyncpg`** | • `app/database.py`: `get_async_qdrant_client()` (`prefer_grpc=True`), `HARD_CONSTRAINT_PAYLOAD_FIELDS`, `init_qdrant_collections()`, `build_qdrant_hard_constraint_filter()`, `create_async_engine(ASYNC_DATABASE_URL)`, `AsyncSessionLocal = async_sessionmaker(...)`, `get_async_db()`, `init_async_db()`.<br/>• `app/models.py`: `JobDescription` (`Vector(VECTOR_DIM)`), `Candidate`.<br/>• `scripts/ingest_*.py`: `upload_candidates_to_qdrant(..., parallel=4)`, `upload_job_descriptions_to_qdrant(..., parallel=4)`. |
| **Tầng 4** | **Chấm điểm Hai phía & Giải thuật Ghép cặp (Scoring & Matching Engine)** | Vòng lặp lồng nhau $O(N \times M)$ thuần Python, tra cứu `dict` động và `heapq` | **Biên dịch mã máy LLVM JIT (`@numba.njit`)** + Đa luồng **`numba.prange`** + Bảng hạng $O(1)$ trên mảng liên tục `int32[:, :]` | • `src/core/scoring.py`: `_cosine_similarity_numba()`, `compute_cosine_similarity_matrix_numba()`, `compute_composite_score_matrices_numba()`, `compute_batch_composite_scores_matrix()`, `calculate_composite_e_score()`, `calculate_composite_c_score()`, `check_hard_constraints()`, `explain_match()`.<br/>• `src/core/matching_engine.py`: `GaleShapleyMatching._build_rank_tables()`, `_gs_candidate_proposing_numba()`, `_gs_employer_proposing_numba()`, `MatchingResult`.<br/>• `app/matching.py`: `_app_gs_candidate_proposing_numba()`, `_app_gs_employer_proposing_numba()`. |
| **Tầng 5** | **Cấu trúc Dữ liệu, Tuần tự hóa & API Web (Data Models & Serialization)** | Dataclass cấp phát `__dict__` động, `JSONResponse` chuẩn thư viện gốc | **`@dataclass(slots=True)`** + **Pydantic V2 (`ConfigDict`, `TypeAdapter`)** + **FastAPI `ORJSONResponse`** | • `src/core/profile_models.py`: `CandidateProfile`, `JobProfile`, `ExecutionAudit` (`@dataclass(slots=True)`), `CandidateProfileAdapter`, `JobProfileAdapter`, `ExecutionAuditAdapter`.<br/>• `src/core/preprocessor.py` & `postprocessor.py`: `PreprocessConfig`, `PreferenceProfile`, `Recommendation(frozen=True, slots=True)`.<br/>• `app/schemas.py`: `CandidateProfileListAdapter`, `JobProfileListAdapter`, `MatchedPairListAdapter`, `StableMatchingResponseAdapter`.<br/>• `app/main.py`: `FastAPI(..., default_response_class=ORJSONResponse)`. |
| **Tầng 6** | **Đóng gói Container & Giao diện Thực nghiệm (Packaging & UI)** | `Dockerfile` đơn tầng cài đặt bằng `pip`, Streamlit tính toán lại mỗi lần tương tác | **Multi-stage `Dockerfile` tích hợp Astral `uv`** + Streamlit Caching (`@st.cache_resource`, `@st.cache_data`) | • `Dockerfile`: Multi-stage (`builder` & `runtime` trên `python:3.12-slim`) dùng `ghcr.io/astral-sh/uv:latest`.<br/>• `docker-compose.yml`: Điều phối `db` (`pgvector/pgvector:pg16`), `qdrant` (`6333`/`6334`), `web` (`8000`).<br/>• `dashboard.py`: `get_cached_preprocessor_resource()`, `get_cached_embedding_model_resource()`, `run_cached_benchmark()`, `build_synthetic_market()`. |

---

## 3. Đặc tả Toán học & Thuật toán trong `src/core/`

### 3.1. Mô hình Dữ liệu Đầu vào (`src/core/profile_models.py` & `src/core/preprocessor.py`)

Toàn bộ cấu trúc dữ liệu lõi được định nghĩa với `@dataclass(slots=True)` để triệt tiêu chi phí cấp phát từ điển động `__dict__` trên từng thực thể, tăng mật độ lưu trữ trên bộ nhớ đệm CPU (L1/L2 Cache):

1. **`CandidateProfile` (`src/core/profile_models.py`):**
   * Định danh & Năng lực: `candidate_id`, `skills` (`list[float] | str | list[str]`), `experience_years`, `career_orientation`.
   * Điều kiện tiếp cận: `essential_access_needs` (*Hard constraint* — nhu cầu bắt buộc), `nice_to_have_access_needs` (*Soft component* — nhu cầu mong muốn), `disability_group` (chỉ dùng làm metadata hỗ trợ tiếp cận, **tuyệt đối không** dùng để suy diễn hay hạ thấp năng lực).
   * Nguyện vọng làm việc: `expected_salary_min`, `expected_salary_max`, `work_mode` (`'remote' | 'hybrid' | 'on-site'`), `work_hours` (`'full-time' | 'part-time' | 'flexible'`), `location`, `priority_ranking` (thứ tự 3 yếu tố ưu tiên tự xếp: `'income'`, `'distance_or_remote'`, `'career_path'`).
   * Tuyển thẳng: `direct_track: bool = False`, `hired_job_id: str | None = None`. Hàm `__post_init__` kiểm tra nghiêm ngặt: nếu `direct_track=True` mà thiếu `hired_job_id` sẽ lập tức ném `ValueError`.
2. **`JobProfile` (`src/core/profile_models.py`):**
   * Định danh & Hạn mức: `job_id`, `title`, `quota` ($q_j \ge 0$), `filled_slots` (số chỉ tiêu đã tuyển trước từ ngoài hệ thống, không tính ứng viên Direct Track trong batch hiện tại; thuộc tính `remaining_quota = max(0, quota - filled_slots)`).
   * Yêu cầu & Hỗ trợ: `skills_required`, `access_support_provided`, `salary_min`, `salary_max`, `work_mode`, `work_hours`, `location`, `experience_required_years`, `career_domain`, `mandatory_requirements`.
3. **`EmployerWeightsConfig` (`src/core/scoring.py`):**
   * Lưu bộ 6 trọng số đánh giá của nhà tuyển dụng: `skill = 0.35`, `work_mode_hours = 0.20`, `nice_to_have_access = 0.20`, `salary = 0.10`, `experience = 0.10`, `location = 0.05`. Phương thức `validate()` đảm bảo tổng trọng số bằng `1.0` ($\pm 10^{-4}$).
4. **`PreprocessConfig` & `PreferenceProfile` (`src/core/preprocessor.py`):**
   * `PreprocessConfig`: Cấu hình cắt ngắn danh sách nguyện vọng (`max_candidate_preferences: int = 40`, `max_employer_preferences: int | None = None`), ngưỡng điểm sàn (`min_e_score = 0.0`, `min_c_score = 0.0`, `require_positive_scores = True`), hạt giống phá hòa (`tie_break_seed = 20_240_517`) và cờ ưu tiên phá hòa theo kỹ năng (`use_skill_tie_break = True`).
   * `PreferenceProfile`: Chứa `candidate_ids`, `job_ids`, ma trận thưa `e_scores`, `c_scores`, từ điển `quotas`, danh sách ưu tiên đã sắp xếp `candidate_preferences`, `job_preferences` cùng `metadata` kiểm toán (`ExecutionAudit`, `explanations`, `breakdowns`, `rejection_reasons`).

### 3.2. Bộ lọc Điều kiện Cứng (Hard Constraints) & Công thức Chấm điểm Hai phía (`src/core/scoring.py`)

#### A. Bộ lọc Điều kiện Bắt buộc (`check_hard_constraints` & `apply_hard_constraints`)
Trước khi tính điểm hoặc xếp hạng, một cặp $(c, j)$ sẽ bị loại bỏ vĩnh viễn khỏi không gian ghép cặp (tương đương hàm chỉ báo $\mathbb{I}_D(c, j) = 0$) nếu vi phạm bất kỳ điều kiện nào sau đây:
1. **Ứng viên thuộc diện Direct Track (`c.direct_track == True`):** Được rút khỏi bể ghép cặp chung và trừ trực tiếp 1 chỉ tiêu vào `quotas[c.hired_job_id]`.
2. **Xung đột hình thức làm việc tuyệt đối:** Ứng viên bắt buộc làm từ xa (`cand.work_mode == "remote"`) nhưng vị trí yêu cầu làm việc tại chỗ 100% (`job.work_mode == "on-site"`).
3. **Thiếu điều kiện tiếp cận thiết yếu (`essential_access_needs`):** Khi công việc không phải `remote`, nếu tập nhu cầu thiết yếu của ứng viên không được bao phủ hoàn toàn bởi tập tiện nghi hỗ trợ của doanh nghiệp ($\text{cand\_essentials} \setminus \text{job\_supports} \neq \emptyset$), cặp $(c, j)$ bị loại ngay lập tức.
4. **Hết chỉ tiêu hoặc dưới ngưỡng sàn:** Vị trí có $q_j^{\text{remaining}} \le 0$, hoặc điểm $E[c][j] \le 0$, $C[c][j] \le 0$, hoặc thấp hơn `min_e_score` / `min_c_score`.

#### B. Công thức Điểm Phía Nhà tuyển dụng $\text{Score}_J(j, c) \equiv E[c][j]$ (`calculate_composite_e_score`)
Đối với mọi cặp $(c, j)$ thỏa mãn Hard Constraints, điểm đánh giá của nhà tuyển dụng $j$ dành cho ứng viên $c$ là tổ hợp tuyến tính của 6 thành phần chuẩn hóa trên đoạn $[0, 1]$:

$$\text{Score}_J(j, c) = E[c][j] = \text{clip}_{[0.001, 1.0]}\left(\text{round}_6\left(\sum_{k=1}^{6} w_k^{(E)} \cdot S_k^{(E)}(c, j)\right)\right)$$

Trong đó:
* $S_{\text{skill}}(c, j)$ ($w_1 = 0.35$ — `calculate_skill_similarity`): Độ tương đồng Cosine giữa hai vector kỹ năng ($\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ biên dịch bằng `_cosine_similarity_numba`) hoặc độ tương đồng Jaccard trên tập từ khóa kỹ năng đã chuẩn hóa (`calculate_token_jaccard`).
* $S_{\text{work}}(c, j)$ ($w_2 = 0.20$ — `calculate_work_mode_hours_match`): Mức khớp kết hợp giữa hình thức làm việc (`remote`/`hybrid`/`on-site`) và thời gian làm việc (`full-time`/`part-time`/`flexible`), nhận giá trị rời rạc trong $\{1.0, 0.5, 0.0\}$.
* $S_{\text{access}}(c, j)$ ($w_3 = 0.20$ — `calculate_nice_to_have_access_ratio`): Tỷ lệ nhu cầu tiếp cận "nên có" được đáp ứng: $\frac{|\text{cand\_nice\_to\_have} \cap \text{job\_supports}|}{|\text{cand\_nice\_to\_have}|}$ (bằng $1.0$ nếu ứng viên không yêu cầu thêm).
* $S_{\text{salary}}(c, j)$ ($w_4 = 0.10$ — `calculate_salary_overlap`): Tỷ lệ chồng lấn khoảng lương (IoU): $\frac{\max(0, \min(c_{\max}, j_{\max}) - \max(c_{\min}, j_{\min}))}{\max(c_{\max}, j_{\max}) - \min(c_{\min}, j_{\min})}$.
* $S_{\text{exp}}(c, j)$ ($w_5 = 0.10$ — `calculate_experience_score`): Tỷ lệ đáp ứng kinh nghiệm: $\min\left(1.0, \frac{\text{cand\_exp}}{\text{req\_exp}}\right)$ (bằng $1.0$ nếu $\text{req\_exp} \le 0$).
* $S_{\text{loc}}(c, j)$ ($w_6 = 0.05$ — `calculate_location_match`): Bằng $1.0$ nếu cùng địa phương hoặc một trong hai bên làm `remote`, $0.5$ nếu khuyết thông tin địa điểm, và $0.0$ nếu khác khu vực.

#### C. Công thức Điểm Phía Ứng viên $\text{Score}_C(c, j) \equiv C[c][j]$ (`calculate_composite_c_score`)
Điểm yêu thích của ứng viên $c$ đối với công việc $j$ được tổng hợp từ 3 chiều nguyện vọng cá nhân hóa:

$$\text{Score}_C(c, j) = C[c][j] = \text{clip}_{[0.001, 1.0]}\left(\text{round}_6\left(w_{\text{inc}}^{(c)} S_{\text{inc}}(c, j) + w_{\text{dist}}^{(c)} S_{\text{dist}}(c, j) + w_{\text{car}}^{(c)} S_{\text{car}}(c, j)\right)\right)$$

* **Trọng số cá nhân hóa (`CandidateProfile.get_priority_weights`):** Dựa trên danh sách `priority_ranking` do ứng viên tự xếp hạng, yếu tố ưu tiên số 1 nhận trọng số **$0.50$**, số 2 nhận **$0.30$**, số 3 nhận **$0.20$**. Nếu ứng viên không xếp hoặc thiếu dữ liệu, hệ thống dùng trọng số mặc định chuẩn: $(w_{\text{inc}}, w_{\text{dist}}, w_{\text{car}}) = (0.35, 0.30, 0.35)$.
* **Ba thành phần điểm:**
  1. $S_{\text{inc}}(c, j) = \min\left(1.0, \frac{j_{\max}}{c_{\min}}\right)$ (bằng $1.0$ nếu $c_{\min} \le 0$).
  2. $S_{\text{dist}}(c, j) = S_{\text{loc}}(c, j)$ (độ tương thích khoảng cách địa lý hoặc làm việc từ xa).
  3. $S_{\text{car}}(c, j)$: Độ tương đồng ngữ nghĩa giữa `career_orientation` của ứng viên và `career_domain` của vị trí (hoặc fallback về $S_{\text{skill}}(c, j)$ nếu chưa khai báo định hướng nghề).

#### D. Tăng tốc Ma trận Điểm bằng Numba LLVM JIT (`compute_composite_score_matrices_numba`)
Khi xử lý theo lô cho $N$ ứng viên và $M$ công việc, hàm điều phối `compute_batch_composite_scores_matrix` chuẩn bị các ma trận thành phần $\mathbb{R}^{N \times M}$ và gọi trực tiếp kernel biên dịch mã máy `compute_composite_score_matrices_numba` cùng `compute_cosine_similarity_matrix_numba` với decorator `@numba.njit(parallel=True, fastmath=True, cache=True)`. Vòng lặp ngoài phân tách theo `numba.prange(n_cands)` chạy đa luồng trên tất cả lõi CPU vật lý mà không bị khóa bởi Python GIL.

### 3.3. Tối ưu hóa Giải thuật Gale–Shapley Many-to-One bằng Numba JIT (`src/core/matching_engine.py`)

#### A. Bảng Tra cứu Thứ hạng $O(1)$ trên Mảng 2 Chiều `int32[:, :]` (`_build_rank_tables`)
Trong thuật toán Gale–Shapley truyền thống, việc kiểm tra xem vị trí $j$ có thích ứng viên mới $c$ hơn ứng viên kém nhất đang nắm giữ $c_{\text{worst}}$ hay không đòi hỏi tìm kiếm tuyến tính $O(N)$ hoặc tra cứu bảng băm `dict` lồng nhau tốn kém bộ nhớ.

Phương thức `GaleShapleyMatching._build_rank_tables()` ánh xạ toàn bộ định danh ứng viên và công việc sang chỉ số nguyên liên tục $0 \dots N-1$ và $0 \dots M-1$, đồng thời khởi tạo bộ mảng NumPy 2 chiều kiểu `np.int32` phẳng trong RAM (và lưu đệm vào `profile.metadata["_numba_rank_tables"]`):
* `_cand_prefs_arr` (`int32[N, max_L]`) & `_cand_pref_lens` (`int32[N]`): Danh sách nguyện vọng của từng ứng viên.
* `_job_prefs_arr` (`int32[M, max_M]`) & `_job_pref_lens` (`int32[M]`): Danh sách ưu tiên ứng viên của từng vị trí.
* `_job_rank_arr` (`int32[M, N]`): Bảng tra cứu nghịch đảo hạng của ứng viên $c$ tại vị trí $j$ (`_job_rank_arr[j, c] = rank`, giá trị `-1` biểu thị không chấp nhận). Mọi phép so sánh ưu tiên trong vòng lặp `while` đều đạt độ phức tạp hằng số **$O(1)$** chỉ bằng một phép truy cập ô nhớ mảng 2D.
* `_cand_rank_arr` (`int32[N, M]`): Bảng tra cứu nghịch đảo hạng của vị trí $j$ trong mắt ứng viên $c$ (`_cand_rank_arr[c, j] = rank`, `-1` nếu không chấp nhận).

#### B. Hai Chế độ Thực thi Biên dịch Mã máy (`@numba.njit(cache=True)`)
1. **Chế độ Candidate-Proposing (`C-Optimal` — `_gs_candidate_proposing_numba` & `_app_gs_candidate_proposing_numba`):**
   * Sử dụng hàng đợi vòng FIFO tĩnh `queue` (`np.empty(max_q, dtype=np.int32)`) với hai con trỏ `head`, `tail` và mảng trạng thái `held_cands[M, max_quota]`, `held_ranks[M, max_quota]`, `held_counts[M]`.
   * Từng ứng viên tự do $c$ lần lượt nộp đơn tới công việc tốt nhất tiếp theo $j = \text{cand\_prefs\_arr}[c, \text{next\_proposal}[c]]$.
   * Nếu vị trí $j$ còn chỗ trống ($\text{held\_counts}[j] < q_j$), $j$ tạm giữ $c$. Nếu $j$ đã đầy, kernel tìm vị trí `worst_slot` có `worst_rank` lớn nhất (kém nhất) đang giữ; nếu $\text{job\_rank\_arr}[j, c] < \text{worst\_rank}$, $c$ thế chỗ `worst_cand` và đẩy `worst_cand` trở lại hàng đợi `queue`.
   * **Tính chất toán học:** Đạt nghiệm ổn định tốt nhất cho mọi ứng viên (**C-Optimal**) và chống khai gian chiến lược (**Strategy-Proof**) đối với phía ứng viên.
2. **Chế độ Employer-Proposing (`J-Optimal` — `_gs_employer_proposing_numba` & `_app_gs_employer_proposing_numba`):**
   * Các vị trí tuyển dụng còn thiếu chỉ tiêu ($\text{assigned\_counts}[j] < q_j$) chủ động gửi lời mời tới ứng viên tiềm năng nhất tiếp theo trong `job_prefs_arr[j, p_idx]`.
   * Ứng viên $c$ đối chiếu qua `cand_rank_arr[cand, j]`: chỉ nhận lời nếu $j$ hợp lệ ($\text{rank} \ge 0$) và tốt hơn công việc hiện tại `candidate_holder[cand]`. Công việc bị từ chối được đưa lại vào hàng đợi nếu vẫn còn nguyện vọng chưa xét.
   * **Tính chất toán học:** Đạt nghiệm ổn định tốt nhất cho phía doanh nghiệp (**J-Optimal**).

### 3.4. Quy trình Hậu xử lý (`src/core/postprocessor.py`) & Kiểm định Độ ổn định (`src/evaluation/metrics.py`)

1. **Sinh Gợi ý Việc làm Dự phòng Top-$K$ (`src/core/postprocessor.py`):**
   * **`normalized_surplus_scores(profile, weight_c=0.5, weight_e=0.5)`:** Chuẩn hóa Min–Max điểm $C$ theo từng ứng viên ($\hat{C}(c, j) \in [0, 1]$) và chuẩn hóa Min–Max điểm $E$ theo từng vị trí ($\hat{E}(c, j) \in [0, 1]$), sau đó tính điểm thặng dư chuẩn hóa $\text{surplus}(c, j) = w_c \hat{C}(c, j) + w_e \hat{E}(c, j)$.
   * **`generate_top_k(profile, result, k=3)`:** Đặt vị trí ghép chính thức (`is_primary=True`, `position=1`) lên đầu tiên, tiếp theo là $k-1$ vị trí dự phòng có điểm `surplus` cao nhất kèm chuỗi giải thích `explanation` và bảng phân rã `breakdown` trong đối tượng bất biến `Recommendation` (`@dataclass(frozen=True, slots=True)`).
   * **`iterative_clearance_rounds(profile, k=3, optimal="candidate")`:** Chạy lặp $K$ vòng Gale–Shapley; sau mỗi vòng loại bỏ các cạnh $(c, j)$ vừa được ghép khỏi danh sách ưu tiên của cả hai phía để lộ diện nguyện vọng tốt nhất kế tiếp.
2. **Kiểm định Độ ổn định & Chỉ số Đánh giá Thị trường (`src/evaluation/metrics.py`):**
   * **`find_blocking_pairs` & `calculate_blocking_pairs`:** Duyệt kiểm tra toàn bộ cặp khả thi $(c, j)$. Một cặp bị coi là **Blocking Pair** nếu: (1) cả hai chấp nhận nhau, (2) $c$ thích $j$ hơn vị trí hiện tại $\mu(c)$ (hoặc $c$ chưa có việc), và (3) $j$ còn trống chỉ tiêu ($|\mu(j)| < q_j$) hoặc $j$ thích $c$ hơn ít nhất một ứng viên $c' \in \mu(j)$. Trả về `BlockingPairReport(count=0, rate=0.0, is_stable=True)`.
   * **`evaluate(profile, result)`:** Tổng hợp báo cáo `EvaluationReport` toàn diện gồm 6 thành phần: `BlockingPairReport`, `SatisfactionReport` (`average_e`, `average_c`, `coverage`), `FillRateReport` (`overall`, `filled_slots`, `total_quota`), `RankReport` (`average_candidate_rank`, `average_job_rank`), `CoverageReport` (`unmatched_candidate_rate`, `unmatched_job_rate`) và `ConflictReport` (`congestion_rate = 0.0`). Ngoài ra cung cấp `audit_fairness` (kiểm toán công bằng theo *Four-Fifths Rule* $\text{DI} \ge 0.80$), `calculate_vector_recall`, `calculate_precision_at_k` và `calculate_ndcg_at_k`.

---

## 4. Cấu trúc Thư mục Chi tiết (Repository Structure)

```text
AVORA-MATCHING/
├── Dockerfile                                    # Multi-stage Docker build (builder + runtime) tích hợp Astral uv trên python:3.12-slim
├── docker-compose.yml                            # Điều phối 3 services: PostgreSQL 16 (pgvector), Qdrant (HTTP 6333 / gRPC 6334), FastAPI Web (8000)
├── requirements.txt                              # Dependencies cho Production API & Core Engine (FastAPI, Numba, Polars, ONNX Runtime, Qdrant, asyncpg)
├── requirements-dashboard.txt                    # Dependencies mở rộng cho giao diện Streamlit Dashboard local (streamlit>=1.30)
├── conftest.py                                   # Cấu hình sys.path gốc cho Pytest
├── dashboard.py                                  # Giao diện tương tác Streamlit (Port 8501) với @st.cache_resource & @st.cache_data
├── document-tech.md                              # Tài liệu đặc tả kỹ thuật chi tiết cấp hàm/module của hệ thống
├── README.md                                     # Tài liệu hướng dẫn toàn diện kiến trúc, toán học, thực nghiệm và vận hành
│
├── src/                                          # PHÂN HỆ LÕI THUẬT TOÁN & ĐÁNH GIÁ ĐỘC LẬP
│   ├── __init__.py                               # Đánh dấu package gốc src
│   ├── core/                                     # Lõi Tiền xử lý, Chấm điểm Hai phía, Gale-Shapley Engine & Hậu xử lý Top-K
│   │   ├── __init__.py                           # Tái xuất (re-export) toàn bộ API công khai của src.core
│   │   ├── profile_models.py                     # @dataclass(slots=True): CandidateProfile, JobProfile, ExecutionAudit & Pydantic V2 TypeAdapters
│   │   ├── preprocessor.py                       # Polars LazyFrame ETL (preprocess_dataframe_polars, load_market_from_csv), Preprocessor, PreferenceProfile
│   │   ├── scoring.py                            # Công thức điểm E/C, Hard Constraints & các kernel @numba.njit(parallel=True, fastmath=True)
│   │   ├── matching_engine.py                    # GaleShapleyMatching, bảng hạng O(1) _build_rank_tables & _gs_*_proposing_numba (@numba.njit)
│   │   └── postprocessor.py                      # Hậu xử lý Recommendation: normalized_surplus_scores, generate_top_k, iterative_clearance_rounds
│   │
│   └── evaluation/                               # Phân hệ Đo lường Chất lượng & Thực nghiệm Benchmark
│       ├── __init__.py                           # Tái xuất các lớp báo cáo và hàm đánh giá
│       ├── metrics.py                            # Kiểm tra Blocking Pairs, Satisfaction, Fill Rate, Rank, Congestion, Recall@N, nDCG@K, Fairness Audit
│       └── benchmark.py                          # CLI Benchmark so sánh Gale-Shapley (C-Optimal, J-Optimal) với GreedyBaselineMatcher
│
├── app/                                          # PHÂN HỆ FASTAPI BACKEND, EMBEDDING, VECTOR DB & ORM
│   ├── __init__.py                               # Đánh dấu package app
│   ├── config.py                                 # Cấu hình AppSettings (Pydantic V2 BaseSettings) quản lý PostgreSQL, Qdrant gRPC, ONNX/FastEmbed
│   ├── database.py                               # SQLAlchemy 2.0 AsyncEngine (asyncpg) + AsyncQdrantClient (gRPC 6334) + Filterable HNSW Payload Index
│   ├── models.py                                 # SQLAlchemy 2.0 ORM Models: JobDescription (Vector 1024 chiều) và Candidate
│   ├── schemas.py                                # Pydantic V2 Request/Response Schemas, Vector3D, HardConstraints & Batch TypeAdapters
│   ├── embeddings.py                             # OnnxOrFastEmbedModel (INT8/FP16), SHA256EmbeddingCache (LRU Thread-Safe) & get_embeddings_batch
│   ├── matching.py                               # Pipeline kết nối API với Gale-Shapley Numba JIT (_app_gs_*_proposing_numba) & đại số Vector Oxyz
│   └── main.py                                   # FastAPI Application (ORJSONResponse) cung cấp 10 RESTful Endpoints
│
├── scripts/                                      # PHÂN HỆ KỊCH BẢN ETL, CHUYỂN ĐỔI DỮ LIỆU & INGESTION
│   ├── convert_xlsx_to_csv.py                    # Chuyển đổi siêu tốc .xlsx -> .csv bằng Polars engine="calamine" (fastexcel)
│   ├── generate_mock_candidates.py               # Sinh dữ liệu ứng viên giả lập cân bằng giữa 7 nhóm khuyết tật bằng Polars
│   ├── generate_mock_data.py                     # Sinh dữ liệu mô tả công việc thực tế chuẩn hóa tiếng Việt bằng Polars
│   ├── ingest_candidates.py                      # Đọc CSV bằng pl.scan_csv và nạp vào FastAPI hoặc Qdrant gRPC (parallel=4)
│   ├── ingest_job_descriptions.py                # Đọc CSV bằng pl.scan_csv, sinh batch embedding và nạp vào FastAPI hoặc Qdrant gRPC (parallel=4)
│   └── data/                                     # Kho dữ liệu Golden Dataset chuẩn của dự án
│       ├── candidates.csv                        # 3,000 hồ sơ ứng viên mẫu (Họ và tên, Nhóm khuyết tật)
│       └── job_descriptions.csv                  # 275 bản ghi mô tả công việc mẫu gắn nhãn nhóm khuyết tật
│
└── tests/                                        # BỘ KIỂM THỬ TỰ ĐỘNG TOÀN DIỆN (93 PASSED, 48 SUBTESTS PASSED)
    ├── __init__.py                               # Đánh dấu package tests
    ├── test_matching.py                          # 31 bài kiểm thử lõi tiền xử lý, Gale-Shapley C/J-Optimal, Top-K và CSV Market
    ├── test_app_stable_matching.py               # 16 bài kiểm thử đại số Vector Oxyz, API /matching/stable, /matching/matrix và KPIs
    ├── test_enhanced_matching.py                 # 23 bài kiểm thử công thức điểm Composite E/C, thang ưu tiên 0.5/0.3/0.2, Fairness & Explainability
    ├── test_regression_fixes.py                  # 9 bài kiểm thử hồi quy + 32 subtests vét cạn (Direct Track, filled_slots, HTTP 422, quota=0)
    ├── test_exhaustive_stability_checker.py      # 7 bài kiểm thử + 16 subtests đối chiếu độc lập 100% từ nguyên lý toán học gốc
    └── test_dashboard.py                         # 7 bài kiểm thử giao diện Streamlit Dashboard (bao gồm AppTest headless smoke test)
```

---

## 5. Kết quả Thực nghiệm (Benchmarks) & Kiểm chứng Golden Dataset

### 5.1. Bảng So sánh Hiệu năng Trước và Sau Nâng cấp (6 Tầng Công nghệ)

Số liệu thực đo trên cùng môi trường phần cứng và cùng bộ dữ liệu đầu vào trước và sau khi nâng cấp 6 tầng hệ thống:

| Phân hệ / Tác vụ Thực nghiệm | Quy mô Dữ liệu | Trước Nâng cấp (Baseline) | Sau Nâng cấp (Optimized) | Hệ số Tăng tốc (Speedup) | Công nghệ Cốt lõi Tạo Đột phá |
| :--- | :--- | ---: | ---: | :---: | :--- |
| **1. Chuyển đổi Excel $\rightarrow$ CSV** (`convert_xlsx_to_csv.py`) | `3,000` dòng `.xlsx` | `130.292 ms` | **`6.697 ms`** | **19.45x** | Polars `pl.read_excel(engine="calamine")` (Rust `fastexcel`) thay cho `pandas` + `openpyxl` |
| **2. Đọc & chuẩn hóa CSV Market** (`load_market_from_csv`) | `300 × 275` (`82,500` cặp) | `45.800 ms` | **`4.754 ms`** | **9.63x** | Polars `pl.scan_csv` (`LazyFrame` Apache Arrow) + Vectorized String Expressions |
| **3. Sinh Vector Embedding theo lô** (`get_embeddings_batch`, Cache Warm) | `275` mô tả công việc (`1024`-dim) | `26,654.006 ms` | **`6.505 ms`** | **4,097.46x** | `OnnxOrFastEmbedModel` + `SHA256EmbeddingCache` (LRU Thread-Safe, sai số $< 10^{-7}$) |
| **4. Chấm điểm Ma trận Composite 4-tầng** (`compute_batch_composite_scores_matrix`) | `200 × 50` (`10,000` cặp hồ sơ đầy đủ) | `244.669 ms` | **`115.171 ms`** | **2.12x** | `@numba.njit(parallel=True, fastmath=True)` + đa luồng `numba.prange` |
| **5. Giải ghép cặp ổn định `C-Optimal`** (CSV Market) | `300 × 275` (`11,808` cặp khả thi) | `1.853 ms` | **`0.375 ms`** | **4.94x** | `_gs_candidate_proposing_numba` (`@numba.njit`) + Bảng hạng $O(1)$ `int32[:, :]` |
| **6. Giải ghép cặp ổn định `J-Optimal`** (CSV Market) | `300 × 275` (`11,808` cặp khả thi) | `7.188 ms` | **`0.598 ms`** | **12.03x** | `_gs_employer_proposing_numba` (`@numba.njit`) + Bảng hạng $O(1)$ `int32[:, :]` |
| **7. Giải ghép cặp ổn định `C-Optimal`** (Composite Profiles) | `200 × 50` (`10,000` cặp) | `2.142 ms` | **`0.275 ms`** | **7.80x** | `@dataclass(slots=True)` + Numba JIT Deferred Acceptance |
| **8. Giải ghép cặp ổn định `J-Optimal`** (Synthetic Market) | `2,000 × 400` (`800,000` cặp) | `102.400 ms` | **`7.320 ms`** | **13.99x** | Bảng tra cứu hạng phẳng `_cand_rank_arr` + Hàng đợi vòng FIFO tĩnh trong Numba |
| **9. Tổng thời gian chạy 93 bài kiểm thử** (`pytest tests/`) | `93 tests` + `48 subtests` | `27.820 s` | **`19.130 s`** | **1.45x** | Tối ưu hóa đồng bộ trên toàn bộ 6 tầng kiến trúc |

### 5.2. Bảng Đối chiếu 19/19 Tiêu chí Bất biến trên Golden Dataset (Bit-for-Bit Verification)

Kiểm chứng thực nghiệm trên bộ **Golden Dataset** của dự án (`N = 300` ứng viên từ `scripts/data/candidates.csv`, `M = 275` vị trí công việc từ `scripts/data/job_descriptions.csv` với `default_quota = 5`, `seed = 42`, `max_candidate_preferences = 30`, cùng kịch bản `Composite Profiles 200 × 50`):

| STT | Tiêu chí Kiểm định (Golden Metric) | Giá trị Trước Nâng cấp | Giá trị Sau Nâng cấp | Độ lệch ($\Delta$) | Trạng thái |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **1** | Số lượng ứng viên đầu vào ($N$) | `300` | `300` | `0` | ✅ **KHỚP 100%** |
| **2** | Số lượng vị trí tuyển dụng đầu vào ($M$) | `275` | `275` | `0` | ✅ **KHỚP 100%** |
| **3** | Tổng hạn mức tuyển dụng toàn thị trường ($\sum q_j$) | `1,375` | `1,375` | `0` | ✅ **KHỚP 100%** |
| **4** | Tổng số cặp khả thi sau lọc nhóm khuyết tật (`feasible_pairs`) | `11,808` | `11,808` | `0` | ✅ **KHỚP 100%** |
| **5** | Số cặp ghép ổn định thành công (`C-Optimal` CSV `300 × 275`) | `257` | `257` | `0` | ✅ **KHỚP 100%** |
| **6** | Số cặp ghép ổn định thành công (`J-Optimal` CSV `300 × 275`) | `257` | `257` | `0` | ✅ **KHỚP 100%** |
| **7** | Số lượng cặp chặn (`Blocking Pairs` — `C-Optimal` CSV) | `0` (`0.00%`) | `0` (`0.00%`) | `0` | ✅ **KHỚP 100%** |
| **8** | Số lượng cặp chặn (`Blocking Pairs` — `J-Optimal` CSV) | `0` (`0.00%`) | `0` (`0.00%`) | `0` | ✅ **KHỚP 100%** |
| **9** | Điểm hài lòng Nhà tuyển dụng trung bình (`Average E` — `C-Optimal`) | `0.5474` | `0.5474` | `0.0000` | ✅ **KHỚP 100%** |
| **10** | Điểm hài lòng Ứng viên trung bình (`Average C` — `C-Optimal`) | `0.9749` | `0.9749` | `0.0000` | ✅ **KHỚP 100%** |
| **11** | Tỷ lệ lấp đầy chỉ tiêu toàn thị trường (`Fill Rate` — CSV `300 × 275`) | `0.1869` (`18.69%`) | `0.1869` (`18.69%`) | `0.0000` | ✅ **KHỚP 100%** |
| **12** | Tổng số lời đề nghị phát sinh (`Proposals` — `C-Optimal` CSV) | `578` | `578` | `0` | ✅ **KHỚP 100%** |
| **13** | Tổng số lời đề nghị phát sinh (`Proposals` — `J-Optimal` CSV) | `945` | `945` | `0` | ✅ **KHỚP 100%** |
| **14** | Số ứng viên chưa được phân bổ (`Unmatched Candidates` — CSV) | `43` | `43` | `0` | ✅ **KHỚP 100%** |
| **15** | Tỷ lệ xung đột phân bổ trùng lặp (`Congestion Rate`) | `0.00%` | `0.00%` | `0.00%` | ✅ **KHỚP 100%** |
| **16** | Số cặp ghép ổn định thành công (`Composite 200 × 50`, `C-Optimal`) | `170` | `170` | `0` | ✅ **KHỚP 100%** |
| **17** | Số lượng cặp chặn (`Blocking Pairs` — `Composite 200 × 50`) | `0` (`0.00%`) | `0` (`0.00%`) | `0` | ✅ **KHỚP 100%** |
| **18** | Sai số chuẩn $L_\infty$ của Vector Embedding (`ONNX/FastEmbed` + Cache) | Baseline | $< 10^{-7}$ | $< 10^{-7}$ | ✅ **KHỚP 100%** |
| **19** | Đối chiếu toàn bộ ánh xạ `result.matches` & `job_assignments` | Tập chiếu chuẩn | Trùng khớp từng cặp ID | `0` khác biệt | ✅ **KHỚP 100%** |

---

## 6. Hướng dẫn Cài đặt, Khởi chạy & Pipeline Dữ liệu

### 6.1. Triển khai bằng Docker Compose (Khuyến nghị cho Production)

Hệ thống đóng gói bằng **Multi-stage `Dockerfile`** sử dụng trình quản lý gói **Astral `uv`** (`ghcr.io/astral-sh/uv:latest`) trên nền `python:3.12-slim` (loại bỏ hoàn toàn các gói PyTorch/CUDA cồng kềnh). File `docker-compose.yml` điều phối đồng thời 3 dịch vụ:
* **`db` (`pgvector/pgvector:pg16`):** PostgreSQL 16 tích hợp sẵn `pgvector` tại cổng `5432`.
* **`qdrant` (`qdrant/qdrant:latest`):** Vector Database hiệu năng cao mở cổng HTTP `6333` và cổng **gRPC `6334`** (`QDRANT__SERVICE__GRPC_PORT: "6334"`).
* **`web` (FastAPI Application):** Chạy Uvicorn tại cổng `8000`, kết nối bất đồng bộ tới `db` qua `postgresql+asyncpg://` và tới `qdrant` qua gRPC.

```bash
# 1. Build và khởi chạy ngầm toàn bộ 3 services (db, qdrant, web)
docker compose up --build -d

# 2. Kiểm tra trạng thái hoạt động của các containers
docker compose ps

# 3. Theo dõi log khởi động của FastAPI server
docker compose logs -f web
```

Sau khi khởi chạy:
* **Swagger UI (OpenAPI Docs):** `http://localhost:8000/docs`
* **Health Check Endpoint:** `http://localhost:8000/health`
* **Qdrant Dashboard:** `http://localhost:6333/dashboard`

### 6.2. Cài đặt Môi trường Local bằng `uv` & Cấu hình Biến Môi trường (`.env`)

Nếu phát triển hoặc chạy trực tiếp trên máy host (Windows / Linux / macOS), sử dụng **`uv`** (hoặc `pip`) để cài đặt siêu tốc:

```powershell
# Cài đặt uv (nếu chưa có)
pip install uv

# Tạo môi trường ảo Python 3.12 và kích hoạt (Windows PowerShell)
uv venv .venv --python 3.12
.\.venv\Scripts\Activate.ps1

# Cài đặt dependencies cho Backend API & Core Engine
uv pip install -r requirements.txt

# (Tùy chọn) Cài đặt thêm thư viện cho giao diện Streamlit Dashboard
uv pip install -r requirements-dashboard.txt
```

Các biến môi trường được quản lý tập trung qua lớp `AppSettings(BaseSettings)` trong `app/config.py` (tự động đọc từ tệp `.env` nếu có):

```ini
# Kết nối PostgreSQL (tự động chuẩn hóa sang postgresql+asyncpg:// và postgresql+psycopg2://)
DATABASE_URL=postgresql://avora:avora@localhost:5432/avora_matching
ASYNC_DATABASE_URL=postgresql+asyncpg://avora:avora@localhost:5432/avora_matching

# Cấu hình Qdrant Vector Database (ưu tiên giao thức gRPC cổng 6334)
QDRANT_HOST=localhost
QDRANT_HTTP_PORT=6333
QDRANT_GRPC_PORT=6334
QDRANT_PREFER_GRPC=true
QDRANT_COLLECTION_JOBS=avora_job_descriptions
QDRANT_COLLECTION_CANDIDATES=avora_candidates

# Cấu hình Embedding ONNX Runtime / FastEmbed & Bộ nhớ đệm SHA-256
# Hỗ trợ Qwen/Qwen3-Embedding-0.6B hoặc AITeamVN/Vietnamese_Embedding (1024 chiều)
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
ONNX_MODEL_PATH=
ONNX_QUANTIZATION=INT8
DYNAMIC_BATCH_SIZE=32
EMBEDDING_CACHE_MAX_SIZE=16384
VECTOR_DIM=1024
TOP_K_JOBS=10
TOP_K_CANDIDATES=20
```

### 6.3. Quy trình Chạy các Kịch bản ETL & Nạp Dữ liệu (`scripts/`)

```powershell
# 1. Chuyển đổi file Excel (.xlsx) sang CSV chuẩn UTF-8 BOM bằng Polars + Calamine
python scripts/convert_xlsx_to_csv.py input_candidates.xlsx scripts/data/candidates.csv 0

# 2. Sinh dữ liệu mô tả công việc mẫu (mặc định 100 mẫu duy nhất -> ~275 dòng gắn nhãn nhóm khuyết tật)
python scripts/generate_mock_data.py --count 100 --seed 42 --output scripts/data/job_descriptions.csv

# 3. Sinh dữ liệu ứng viên mẫu (mặc định 3,000 ứng viên phân bố đều giữa 7 nhóm khuyết tật)
python scripts/generate_mock_candidates.py --count 3000 --seed 42 --output scripts/data/candidates.csv

# 4. Kiểm tra tính hợp lệ của file CSV bằng chế độ Dry-Run (đọc qua pl.scan_csv)
python scripts/ingest_job_descriptions.py --dry-run
python scripts/ingest_candidates.py --dry-run

# 5. Nạp dữ liệu vào PostgreSQL qua FastAPI REST Endpoints
python scripts/ingest_job_descriptions.py --timeout 300
python scripts/ingest_candidates.py

# 6. (Tùy chọn) Nạp song song trực tiếp vào Qdrant qua gRPC (upload_points với parallel=4)
python scripts/ingest_job_descriptions.py --qdrant
python scripts/ingest_candidates.py --qdrant
```

### 6.4. Khởi chạy FastAPI Server (`app/main.py`) & Streamlit Dashboard (`dashboard.py`)

```powershell
# Khởi chạy FastAPI Server tại http://localhost:8000
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Khởi chạy giao diện Streamlit Dashboard tại http://localhost:8501 (hoạt động độc lập, không cần DB)
streamlit run dashboard.py --server.port 8501
```

---

## 7. Tài liệu API Endpoints & Ví dụ Sử dụng Trực tiếp `src/core/`

### 7.1. Danh mục Toàn bộ 10 API Endpoints trong `app/main.py`

Ứng dụng FastAPI được cấu hình với `default_response_class=ORJSONResponse` để tuần tự hóa JSON và mảng `numpy.ndarray` ở tốc độ C/Rust:

| Phương thức | Đường dẫn (Endpoint) | Nhóm (Tag) | Chức năng Chi tiết |
| :---: | :--- | :--- | :--- |
| `GET` | `/` | `Hệ thống` | Kiểm tra trạng thái hoạt động cơ bản của dịch vụ và thuật toán mặc định. |
| `GET` | `/health` | `Hệ thống` | Kiểm tra kết nối Database và trả về tổng số bản ghi `job_descriptions`, `candidates`. |
| `POST` | `/job-descriptions` | `Job Descriptions` | Tạo mới mô tả công việc, sinh vector embedding `1024` chiều và lưu vào `pgvector`. |
| `GET` | `/job-descriptions` | `Job Descriptions` | Truy vấn danh sách mô tả công việc (hỗ trợ phân trang `skip`, `limit` và lọc `nhom_khuyet_tat`). |
| `POST` | `/candidates` | `Candidates` | Thêm mới hồ sơ ứng viên (`ho_ten`, `nhom_khuyet_tat`) vào cơ sở dữ liệu. |
| `GET` | `/candidates` | `Candidates` | Truy vấn danh sách ứng viên (hỗ trợ phân trang `skip`, `limit` và lọc `nhom_khuyet_tat`). |
| `POST` | `/match` | `Semantic Matching` | Tìm kiếm ngữ nghĩa theo query tuyển dụng, tổng hợp điểm nhóm ($0.6 \times \text{Top}_1 + 0.4 \times \text{AvgTop}_3$) và lọc ứng viên. |
| `POST` | `/matching/stable` | `Stable Matching Theory` | Thực thi pipeline Gale–Shapley toàn diện (hỗ trợ hồ sơ Composite 6 thành phần, Vector Oxyz, Direct Track, Top-$K$). |
| `POST` | `/matching/matrix` | `Stable Matching Theory` | Thực thi Gale–Shapley trực tiếp từ hai ma trận điểm `candidate_scores` ($C$), `employer_scores` ($E$) và `job_quotas`. |
| `POST` | `/matching/database` | `Stable Matching Theory` | Trích xuất trực tiếp ứng viên và công việc từ PostgreSQL để chạy ghép cặp ổn định Gale–Shapley. |

#### Ví dụ Request & Response cho `POST /matching/stable`

**Request (`POST http://localhost:8000/matching/stable`):**
```json
{
  "optimal": "candidate",
  "min_e_score": 0.1,
  "min_c_score": 0.1,
  "top_k": 3,
  "tie_break_seed": 20240517,
  "candidates": [
    {
      "id": "cand_01",
      "ho_ten": "Nguyễn Minh An",
      "nhom_khuyet_tat": "Khiếm thính",
      "skills": ["python", "sql", "fastapi", "git"],
      "essential_access_needs": ["giao tiếp văn bản"],
      "nice_to_have_access_needs": ["làm việc linh hoạt"],
      "expected_salary_min": 12.0,
      "expected_salary_max": 18.0,
      "work_mode": "hybrid",
      "work_hours": "full-time",
      "location": "Hà Nội",
      "priority_ranking": ["career_path", "income", "distance_or_remote"],
      "experience_years": 2.5,
      "career_orientation": "backend python sql"
    }
  ],
  "jobs": [
    {
      "id": "job_backend_01",
      "title": "Lập trình viên Backend Python",
      "quota": 2,
      "filled_slots": 0,
      "skills_required": ["python", "sql", "fastapi"],
      "access_support_provided": ["giao tiếp văn bản", "làm việc linh hoạt"],
      "salary_min": 14.0,
      "salary_max": 20.0,
      "work_mode": "hybrid",
      "work_hours": "full-time",
      "location": "Hà Nội",
      "experience_required_years": 2.0,
      "career_domain": "backend python sql"
    }
  ]
}
```

**Response (`200 OK` — `ORJSONResponse`):**
```json
{
  "status": "success",
  "optimal": "candidate",
  "matches": {
    "cand_01": "job_backend_01"
  },
  "matched_pairs": [
    {
      "candidate_id": "cand_01",
      "job_id": "job_backend_01",
      "c_score": 1.0,
      "e_score": 0.9038,
      "candidate_rank": 1,
      "job_rank": 1,
      "explanation": "Kỹ năng chuyên môn phù hợp tốt với yêu cầu vị trí (75.0%); Đáp ứng 100% các điều kiện hỗ trợ tiếp cận mong muốn; Hoàn toàn tương thích về hình thức (Hybrid) và địa điểm làm việc; Khoảng lương hai bên có mức độ đồng thuận cao; Phù hợp cao với định hướng phát triển nghề nghiệp của ứng viên.",
      "e_breakdown": {
        "skill": 0.75,
        "work_mode_hours": 1.0,
        "nice_to_have_access": 1.0,
        "salary": 0.5,
        "experience": 1.0,
        "location": 1.0
      },
      "c_breakdown": {
        "income": 1.0,
        "distance_or_remote": 1.0,
        "career_path": 1.0
      }
    }
  ],
  "job_assignments": {
    "job_backend_01": ["cand_01"]
  },
  "unmatched_candidates": [],
  "unmatched_jobs": ["job_backend_01"],
  "recommendations": {
    "cand_01": [
      {
        "candidate_id": "cand_01",
        "job_id": "job_backend_01",
        "position": 1,
        "score": 1.0,
        "is_primary": True,
        "explanation": "Kỹ năng chuyên môn phù hợp tốt với yêu cầu vị trí (75.0%); Đáp ứng 100% các điều kiện hỗ trợ tiếp cận mong muốn; Hoàn toàn tương thích về hình thức (Hybrid) và địa điểm làm việc; Khoảng lương hai bên có mức độ đồng thuận cao; Phù hợp cao với định hướng phát triển nghề nghiệp của ứng viên."
      }
    ]
  },
  "metrics": {
    "total_candidates": 1,
    "total_jobs": 1,
    "total_quota": 2,
    "matched_pairs_count": 1,
    "unmatched_candidates_count": 0,
    "unmatched_jobs_count": 1,
    "blocking_pairs_count": 0,
    "blocking_pair_rate": 0.0,
    "is_stable": true,
    "candidate_satisfaction_avg": 1.0,
    "employer_satisfaction_avg": 0.9038,
    "fill_rate": 0.5,
    "proposals_count": 1,
    "average_candidate_rank": 1.0,
    "average_job_rank": 1.0,
    "unmatched_candidate_rate": 0.0,
    "congestion_rate": 0.0
  },
  "execution_time_ms": 0.85,
  "warnings": []
}
```

### 7.2. Ví dụ Sử dụng Trực tiếp Lõi Thuật toán `src/core/` (Python Quickstart Snippet)

Bạn có thể nhúng và gọi trực tiếp các hàm trong `src/core/` và `src/evaluation/` mà không cần khởi chạy FastAPI hay Database:

```python
from src.core import (
    CandidateProfile,
    EmployerWeightsConfig,
    GaleShapleyMatching,
    JobProfile,
    PreprocessConfig,
    Preprocessor,
    build_profile_from_profiles,
    generate_top_k,
    load_market_from_csv,
)
from src.core.scoring import compute_batch_composite_scores_matrix
from src.evaluation.metrics import evaluate

# ===========================================================================
# CÁCH 1: Đọc dữ liệu từ CSV Golden Dataset bằng Polars LazyFrame (scan_csv)
# ===========================================================================
e_scores, c_scores, quotas, feasible_pairs = load_market_from_csv(
    candidates_csv="scripts/data/candidates.csv",
    job_descriptions_csv="scripts/data/job_descriptions.csv",
    default_quota=5,
    seed=42,
    candidate_limit=300,
)

profile_csv = Preprocessor(
    PreprocessConfig(max_candidate_preferences=30, tie_break_seed=20_240_517)
).build_profile(e_scores, c_scores, quotas, feasible_pairs=feasible_pairs)

# Chạy Gale-Shapley Many-to-One biên dịch bằng @numba.njit (C-Optimal & J-Optimal)
result_c = GaleShapleyMatching(profile_csv, optimal="candidate").solve()
result_j = GaleShapleyMatching(profile_csv, optimal="employer").solve()

# Xuất báo cáo đánh giá toàn diện EvaluationReport
report_c = evaluate(profile_csv, result_c)
assert report_c.blocking.count == 0 and report_c.blocking.is_stable
print(
    f"[CSV Market] Matched: {result_c.matched_pair_count()}, "
    f"Blocking Pairs: {report_c.blocking.count}, "
    f"E_avg: {report_c.satisfaction.average_e:.4f}, "
    f"C_avg: {report_c.satisfaction.average_c:.4f}, "
    f"Fill Rate: {report_c.fill_rate.overall:.4f}, "
    f"Proposals (C/J): {result_c.proposals}/{result_j.proposals}"
)

# ===========================================================================
# CÁCH 2: Chấm điểm Ma trận Composite bằng Numba JIT & Gợi ý Top-K
# ===========================================================================
candidates = [
    CandidateProfile(
        candidate_id="C1",
        skills=["python", "sql", "docker"],
        essential_access_needs=["giao tiếp văn bản"],
        nice_to_have_access_needs=["giờ linh hoạt"],
        expected_salary_min=15.0,
        expected_salary_max=22.0,
        work_mode="remote",
        work_hours="flexible",
        location="TP.HCM",
        priority_ranking=["income", "distance_or_remote", "career_path"],
        experience_years=3.0,
    ),
]
jobs = [
    JobProfile(
        job_id="J1",
        title="Data & Backend Engineer",
        quota=2,
        skills_required=["python", "sql", "polars"],
        access_support_provided=["giao tiếp văn bản", "giờ linh hoạt"],
        salary_min=18.0,
        salary_max=25.0,
        work_mode="remote",
        work_hours="flexible",
        location="Hà Nội",
        experience_required_years=2.0,
    ),
]

# Tính ma trận điểm E[N, M] và C[N, M] đa luồng qua compute_composite_score_matrices_numba
e_mat, c_mat, skill_mat = compute_batch_composite_scores_matrix(
    candidates, jobs, EmployerWeightsConfig()
)

# Hoặc dựng trực tiếp PreferenceProfile kèm giải thích từng cặp và sinh Top-K
profile_comp = build_profile_from_profiles(candidates, jobs)
match_res = GaleShapleyMatching(profile_comp, optimal="candidate").solve()
top_k_recs = generate_top_k(profile_comp, match_res, k=3)
print("Gợi ý số 1 cho C1:", top_k_recs["C1"][0])
```

---

## 8. Hướng dẫn Kiểm thử & Đánh giá Hiệu năng (Testing & Benchmarking)

### 8.1. Chạy Toàn bộ Bộ Kiểm thử Tự động (`93 passed, 48 subtests passed`)

Sử dụng `pytest` để thực thi toàn bộ 6 tệp kiểm thử trong thư mục `tests/`:

```powershell
# Chạy toàn bộ bộ kiểm thử với thông tin tóm tắt
python -m pytest tests/ -q

# Chạy chi tiết từng bài kiểm thử
python -m pytest tests/ -v
```

Kết quả chuẩn cần đạt: **`93 passed, 48 subtests passed`**. Ý nghĩa của từng tệp kiểm thử:

| Tệp Kiểm thử | Số lượng Test | Nội dung Kiểm chứng Chuyên sâu |
| :--- | :---: | :--- |
| `tests/test_matching.py` | `31 tests` | Kiểm thử chuẩn hóa ma trận thưa/đặc, lọc Hard Constraints, cắt danh sách ưu tiên (`truncate_preferences`), đối ngẫu `C-Optimal` vs `J-Optimal`, bẫy tham lam (`GREEDY_TRAP`), hạn mức Many-to-One, Top-$K$ (`generate_top_k` & `iterative_clearance_rounds`) và tích hợp `load_market_from_csv`. |
| `tests/test_enhanced_matching.py` | `23 tests` | Kiểm thử mô hình hồ sơ `@dataclass(slots=True)`, thang trọng số ưu tiên `0.5 / 0.3 / 0.2`, công thức 6 thành phần điểm $E$ và 3 thành phần điểm $C$, phá hòa theo điểm kỹ năng (`use_skill_tie_break`), sinh giải thích `explain_match`, chỉ số `Recall@N`, `Precision@K`, `nDCG@K` và kiểm toán công bằng `audit_fairness`. |
| `tests/test_app_stable_matching.py` | `16 tests` | Kiểm thử đại số không gian Vector Oxyz (`Vector3D`), hàm phá hòa `blake2b` (`deterministic_tie_breaker`), thuật toán Gale–Shapley trong `app/matching.py`, và gọi trực tiếp các FastAPI endpoints `/matching/stable`, `/matching/matrix` qua `TestClient`. |
| `tests/test_regression_fixes.py` | `9 tests` (`32 subtests`) | Kiểm thử hồi quy và vét cạn cho các trường hợp biên: đảm bảo `Employer-Proposing` không bao giờ gán ứng viên vào việc ngoài nguyện vọng khi cắt ngắn `L`, xử lý vị trí có `quota = 0`, trừ đúng chỉ tiêu `Direct Track` + `filled_slots`, và trả mã lỗi **HTTP 422** khi dữ liệu Direct Track không hợp lệ. |
| `tests/test_exhaustive_stability_checker.py` | `7 tests` (`16 subtests`) | Lớp kiểm định độc lập thứ hai (`independent_find_blocking_pairs` viết tách biệt 100% khỏi mã nguồn engine), đối chiếu kết quả vét cạn trên hàng loạt thị trường ngẫu nhiên, thị trường đồng điểm hoàn toàn (100% ties) và thị trường có `Direct Track` / `filled_slots`. |
| `tests/test_dashboard.py` | `7 tests` | Kiểm thử bộ đọc ma trận văn bản, bộ phân tích quota, tính tất định của `build_synthetic_market` và chạy **AppTest Headless Smoke Test** (`streamlit.testing.v1.AppTest`) xác nhận giao diện `dashboard.py` render thành công với `Blocking pairs = 0`. |

### 8.2. Chạy CLI Benchmark & Đối chiếu với Greedy Baseline (`src/evaluation/benchmark.py`)

Module `src/evaluation/benchmark.py` cung cấp công cụ dòng lệnh để đo đạc trực tiếp tốc độ và chất lượng ghép cặp giữa `GaleShapley (C-optimal)`, `GaleShapley (J-optimal)` và `Greedy Baseline`:

```powershell
# Chạy benchmark trên thị trường mô phỏng 2,000 ứng viên x 400 vị trí (800,000 cặp) và in bảng Markdown
python -m src.evaluation.benchmark --candidates 2000 --jobs 400 --avg-quota 5 --density 0.4 --max-preferences 40 --seed 0

# Xuất kết quả benchmark dưới định dạng JSON phục vụ CI/CD hoặc tự động hóa báo cáo
python -m src.evaluation.benchmark --candidates 2000 --jobs 400 --json
```

### 8.3. Tài liệu Kỹ thuật Tham chiếu

Để tra cứu sâu hơn về lược đồ cơ sở dữ liệu, luồng xử lý nội bộ của từng hàm và thiết kế chi tiết cấp module, vui lòng xem thêm tại:
* 📄 **[Tài liệu Kỹ thuật Chi tiết (`document-tech.md`)](./document-tech.md)**
>>>>>>> 2ee0720 (update README.md)
