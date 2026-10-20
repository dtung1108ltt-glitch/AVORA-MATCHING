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
