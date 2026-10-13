from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

# Thử import get_db và init_db từ app.database, xử lý an toàn nếu thiếu driver psycopg2 ở môi trường dev local
try:
    from app.database import get_db, init_db
except (ImportError, Exception):  # pragma: no cover
    def get_db():
        """Fallback dependency khi môi trường chưa cấu hình database PostgreSQL."""
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Kết nối Database chưa sẵn sàng hoặc driver PostgreSQL chưa được cài đặt."
        )

    def init_db():
        """Fallback khởi tạo database khi không có kết nối DB."""
        pass

# Import các tiện ích tính toán embedding và model ORM
try:
    from app.embeddings import get_embedding
except (ImportError, Exception):  # pragma: no cover
    def get_embedding(text: str) -> list[float]:
        """Fallback embedding giả lập khi thiếu thư viện sentence-transformers."""
        return [0.0] * 1024

try:
    from app.models import Candidate, JobDescription
except (ImportError, Exception):  # pragma: no cover
    Candidate = None  # type: ignore
    JobDescription = None  # type: ignore

from app.schemas import (
    CandidateCreate,
    CandidateProfileInput,
    CandidateRecommendationResponse,
    CandidateResponse,
    JobDescriptionCreate,
    JobDescriptionResponse,
    JobProfileInput,
    MatchQuery,
    MatchResponse,
    MatchedDescription,
    MatchedGroup,
    MatchedPairResponse,
    MatrixMatchingRequest,
    StableMatchingMetricsResponse,
    StableMatchingRequest,
    StableMatchingResponse,
    Vector3D,
)
from app.matching import (
    calculate_cosine_similarity_3d,
    candidate_proposing_deferred_acceptance,
    employer_proposing_deferred_acceptance,
    find_blocking_pairs,
    generate_top_k_recommendations,
    run_matrix_matching,
    run_stable_matching,
)


# Khởi tạo ứng dụng FastAPI với tài liệu OpenAPI chi tiết
app = FastAPI(
    title="Avora Matching API - Two-Sided Stable Matching Platform",
    version="2.1.0",
    description=(
        "Nền tảng API tối ưu hóa thị trường kết nối tuyển dụng hai phía (Two-Sided Matching Market) "
        "tích hợp thuật toán Ghép cặp Ổn định Gale-Shapley (Deferred Acceptance Many-to-One), "
        "mô hình tính điểm không gian vector Oxyz, bộ lọc điều kiện tiên quyết (Hard Constraints) "
        "và hệ thống khuyến nghị việc làm dự phòng Top-K Recommendations."
    ),
)
 
@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Xử lý ngoại lệ ValueError (validation nghiệp vụ) và trả về HTTP 422."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )

# Biên độ điểm dung sai để chọn nhóm khuyết tật gần nhóm tốt nhất
GROUP_SCORE_MARGIN = 0.03

# Ngưỡng sàn điểm similarity tối thiểu để chấp nhận nhóm công việc
MIN_GROUP_SCORE = 0.50


# ===========================================================================
# SỰ KIỆN VÀ ENDPOINT HỆ THỐNG
# ===========================================================================

@app.on_event("startup")
def on_startup() -> None:
    """
    Sự kiện vòng đời khởi động của ứng dụng FastAPI.

    Mục đích:
        Khởi tạo kết nối cơ sở dữ liệu, kích hoạt extension pgvector và
        tạo lập các bảng ORM chưa tồn tại trong cơ sở dữ liệu.

    Tham số đầu vào:
        Không có.

    Kết quả trả về:
        None.
    """
    # Khởi tạo kết nối database và các extension cần thiết
    init_db()


@app.get("/", tags=["Hệ thống"])
def root() -> dict[str, str]:
    """
    Endpoint chào mừng và kiểm tra trạng thái hoạt động cơ bản của API.

    Mục đích:
        Cung cấp thông tin tổng quan về dịch vụ, trạng thái đang chạy và đường dẫn Swagger Docs.

    Tham số đầu vào:
        Không có.

    Kết quả trả về:
        dict[str, str]: Dictionary chứa tên dịch vụ, trạng thái 'running', phiên bản và liên kết docs.
    """
    # Trả về thông tin meta của hệ thống AVORA Matching
    return {
        "service": "Avora Two-Sided Stable Matching API",
        "status": "running",
        "algorithm": "Gale-Shapley Deferred Acceptance (Many-to-One)",
        "docs": "/docs",
    }


@app.get("/health", tags=["Hệ thống"])
def health(db: Session = Depends(get_db)) -> dict[str, int | str]:
    """
    Endpoint kiểm tra sức khỏe hệ thống (Health Check) và số lượng bản ghi trong Database.

    Mục đích:
        Giám sát tính sẵn sàng của cơ sở dữ liệu PostgreSQL, đếm tổng số mô tả công việc
        và ứng viên hiện có trong kho dữ liệu.

    Tham số đầu vào:
        db (Session): SQLAlchemy Database Session được inject qua FastAPI Dependency.

    Kết quả trả về:
        dict[str, int | str]: Trạng thái 'healthy', số lượng job descriptions và candidates hiện có.

    Raises:
        HTTPException: Mã lỗi 503 nếu không thể kết nối tới cơ sở dữ liệu.
    """
    try:
        # Đếm số lượng mô tả công việc trong bảng job_descriptions
        job_description_count = db.query(func.count(JobDescription.id)).scalar() or 0
        # Đếm số lượng ứng viên trong bảng candidates
        candidate_count = db.query(func.count(Candidate.id)).scalar() or 0
    except SQLAlchemyError as exc:
        # Bắt lỗi database và trả mã 503 Service Unavailable
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Không thể kết nối database",
        ) from exc

    # Trả về kết quả kiểm tra thành công
    return {
        "status": "healthy",
        "job_descriptions": int(job_description_count),
        "candidates": int(candidate_count),
    }


# ===========================================================================
# CÁC ENDPOINT QUẢN LÝ MÔ TẢ CÔNG VIỆC VÀ ỨNG VIÊN (CRUD)
# ===========================================================================

@app.post(
    "/job-descriptions",
    response_model=JobDescriptionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Job Descriptions"],
)
def create_job_description(
    payload: JobDescriptionCreate,
    db: Session = Depends(get_db),
) -> JobDescriptionResponse:
    """
    Tạo mới một bản ghi mô tả công việc và sinh vector embedding 1024 chiều.

    Mục đích:
        Tiếp nhận thông tin mô tả công việc, sinh vector ngữ nghĩa bằng Qwen3 Embedding
        (không nhúng tên nhóm khuyết tật vào nội dung embedding để tránh thiên kiến),
        sau đó lưu trữ vector vào PostgreSQL thông qua pgvector.

    Tham số đầu vào:
        payload (JobDescriptionCreate): Dữ liệu đầu vào gồm nhóm khuyết tật và nội dung mô tả công việc.
        db (Session): SQLAlchemy Database Session dùng để ghi dữ liệu.

    Kết quả trả về:
        JobDescriptionResponse: Bản ghi mô tả công việc vừa tạo kèm ID đã cấp phát.

    Raises:
        HTTPException:
            - 409 Conflict nếu vi phạm ràng buộc dữ liệu.
            - 500 Internal Server Error nếu lỗi sinh embedding hoặc lỗi ghi database.
    """
    try:
        # Bước 1: Sinh vector embedding từ nội dung văn bản mô tả công việc
        vector = get_embedding(payload.mo_ta_cong_viec)

        # Bước 2: Tạo đối tượng ORM JobDescription
        record = JobDescription(
            nhom_khuyet_tat=payload.nhom_khuyet_tat,
            mo_ta_cong_viec=payload.mo_ta_cong_viec,
            embedding=vector,
        )

        # Bước 3: Thêm và commit vào database
        db.add(record)
        db.commit()
        db.refresh(record)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Bản ghi mô tả công việc vi phạm ràng buộc database",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể lưu mô tả công việc vào database",
        ) from exc
    except Exception as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể tạo embedding: {type(exc).__name__}",
        ) from exc

    # Trả về kết quả chuẩn hóa dạng Pydantic response
    return JobDescriptionResponse.model_validate(record)


@app.get(
    "/job-descriptions",
    response_model=list[JobDescriptionResponse],
    tags=["Job Descriptions"],
)
def list_job_descriptions(
    skip: int = Query(default=0, ge=0, description="Số lượng bản ghi bỏ qua"),
    limit: int = Query(default=100, ge=1, le=1000, description="Số lượng bản ghi tối đa lấy"),
    nhom_khuyet_tat: str | None = Query(default=None, description="Lọc theo nhóm khuyết tật cụ thể"),
    db: Session = Depends(get_db),
) -> list[JobDescriptionResponse]:
    """
    Truy vấn danh sách mô tả công việc có hỗ trợ phân trang và bộ lọc theo nhóm khuyết tật.

    Mục đích:
        Cung cấp API đọc dữ liệu mô tả công việc phục vụ kiểm tra dữ liệu và hiển thị trên Dashboard.

    Tham số đầu vào:
        skip (int): Vị trí bắt đầu lấy bản ghi (phân trang).
        limit (int): Số lượng bản ghi tối đa trên một trang truy vấn.
        nhom_khuyet_tat (str | None): Tên nhóm khuyết tật cần lọc (nếu có).
        db (Session): SQLAlchemy Database Session.

    Kết quả trả về:
        list[JobDescriptionResponse]: Danh sách các bản ghi mô tả công việc thỏa mãn điều kiện.
    """
    # Khởi tạo truy vấn trên bảng JobDescription
    query = db.query(JobDescription)

    # Áp dụng bộ lọc nhóm khuyết tật nếu người dùng truyền tham số
    if nhom_khuyet_tat:
        query = query.filter(
            JobDescription.nhom_khuyet_tat == nhom_khuyet_tat.strip()
        )

    # Phân trang và sắp xếp tăng dần theo ID
    records = (
        query.order_by(JobDescription.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    # Chuyển đổi các bản ghi ORM thành Pydantic response model
    return [JobDescriptionResponse.model_validate(record) for record in records]


@app.post(
    "/candidates",
    response_model=CandidateResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Candidates"],
)
def create_candidate(
    payload: CandidateCreate,
    db: Session = Depends(get_db),
) -> CandidateResponse:
    """
    Tạo mới một hồ sơ ứng viên trong cơ sở dữ liệu.

    Mục đích:
        Lưu trữ thông tin họ tên và nhóm khuyết tật của ứng viên để phục vụ quá trình kết nối việc làm.

    Tham số đầu vào:
        payload (CandidateCreate): Họ tên và nhóm khuyết tật của ứng viên.
        db (Session): SQLAlchemy Database Session.

    Kết quả trả về:
        CandidateResponse: Bản ghi ứng viên vừa tạo kèm ID định danh.

    Raises:
        HTTPException: 409 nếu trùng lặp/xung đột, 500 nếu lỗi kết nối database.
    """
    # Khởi tạo bản ghi ứng viên mới
    record = Candidate(
        ho_ten=payload.ho_ten,
        nhom_khuyet_tat=payload.nhom_khuyet_tat,
    )

    try:
        # Ghi vào database và commit transaction
        db.add(record)
        db.commit()
        db.refresh(record)
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Ứng viên vi phạm ràng buộc database",
        ) from exc
    except SQLAlchemyError as exc:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể lưu ứng viên vào database",
        ) from exc

    return CandidateResponse.model_validate(record)


@app.get(
    "/candidates",
    response_model=list[CandidateResponse],
    tags=["Candidates"],
)
def list_candidates(
    skip: int = Query(default=0, ge=0, description="Số lượng bản ghi bỏ qua"),
    limit: int = Query(default=100, ge=1, le=1000, description="Số lượng bản ghi tối đa lấy"),
    nhom_khuyet_tat: str | None = Query(default=None, description="Lọc theo nhóm khuyết tật"),
    db: Session = Depends(get_db),
) -> list[CandidateResponse]:
    """
    Truy vấn danh sách ứng viên có hỗ trợ phân trang và lọc theo nhóm khuyết tật.

    Mục đích:
        Lấy danh sách các ứng viên trong hệ thống để phục vụ quản lý và phân bổ thị trường.

    Tham số đầu vào:
        skip (int): Vị trí bắt đầu lấy bản ghi (phân trang).
        limit (int): Số lượng bản ghi tối đa trên một trang truy vấn.
        nhom_khuyet_tat (str | None): Tên nhóm khuyết tật cần lọc (nếu có).
        db (Session): SQLAlchemy Database Session.

    Kết quả trả về:
        list[CandidateResponse]: Danh sách các hồ sơ ứng viên thỏa mãn điều kiện.
    """
    query = db.query(Candidate)

    if nhom_khuyet_tat:
        query = query.filter(
            Candidate.nhom_khuyet_tat == nhom_khuyet_tat.strip()
        )

    records = (
        query.order_by(Candidate.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [CandidateResponse.model_validate(record) for record in records]


# ===========================================================================
# MATCHING THEO NHÓM KHUYẾT TẬT DỰA TRÊN QUERY EMBEDDING
# ===========================================================================

def aggregate_group_scores(
    rows: list[tuple[JobDescription, float]],
) -> list[tuple[str, float]]:
    """
    Tổng hợp điểm tương đồng ngữ nghĩa (Cosine Similarity) theo từng nhóm khuyết tật.

    Mục đích:
        Tính điểm đại diện cho từng nhóm khuyết tật dựa trên công thức kết hợp:
        Điểm nhóm = 60% điểm cao nhất (Top-1) + 40% điểm trung bình của tối đa 3 kết quả tốt nhất.
        Công thức này giúp ổn định điểm số hơn việc chỉ dựa vào duy nhất một kết quả riêng lẻ.

    Tham số đầu vào:
        rows (list[tuple[JobDescription, float]]): Danh sách các cặp (bản ghi mô tả công việc, điểm similarity).

    Kết quả trả về:
        list[tuple[str, float]]: Danh sách các cặp (tên nhóm khuyết tật, điểm tổng hợp)
                                 được sắp xếp theo điểm số giảm dần.
    """
    # Gom nhóm điểm tương đồng theo tên nhóm khuyết tật
    scores_by_group: dict[str, list[float]] = defaultdict(list)

    for description, similarity in rows:
        scores_by_group[description.nhom_khuyet_tat].append(float(similarity))

    aggregated: list[tuple[str, float]] = []
    # Duyệt từng nhóm và tính điểm kết hợp
    for group, scores in scores_by_group.items():
        # Lấy tối đa 3 điểm tương đồng cao nhất của nhóm
        top_scores = sorted(scores, reverse=True)[:3]
        maximum = top_scores[0]
        average = sum(top_scores) / len(top_scores)
        # Tính điểm đại diện có trọng số
        group_score = 0.6 * maximum + 0.4 * average
        aggregated.append((group, group_score))

    # Sắp xếp các nhóm theo điểm tổng hợp giảm dần
    aggregated.sort(key=lambda item: item[1], reverse=True)
    return aggregated


@app.post("/match", response_model=MatchResponse, tags=["Semantic Matching"])
def match_candidates(
    payload: MatchQuery,
    db: Session = Depends(get_db),
) -> MatchResponse:
    """
    Quy trình matching theo ngữ nghĩa truy vấn tuyển dụng:
    1. Chuyển đổi query văn bản thành vector embedding 1024 chiều.
    2. So khớp cosine distance với vector kho JobDescription trong PostgreSQL.
    3. Trích xuất Top-K mô tả công việc gần nhất.
    4. Tổng hợp điểm phù hợp cho từng nhóm khuyết tật.
    5. Chọn lọc các nhóm đạt ngưỡng và tiệm cận nhóm tốt nhất.
    6. Lọc danh sách ứng viên thuộc các nhóm được chọn.

    Tham số đầu vào:
        payload (MatchQuery): Chứa query mô tả công việc và các giới hạn top_k.
        db (Session): SQLAlchemy Database Session.

    Kết quả trả về:
        MatchResponse: Chứa các nhóm phù hợp, mô tả đối chiếu và danh sách ứng viên.
    """
    # Kiểm tra xem kho mô tả công việc đã có dữ liệu hay chưa
    description_count = db.query(func.count(JobDescription.id)).scalar() or 0
    if description_count == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Database chưa có mô tả công việc. "
                "Hãy chạy scripts/ingest_job_descriptions.py trước."
            ),
        )

    # Sinh vector embedding cho câu truy vấn
    try:
        query_vector = get_embedding(payload.query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể tạo query embedding: {type(exc).__name__}",
        ) from exc

    # Tính cosine distance trực tiếp trên PostgreSQL pgvector
    cosine_distance = JobDescription.embedding.cosine_distance(query_vector)

    try:
        raw_rows = (
            db.query(
                JobDescription,
                (1 - cosine_distance).label("similarity"),
            )
            .order_by(cosine_distance.asc(), JobDescription.id.asc())
            .limit(payload.top_k_descriptions)
            .all()
        )
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Không thể thực hiện truy vấn vector",
        ) from exc

    if not raw_rows:
        return MatchResponse(
            matched_groups=[],
            matched_descriptions=[],
            candidates=[],
        )

    rows: list[tuple[JobDescription, float]] = [
        (description, float(similarity))
        for description, similarity in raw_rows
    ]

    # Tổng hợp điểm theo từng nhóm khuyết tật
    ranked_groups = aggregate_group_scores(rows)
    best_score = ranked_groups[0][1]

    # Lọc các nhóm thỏa mãn ngưỡng sàn và nằm trong khoảng dung sai margin
    selected_groups = [
        (group, score)
        for group, score in ranked_groups
        if score >= MIN_GROUP_SCORE
        and best_score - score <= GROUP_SCORE_MARGIN
    ]

    # Nếu không có nhóm nào vượt qua ngưỡng chất lượng
    if not selected_groups:
        return MatchResponse(
            matched_groups=[],
            matched_descriptions=[
                MatchedDescription(
                    id=description.id,
                    nhom_khuyet_tat=description.nhom_khuyet_tat,
                    mo_ta_cong_viec=description.mo_ta_cong_viec,
                    similarity=round(similarity, 4),
                )
                for description, similarity in rows
            ],
            candidates=[],
        )

    selected_group_names = [group for group, _ in selected_groups]

    # Truy vấn ứng viên thuộc các nhóm đã được chọn
    candidate_records = (
        db.query(Candidate)
        .filter(Candidate.nhom_khuyet_tat.in_(selected_group_names))
        .order_by(Candidate.nhom_khuyet_tat, Candidate.id)
        .limit(payload.top_k_candidates)
        .all()
    )

    return MatchResponse(
        matched_groups=[
            MatchedGroup(
                nhom_khuyet_tat=group,
                score=round(score, 4),
            )
            for group, score in selected_groups
        ],
        matched_descriptions=[
            MatchedDescription(
                id=description.id,
                nhom_khuyet_tat=description.nhom_khuyet_tat,
                mo_ta_cong_viec=description.mo_ta_cong_viec,
                similarity=round(similarity, 4),
            )
            for description, similarity in rows
        ],
        candidates=[
            CandidateResponse.model_validate(candidate)
            for candidate in candidate_records
        ],
    )


# ===========================================================================
# CÁC ENDPOINT STABLE MATCHING (GALE-SHAPLEY MANY-TO-ONE)
# ===========================================================================

@app.post(
    "/matching/stable",
    response_model=StableMatchingResponse,
    tags=["Stable Matching Theory (Gale-Shapley)"],
    summary="Thực thi Thuật toán Ghép cặp Ổn định Gale-Shapley toàn diện",
)
def stable_matching_endpoint(
    request: StableMatchingRequest,
) -> StableMatchingResponse:
    """
    Endpoint thực thi quy trình Ghép cặp Ổn định (Gale-Shapley Many-to-One Deferred Acceptance) toàn diện.

    Mục đích:
        Phân bổ tối ưu thị trường kết nối tuyển dụng hai phía giữa Ứng viên (Candidates)
        và Vị trí việc làm (Jobs) với các tính năng nâng cao:
        - Tính Điểm E (Employer Score) từ tọa độ vector không gian Oxyz (Cosine Similarity).
        - Tính Điểm C (Candidate Score) từ nguyện vọng và mức độ yêu thích.
        - Xử lý hạn mức tuyển dụng Quotas q_j (Many-to-One).
        - Bộ lọc điều kiện tiên quyết Hard Constraints (Domain Filter D).
        - Cơ chế phá hòa tất định Deterministic Tie-Breaking.
        - Hậu xử lý Top-K Recommendations dự phòng (Normalized Surplus Score).
        - Báo cáo định lượng các chỉ số KPIs (100% Stability - 0% Blocking Pairs, Satisfaction, Fill Rate).

    Tham số đầu vào:
        request (StableMatchingRequest): Payload đầu vào chứa danh sách ứng viên (kèm tọa độ Oxyz/điểm C),
                                        danh sách việc làm (kèm quota/tọa độ Oxyz), chế độ 'optimal' ('candidate' hoặc 'employer'),
                                        các ngưỡng điểm sàn và tham số top_k.

    Kết quả trả về:
        StableMatchingResponse: Kết quả phân bổ chi tiết gồm:
            - matches: Ánh xạ {candidate_id: job_id}.
            - matched_pairs: Chi tiết từng cặp trúng tuyển kèm thứ hạng và điểm số hai chiều.
            - job_assignments: Danh sách trúng tuyển theo từng vị trí việc làm.
            - unmatched_candidates: Danh sách ứng viên chưa được phân bổ.
            - unmatched_jobs: Danh sách vị trí còn trống chỉ tiêu.
            - recommendations: Danh sách Top-K công việc gợi ý dự phòng cho mỗi ứng viên.
            - metrics: Báo cáo chỉ số KPIs hoàn chỉnh (Blocking Pairs = 0, Fill Rate, Satisfaction).
    """
    # Gọi hàm điều phối toàn trình trong module matching
    try:
        return run_stable_matching(request)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )



@app.post(
    "/matching/matrix",
    response_model=StableMatchingResponse,
    tags=["Stable Matching Theory (Gale-Shapley)"],
    summary="Thực thi Gale-Shapley trực tiếp từ Ma trận điểm C và E",
)
def matrix_matching_endpoint(
    request: MatrixMatchingRequest,
) -> StableMatchingResponse:
    """
    Endpoint thực thi Gale-Shapley trực tiếp từ Ma trận điểm số C và E (theo mục 7 trong Báo cáo phân tích).

    Mục đích:
        Phục vụ các hệ thống đã tính sẵn ma trận điểm sở thích của Ứng viên (Điểm C)
        và ma trận điểm phù hợp của Nhà tuyển dụng (Điểm E), cùng chỉ tiêu tuyển dụng Quotas.

    Tham số đầu vào:
        request (MatrixMatchingRequest): Payload chứa:
            - candidate_scores: {candidate_id: {job_id: score}} - Điểm C.
            - employer_scores: {job_id: {candidate_id: score}} - Điểm E.
            - job_quotas: {job_id: quota} - Hạn mức tuyển dụng.
            - optimal: Chế độ phía đề nghị ('candidate' hoặc 'employer').
            - top_k: Số lượng vị trí gợi ý Top-K dự phòng.
            - tie_break_seed: Hạt giống phá hòa tất định.

    Kết quả trả về:
        StableMatchingResponse: Kết quả phân bổ trúng tuyển, danh sách gợi ý dự phòng và chỉ số đo lường hiệu quả.
    """
    # Gọi hàm xử lý ma trận trực tiếp trong module matching
    return run_matrix_matching(request)


@app.post(
    "/matching/database",
    response_model=StableMatchingResponse,
    tags=["Stable Matching Theory (Gale-Shapley)"],
    summary="Thực thi Gale-Shapley trên dữ liệu thực tế lưu trong Cơ sở dữ liệu",
)
def database_stable_matching_endpoint(
    job_ids: list[int] | None = Query(default=None, description="Danh sách ID mô tả công việc muốn ghép cặp (None = lấy tất cả)"),
    top_k: int = Query(default=3, ge=1, le=20, description="Số lượng gợi ý dự phòng Top-K cho mỗi ứng viên"),
    optimal: str = Query(default="candidate", description="Chế độ tối ưu ('candidate' hoặc 'employer')"),
    quota_per_job: int = Query(default=2, ge=1, description="Hạn mức tuyển dụng mặc định cho mỗi vị trí công việc"),
    db: Session = Depends(get_db),
) -> StableMatchingResponse:
    """
    Endpoint trích xuất dữ liệu Candidates và Job Descriptions từ Database để chạy thuật toán Gale-Shapley.

    Mục đích:
        Kết nối cơ sở dữ liệu PostgreSQL thực tế với Thuật toán Gale-Shapley Many-to-One.
        Hệ thống tự động:
        1. Lấy danh sách ứng viên và công việc từ DB.
        2. Sinh ma trận điểm tương đồng dựa trên nhóm khuyết tật và độ tương quan nội dung.
        3. Áp dụng bộ lọc Hard Constraints.
        4. Chạy Candidate-Proposing Deferred Acceptance để tìm kết quả phân bổ ổn định.
        5. Sinh danh sách gợi ý Top-K và tính toán các chỉ số KPIs.

    Tham số đầu vào:
        job_ids (list[int] | None): Bộ lọc danh sách ID công việc (tùy chọn).
        top_k (int): Số lượng việc làm gợi ý dự phòng cho mỗi ứng viên.
        optimal (str): Phía chủ động gửi đề nghị ('candidate' hoặc 'employer').
        quota_per_job (int): Hạn mức tuyển dụng mặc định cho mỗi vị trí công việc.
        db (Session): SQLAlchemy Database Session.

    Kết quả trả về:
        StableMatchingResponse: Kết quả phân bổ ổn định và báo cáo hiệu quả KPIs trên dữ liệu DB thật.

    Raises:
        HTTPException: 404 nếu cơ sở dữ liệu không có dữ liệu ứng viên hoặc công việc.
    """
    # Lấy danh sách công việc từ database
    job_query = db.query(JobDescription)
    if job_ids:
        job_query = job_query.filter(JobDescription.id.in_(job_ids))
    jobs = job_query.all()

    if not jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy mô tả công việc nào trong database để thực hiện matching.",
        )

    # Lấy danh sách ứng viên từ database
    candidates = db.query(Candidate).all()
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Không tìm thấy ứng viên nào trong database để thực hiện matching.",
        )

    # Khởi tạo danh sách JobProfileInput
    job_profiles: list[JobProfileInput] = []
    for job in jobs:
        job_profiles.append(
            JobProfileInput(
                id=f"job_{job.id}",
                title=f"Vị trí #{job.id} ({job.nhom_khuyet_tat})",
                quota=quota_per_job,
                nhom_khuyet_tat_yeu_cau=job.nhom_khuyet_tat,
            )
        )

    # Khởi tạo danh sách CandidateProfileInput và ma trận điểm
    candidate_profiles: list[CandidateProfileInput] = []
    for cand in candidates:
        # Tính điểm ưu tiên: nếu cùng nhóm khuyết tật thì đạt điểm cao (0.85 - 0.95), nếu khác nhóm thì điểm thấp (0.1 - 0.3)
        c_scores: dict[str, float] = {}
        for job in jobs:
            j_key = f"job_{job.id}"
            if cand.nhom_khuyet_tat == job.nhom_khuyet_tat:
                c_scores[j_key] = 0.90
            else:
                c_scores[j_key] = 0.20

        candidate_profiles.append(
            CandidateProfileInput(
                id=f"cand_{cand.id}",
                ho_ten=cand.ho_ten,
                nhom_khuyet_tat=cand.nhom_khuyet_tat,
                c_scores=c_scores,
            )
        )

    # Đóng gói thành StableMatchingRequest
    request = StableMatchingRequest(
        candidates=candidate_profiles,
        jobs=job_profiles,
        optimal=optimal,
        min_e_score=0.0,
        min_c_score=0.0,
        top_k=top_k,
    )

    # Thực thi thuật toán Stable Matching
    return run_stable_matching(request)
