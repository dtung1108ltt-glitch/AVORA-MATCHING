from __future__ import annotations

from collections import defaultdict

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db, init_db
from app.embeddings import get_embedding
from app.models import Candidate, JobDescription
from app.schemas import (
    CandidateCreate,
    CandidateResponse,
    JobDescriptionCreate,
    JobDescriptionResponse,
    MatchQuery,
    MatchResponse,
    MatchedDescription,
    MatchedGroup,
)


app = FastAPI(
    title="Avora Matching API",
    version="2.0.0",
    description=(
        "Matching query tuyển dụng với kho mô tả công việc, xác định nhóm "
        "khuyết tật phù hợp và trả danh sách ứng viên thuộc nhóm đó."
    ),
)

# Một nhóm chỉ được chọn khi điểm không thấp hơn nhóm tốt nhất quá khoảng này.
GROUP_SCORE_MARGIN = 0.03

# Ngưỡng mặc định ban đầu. Cần hiệu chỉnh bằng bộ query kiểm thử thực tế.
MIN_GROUP_SCORE = 0.50


@app.on_event("startup")
def on_startup() -> None:
    """Bật pgvector và tạo các bảng chưa tồn tại khi API khởi động."""
    init_db()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Avora Matching API",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
def health(db: Session = Depends(get_db)) -> dict[str, int | str]:
    """Kiểm tra API và database, đồng thời trả số bản ghi hiện có."""
    try:
        job_description_count = db.query(func.count(JobDescription.id)).scalar() or 0
        candidate_count = db.query(func.count(Candidate.id)).scalar() or 0
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Không thể kết nối database",
        ) from exc

    return {
        "status": "healthy",
        "job_descriptions": int(job_description_count),
        "candidates": int(candidate_count),
    }


# ---------------------------------------------------------------------------
# JOB DESCRIPTIONS
# ---------------------------------------------------------------------------

@app.post(
    "/job-descriptions",
    response_model=JobDescriptionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_job_description(
    payload: JobDescriptionCreate,
    db: Session = Depends(get_db),
) -> JobDescriptionResponse:
    """
    Tạo embedding từ riêng mo_ta_cong_viec rồi lưu cùng nhãn nhóm khuyết tật.

    Không đưa nhom_khuyet_tat vào nội dung embedding.
    """
    try:
        vector = get_embedding(
            payload.mo_ta_cong_viec,
        )

        record = JobDescription(
            nhom_khuyet_tat=payload.nhom_khuyet_tat,
            mo_ta_cong_viec=payload.mo_ta_cong_viec,
            embedding=vector,
        )

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

    return JobDescriptionResponse.model_validate(record)


@app.get(
    "/job-descriptions",
    response_model=list[JobDescriptionResponse],
)
def list_job_descriptions(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    nhom_khuyet_tat: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[JobDescriptionResponse]:
    """Liệt kê mô tả công việc, hỗ trợ phân trang và lọc theo nhóm."""
    query = db.query(JobDescription)

    if nhom_khuyet_tat:
        query = query.filter(
            JobDescription.nhom_khuyet_tat == nhom_khuyet_tat.strip()
        )

    records = (
        query.order_by(JobDescription.id)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [JobDescriptionResponse.model_validate(record) for record in records]


# ---------------------------------------------------------------------------
# CANDIDATES
# ---------------------------------------------------------------------------

@app.post(
    "/candidates",
    response_model=CandidateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_candidate(
    payload: CandidateCreate,
    db: Session = Depends(get_db),
) -> CandidateResponse:
    """Lưu ứng viên; không tạo embedding cho ứng viên."""
    record = Candidate(
        ho_ten=payload.ho_ten,
        nhom_khuyet_tat=payload.nhom_khuyet_tat,
    )

    try:
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
)
def list_candidates(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=1000),
    nhom_khuyet_tat: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> list[CandidateResponse]:
    """Liệt kê ứng viên, hỗ trợ phân trang và lọc theo nhóm."""
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


# ---------------------------------------------------------------------------
# MATCHING
# ---------------------------------------------------------------------------

def aggregate_group_scores(
    rows: list[tuple[JobDescription, float]],
) -> list[tuple[str, float]]:
    """
    Tổng hợp similarity theo nhóm.

    Điểm nhóm = 60% điểm cao nhất + 40% trung bình tối đa 3 kết quả tốt nhất.
    Cách này ổn định hơn việc chỉ dùng một dòng Top-1.
    """
    scores_by_group: dict[str, list[float]] = defaultdict(list)

    for description, similarity in rows:
        scores_by_group[description.nhom_khuyet_tat].append(float(similarity))

    aggregated: list[tuple[str, float]] = []
    for group, scores in scores_by_group.items():
        top_scores = sorted(scores, reverse=True)[:3]
        maximum = top_scores[0]
        average = sum(top_scores) / len(top_scores)
        group_score = 0.6 * maximum + 0.4 * average
        aggregated.append((group, group_score))

    aggregated.sort(key=lambda item: item[1], reverse=True)
    return aggregated


@app.post("/match", response_model=MatchResponse)
def match_candidates(
    payload: MatchQuery,
    db: Session = Depends(get_db),
) -> MatchResponse:
    """
    Workflow:
    1. Encode query với is_query=True.
    2. So cosine với JobDescription.embedding.
    3. Lấy Top-K mô tả gần nhất.
    4. Tổng hợp điểm theo nhóm khuyết tật.
    5. Chọn các nhóm đạt ngưỡng và gần nhóm tốt nhất.
    6. Lọc Candidate theo các nhóm đã chọn.
    """
    description_count = db.query(func.count(JobDescription.id)).scalar() or 0
    if description_count == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Database chưa có mô tả công việc. "
                "Hãy chạy scripts/ingest_job_descriptions.py trước."
            ),
        )

    try:
        query_vector = get_embedding(payload.query)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Không thể tạo query embedding: {type(exc).__name__}",
        ) from exc

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

    ranked_groups = aggregate_group_scores(rows)
    best_score = ranked_groups[0][1]

    selected_groups = [
        (group, score)
        for group, score in ranked_groups
        if score >= MIN_GROUP_SCORE
        and best_score - score <= GROUP_SCORE_MARGIN
    ]

    # Nếu không nhóm nào đạt ngưỡng, không trả ứng viên.
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
