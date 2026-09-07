from fastapi import FastAPI, Depends, HTTPException   # framework API + dependency injection + lỗi HTTP
from sqlalchemy.orm import Session                       # kiểu Session dùng để gõ type hint

from app.database import get_db, init_db                 # session DB + hàm khởi tạo bảng
from app.models import Candidate                         # bảng ORM duy nhất
from app.embeddings import get_embedding                 # hàm encode text -> vector
from app.config import TOP_K_CANDIDATES                  # giá trị mặc định từ config
from app.schemas import (
    CandidateCreate,
    CandidateResponse,
    MatchQuery,
    MatchResponse,
    CandidateMatchResult,
)

app = FastAPI(title="Avora Matching API")   # khởi tạo ứng dụng FastAPI


@app.on_event("startup")
def on_startup():
    """Chạy 1 lần khi app khởi động: bật extension pgvector + tạo bảng nếu chưa có."""
    init_db()


# ---------------------------------------------------------------------------
# Nạp dữ liệu ứng viên -- text (mo_ta) -> embedding vector -> lưu vào DB
# ---------------------------------------------------------------------------
@app.post("/candidates", response_model=CandidateResponse)
def create_candidate(payload: CandidateCreate, db: Session = Depends(get_db)):
    """Nhận 1 ứng viên, tự tính embedding từ `mo_ta` và lưu vào bảng candidates."""
    vector = get_embedding(payload.mo_ta, is_query=False)   # text -> embedding vector

    candidate = Candidate(                                   # tạo bản ghi ứng viên mới
        ho_ten=payload.ho_ten,
        mo_ta=payload.mo_ta,
        khu_vuc=payload.khu_vuc,
        lien_he=payload.lien_he,
        ghi_chu=payload.ghi_chu,
        embedding=vector,
    )
    db.add(candidate)                                          # thêm vào session
    db.commit()                                                 # lưu xuống DB
    db.refresh(candidate)                                        # load lại id/created_at

    return CandidateResponse.model_validate(candidate)           # build response từ ORM object


@app.get("/candidates", response_model=list[CandidateResponse])
def list_candidates(db: Session = Depends(get_db)):
    """Liệt kê toàn bộ ứng viên đã lưu (dùng để kiểm tra dữ liệu khi cần)."""
    candidates = db.query(Candidate).all()
    return [CandidateResponse.model_validate(c) for c in candidates]


# ---------------------------------------------------------------------------
# LUỒNG DUY NHẤT: nhà tuyển dụng nhập yêu cầu tuyển dụng
#   -> model embedding (Qwen3-Embedding-0.6B)
#   -> chuyển thành vector embedding
#   -> query trực tiếp vào embedding của TỪNG ứng viên (cosine similarity)
#   -> trả về top-K ứng viên phù hợp nhất, xếp hạng theo similarity
# ---------------------------------------------------------------------------
@app.post("/match", response_model=MatchResponse)
def match_candidates(payload: MatchQuery, db: Session = Depends(get_db)):
    """Nhận mô tả yêu cầu tuyển dụng, trả về danh sách ứng viên phù hợp nhất."""
    if not payload.query.strip():                          # chặn query rỗng, tránh embedding vô nghĩa
        raise HTTPException(status_code=400, detail="Query không được để trống")

    top_k = payload.top_k or TOP_K_CANDIDATES                 # dùng override nếu có, không thì lấy default

    query_vector = get_embedding(payload.query, is_query=True)   # bước 'model embedding' cho câu query

    cosine_dist = Candidate.embedding.cosine_distance(query_vector)   # công thức khoảng cách cosine của pgvector

    rows = (                                                  # query thẳng vào bảng candidates, không qua bảng nào khác
        db.query(
            Candidate,
            (1 - cosine_dist).label("similarity"),               # similarity = 1 - distance, càng gần 1 càng giống
        )
        .order_by(cosine_dist)                                    # khoảng cách nhỏ nhất xếp lên đầu = giống nhất
        .limit(top_k)
        .all()
    )

    results = [                                               # ghép object Candidate + điểm similarity vào 1 response
        CandidateMatchResult(
            id=c.id,
            ho_ten=c.ho_ten,
            mo_ta=c.mo_ta,
            khu_vuc=c.khu_vuc,
            lien_he=c.lien_he,
            similarity=round(float(sim), 4),
        )
        for c, sim in rows
    ]

    return MatchResponse(candidates=results)