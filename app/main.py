from fastapi import FastAPI, Depends, HTTPException   # framework API + dependency injection + lỗi HTTP
from sqlalchemy.orm import Session                       # kiểu Session dùng để gõ type hint

from app.database import get_db, init_db                 # session DB + hàm khởi tạo bảng
from app.models import JobListing, Candidate, DisabilityGroup   # 3 bảng ORM
from app.embeddings import get_embedding                 # hàm encode text -> vector
from app.config import TOP_K_JOBS, TOP_K_CANDIDATES       # giá trị mặc định từ config
from app.schemas import (
    JobListingCreate,
    JobListingResponse,
    CandidateCreate,
    CandidateResponse,
    MatchQuery,
    MatchResponse,
    GroupScore,
    JobMatchResult,
)

app = FastAPI(title="Avora Matching API")   # khởi tạo ứng dụng FastAPI


@app.on_event("startup")
def on_startup():
    """Chạy 1 lần khi app khởi động: bật extension pgvector + tạo bảng nếu chưa có."""
    init_db()


def get_or_create_group(db: Session, ten_nhom: str) -> DisabilityGroup:
    """
    Tìm nhóm khuyết tật theo tên, nếu chưa tồn tại thì tạo mới.
    Dùng chung cho cả 2 luồng tạo job và tạo candidate, vì cả 2 đều
    tham chiếu (FK) tới cùng 1 bảng disability_groups theo sơ đồ.
    """
    group = db.query(DisabilityGroup).filter(DisabilityGroup.ten_nhom == ten_nhom).first()  # tìm theo tên
    if group is None:                              # nếu chưa có nhóm này trong DB
        group = DisabilityGroup(ten_nhom=ten_nhom)  # tạo object nhóm mới
        db.add(group)                               # thêm vào session
        db.flush()                                  # flush để lấy được group.id ngay (chưa cần commit hẳn)
    return group                                    # trả về nhóm đã có sẵn hoặc vừa tạo


def build_job_content(payload: JobListingCreate) -> str:
    """
    Ghép các trường của job thành 1 đoạn text -- ứng với ô 'lấy mô tả dạng text'
    -> 'mô tả cv' trong sơ đồ, trước khi đưa vào model embedding.
    """
    parts = [
        f"Vị trí: {payload.nghe_cong_viec}",                       # tên nghề
        f"Ngành: {payload.nganh_linh_vuc or ''}",                   # ngành/lĩnh vực
        f"Mô tả: {payload.mo_ta_cong_viec}",                        # mô tả chi tiết
        f"Mức lương: {payload.muc_luong or ''}",                    # mức lương
        f"Khu vực: {payload.khu_vuc_pho_bien or ''}",                # khu vực
        f"Nhóm khuyết tật phù hợp: {payload.nhom_khuyet_tat}",       # nhóm khuyết tật
    ]
    return "\n".join(parts)                          # nối các dòng lại thành 1 đoạn văn bản


# ---------------------------------------------------------------------------
# LUỒNG 1: nạp dữ liệu công việc -- database -> mô tả công việc -> mô tả cv
# -> model embedding -> vector embedding -> lưu vào DB
# ---------------------------------------------------------------------------
@app.post("/documents", response_model=JobListingResponse)
def create_job_listing(payload: JobListingCreate, db: Session = Depends(get_db)):
    """Nhận 1 công việc, tự tính embedding và lưu vào bảng job_listings."""
    group = get_or_create_group(db, payload.nhom_khuyet_tat)   # lấy/tạo nhóm khuyết tật tương ứng

    content = build_job_content(payload)                        # bước 'lấy mô tả dạng text' -> 'mô tả cv'
    vector = get_embedding(content, is_query=False)              # bước 'model embedding' -> 'vector embedding'

    job = JobListing(                                             # tạo bản ghi job mới
        nghe_cong_viec=payload.nghe_cong_viec,
        nganh_linh_vuc=payload.nganh_linh_vuc,
        mo_ta_cong_viec=payload.mo_ta_cong_viec,
        muc_luong=payload.muc_luong,
        khu_vuc_pho_bien=payload.khu_vuc_pho_bien,
        disability_group_id=group.id,                             # gắn FK tới nhóm khuyết tật
        content=content,
        embedding=vector,
    )
    db.add(job)                                                    # thêm vào session
    db.commit()                                                     # bước 'Lưu vào DB'
    db.refresh(job)                                                  # load lại id/created_at vừa sinh ra

    return JobListingResponse(                                      # build response, join tên nhóm ra cho dễ đọc
        id=job.id,
        nghe_cong_viec=job.nghe_cong_viec,
        nganh_linh_vuc=job.nganh_linh_vuc,
        mo_ta_cong_viec=job.mo_ta_cong_viec,
        muc_luong=job.muc_luong,
        khu_vuc_pho_bien=job.khu_vuc_pho_bien,
        nhom_khuyet_tat=group.ten_nhom,
    )


@app.get("/documents", response_model=list[JobListingResponse])
def list_job_listings(db: Session = Depends(get_db)):
    """Liệt kê toàn bộ công việc đã lưu (dùng để kiểm tra dữ liệu khi cần)."""
    jobs = db.query(JobListing).all()                    # lấy toàn bộ job trong DB
    return [
        JobListingResponse(
            id=j.id,
            nghe_cong_viec=j.nghe_cong_viec,
            nganh_linh_vuc=j.nganh_linh_vuc,
            mo_ta_cong_viec=j.mo_ta_cong_viec,
            muc_luong=j.muc_luong,
            khu_vuc_pho_bien=j.khu_vuc_pho_bien,
            nhom_khuyet_tat=j.disability_group.ten_nhom,     # join qua relationship để lấy tên nhóm
        )
        for j in jobs
    ]


# ---------------------------------------------------------------------------
# Nạp dữ liệu ứng viên -- database -> danh sách ứng viên
# ---------------------------------------------------------------------------
@app.post("/candidates", response_model=CandidateResponse)
def create_candidate(payload: CandidateCreate, db: Session = Depends(get_db)):
    """Nhận 1 ứng viên và lưu vào bảng candidates (không cần embedding)."""
    group = get_or_create_group(db, payload.nhom_khuyet_tat)   # lấy/tạo nhóm khuyết tật tương ứng

    candidate = Candidate(                                       # tạo bản ghi ứng viên mới
        ho_ten=payload.ho_ten,
        disability_group_id=group.id,                             # gắn FK tới nhóm khuyết tật
        ky_nang=payload.ky_nang,
        khu_vuc=payload.khu_vuc,
        lien_he=payload.lien_he,
        ghi_chu=payload.ghi_chu,
    )
    db.add(candidate)                                              # thêm vào session
    db.commit()                                                     # lưu xuống DB
    db.refresh(candidate)                                            # load lại id/created_at

    return CandidateResponse(                                        # build response
        id=candidate.id,
        ho_ten=candidate.ho_ten,
        nhom_khuyet_tat=group.ten_nhom,
        ky_nang=candidate.ky_nang,
        khu_vuc=candidate.khu_vuc,
        lien_he=candidate.lien_he,
    )


@app.get("/candidates", response_model=list[CandidateResponse])
def list_candidates(db: Session = Depends(get_db)):
    """Liệt kê toàn bộ ứng viên đã lưu (dùng để kiểm tra dữ liệu khi cần)."""
    candidates = db.query(Candidate).all()                # lấy toàn bộ ứng viên trong DB
    return [
        CandidateResponse(
            id=c.id,
            ho_ten=c.ho_ten,
            nhom_khuyet_tat=c.disability_group.ten_nhom,     # join qua relationship để lấy tên nhóm
            ky_nang=c.ky_nang,
            khu_vuc=c.khu_vuc,
            lien_he=c.lien_he,
        )
        for c in candidates
    ]


# ---------------------------------------------------------------------------
# LUỒNG 2: nhà tuyển dụng tìm ứng viên -- đây là phần lõi khớp với nửa dưới
# của sơ đồ flow, từ 'nhập yêu cầu tuyển dụng' tới 'Trả về kết quả cho người dùng'
# ---------------------------------------------------------------------------
@app.post("/match", response_model=MatchResponse)
def match_candidates(payload: MatchQuery, db: Session = Depends(get_db)):
    """
    nhập yêu cầu tuyển dụng
      -> model embedding (Qwen3-Embedding-0.6B)
      -> chuyển thành vector embedding
      -> dùng db query lấy ra K công việc giống nhất với embedding vector
      -> chọn ra nhóm đối tượng phù hợp nhất tương ứng với công việc có độ
         tương đồng cao nhất
      -> lấy ra danh sách ứng viên của nhóm đối tượng đó
      -> trả về kết quả cho người dùng
    """
    if not payload.query.strip():                          # chặn query rỗng, tránh embedding vô nghĩa
        raise HTTPException(status_code=400, detail="Query không được để trống")

    top_k_jobs = payload.top_k_jobs or TOP_K_JOBS            # dùng override nếu có, không thì lấy default
    top_k_candidates = payload.top_k_candidates or TOP_K_CANDIDATES

    query_vector = get_embedding(payload.query, is_query=True)   # bước 'model embedding' cho câu query

    cosine_dist = JobListing.embedding.cosine_distance(query_vector)   # công thức khoảng cách cosine của pgvector

    top_jobs = (                                              # bước 'dùng db query lấy ra K công việc giống nhất'
        db.query(
            JobListing.id,
            JobListing.nghe_cong_viec,
            JobListing.mo_ta_cong_viec,
            JobListing.disability_group_id,
            (1 - cosine_dist).label("similarity"),               # similarity = 1 - distance, càng gần 1 càng giống
        )
        .order_by(cosine_dist)                                    # khoảng cách nhỏ nhất xếp lên đầu = giống nhất
        .limit(top_k_jobs)
        .all()
    )

    if not top_jobs:                                          # nếu DB chưa có job nào thì không thể chọn nhóm
        return MatchResponse(
            selected_group="", group_scores=[], matched_jobs=[], candidates=[]
        )

    group_stats: dict[int, dict] = {}                          # {disability_group_id: {"total": float, "count": int}}
    for job in top_jobs:
        stats = group_stats.setdefault(job.disability_group_id, {"total": 0.0, "count": 0})
        stats["total"] += float(job.similarity)                  # cộng dồn điểm similarity của job này vào nhóm
        stats["count"] += 1                                        # đếm thêm 1 job thuộc nhóm này

    # Lấy tên nhóm tương ứng với từng group_id để hiển thị cho người dùng
    group_ids = list(group_stats.keys())
    groups = db.query(DisabilityGroup).filter(DisabilityGroup.id.in_(group_ids)).all()
    id_to_name = {g.id: g.ten_nhom for g in groups}              # map id -> tên nhóm

    group_scores = [                                              # bước 'chọn ra nhóm đối tượng phù hợp nhất'
        GroupScore(
            nhom_khuyet_tat=id_to_name[gid],
            score=round(stats["total"] / stats["count"], 4),        # điểm trung bình similarity của nhóm
            matched_jobs=stats["count"],
        )
        for gid, stats in group_stats.items()
    ]
    group_scores.sort(key=lambda g: g.score, reverse=True)         # nhóm điểm cao nhất xếp lên đầu

    selected_group = group_scores[0].nhom_khuyet_tat                # nhóm có điểm cao nhất = nhóm được chọn
    selected_group_id = next(gid for gid, name in id_to_name.items() if name == selected_group)

    candidates = (                                                  # bước 'lấy ra danh sách ứng viên của nhóm đó'
        db.query(Candidate)
        .filter(Candidate.disability_group_id == selected_group_id)
        .limit(top_k_candidates)
        .all()
    )

    matched_jobs = [                                                  # danh sách job dùng làm căn cứ, trả kèm để tham khảo
        JobMatchResult(
            id=job.id,
            nghe_cong_viec=job.nghe_cong_viec,
            mo_ta_cong_viec=job.mo_ta_cong_viec,
            nhom_khuyet_tat=id_to_name[job.disability_group_id],
            similarity=round(float(job.similarity), 4),
        )
        for job in top_jobs
    ]

    return MatchResponse(                                            # bước 'trả về kết quả cho người dùng'
        selected_group=selected_group,
        group_scores=group_scores,
        matched_jobs=matched_jobs,
        candidates=[
            CandidateResponse(
                id=c.id,
                ho_ten=c.ho_ten,
                nhom_khuyet_tat=selected_group,
                ky_nang=c.ky_nang,
                khu_vuc=c.khu_vuc,
                lien_he=c.lien_he,
            )
            for c in candidates
        ],
    )
