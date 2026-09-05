from pydantic import BaseModel, ConfigDict   # BaseModel để định nghĩa schema, ConfigDict để bật from_attributes


class JobListingCreate(BaseModel):
    """Input khi thêm 1 công việc mới -- ứng với bước 'mô tả công việc' đọc từ database nguồn."""
    nghe_cong_viec: str          # tên nghề / vị trí
    nganh_linh_vuc: str | None = None    # ngành / lĩnh vực
    mo_ta_cong_viec: str          # mô tả chi tiết công việc (bắt buộc, dùng để build content)
    muc_luong: str | None = None      # mức lương
    khu_vuc_pho_bien: str | None = None   # khu vực làm việc
    nhom_khuyet_tat: str          # tên nhóm khuyết tật (client gửi tên, server tự tra/tạo id)


class JobListingResponse(BaseModel):
    """Output trả về sau khi tạo/lấy 1 công việc, không kèm vector cho gọn."""
    id: int                       # id bản ghi
    nghe_cong_viec: str           # tên nghề
    nganh_linh_vuc: str | None    # ngành/lĩnh vực
    mo_ta_cong_viec: str          # mô tả công việc
    muc_luong: str | None         # mức lương
    khu_vuc_pho_bien: str | None  # khu vực
    nhom_khuyet_tat: str          # tên nhóm khuyết tật (join ra để dễ đọc)

    model_config = ConfigDict(from_attributes=True)   # cho phép tạo trực tiếp từ object ORM


class CandidateCreate(BaseModel):
    """Input khi thêm 1 ứng viên mới -- ứng với bảng 'danh sách ứng viên'."""
    ho_ten: str                   # họ tên ứng viên (bắt buộc)
    nhom_khuyet_tat: str          # tên nhóm khuyết tật
    ky_nang: str | None = None        # kỹ năng / kinh nghiệm
    khu_vuc: str | None = None         # khu vực sinh sống / mong muốn làm việc
    lien_he: str | None = None         # thông tin liên hệ
    ghi_chu: str | None = None         # ghi chú thêm


class CandidateResponse(BaseModel):
    """Thông tin 1 ứng viên trả về cho nhà tuyển dụng ở bước cuối flow."""
    id: int                       # id ứng viên
    ho_ten: str                   # họ tên
    nhom_khuyet_tat: str          # tên nhóm khuyết tật (để verify đúng nhóm đã chọn)
    ky_nang: str | None           # kỹ năng
    khu_vuc: str | None           # khu vực
    lien_he: str | None           # liên hệ

    model_config = ConfigDict(from_attributes=True)


class JobMatchResult(BaseModel):
    """1 công việc trong danh sách K công việc giống nhất, kèm điểm similarity."""
    id: int                       # id job
    nghe_cong_viec: str           # tên nghề
    mo_ta_cong_viec: str          # mô tả công việc
    nhom_khuyet_tat: str          # nhóm khuyết tật của job này
    similarity: float             # điểm giống nhau với query (0..1, càng cao càng giống)


class MatchQuery(BaseModel):
    """Input cho endpoint /match -- ứng với ô 'nhập yêu cầu tuyển dụng' của nhà tuyển dụng."""
    query: str                    # nội dung yêu cầu tuyển dụng (text tự do)
    top_k_jobs: int | None = None        # override số job lấy ra, None thì dùng default trong config
    top_k_candidates: int | None = None  # override số ứng viên trả về tối đa


class GroupScore(BaseModel):
    """Điểm phù hợp của 1 nhóm khuyết tật -- để giải thích vì sao hệ thống chọn nhóm này."""
    nhom_khuyet_tat: str           # tên nhóm khuyết tật
    score: float                   # điểm trung bình similarity của các job thuộc nhóm này trong top K
    matched_jobs: int              # số lượng job trong top K thuộc nhóm này


class MatchResponse(BaseModel):
    """
    Output cuối cùng của endpoint /match -- ứng với ô 'Trả về kết quả cho người dùng'
    ở cuối sơ đồ flow.
    """
    selected_group: str                    # nhóm khuyết tật được chọn là phù hợp nhất
    group_scores: list[GroupScore]         # bảng điểm toàn bộ nhóm xuất hiện trong top K (để debug/giải thích)
    matched_jobs: list[JobMatchResult]     # danh sách job đã dùng làm căn cứ chọn nhóm
    candidates: list[CandidateResponse]    # danh sách ứng viên của nhóm được chọn -- KẾT QUẢ CHÍNH
