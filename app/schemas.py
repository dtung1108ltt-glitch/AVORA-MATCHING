from pydantic import BaseModel, ConfigDict   # BaseModel để định nghĩa schema, ConfigDict để bật from_attributes


class CandidateCreate(BaseModel):
    """Input khi thêm 1 ứng viên mới. `mo_ta` là trường bắt buộc, dùng để tính embedding."""
    ho_ten: str                   # họ tên ứng viên (bắt buộc)
    mo_ta: str                    # mô tả ứng viên: kỹ năng/kinh nghiệm/loại việc phù hợp... (bắt buộc, dùng để embed)
    khu_vuc: str | None = None         # khu vực sinh sống / mong muốn làm việc
    lien_he: str | None = None         # thông tin liên hệ
    ghi_chu: str | None = None         # ghi chú thêm


class CandidateResponse(BaseModel):
    """Thông tin 1 ứng viên trả về, không kèm vector cho gọn."""
    id: int                       # id ứng viên
    ho_ten: str                   # họ tên
    mo_ta: str                    # mô tả ứng viên
    khu_vuc: str | None           # khu vực
    lien_he: str | None           # liên hệ

    model_config = ConfigDict(from_attributes=True)


class MatchQuery(BaseModel):
    """Input cho endpoint /match -- mô tả yêu cầu tuyển dụng của nhà tuyển dụng, dạng text tự do."""
    query: str                    # nội dung yêu cầu tuyển dụng
    top_k: int | None = None      # override số ứng viên trả về, None thì dùng default trong config


class CandidateMatchResult(CandidateResponse):
    """1 ứng viên trong kết quả /match, kèm điểm giống nhau với query."""
    similarity: float             # điểm giống nhau với query (0..1, càng cao càng phù hợp)


class MatchResponse(BaseModel):
    """Output của endpoint /match: danh sách ứng viên phù hợp nhất, xếp hạng theo similarity."""
    candidates: list[CandidateMatchResult]