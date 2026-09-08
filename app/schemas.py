from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


ALLOWED_DISABILITY_GROUPS = {
    "Khiếm thính",
    "Khiếm thị",
    "Khuyết tật vận động",
    "Khuyết tật trí tuệ nhẹ",
    "Khuyết tật ngôn ngữ",
    "Khuyết tật thần kinh, tâm thần (ổn định)",
    "Đa khuyết tật nhẹ",
}


def normalize_required_text(value: str) -> str:
    """Loại khoảng trắng thừa và từ chối chuỗi rỗng."""
    normalized = " ".join(value.split())
    if not normalized:
        raise ValueError("Giá trị không được để trống")
    return normalized


def validate_disability_group(value: str) -> str:
    """Chuẩn hóa và kiểm tra nhóm khuyết tật."""
    normalized = normalize_required_text(value)
    if normalized not in ALLOWED_DISABILITY_GROUPS:
        allowed = ", ".join(sorted(ALLOWED_DISABILITY_GROUPS))
        raise ValueError(
            f"Nhóm khuyết tật không hợp lệ. Các giá trị được phép: {allowed}"
        )
    return normalized


class JobDescriptionCreate(BaseModel):
    """Payload cho POST /job-descriptions."""

    nhom_khuyet_tat: str = Field(min_length=1)
    mo_ta_cong_viec: str = Field(min_length=50)

    @field_validator("nhom_khuyet_tat")
    @classmethod
    def check_group(cls, value: str) -> str:
        return validate_disability_group(value)

    @field_validator("mo_ta_cong_viec")
    @classmethod
    def check_description(cls, value: str) -> str:
        normalized = normalize_required_text(value)
        if len(normalized) < 50:
            raise ValueError("Mô tả công việc phải có ít nhất 50 ký tự")
        return normalized


class JobDescriptionResponse(BaseModel):
    """Response của POST/GET job descriptions, không trả vector."""

    id: int
    nhom_khuyet_tat: str
    mo_ta_cong_viec: str

    model_config = ConfigDict(from_attributes=True)


class CandidateCreate(BaseModel):
    """Payload cho POST /candidates."""

    ho_ten: str = Field(min_length=1)
    nhom_khuyet_tat: str = Field(min_length=1)

    @field_validator("ho_ten")
    @classmethod
    def check_name(cls, value: str) -> str:
        return normalize_required_text(value)

    @field_validator("nhom_khuyet_tat")
    @classmethod
    def check_group(cls, value: str) -> str:
        return validate_disability_group(value)


class CandidateResponse(BaseModel):
    """Ứng viên trả về từ API."""

    id: int
    ho_ten: str
    nhom_khuyet_tat: str

    model_config = ConfigDict(from_attributes=True)


class MatchQuery(BaseModel):
    """Payload cho POST /match."""

    query: str = Field(min_length=1)
    top_k_descriptions: int = Field(default=20, ge=1, le=200)
    top_k_candidates: int = Field(default=20, ge=1, le=500)

    @field_validator("query")
    @classmethod
    def check_query(cls, value: str) -> str:
        return normalize_required_text(value)


class MatchedGroup(BaseModel):
    """Điểm tổng hợp của một nhóm khuyết tật."""

    nhom_khuyet_tat: str
    score: float = Field(ge=-1.0, le=1.0)


class MatchedDescription(BaseModel):
    """Mô tả gần query, dùng để kiểm tra kết quả matching."""

    id: int
    nhom_khuyet_tat: str
    mo_ta_cong_viec: str
    similarity: float = Field(ge=-1.0, le=1.0)


class MatchResponse(BaseModel):
    """Response đầy đủ của POST /match."""

    matched_groups: list[MatchedGroup]
    matched_descriptions: list[MatchedDescription] = Field(default_factory=list)
    candidates: list[CandidateResponse]
