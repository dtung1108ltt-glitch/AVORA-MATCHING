from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator


# Tập hợp các nhóm khuyết tật hợp lệ trong hệ thống AVORA
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
    """
    Chuẩn hóa văn bản bắt buộc: loại bỏ khoảng trắng thừa đầu cuối và giữa các từ.

    Mục đích:
        Làm sạch dữ liệu chuỗi đầu vào, đảm bảo tính nhất quán và từ chối các chuỗi rỗng.

    Tham số đầu vào:
        value (str): Chuỗi ký tự cần chuẩn hóa.

    Kết quả trả về:
        str: Chuỗi ký tự đã được loại bỏ khoảng trắng dư thừa.

    Raises:
        ValueError: Nếu chuỗi rỗng hoặc chỉ toàn khoảng trắng.
    """
    # Tách chuỗi theo khoảng trắng và ghép lại với dấu cách đơn
    normalized = " ".join(value.split())
    # Kiểm tra chuỗi sau khi làm sạch có rỗng không
    if not normalized:
        raise ValueError("Giá trị không được để trống")
    # Trả về chuỗi chuẩn hóa hợp lệ
    return normalized


def validate_disability_group(value: str) -> str:
    """
    Chuẩn hóa và kiểm tra nhóm khuyết tật có nằm trong danh mục cho phép hay không.

    Mục đích:
        Đảm bảo tính toàn vẹn dữ liệu cho trường nhóm khuyết tật theo danh mục chuẩn.

    Tham số đầu vào:
        value (str): Tên nhóm khuyết tật đầu vào cần kiểm tra.

    Kết quả trả về:
        str: Tên nhóm khuyết tật chuẩn xác sau khi kiểm tra hợp lệ.

    Raises:
        ValueError: Nếu nhóm khuyết tật không thuộc danh mục ALLOWED_DISABILITY_GROUPS.
    """
    # Bước 1: Chuẩn hóa khoảng trắng của chuỗi nhập vào
    normalized = normalize_required_text(value)
    # Bước 2: Đối chiếu với tập hợp các nhóm khuyết tật được định nghĩa trước
    if normalized not in ALLOWED_DISABILITY_GROUPS:
        allowed = ", ".join(sorted(ALLOWED_DISABILITY_GROUPS))
        raise ValueError(
            f"Nhóm khuyết tật không hợp lệ. Các giá trị được phép: {allowed}"
        )
    # Bước 3: Trả về kết quả nhóm khuyết tật đã được xác thực
    return normalized


# ---------------------------------------------------------------------------
# CÁC SCHEMA CŨ CHO JOB DESCRIPTIONS VÀ CANDIDATES (GIỮ NGUYÊN TƯƠNG THÍCH)
# ---------------------------------------------------------------------------

class JobDescriptionCreate(BaseModel):
    """Payload cho POST /job-descriptions."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    nhom_khuyet_tat: str = Field(min_length=1)
    mo_ta_cong_viec: str = Field(min_length=50)

    @field_validator("nhom_khuyet_tat")
    @classmethod
    def check_group(cls, value: str) -> str:
        """
        Kiểm tra tính hợp lệ của nhóm khuyết tật trong mô tả công việc.

        Tham số đầu vào:
            value (str): Nhóm khuyết tật truyền vào từ request body.

        Kết quả trả về:
            str: Tên nhóm khuyết tật chuẩn hóa.
        """
        # Gọi hàm kiểm tra nhóm khuyết tật dùng chung
        return validate_disability_group(value)

    @field_validator("mo_ta_cong_viec")
    @classmethod
    def check_description(cls, value: str) -> str:
        """
        Kiểm tra và chuẩn hóa độ dài của mô tả công việc.

        Tham số đầu vào:
            value (str): Nội dung văn bản mô tả công việc.

        Kết quả trả về:
            str: Nội dung mô tả công việc đã loại bỏ khoảng trắng thừa.
        """
        # Loại bỏ khoảng trắng thừa ở đầu/cuối và giữa các từ
        normalized = normalize_required_text(value)
        # Đảm bảo độ dài tối thiểu 50 ký tự để mô hình embedding có đủ ngữ cảnh
        if len(normalized) < 50:
            raise ValueError("Mô tả công việc phải có ít nhất 50 ký tự")
        return normalized


class JobDescriptionResponse(BaseModel):
    """Response của POST/GET job descriptions, không trả vector."""

    id: int
    nhom_khuyet_tat: str
    mo_ta_cong_viec: str

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class CandidateCreate(BaseModel):
    """Payload cho POST /candidates."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    ho_ten: str = Field(min_length=1)
    nhom_khuyet_tat: str = Field(min_length=1)

    @field_validator("ho_ten")
    @classmethod
    def check_name(cls, value: str) -> str:
        """
        Kiểm tra và chuẩn hóa họ tên ứng viên.

        Tham số đầu vào:
            value (str): Họ và tên ứng viên cần kiểm tra.

        Kết quả trả về:
            str: Họ tên ứng viên chuẩn hóa.
        """
        # Làm sạch khoảng trắng của chuỗi tên
        return normalize_required_text(value)

    @field_validator("nhom_khuyet_tat")
    @classmethod
    def check_group(cls, value: str) -> str:
        """
        Kiểm tra nhóm khuyết tật của ứng viên.

        Tham số đầu vào:
            value (str): Nhóm khuyết tật của ứng viên.

        Kết quả trả về:
            str: Nhóm khuyết tật chuẩn hóa hợp lệ.
        """
        # Gọi hàm xác thực nhóm khuyết tật
        return validate_disability_group(value)


class CandidateResponse(BaseModel):
    """Ứng viên trả về từ API."""

    id: int
    ho_ten: str
    nhom_khuyet_tat: str

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class MatchQuery(BaseModel):
    """Payload cho POST /match."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    query: str = Field(min_length=1)
    top_k_descriptions: int = Field(default=20, ge=1, le=200)
    top_k_candidates: int = Field(default=20, ge=1, le=500)

    @field_validator("query")
    @classmethod
    def check_query(cls, value: str) -> str:
        """
        Kiểm tra chuỗi truy vấn tuyển dụng.

        Tham số đầu vào:
            value (str): Chuỗi query mô tả công việc cần tìm kiếm.

        Kết quả trả về:
            str: Chuỗi query đã chuẩn hóa.
        """
        # Làm sạch chuỗi truy vấn đầu vào
        return normalize_required_text(value)


class MatchedGroup(BaseModel):
    """Điểm tổng hợp của một nhóm khuyết tật."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    nhom_khuyet_tat: str
    score: float = Field(ge=-1.0, le=1.0)


class MatchedDescription(BaseModel):
    """Mô tả gần query, dùng để kiểm tra kết quả matching."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: int
    nhom_khuyet_tat: str
    mo_ta_cong_viec: str
    similarity: float = Field(ge=-1.0, le=1.0)


class MatchResponse(BaseModel):
    """Response đầy đủ của POST /match."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    matched_groups: list[MatchedGroup]
    matched_descriptions: list[MatchedDescription] = Field(default_factory=list)
    candidates: list[CandidateResponse]


# ---------------------------------------------------------------------------
# CÁC SCHEMA MỚI DÀNH CHO THUẬT TOÁN STABLE MATCHING (GALE-SHAPLEY)
# ---------------------------------------------------------------------------

class Vector3D(BaseModel):
    """
    Biểu diễn vector trong không gian Oxyz ba chiều.

    Mục đích:
        Dùng để mô hình hóa vector năng lực ứng viên u = (x1, y1, z1)
        và vector yêu cầu tuyển dụng v = (x2, y2, z2) trong không gian Oxyz.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    x: float = Field(description="Tọa độ x (ví dụ: Chuyên môn / Kỹ năng cứng)")
    y: float = Field(description="Tọa độ y (ví dụ: Kỹ năng mềm / Thích ứng)")
    z: float = Field(description="Tọa độ z (ví dụ: Độ phù hợp môi trường / Văn hóa)")

    def norm(self) -> float:
        """
        Tính độ dài (Euclidean norm) của vector trong không gian Oxyz: |v| = sqrt(x^2 + y^2 + z^2).

        Mục đích:
            Xác định độ lớn của vector để chuẩn hóa khi tính Cosine Similarity.

        Tham số đầu vào:
            Không có (sử dụng thuộc tính nội tại của đối tượng Vector3D).

        Kết quả trả về:
            float: Độ dài Euclidean của vector. Trả về 0.0 nếu vector gốc (0, 0, 0).
        """
        # Áp dụng công thức căn bậc hai tổng bình phương các tọa độ
        return math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)

    def dot_product(self, other: Vector3D) -> float:
        """
        Tính tích vô hướng (Dot Product) giữa hai vector: u . v = x1*x2 + y1*y2 + z1*z2.

        Mục đích:
            Phục vụ tính điểm Employer Preference Score (Điểm E) theo đại số tuyến tính.

        Tham số đầu vào:
            other (Vector3D): Vector đối tác cần nhân vô hướng.

        Kết quả trả về:
            float: Giá trị tích vô hướng giữa hai vector.
        """
        # Nhân từng thành phần tương ứng và cộng dồn lại
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cosine_similarity(self, other: Vector3D) -> float:
        """
        Tính Cosine Similarity giữa hai vector: cos(u, v) = (u . v) / (|u| * |v|).

        Mục đích:
            Đo lường góc giữa vector năng lực ứng viên và yêu cầu tuyển dụng trong không gian Oxyz,
            tạo nên Điểm E (Employer Preference Score) chuẩn hóa trong đoạn [-1.0, 1.0].

        Tham số đầu vào:
            other (Vector3D): Vector đối tác cần so sánh góc.

        Kết quả trả về:
            float: Giá trị cosin tương đồng trong khoảng [-1.0, 1.0]. Trả về 0.0 nếu một trong hai vector có độ dài 0.
        """
        # Bước 1: Tính độ dài của vector hiện tại
        norm_self = self.norm()
        # Bước 2: Tính độ dài của vector đối tác
        norm_other = other.norm()
        # Bước 3: Kiểm tra nếu một trong hai vector có độ dài bằng 0 để tránh chia cho 0
        if norm_self == 0.0 or norm_other == 0.0:
            return 0.0
        # Bước 4: Áp dụng công thức Cosine Similarity
        cosine = self.dot_product(other) / (norm_self * norm_other)
        # Giới hạn trong khoảng chuẩn [-1.0, 1.0] để tránh sai số số thực
        return max(-1.0, min(1.0, cosine))

    def to_tuple(self) -> tuple[float, float, float]:
        """
        Chuyển đổi đối tượng Vector3D thành tuple 3 phần tử (x, y, z).

        Mục đích:
            Tương thích nhanh với các hàm tính toán đại số dạng tuple.

        Tham số đầu vào:
            Không có.

        Kết quả trả về:
            tuple[float, float, float]: Bộ 3 giá trị tọa độ (x, y, z).
        """
        # Trả về bộ giá trị float của 3 tọa độ
        return (self.x, self.y, self.z)


class HardConstraints(BaseModel):
    """
    Ràng buộc điều kiện bắt buộc (Hard Constraints) thuộc Tập xác định D.

    Mục đích:
        Lọc biên sơ loại ngay từ đầu: bất kỳ cặp nào vi phạm ràng buộc
        sẽ có điểm ưu tiên bị triệt tiêu bằng 0 thông qua hàm chỉ báo (Indicator Function).
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    min_salary: float | None = Field(default=None, description="Mức lương tối thiểu mong muốn")
    max_salary: float | None = Field(default=None, description="Mức lương tối đa nhà tuyển dụng chi trả")
    max_distance_km: float | None = Field(default=None, description="Khoảng cách địa lý tối đa cho phép (km)")
    disability_group_allowed: list[str] | None = Field(default=None, description="Danh sách nhóm khuyết tật được chấp nhận")


class CandidateProfileInput(BaseModel):
    """
    Dữ liệu đầu vào của một ứng viên trong bài toán Stable Matching.

    Mục đích:
        Chứa thông tin định danh, sở thích, điểm ưu tiên và vector năng lực Oxyz của ứng viên.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(default="", description="Định danh duy nhất của ứng viên (ví dụ: 'C1', 'cand_01')")
    candidate_id: str | None = Field(default=None, description="Định danh ứng viên (alias cho id)")
    ho_ten: str | None = Field(default=None, description="Họ và tên của ứng viên")
    nhom_khuyet_tat: str | None = Field(default=None, description="Nhóm khuyết tật của ứng viên")
    vector_oxyz: Vector3D | None = Field(default=None, description="Tọa độ vector năng lực trong không gian Oxyz")
    skills: list[float] | str | list[str] | None = Field(default=None, description="Kỹ năng chuyên môn (vector, text hoặc list)")
    essential_access_needs: list[str] | None = Field(default=None, description="Nhu cầu tiếp cận thiết yếu (Hard constraint)")
    nice_to_have_access_needs: list[str] | None = Field(default=None, description="Nhu cầu tiếp cận nên có (Soft component)")
    expected_salary_min: float | None = Field(default=None, description="Mức lương tối thiểu mong muốn")
    expected_salary_max: float | None = Field(default=None, description="Mức lương tối đa mong muốn")
    work_mode: str | None = Field(default=None, description="Hình thức làm việc ('remote', 'hybrid', 'on-site')")
    work_hours: str | None = Field(default=None, description="Thời gian làm việc ('full-time', 'part-time', 'flexible')")
    location: str | None = Field(default=None, description="Khu vực sinh sống / làm việc")
    priority_ranking: list[str] | None = Field(default=None, description="Thứ tự ưu tiên 3 yếu tố tự xếp (0.5, 0.3, 0.2)")
    experience_years: float | None = Field(default=None, description="Số năm kinh nghiệm")
    career_orientation: list[float] | str | None = Field(default=None, description="Mục tiêu / định hướng nghề nghiệp")
    direct_track: bool = Field(default=False, description="True nếu đã trúng tuyển Direct Track (loại khỏi pool)")
    hired_job_id: str | None = Field(default=None, description="ID vị trí đã trúng tuyển nếu là Direct Track")
    preferences: list[str] | None = Field(
        default=None,
        description="Danh sách ID công việc được sắp xếp theo thứ tự ưu tiên giảm dần của ứng viên"
    )
    c_scores: dict[str, float] | None = Field(
        default=None,
        description="Ma trận điểm C: {job_id: score} đo lường mức độ yêu thích công việc của ứng viên"
    )
    expected_salary: float | None = Field(default=None, description="Kỳ vọng lương của ứng viên")
    distance_km: float | None = Field(default=None, description="Khoảng cách từ nơi ở tới nơi làm việc (km)")

    @model_validator(mode="before")
    @classmethod
    def resolve_identifiers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("id") and data.get("candidate_id"):
                data["id"] = data["candidate_id"]
        return data

    @model_validator(mode="after")
    def validate_direct_track(self) -> CandidateProfileInput:
        if self.direct_track:
            c_id = self.id or self.candidate_id or "unknown"
            if not self.hired_job_id or not str(self.hired_job_id).strip():
                raise ValueError(
                    f"Ứng viên '{c_id}' có direct_track=True nhưng thiếu hired_job_id "
                    "(bắt buộc phải chỉ định vị trí đã trúng tuyển)."
                )
        return self


class JobProfileInput(BaseModel):
    """
    Dữ liệu đầu vào của một vị trí tuyển dụng trong bài toán Stable Matching.

    Mục đích:
        Chứa thông tin vị trí, hạn mức tuyển dụng (Quota), danh sách ưu tiên, và vector yêu cầu Oxyz.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    id: str = Field(default="", description="Định danh duy nhất của vị trí việc làm (ví dụ: 'J1', 'job_dev')")
    job_id: str | None = Field(default=None, description="Định danh vị trí (alias cho id)")
    title: str | None = Field(default=None, description="Tên hoặc chức danh công việc")
    quota: int = Field(default=1, ge=0, description="Hạn mức tuyển dụng q_j: số lượng chỉ tiêu tối đa cho vị trí (Many-to-One, >= 0)")
    filled_slots: int = Field(
        default=0,
        ge=0,
        description="Số lượng chỉ tiêu đã tuyển trước từ nguồn bên ngoài hệ thống (external hires), KHÔNG tính ứng viên Direct Track trong batch hiện tại để tránh bị đếm hai lần"
    )
    vector_oxyz: Vector3D | None = Field(default=None, description="Tọa độ vector yêu cầu tuyển dụng trong không gian Oxyz")
    skills_required: list[float] | str | list[str] | None = Field(default=None, description="Kỹ năng yêu cầu")
    access_support_provided: list[str] | None = Field(default=None, description="Các điều kiện hỗ trợ tiếp cận cung cấp")
    salary_min: float | None = Field(default=None, description="Mức lương tối thiểu")
    salary_max: float | None = Field(default=None, description="Mức lương tối đa")
    work_mode: str | None = Field(default=None, description="Hình thức làm việc ('remote', 'hybrid', 'on-site')")
    work_hours: str | None = Field(default=None, description="Thời gian làm việc ('full-time', 'part-time', 'flexible')")
    location: str | None = Field(default=None, description="Khu vực làm việc")
    experience_required_years: float | None = Field(default=None, description="Số năm kinh nghiệm yêu cầu")
    career_domain: list[float] | str | None = Field(default=None, description="Lĩnh vực ngành nghề của vị trí")

    @model_validator(mode="before")
    @classmethod
    def resolve_identifiers(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("id") and data.get("job_id"):
                data["id"] = data["job_id"]
        return data

    @model_validator(mode="after")
    def validate_filled_slots(self) -> JobProfileInput:
        j_id = self.id or self.job_id or "unknown"
        if self.filled_slots > self.quota:
            raise ValueError(
                f"Vị trí '{j_id}' có số chỗ đã lấp ngoài hệ thống (filled_slots={self.filled_slots}) "
                f"vượt quá hạn mức tuyển dụng (quota={self.quota})."
            )
        return self

    preferences: list[str] | None = Field(
        default=None,
        description="Danh sách ID ứng viên được sắp xếp theo thứ tự ưu tiên giảm dần của nhà tuyển dụng"
    )
    e_scores: dict[str, float] | None = Field(
        default=None,
        description="Ma trận điểm E: {candidate_id: score} đo lường độ phù hợp kỹ năng của ứng viên"
    )
    min_e_score: float = Field(default=0.0, ge=0.0, description="Ngưỡng sàn điểm E tối thiểu để chấp nhận ứng viên")
    salary_offered: float | None = Field(default=None, description="Mức lương chi trả cho vị trí")
    allowed_disability_groups: list[str] | None = Field(default=None, description="Các nhóm khuyết tật phù hợp với công việc")
    nhom_khuyet_tat_yeu_cau: str | None = Field(default=None, description="Nhóm khuyết tật yêu cầu cho vị trí (nếu có)")


class StableMatchingRequest(BaseModel):
    """
    Payload yêu cầu chạy thuật toán Gale-Shapley Stable Matching toàn diện.

    Mục đích:
        Nhận danh sách ứng viên, vị trí tuyển dụng, thiết lập cơ chế đề nghị (optimal)
        và các tham số lọc ngưỡng, hậu xử lý Top-K.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    candidates: list[CandidateProfileInput] = Field(description="Danh sách hồ sơ các ứng viên tham gia ghép cặp")
    jobs: list[JobProfileInput] = Field(description="Danh sách các vị trí tuyển dụng kèm quota")
    optimal: str = Field(
        default="candidate",
        description="Chiến lược chọn phía đề nghị: 'candidate' (Candidate-Proposing / C-optimal) hoặc 'employer' (Employer-Proposing / J-optimal)"
    )
    min_e_score: float = Field(default=0.0, ge=0.0, description="Ngưỡng sàn điểm E (Hard constraint)")
    min_c_score: float = Field(default=0.0, ge=0.0, description="Ngưỡng sàn điểm C (Hard constraint)")
    top_k: int = Field(default=3, ge=1, le=50, description="Số lượng gợi ý việc làm dự phòng (Top-K Recommendations) cho mỗi ứng viên")
    tie_break_seed: int = Field(default=20240517, description="Hạt giống băm số học để phá hòa tất định (Deterministic Tie-Breaking)")

    @field_validator("optimal")
    @classmethod
    def check_optimal_mode(cls, value: str) -> str:
        """
        Kiểm tra và chuẩn hóa chế độ đề nghị optimal.

        Tham số đầu vào:
            value (str): Giá trị chế độ ('candidate' hoặc 'employer').

        Kết quả trả về:
            str: Giá trị chữ thường chuẩn hóa.

        Raises:
            ValueError: Nếu giá trị không thuộc 'candidate' hoặc 'employer'.
        """
        # Chuẩn hóa về chữ thường không chứa khoảng trắng
        cleaned = value.strip().lower()
        # Đối chiếu với hai chế độ toán học được hỗ trợ
        if cleaned not in ("candidate", "employer"):
            raise ValueError("optimal phải là 'candidate' (C-optimal) hoặc 'employer' (J-optimal)")
        return cleaned

    @model_validator(mode="after")
    def validate_direct_track_and_quotas(self) -> StableMatchingRequest:
        valid_job_ids = {j.id for j in self.jobs if j.id} | {j.job_id for j in self.jobs if j.job_id}
        valid_job_ids.discard(None)

        direct_hired_counts: dict[str, int] = {}
        for c in self.candidates:
            c_id = c.id or c.candidate_id or "unknown"
            if c.direct_track:
                if not c.hired_job_id or not str(c.hired_job_id).strip():
                    raise ValueError(
                        f"Ứng viên '{c_id}' có direct_track=True nhưng thiếu hired_job_id "
                        "(bắt buộc phải chỉ định vị trí đã trúng tuyển)."
                    )
                if c.hired_job_id not in valid_job_ids:
                    raise ValueError(
                        f"Ứng viên '{c_id}' Direct Track trỏ tới vị trí không tồn tại: '{c.hired_job_id}'. "
                        f"Danh sách vị trí hợp lệ: {sorted(valid_job_ids)}"
                    )
                direct_hired_counts[c.hired_job_id] = (
                    direct_hired_counts.get(c.hired_job_id, 0) + 1
                )

        for j in self.jobs:
            j_id = j.id or j.job_id or "unknown"
            direct_count = direct_hired_counts.get(j_id, 0)
            total_filled = direct_count + j.filled_slots
            if total_filled > j.quota:
                raise ValueError(
                    f"Vị trí '{j_id}' có số lượng tuyển trực tiếp ({direct_count}) "
                    f"cộng số chỗ đã lấp ngoài hệ thống ({j.filled_slots}) = {total_filled} "
                    f"vượt quá hạn mức tuyển dụng (quota={j.quota})."
                )
        return self


class MatrixMatchingRequest(BaseModel):
    """
    Payload chạy Stable Matching từ ma trận điểm thô E và C (theo đúng ví dụ mục 7 của Báo cáo).

    Mục đích:
        Cho phép chạy Gale-Shapley trực tiếp khi người dùng đã có sẵn ma trận điểm
        Candidate Preference Scores và Employer Preference Scores cùng Quotas.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    candidate_scores: dict[str, dict[str, float]] = Field(
        description="Ma trận điểm C: {candidate_id: {job_id: score}} - mức độ yêu thích công việc của ứng viên"
    )
    employer_scores: dict[str, dict[str, float]] = Field(
        description="Ma trận điểm E: {job_id: {candidate_id: score}} - mức độ phù hợp của ứng viên trong mắt NTD"
    )
    job_quotas: dict[str, int] = Field(
        description="Hạn mức tuyển dụng của từng công việc: {job_id: quota}"
    )
    optimal: str = Field(
        default="candidate",
        description="Chế độ phía đề nghị: 'candidate' (mặc định) hoặc 'employer'"
    )
    top_k: int = Field(default=3, ge=1, le=50, description="Số lượng vị trí gợi ý Top-K dự phòng")
    tie_break_seed: int = Field(default=20240517, description="Hạt giống phá hòa tất định")

    @field_validator("optimal")
    @classmethod
    def check_optimal_mode(cls, value: str) -> str:
        """
        Kiểm tra và chuẩn hóa chế độ đề nghị cho ma trận điểm.

        Tham số đầu vào:
            value (str): Chuỗi chế độ truyền vào.

        Kết quả trả về:
            str: Chế độ chuẩn hóa ('candidate' hoặc 'employer').
        """
        cleaned = value.strip().lower()
        if cleaned not in ("candidate", "employer"):
            raise ValueError("optimal phải là 'candidate' hoặc 'employer'")
        return cleaned


class CandidateRecommendationResponse(BaseModel):
    """
    Thông tin một vị trí công việc được gợi ý trong danh sách Top-K cho ứng viên.

    Mục đích:
        Cung cấp các phương án dự phòng khi vị trí chính không thành công (Hậu xử lý Giai đoạn 4).
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    candidate_id: str = Field(description="ID ứng viên nhận gợi ý")
    job_id: str = Field(description="ID vị trí công việc được gợi ý")
    position: int = Field(description="Thứ hạng gợi ý (1 là ưu tiên cao nhất)")
    score: float = Field(description="Điểm phù hợp / thặng dư danh dư (Normalized Surplus Score)")
    is_primary: bool = Field(default=False, description="True nếu đây là cặp ghép cặp chính thức từ Gale-Shapley")
    explanation: str | None = Field(default=None, description="Lý do giải thích vì sao ứng viên được gợi ý (Mục 6.1 bước 7)")
    e_breakdown: dict[str, float] | None = Field(default=None, description="Bảng phân rã điểm thành phần phía NTD")
    c_breakdown: dict[str, float] | None = Field(default=None, description="Bảng phân rã điểm thành phần phía ứng viên")


class MatchedPairResponse(BaseModel):
    """
    Chi tiết một cặp ghép cặp ổn định thành công (Candidate, Job).

    Mục đích:
        Cung cấp thông tin minh bạch về điểm số hai chiều và thứ hạng của nhau trong danh sách ưu tiên.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    candidate_id: str = Field(description="ID ứng viên")
    job_id: str = Field(description="ID vị trí tuyển dụng")
    c_score: float | None = Field(default=None, description="Điểm C (mức độ hài lòng của ứng viên)")
    e_score: float | None = Field(default=None, description="Điểm E (mức độ phù hợp kỹ năng từ phía NTD)")
    candidate_rank: int | None = Field(default=None, description="Thứ bậc của job trong danh sách ưu tiên của ứng viên (1 là số 1)")
    job_rank: int | None = Field(default=None, description="Thứ bậc của ứng viên trong danh sách ưu tiên của job (1 là số 1)")
    explanation: str | None = Field(default=None, description="Lý do giải thích vì sao ứng viên được gợi ý (Mục 6.1 bước 7)")
    e_breakdown: dict[str, float] | None = Field(default=None, description="Phân rã điểm thành phần E")
    c_breakdown: dict[str, float] | None = Field(default=None, description="Phân rã điểm thành phần C")


class StableMatchingMetricsResponse(BaseModel):
    """
    Các chỉ số đo lường hiệu quả (KPIs & Metrics) của bài toán Ghép cặp Ổn định.

    Mục đích:
        Đánh giá tính ổn định toán học (0% Blocking Pairs), mức độ hài lòng hai phía và tỷ lệ lấp đầy Quota.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    total_candidates: int = Field(description="Tổng số ứng viên tham gia thị trường")
    total_jobs: int = Field(description="Tổng số vị trí tuyển dụng")
    total_quota: int = Field(description="Tổng số chỉ tiêu tuyển dụng của toàn hệ thống")
    matched_pairs_count: int = Field(description="Số lượng cặp ghép thành công")
    unmatched_candidates_count: int = Field(description="Số ứng viên chưa được phân bổ việc làm")
    unmatched_jobs_count: int = Field(description="Số công việc còn vị trí trống")
    blocking_pairs_count: int = Field(description="Số cặp gây bất ổn (Target: 0 theo định lý Gale-Shapley)")
    blocking_pair_rate: float = Field(description="Tỷ lệ cặp gây bất ổn (phải đạt 0.0%)")
    is_stable: bool = Field(description="Khẳng định kết quả ghép cặp đạt 100% tính ổn định toán học")
    candidate_satisfaction_avg: float = Field(description="Mức độ hài lòng trung bình phía ứng viên (thang điểm 0 - 1)")
    employer_satisfaction_avg: float = Field(description="Mức độ hài lòng trung bình phía nhà tuyển dụng (thang điểm 0 - 1)")
    fill_rate: float = Field(description="Tỷ lệ lấp đầy vị trí tuyển dụng (tổng tuyển / tổng quota)")
    proposals_count: int = Field(description="Tổng số lượt đề xuất phát sinh trong quá trình chạy thuật toán")
    average_candidate_rank: float | None = Field(default=None, description="Hạng trung bình của công việc trong danh sách ưu tiên ứng viên (Mục 6.5)")
    average_job_rank: float | None = Field(default=None, description="Hạng trung bình của ứng viên trong danh sách ưu tiên NTD (Mục 6.5)")
    unmatched_candidate_rate: float | None = Field(default=None, description="Tỷ lệ ứng viên chưa được ghép (Mục 6.5)")
    congestion_rate: float | None = Field(default=None, description="Mức độ xung đột giới thiệu nhiều nơi (Mục 6.5)")


class StableMatchingResponse(BaseModel):
    """
    Response kết quả hoàn chỉnh của thuật toán Gale-Shapley Stable Matching.

    Mục đích:
        Đóng gói toàn diện kết quả phân bổ, gợi ý Top-K và báo cáo chỉ số KPIs trả về cho người dùng/API Gateway.
    """

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    status: str = Field(default="success", description="Trạng thái thực thi ('success')")
    optimal: str = Field(description="Chế độ tối ưu đã thực hiện ('candidate' hoặc 'employer')")
    matches: dict[str, str] = Field(
        description="Bảng phân bổ ánh xạ trực tiếp: {candidate_id: job_id}"
    )
    matched_pairs: list[MatchedPairResponse] = Field(
        description="Danh sách chi tiết từng cặp ghép đôi ổn định"
    )
    job_assignments: dict[str, list[str]] = Field(
        description="Danh sách trúng tuyển theo từng vị trí: {job_id: [candidate_id_1, candidate_id_2, ...]}"
    )
    unmatched_candidates: list[str] = Field(
        description="Danh sách các ứng viên không được phân bổ"
    )
    unmatched_jobs: list[str] = Field(
        description="Danh sách các công việc chưa tuyển đủ chỉ tiêu quota"
    )
    recommendations: dict[str, list[CandidateRecommendationResponse]] = Field(
        description="Danh sách Top-K vị trí gợi ý dự phòng cho từng ứng viên"
    )
    metrics: StableMatchingMetricsResponse = Field(
        description="Báo cáo định lượng các chỉ số hiệu quả KPIs"
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Thời gian thực thi toàn bộ pipeline thuật toán (mili-giây)"
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Danh sách các cảnh báo dữ liệu hoặc nghiệp vụ phát hiện trong quá trình xử lý"
    )


# ---------------------------------------------------------------------------
# PYDANTIC V2 TYPE ADAPTERS CHO XÁC THỰC & TUẦN TỰ HÓA BATCH TỐC ĐỘ CAO
# ---------------------------------------------------------------------------

CandidateProfileListAdapter: TypeAdapter[list[CandidateProfileInput]] = TypeAdapter(
    list[CandidateProfileInput]
)
JobProfileListAdapter: TypeAdapter[list[JobProfileInput]] = TypeAdapter(
    list[JobProfileInput]
)
MatchedPairListAdapter: TypeAdapter[list[MatchedPairResponse]] = TypeAdapter(
    list[MatchedPairResponse]
)
CandidateRecommendationListAdapter: TypeAdapter[list[CandidateRecommendationResponse]] = (
    TypeAdapter(list[CandidateRecommendationResponse])
)
StableMatchingResponseAdapter: TypeAdapter[StableMatchingResponse] = TypeAdapter(
    StableMatchingResponse
)
