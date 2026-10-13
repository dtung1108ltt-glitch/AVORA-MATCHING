#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mô hình dữ liệu hồ sơ chuẩn hóa cho Ứng viên và Nhà tuyển dụng.

Theo đặc tả Nâng cấp thiết kế mục 6.1 (Thu thập hồ sơ chuẩn hóa) và 6.2 (Công thức chấm điểm):
- Ứng viên (CandidateProfile): kỹ năng, nhu cầu tiếp cận (thiết yếu & nên có),
  lương mong muốn, hình thức làm việc, thời gian, khu vực, thứ tự ưu tiên tự xếp, kinh nghiệm.
- Nhà tuyển dụng (JobProfile): yêu cầu bắt buộc và mong muốn, số vị trí (quota),
  kỹ năng, hỗ trợ tiếp cận, khoảng lương chi trả, hình thức, khu vực, kinh nghiệm yêu cầu.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class CandidateProfile:
    """Hồ sơ ứng viên chuẩn hóa theo Mục 6.1.

    Attributes:
        candidate_id: Định danh duy nhất của ứng viên.
        skills: Kỹ năng dạng vector float list, chuỗi văn bản mô tả, hoặc danh sách từ khóa.
        essential_access_needs: Danh sách nhu cầu tiếp cận thiết yếu (Hard constraint).
        nice_to_have_access_needs: Danh sách nhu cầu tiếp cận nên có (Soft component trong E).
        expected_salary_min: Mức lương mong muốn tối thiểu (triệu VNĐ hoặc VNĐ).
        expected_salary_max: Mức lương mong muốn tối đa.
        work_mode: Hình thức làm việc ('remote', 'hybrid', 'on-site').
        work_hours: Thời gian làm việc ('full-time', 'part-time', 'flexible').
        location: Khu vực sinh sống / làm việc (ví dụ: 'Hà Nội', 'TP.HCM', 'Đà Nẵng').
        priority_ranking: Thứ tự ưu tiên 3 yếu tố tự xếp ('income', 'distance_or_remote', 'career_path').
        experience_years: Số năm kinh nghiệm làm việc hiện tại.
        career_orientation: Định hướng nghề nghiệp (chuỗi văn bản hoặc vector).
        disability_group: Nhóm khuyết tật (metadata chỉ dùng để hỗ trợ tiếp cận, KHÔNG suy diễn năng lực).
        direct_track: True nếu ứng viên đã được tuyển dụng qua kênh trực tiếp (loại khỏi pool).
        metadata: Dữ liệu mở rộng khác.
    """

    candidate_id: str
    skills: list[float] | str | list[str] = field(default_factory=list)
    essential_access_needs: list[str] = field(default_factory=list)
    nice_to_have_access_needs: list[str] = field(default_factory=list)
    expected_salary_min: float = 0.0
    expected_salary_max: float = 0.0
    work_mode: str = "on-site"
    work_hours: str = "full-time"
    location: str = ""
    priority_ranking: list[str] = field(default_factory=lambda: ["income", "distance_or_remote", "career_path"])
    experience_years: float = 0.0
    career_orientation: list[float] | str = ""
    disability_group: str | None = None
    direct_track: bool = False
    hired_job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_salary_range(self) -> tuple[float, float]:
        """Trả về khoảng lương mong muốn (min, max)."""
        s_min = max(0.0, float(self.expected_salary_min))
        s_max = max(s_min, float(self.expected_salary_max))
        return s_min, s_max

    def get_priority_weights(self) -> dict[str, float]:
        """Quy thứ hạng ưu tiên thành trọng số cho C theo Mục 6.2:

        Thứ nhất: 0.5, thứ hai: 0.3, thứ ba: 0.2.
        Nếu thiếu dữ liệu hoặc không đủ 3 thứ hạng, dùng mặc định chung:
        - income: 0.35
        - distance_or_remote: 0.30
        - career_path: 0.35
        """
        valid_dims = {"income", "distance_or_remote", "career_path"}
        ranks = [p for p in self.priority_ranking if p in valid_dims]
        if len(set(ranks)) == 3:
            weight_ladder = [0.5, 0.3, 0.2]
            return {dim: weight_ladder[i] for i, dim in enumerate(ranks)}

        # Mặc định chung khi thiếu dữ liệu hoặc danh sách xếp không chuẩn
        return {
            "income": 0.35,
            "distance_or_remote": 0.30,
            "career_path": 0.35,
        }


@dataclass
class JobProfile:
    """Hồ sơ vị trí tuyển dụng chuẩn hóa theo Mục 6.1.

    Attributes:
        job_id: Định danh duy nhất của vị trí việc làm.
        title: Tên chức danh công việc.
        quota: Số vị trí cần tuyển dụng (Many-to-One quota q_j >= 1).
        skills_required: Kỹ năng yêu cầu (vector embedding, chuỗi văn bản hoặc danh sách từ khóa).
        access_support_provided: Danh sách các điều kiện/tiện nghi tiếp cận mà doanh nghiệp đáp ứng.
        salary_min: Mức lương chi trả tối thiểu.
        salary_max: Mức lương chi trả tối đa.
        work_mode: Hình thức làm việc yêu cầu ('remote', 'hybrid', 'on-site').
        work_hours: Thời gian làm việc ('full-time', 'part-time', 'flexible').
        location: Địa điểm / khu vực làm việc (ví dụ: 'Hà Nội', 'TP.HCM').
        experience_required_years: Số năm kinh nghiệm tối thiểu yêu cầu.
        career_domain: Lĩnh vực / định hướng nghề của vị trí (chuỗi văn bản hoặc vector).
        mandatory_requirements: Các yêu cầu bắt buộc bổ sung.
        metadata: Dữ liệu mở rộng khác.
    """

    job_id: str
    title: str = ""
    quota: int = 1
    filled_slots: int = 0
    skills_required: list[float] | str | list[str] = field(default_factory=list)
    access_support_provided: list[str] = field(default_factory=list)
    salary_min: float = 0.0
    salary_max: float = 0.0
    work_mode: str = "on-site"
    work_hours: str = "full-time"
    location: str = ""
    experience_required_years: float = 0.0
    career_domain: list[float] | str = ""
    mandatory_requirements: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_quota(self) -> int:
        """Số chỉ tiêu tuyển dụng thực tế còn lại cho batch matching."""
        return max(0, int(self.quota) - int(self.filled_slots))

    def get_salary_range(self) -> tuple[float, float]:
        """Trả về khoảng lương chi trả (min, max)."""
        s_min = max(0.0, float(self.salary_min))
        s_max = max(s_min, float(self.salary_max))
        return s_min, s_max


@dataclass
class ExecutionAudit:
    """Bản ghi kiểm toán và tái lập kết quả theo Mục 6.4.

    Attributes:
        run_id: Mã định danh duy nhất của lượt chạy.
        timestamp: Thời điểm thực thi ISO-8601.
        seed: Hạt giống ngẫu nhiên cố định.
        weights_config: Cấu hình trọng số E và C đã dùng.
        data_version: Phiên bản bộ dữ liệu đầu vào.
        metadata: Thông tin kiểm toán bổ sung.
    """

    run_id: str
    timestamp: str
    seed: int = 20_240_517
    weights_config: dict[str, Any] = field(default_factory=dict)
    data_version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Chuyển bản ghi kiểm toán thành dict."""
        return asdict(self)

