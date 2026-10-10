#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Đo lường chất lượng matching - Blocking Pairs, Satisfaction, Fill Rate.

Module độc lập với engine: chỉ cần một ``PreferenceProfile`` và một
``MatchingResult`` là tính được toàn bộ chỉ số. Nhờ đó có thể so sánh kết quả
của Gale-Shapley với bất kỳ baseline nào.

Ba chỉ số cốt lõi:

* **Blocking Pairs** - với matching ổn định, tỷ lệ này **bắt buộc bằng 0%**
  (Gale & Shapley 1962). Bất kỳ blocking pair nào cũng là bằng chứng thuật toán
  sai.
* **Average Satisfaction** - điểm ``E`` và ``C`` trung bình của các cặp đã ghép;
  dùng để so sánh *chất lượng* giữa các matching cùng ổn định.
* **Fill Rate** - tỷ lệ slot được lấp đầy trên tổng hạn mức (hiệu quả sử dụng
  nguồn lực).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.core.matching_engine import MatchingResult
from src.core.preprocessor import Id, PreferenceProfile

__all__ = [
    "BlockingPairReport",
    "EvaluationReport",
    "FillRateReport",
    "SatisfactionReport",
    "average_satisfaction",
    "calculate_blocking_pairs",
    "evaluate",
    "fill_rate",
    "find_blocking_pairs",
]


@dataclass
class BlockingPairReport:
    """Kết quả kiểm tra ổn định.

    Attributes:
        count: Số blocking pair phát hiện được (phải là 0 với matching ổn định).
        pairs: Danh sách cặp ``(candidate, job)`` chặn.
        rate: ``count / tổng số cặp khả thi hai chiều``.
        is_stable: ``True`` khi ``count == 0``.
    """

    count: int
    pairs: list[tuple[Id, Id]] = field(default_factory=list)
    rate: float = 0.0
    is_stable: bool = True


@dataclass
class SatisfactionReport:
    """Mức hài lòng trung bình của hai phía.

    Attributes:
        average_e: Điểm ``E`` trung bình trên các cặp đã ghép (phía nhà tuyển dụng).
        average_c: Điểm ``C`` trung bình trên các cặp đã ghép (phía ứng viên).
        min_e: Điểm ``E`` thấp nhất trong các cặp đã ghép.
        min_c: Điểm ``C`` thấp nhất trong các cặp đã ghép.
        matched_pairs: Số cặp đã ghép.
        coverage: Tỷ lệ ứng viên được ghép trên tổng số ứng viên.
    """

    average_e: float = 0.0
    average_c: float = 0.0
    min_e: float = 0.0
    min_c: float = 0.0
    matched_pairs: int = 0
    coverage: float = 0.0


@dataclass
class FillRateReport:
    """Hiệu quả lấp đầy slot tuyển dụng.

    Attributes:
        overall: Tổng slot đã lấp / tổng hạn mức.
        by_job: Tỷ lệ lấp đầy của từng vị trí có quota dương.
        filled_slots: Số slot đã lấp.
        total_quota: Tổng hạn mức.
    """

    overall: float = 0.0
    by_job: dict[Id, float] = field(default_factory=dict)
    filled_slots: int = 0
    total_quota: int = 0


@dataclass
class EvaluationReport:
    """Gói toàn bộ chỉ số đánh giá cho một matching."""

    blocking: BlockingPairReport = field(default_factory=BlockingPairReport)
    satisfaction: SatisfactionReport = field(default_factory=SatisfactionReport)
    fill_rate: FillRateReport = field(default_factory=FillRateReport)

    def as_dict(self) -> dict[str, Any]:
        """Chuyển báo cáo thành ``dict`` lồng nhau (tiện xuất JSON/CSV)."""

        return asdict(self)


def find_blocking_pairs(
    profile: PreferenceProfile,
    result: MatchingResult,
    *,
    max_pairs: int | None = None,
) -> list[tuple[Id, Id]]:
    """Liệt kê mọi blocking pair của một matching.

    Cặp ``(c, j)`` là **blocking pair** khi thoả mãn đồng thời:

    1. ``c`` và ``j`` chấp nhận được lẫn nhau (nằm trong danh sách ưu tiên của
       nhau).
    2. ``c`` thích ``j`` hơn đối tác hiện tại (hoặc ``c`` đang thất nghiệp).
    3. ``j`` còn slot trống, **hoặc** ``j`` thích ``c`` hơn ứng viên kém nhất
       trong số những người nó đang giữ.

    Args:
        profile: Hồ sơ ưu tiên.
        result: Kết quả matching cần kiểm tra.
        max_pairs: Dừng sớm sau khi tìm đủ số cặp này (tiện cho thị trường lớn).

    Returns:
        Danh sách ``(candidate, job)`` chặn. Rỗng nghĩa là matching ổn định.
    """

    candidate_rank: dict[Id, dict[Id, int]] = {
        candidate: {job: rank for rank, job in enumerate(jobs)}
        for candidate, jobs in profile.candidate_preferences.items()
    }
    job_rank: dict[Id, dict[Id, int]] = {
        job: {candidate: rank for rank, candidate in enumerate(candidates)}
        for job, candidates in profile.job_preferences.items()
    }

    blocking: list[tuple[Id, Id]] = []

    for candidate in profile.candidate_ids:
        current_job = result.matches.get(candidate)
        ranks = candidate_rank.get(candidate, {})
        current_rank = ranks.get(current_job) if current_job is not None else None

        for job in profile.candidate_preferences.get(candidate, ()):
            if job == current_job:
                continue

            # Điều kiện 2: ứng viên thích job này hơn đối tác hiện tại.
            if current_rank is not None and not (ranks[job] < current_rank):
                continue

            # Điều kiện 1: vị trí cũng phải chấp nhận ứng viên.
            if candidate not in job_rank.get(job, {}):
                continue

            assigned = result.job_assignments.get(job, [])
            quota = profile.quotas.get(job, 0)

            # Điều kiện 3.
            if len(assigned) < quota:
                blocking.append((candidate, job))
            else:
                worst_rank = max(job_rank[job].get(held, -1) for held in assigned)
                if job_rank[job][candidate] < worst_rank:
                    blocking.append((candidate, job))

            if max_pairs is not None and len(blocking) >= max_pairs:
                return blocking

    return blocking


def calculate_blocking_pairs(
    profile: PreferenceProfile,
    result: MatchingResult,
    *,
    max_pairs: int | None = None,
) -> BlockingPairReport:
    """Tính báo cáo blocking pair kèm tỷ lệ trên tổng cặp khả thi.

    Args:
        profile: Hồ sơ ưu tiên.
        result: Kết quả matching.
        max_pairs: Giới hạn số cặp liệt kê chi tiết.

    Returns:
        ``BlockingPairReport`` với ``rate = count / |feasible_pairs|``.
    """

    pairs = find_blocking_pairs(profile, result, max_pairs=max_pairs)
    total_feasible = len(profile.feasible_pairs())
    rate = (len(pairs) / total_feasible) if total_feasible else 0.0
    return BlockingPairReport(
        count=len(pairs),
        pairs=pairs,
        rate=rate,
        is_stable=(len(pairs) == 0),
    )


def average_satisfaction(
    profile: PreferenceProfile,
    result: MatchingResult,
) -> SatisfactionReport:
    """Tính điểm ``E`` và ``C`` trung bình của các cặp đã ghép.

    Args:
        profile: Hồ sơ ưu tiên.
        result: Kết quả matching.

    Returns:
        ``SatisfactionReport``; các trường bằng 0 nếu chưa ghép được cặp nào.
    """

    e_values: list[float] = []
    c_values: list[float] = []

    for candidate, job in result.matches.items():
        e_value = profile.e_scores.get(candidate, {}).get(job)
        if e_value is not None:
            e_values.append(e_value)
        c_value = profile.c_scores.get(candidate, {}).get(job)
        if c_value is not None:
            c_values.append(c_value)

    matched = len(result.matches)
    total_candidates = len(profile.candidate_ids)

    return SatisfactionReport(
        average_e=(sum(e_values) / len(e_values)) if e_values else 0.0,
        average_c=(sum(c_values) / len(c_values)) if c_values else 0.0,
        min_e=min(e_values) if e_values else 0.0,
        min_c=min(c_values) if c_values else 0.0,
        matched_pairs=matched,
        coverage=(matched / total_candidates) if total_candidates else 0.0,
    )


def fill_rate(
    profile: PreferenceProfile,
    result: MatchingResult,
) -> FillRateReport:
    """Tính tỷ lệ lấp đầy slot trên từng vị trí và toàn thị trường.

    Args:
        profile: Hồ sơ ưu tiên.
        result: Kết quả matching.

    Returns:
        ``FillRateReport`` với ``overall = filled_slots / total_quota``.
    """

    by_job: dict[Id, float] = {}
    filled_slots = 0
    total_quota = 0

    for job in profile.job_ids:
        quota = profile.quotas.get(job, 0)
        if quota <= 0:
            continue
        filled = min(len(result.job_assignments.get(job, [])), quota)
        by_job[job] = filled / quota
        filled_slots += filled
        total_quota += quota

    return FillRateReport(
        overall=(filled_slots / total_quota) if total_quota else 0.0,
        by_job=by_job,
        filled_slots=filled_slots,
        total_quota=total_quota,
    )


def evaluate(
    profile: PreferenceProfile,
    result: MatchingResult,
    *,
    max_blocking_pairs: int | None = None,
) -> EvaluationReport:
    """Chạy toàn bộ bộ chỉ số đánh giá.

    Args:
        profile: Hồ sơ ưu tiên.
        result: Kết quả matching.
        max_blocking_pairs: Giới hạn liệt kê blocking pair chi tiết.

    Returns:
        ``EvaluationReport`` gồm blocking, satisfaction và fill rate.
    """

    return EvaluationReport(
        blocking=calculate_blocking_pairs(
            profile, result, max_pairs=max_blocking_pairs
        ),
        satisfaction=average_satisfaction(profile, result),
        fill_rate=fill_rate(profile, result),
    )