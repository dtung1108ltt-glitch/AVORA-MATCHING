#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Giai đoạn 4 - Hậu xử lý Top-K Recommendation.

Deferred Acceptance chỉ gán mỗi ứng viên **đúng một** vị trí. Trong thực tế
tuyển dụng, ứng viên cần danh sách **Top-K việc làm dự phòng** (backup options)
để chủ động liên hệ khi vị trí chính không thành công.

Module cung cấp hai cách sinh Top-K:

1. :func:`generate_top_k` - **Normalized Surplus Score**: chuẩn hóa (min-max)
   điểm ``C`` theo từng ứng viên và điểm ``E`` theo từng vị trí, rồi trộn có
   trọng số thành một điểm danh dư. Ghép cặp chính (primary) luôn đứng đầu,
   sau đó là các vị trí khả thi điểm cao nhất.
2. :func:`iterative_clearance_rounds` - **Iterative Clearance Rounds**: chạy
   lại DA nhiều vòng; sau mỗi vòng loại bỏ cặp vừa chốt khỏi cả hai danh sách ưu
   tiên để vòng sau lộ ra nguyện vọng dự phòng kế tiếp.

Cả hai hàm đều trả về ``dict[candidate, list[Recommendation]]`` với ``position``
bắt đầu từ 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.core.matching_engine import GaleShapleyMatching, MatchingResult
from src.core.preprocessor import Id, PreferenceProfile

__all__ = [
    "Recommendation",
    "generate_top_k",
    "iterative_clearance_rounds",
    "normalized_surplus_scores",
]


@dataclass(frozen=True, slots=True)
class Recommendation:
    """Một gợi ý việc làm cho ứng viên.

    Attributes:
        candidate_id: ID ứng viên nhận gợi ý.
        job_id: ID vị trí được gợi ý.
        position: Thứ hạng gợi ý, bắt đầu từ 1.
        score: Điểm danh dư đã chuẩn hóa trong ``[0, 1]``.
        is_primary: ``True`` nếu đây là vị trí ứng viên được ghép chính thức.
        explanation: Lý do giải thích vì sao ứng viên được gợi ý (Mục 6.1 bước 7).
        breakdown: Bảng phân rã điểm thành phần.
    """

    candidate_id: Id
    job_id: Id
    position: int
    score: float
    is_primary: bool = False
    explanation: str = ""
    breakdown: dict[str, float] = field(default_factory=dict)


def normalized_surplus_scores(
    profile: PreferenceProfile,
    *,
    weight_c: float = 0.5,
    weight_e: float = 0.5,
) -> dict[tuple[Id, Id], float]:
    """Tính điểm danh dư chuẩn hóa cho mọi cặp khả thi hai chiều.

    Công thức:

    .. math::
        surplus(c, j) = w_c \\cdot \\hat{C}(c, j) + w_e \\cdot \\hat{E}(c, j)

    trong đó ``\\hat{C}`` chuẩn hóa min-max điểm ``C`` *theo từng ứng viên* (để
    mỗi người có thang đo riêng) và ``\\hat{E}`` chuẩn hóa min-max điểm ``E``
    *theo từng vị trí*. Nếu mọi giá trị bằng nhau, chuẩn hóa bằng ``1.0``.

    Args:
        profile: Hồ sơ ưu tiên.
        weight_c: Trọng số phía ứng viên.
        weight_e: Trọng số phía nhà tuyển dụng.

    Returns:
        Mapping ``(candidate, job) -> surplus`` chỉ gồm cặp *acceptable* hai chiều.
    """

    candidate_norm: dict[Id, dict[Id, float]] = {}
    for candidate, jobs in profile.candidate_preferences.items():
        row = profile.c_scores.get(candidate, {})
        values = [row[job] for job in jobs if job in row]
        low = min(values) if values else 0.0
        high = max(values) if values else 0.0
        span = high - low
        candidate_norm[candidate] = {
            job: (1.0 if span <= 0 else (row[job] - low) / span)
            for job in jobs
            if job in row
        }

    job_norm: dict[Id, dict[Id, float]] = {}
    for job, candidates in profile.job_preferences.items():
        values = [
            profile.e_scores[candidate][job]
            for candidate in candidates
            if job in profile.e_scores.get(candidate, {})
        ]
        low = min(values) if values else 0.0
        high = max(values) if values else 0.0
        span = high - low
        job_norm[job] = {}
        for candidate in candidates:
            if job not in profile.e_scores.get(candidate, {}):
                continue
            job_norm[job][candidate] = (
                1.0 if span <= 0
                else (profile.e_scores[candidate][job] - low) / span
            )

    surplus: dict[tuple[Id, Id], float] = {}
    for candidate, jobs in profile.candidate_preferences.items():
        for job in jobs:
            e_lookup = job_norm.get(job, {})
            if candidate not in e_lookup:
                continue
            surplus[(candidate, job)] = (
                weight_c * candidate_norm.get(candidate, {}).get(job, 0.0)
                + weight_e * e_lookup[candidate]
            )
    return surplus


def generate_top_k(
    profile: PreferenceProfile,
    result: MatchingResult,
    *,
    k: int = 3,
    weight_c: float = 0.5,
    weight_e: float = 0.5,
    include_primary: bool = True,
) -> dict[Id, list[Recommendation]]:
    """Sinh danh sách Top-K việc làm gợi ý cho từng ứng viên.

    Cách hoạt động:

    1. Nếu ứng viên đã được ghép, cặp ghép chính luôn là gợi ý số 1.
    2. Các vị trí còn lại được xếp theo :func:`normalized_surplus_scores` giảm
       dần (phá hòa theo ``str(job_id)`` để tất định), cắt lấy cho đủ ``k``.

    Args:
        profile: Hồ sơ ưu tiên đã tiền xử lý.
        result: Kết quả matching từ ``GaleShapleyMatching``.
        k: Số gợi ý tối đa cho mỗi ứng viên.
        weight_c: Trọng số điểm ``C`` khi trộn.
        weight_e: Trọng số điểm ``E`` khi trộn.
        include_primary: Có đưa cặp ghép chính vào vị trí đầu hay không.

    Returns:
        ``{candidate: [Recommendation, ...]}`` - danh sách rỗng nếu ``k <= 0``
        hoặc ứng viên không có nguyện vọng khả thi nào.
    """

    if k <= 0:
        return {candidate: [] for candidate in profile.candidate_ids}

    surplus = normalized_surplus_scores(profile, weight_c=weight_c, weight_e=weight_e)
    meta = profile.metadata if isinstance(profile.metadata, dict) else {}
    explanations: dict[object, str] = meta.get("explanations", {})
    breakdowns: dict[object, dict[str, float]] = meta.get("breakdowns", {})

    recommendations: dict[Id, list[Recommendation]] = {}
    for candidate in profile.candidate_ids:
        items: list[Recommendation] = []
        primary = result.matches.get(candidate)

        if include_primary and primary is not None:
            exp_text = explanations.get((candidate, primary), "")
            bk_data = breakdowns.get((candidate, primary), {})
            items.append(
                Recommendation(
                    candidate_id=candidate,
                    job_id=primary,
                    position=1,
                    score=round(surplus.get((candidate, primary), 0.0), 6),
                    is_primary=True,
                    explanation=exp_text,
                    breakdown=bk_data,
                )
            )

        backups = [
            job
            for job in profile.candidate_preferences.get(candidate, ())
            if job != primary
        ]
        backups.sort(
            key=lambda job: (-surplus.get((candidate, job), 0.0), str(job))
        )

        for job in backups:
            if len(items) >= k:
                break
            exp_text = explanations.get((candidate, job), "")
            bk_data = breakdowns.get((candidate, job), {})
            items.append(
                Recommendation(
                    candidate_id=candidate,
                    job_id=job,
                    position=len(items) + 1,
                    score=round(surplus.get((candidate, job), 0.0), 6),
                    is_primary=False,
                    explanation=exp_text,
                    breakdown=bk_data,
                )
            )

        recommendations[candidate] = items[:k]

    return recommendations


def iterative_clearance_rounds(
    profile: PreferenceProfile,
    *,
    k: int = 3,
    optimal: str = "candidate",
    reduce_quota: bool = False,
) -> dict[Id, list[Recommendation]]:
    """Sinh Top-K bằng các vòng lặp xóa dần (Iterative Clearance Rounds).

    Vòng 1 chạy DA đầy đủ. Sau mỗi vòng, với từng ứng viên đã được ghép ở vị trí
    ``j``, cặp ``(c, j)`` bị loại khỏi **cả hai** danh sách ưu tiên, rồi DA được
    chạy lại. Vì vậy vòng sau buộc phải lộ ra nguyện vọng dự phòng kế tiếp.

    Args:
        profile: Hồ sơ ưu tiên đã tiền xử lý.
        k: Số vòng (bằng số gợi ý tối đa mỗi ứng viên).
        optimal: Chế độ DA dùng cho mỗi vòng (``"candidate"`` hoặc ``"employer"``).
        reduce_quota: Nếu ``True``, mỗi cặp chốt còn làm giảm ``q_j`` đi 1 để mô
            phỏng đúng tình trạng khan hiếm slot; mặc định ``False`` (chỉ loại
            cạnh, giữ nguyên quota để mọi ứng viên đều có đủ K gợi ý).

    Returns:
        ``{candidate: [Recommendation, ...]}`` theo thứ tự vòng.
    """

    if k <= 0:
        return {candidate: [] for candidate in profile.candidate_ids}

    candidate_lists: dict[Id, list[Id]] = {
        candidate: list(profile.candidate_preferences.get(candidate, ()))
        for candidate in profile.candidate_ids
    }
    job_lists: dict[Id, list[Id]] = {
        job: list(profile.job_preferences.get(job, ()))
        for job in profile.job_ids
    }
    quotas = dict(profile.quotas)

    recommendations: dict[Id, list[Recommendation]] = {
        candidate: [] for candidate in profile.candidate_ids
    }
    surplus = normalized_surplus_scores(profile)

    for round_index in range(k):
        engine = GaleShapleyMatching(
            profile,
            optimal=optimal,
            candidate_preferences={
                candidate: tuple(jobs) for candidate, jobs in candidate_lists.items()
            },
            job_preferences={
                job: tuple(candidates) for job, candidates in job_lists.items()
            },
            quotas=quotas,
        )
        round_result = engine.solve()

        if not round_result.matches:
            break

        for candidate in profile.candidate_ids:
            job = round_result.matches.get(candidate)
            if job is None:
                continue
            if any(rec.job_id == job for rec in recommendations[candidate]):
                continue

            recommendations[candidate].append(
                Recommendation(
                    candidate_id=candidate,
                    job_id=job,
                    position=len(recommendations[candidate]) + 1,
                    score=round(surplus.get((candidate, job), 0.0), 6),
                    is_primary=(round_index == 0),
                )
            )

            # Loại cạnh đã chốt khỏi cả hai phía cho vòng sau.
            if job in candidate_lists.get(candidate, ()):
                candidate_lists[candidate] = [
                    x for x in candidate_lists[candidate] if x != job
                ]
            if candidate in job_lists.get(job, ()):
                job_lists[job] = [x for x in job_lists[job] if x != candidate]
            if reduce_quota:
                quotas[job] = max(0, quotas[job] - 1)

        if not any(candidate_lists.values()):
            break

    return recommendations
