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

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from src.core.matching_engine import MatchingResult
from src.core.preprocessor import Id, PreferenceProfile

__all__ = [
    "BlockingPairReport",
    "ConflictReport",
    "CoverageReport",
    "EvaluationReport",
    "FairnessAuditReport",
    "FillRateReport",
    "RankReport",
    "SatisfactionReport",
    "audit_fairness",
    "average_ranks",
    "average_satisfaction",
    "calculate_blocking_pairs",
    "calculate_ndcg_at_k",
    "calculate_precision_at_k",
    "calculate_vector_recall",
    "conflict_metrics",
    "coverage_metrics",
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
class RankReport:
    """Hạng trung bình của đối tác trong danh sách ưu tiên theo Mục 6.5.

    Attributes:
        average_candidate_rank: Hạng trung bình của công việc trong mắt ứng viên (1 là số 1).
        average_job_rank: Hạng trung bình của ứng viên trong mắt nhà tuyển dụng (1 là số 1).
        matched_pairs: Số cặp đã ghép.
    """

    average_candidate_rank: float = 0.0
    average_job_rank: float = 0.0
    matched_pairs: int = 0


@dataclass
class CoverageReport:
    """Độ phủ thị trường và tỷ lệ chưa được ghép theo Mục 6.5.

    Attributes:
        unmatched_candidate_count: Số ứng viên chưa được ghép.
        unmatched_candidate_rate: Tỷ lệ ứng viên chưa được ghép (unmatched / total).
        unmatched_job_count: Số vị trí việc làm chưa tuyển đủ chỉ tiêu.
        unmatched_job_rate: Tỷ lệ vị trí chưa tuyển đủ chỉ tiêu.
        total_candidates: Tổng số ứng viên.
        total_jobs: Tổng số vị trí tuyển dụng.
    """

    unmatched_candidate_count: int = 0
    unmatched_candidate_rate: float = 0.0
    unmatched_job_count: int = 0
    unmatched_job_rate: float = 0.0
    total_candidates: int = 0
    total_jobs: int = 0


@dataclass
class ConflictReport:
    """Đo lường mức độ xung đột và tắc nghẽn (Congestion Rate) theo Mục 6.5.

    Attributes:
        multi_assigned_candidates: Số ứng viên bị phân bổ / giới thiệu cho > 1 nơi.
        congestion_rate: Tỷ lệ xung đột (multi_assigned / total_matched).
        max_assignments_per_candidate: Số lượng vị trí tối đa mà 1 ứng viên bị gán.
    """

    multi_assigned_candidates: int = 0
    congestion_rate: float = 0.0
    max_assignments_per_candidate: int = 1


@dataclass
class FairnessAuditReport:
    """Báo cáo kiểm toán công bằng và tỷ lệ ghép giữa các nhóm theo Mục 6.6.

    Attributes:
        group_match_rates: Tỷ lệ trúng tuyển theo từng nhóm (nhóm khuyết tật).
        disparate_impact_ratio: Tỷ số giữa nhóm thấp nhất và nhóm cao nhất (Four-Fifths Rule).
        is_fair: True nếu disparate_impact_ratio >= 0.80.
    """

    group_match_rates: dict[str, float] = field(default_factory=dict)
    disparate_impact_ratio: float = 1.0
    is_fair: bool = True


@dataclass
class EvaluationReport:
    """Gói toàn bộ chỉ số đánh giá cho một matching."""

    blocking: BlockingPairReport = field(default_factory=BlockingPairReport)
    satisfaction: SatisfactionReport = field(default_factory=SatisfactionReport)
    fill_rate: FillRateReport = field(default_factory=FillRateReport)
    ranks: RankReport = field(default_factory=RankReport)
    coverage: CoverageReport = field(default_factory=CoverageReport)
    conflict: ConflictReport = field(default_factory=ConflictReport)

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

    # Kiểm tra tính chấp nhận được (Individual Rationality):
    # Cả hai bên bắt buộc phải nằm trong danh sách ưu tiên của nhau.
    for candidate, job in result.matches.items():
        if job not in candidate_rank.get(candidate, {}):
            blocking.append((candidate, job))
        elif candidate not in job_rank.get(job, {}):
            blocking.append((candidate, job))

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
                result.is_stable = len(blocking) == 0
                return blocking

    result.is_stable = len(blocking) == 0
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


def average_ranks(
    profile: PreferenceProfile,
    result: MatchingResult,
) -> RankReport:
    """Tính thứ hạng trung bình của đối tác trong danh sách ưu tiên theo Mục 6.5.

    Hạng bắt đầu từ 1 (1 là lựa chọn tốt nhất).
    """
    cand_ranks: list[int] = []
    job_ranks: list[int] = []

    for cand, job in result.matches.items():
        c_prefs = profile.candidate_preferences.get(cand, ())
        if job in c_prefs:
            cand_ranks.append(c_prefs.index(job) + 1)

        j_prefs = profile.job_preferences.get(job, ())
        if cand in j_prefs:
            job_ranks.append(j_prefs.index(cand) + 1)

    matched = len(result.matches)
    return RankReport(
        average_candidate_rank=(sum(cand_ranks) / len(cand_ranks)) if cand_ranks else 0.0,
        average_job_rank=(sum(job_ranks) / len(job_ranks)) if job_ranks else 0.0,
        matched_pairs=matched,
    )


def coverage_metrics(
    profile: PreferenceProfile,
    result: MatchingResult,
) -> CoverageReport:
    """Tính tỷ lệ ứng viên và việc làm chưa được ghép theo Mục 6.5."""
    total_c = len(profile.candidate_ids)
    matched_c = len(result.matches)
    unmatched_c = total_c - matched_c

    total_j = len(profile.job_ids)
    unfilled_j = 0
    for j in profile.job_ids:
        q = profile.quotas.get(j, 0)
        filled = len(result.job_assignments.get(j, []))
        if filled < q:
            unfilled_j += 1

    return CoverageReport(
        unmatched_candidate_count=unmatched_c,
        unmatched_candidate_rate=(unmatched_c / total_c) if total_c else 0.0,
        unmatched_job_count=unfilled_j,
        unmatched_job_rate=(unfilled_j / total_j) if total_j else 0.0,
        total_candidates=total_c,
        total_jobs=total_j,
    )


def conflict_metrics(
    job_assignments: MatchingResult | Mapping[Id, Sequence[Id]],
    total_candidates: int = 0,
) -> ConflictReport:
    """Đo lường mức độ xung đột / tắc nghẽn (Congestion Rate) theo Mục 6.5.

    Trong Gale-Shapley, mỗi ứng viên chỉ được nhận tối đa 1 việc (congestion = 0%).
    Trong Baseline sắp xếp tham lam hoặc gợi ý không điều phối, ứng viên giỏi bị
    giới thiệu đồng thời cho nhiều nơi, dẫn đến xung đột tuyển dụng.
    """
    assignments = (
        job_assignments.job_assignments
        if isinstance(job_assignments, MatchingResult)
        else job_assignments
    )
    counts: dict[Id, int] = {}
    for job, cands in assignments.items():
        for c in cands:
            counts[c] = counts.get(c, 0) + 1

    multi_assigned = sum(1 for c, cnt in counts.items() if cnt > 1)
    total_placed = len(counts)
    rate = (multi_assigned / total_placed) if total_placed else 0.0
    max_c = max(counts.values()) if counts else (1 if total_placed else 0)

    return ConflictReport(
        multi_assigned_candidates=multi_assigned,
        congestion_rate=rate,
        max_assignments_per_candidate=max_c,
    )


def calculate_vector_recall(
    approximate_ids: Sequence[Id],
    exact_ids: Sequence[Id],
    top_n: int = 50,
) -> float:
    """Đo Recall Top-N của pgvector ANN so với tìm kiếm vét cạn (brute-force) theo Mục 6.5."""
    if top_n <= 0:
        return 0.0
    approx_set = set(approximate_ids[:top_n])
    exact_set = set(exact_ids[:top_n])
    if not exact_set:
        return 1.0
    return len(approx_set & exact_set) / len(exact_set)


def calculate_precision_at_k(
    recommended_ids: Sequence[Id],
    relevant_ids: set[Id],
    k: int = 5,
) -> float:
    """Tính Precision@K cho danh sách gợi ý theo Mục 6.5."""
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_ids)
    return hits / len(top_k)


def calculate_ndcg_at_k(
    recommended_ids: Sequence[Id],
    relevance_scores: Mapping[Id, float],
    k: int = 5,
) -> float:
    """Tính Normalized Discounted Cumulative Gain (nDCG@K) theo Mục 6.5."""
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    if not top_k:
        return 0.0

    dcg = 0.0
    for idx, item in enumerate(top_k):
        rel = float(relevance_scores.get(item, 0.0))
        dcg += (math.pow(2.0, rel) - 1.0) / math.log2(idx + 2.0)

    sorted_ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for idx, rel in enumerate(sorted_ideal):
        idcg += (math.pow(2.0, float(rel)) - 1.0) / math.log2(idx + 2.0)

    if idcg <= 0.0:
        return 1.0 if dcg <= 0.0 else 0.0
    return dcg / idcg


def audit_fairness(
    result: MatchingResult | Mapping[Id, Id],
    candidate_groups: Mapping[Id, str],
) -> FairnessAuditReport:
    """Kiểm toán công bằng giữa các nhóm ứng viên theo Mục 6.6.

    Đảm bảo không thiên kiến, tính Disparate Impact ratio dựa trên tỷ lệ trúng tuyển.
    """
    matches = result.matches if isinstance(result, MatchingResult) else result
    groups_total: dict[str, int] = {}
    groups_matched: dict[str, int] = {}

    for cand, group in candidate_groups.items():
        groups_total[group] = groups_total.get(group, 0) + 1
        if cand in matches:
            groups_matched[group] = groups_matched.get(group, 0) + 1

    rates: dict[str, float] = {}
    for group, total in groups_total.items():
        matched = groups_matched.get(group, 0)
        rates[group] = (matched / total) if total else 0.0

    if not rates or min(rates.values()) == 0.0:
        di_ratio = 0.0 if (rates and max(rates.values()) > 0.0) else 1.0
    else:
        max_rate = max(rates.values())
        min_rate = min(rates.values())
        di_ratio = (min_rate / max_rate) if max_rate > 0 else 1.0

    is_fair = (di_ratio >= 0.80)
    return FairnessAuditReport(
        group_match_rates={g: round(r, 4) for g, r in rates.items()},
        disparate_impact_ratio=round(di_ratio, 4),
        is_fair=is_fair,
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
        ``EvaluationReport`` gồm blocking, satisfaction, fill rate, ranks, coverage và conflict.
    """

    return EvaluationReport(
        blocking=calculate_blocking_pairs(
            profile, result, max_pairs=max_blocking_pairs
        ),
        satisfaction=average_satisfaction(profile, result),
        fill_rate=fill_rate(profile, result),
        ranks=average_ranks(profile, result),
        coverage=coverage_metrics(profile, result),
        conflict=conflict_metrics(result.job_assignments, len(profile.candidate_ids)),
    )