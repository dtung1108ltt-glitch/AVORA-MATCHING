#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark engine Gale-Shapley so với baseline tham lam (Greedy).

Script sinh thị trường hai phía mô phỏng (candidate x job với mật độ cạnh và
quota cấu hình được), rồi chạy ba cấu hình:

1. ``GaleShapley (C-optimal)`` - engine chính, candidate-proposing.
2. ``GaleShapley (J-optimal)`` - engine chính, employer-proposing.
3. ``Greedy Baseline`` - cách làm cũ: sắp xếp toàn bộ cặp theo điểm trộn rồi gán
   tham lam, **không** đảm bảo ổn định.

Chạy::

    python -m src.evaluation.benchmark --candidates 2000 --jobs 400

Kết quả in ra bảng Markdown (dán trực tiếp vào README) kèm kiểm chứng
*blocking pair rate = 0%* cho hai cấu hình Gale-Shapley.
"""

from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from typing import Any

from src.core.matching_engine import GaleShapleyMatching, MatchingResult
from src.core.preprocessor import (
    Id,
    PreprocessConfig,
    PreferenceProfile,
    Preprocessor,
    ScoreMatrix,
)
from src.evaluation.metrics import EvaluationReport, evaluate

__all__ = [
    "BenchmarkResult",
    "GreedyBaselineMatcher",
    "generate_synthetic_market",
    "run_benchmark",
    "run_profile_benchmark",
    "to_markdown_table",
]


def generate_synthetic_market(
    n_candidates: int,
    n_jobs: int,
    *,
    avg_quota: int = 5,
    density: float = 0.4,
    seed: int = 0,
) -> tuple[ScoreMatrix, ScoreMatrix, dict[Id, int]]:
    """Sinh thị trường mô phỏng với mật độ cạnh và quota cho trước.

    Đảm bảo mỗi ứng viên có ít nhất một công việc khả thi và mỗi vị trí có ít
    nhất một ứng viên khả thi, tránh thị trường suy biến.

    Args:
        n_candidates: Số ứng viên ``N``.
        n_jobs: Số vị trí ``M``.
        avg_quota: Hạn mức trung bình mỗi vị trí.
        density: Xác suất một cặp ``(c, j)`` trở thành *acceptable*.
        seed: Hạt giống ngẫu nhiên.

    Returns:
        ``(e_scores, c_scores, quotas)`` sẵn sàng cho ``Preprocessor``.
    """

    rng = random.Random(seed)
    candidates: list[Id] = [f"C{i}" for i in range(n_candidates)]
    jobs: list[Id] = [f"J{j}" for j in range(n_jobs)]

    e_scores: ScoreMatrix = {}
    c_scores: ScoreMatrix = {}
    for candidate in candidates:
        e_row: dict[Id, float] = {}
        c_row: dict[Id, float] = {}
        for job in jobs:
            if rng.random() < density:
                e_row[job] = round(rng.uniform(0.05, 1.0), 6)
                c_row[job] = round(rng.uniform(0.05, 1.0), 6)
        e_scores[candidate] = e_row
        c_scores[candidate] = c_row

    # Đảm bảo mỗi ứng viên có ≥ 1 lựa chọn.
    for candidate in candidates:
        if not e_scores[candidate]:
            job = rng.choice(jobs)
            value = round(rng.uniform(0.05, 1.0), 6)
            e_scores[candidate][job] = value
            c_scores[candidate][job] = value

    # Đảm bảo mỗi vị trí có ≥ 1 ứng viên.
    for job in jobs:
        if not any(job in e_scores[candidate] for candidate in candidates):
            candidate = rng.choice(candidates)
            value = round(rng.uniform(0.05, 1.0), 6)
            e_scores[candidate][job] = value
            c_scores[candidate][job] = value

    quotas: dict[Id, int] = {}
    for job in jobs:
        raw = int(round(rng.gauss(avg_quota, max(1.0, avg_quota * 0.3))))
        quotas[job] = max(1, raw)

    return e_scores, c_scores, quotas


class GreedyBaselineMatcher:
    """Baseline tham lam không đảm bảo ổn định.

    Thuật toán: sắp xếp **toàn bộ** cặp khả thi theo điểm trộn ``w_e * E + w_c *
    C`` giảm dần, rồi lần lượt gán nếu cả ứng viên còn trống và vị trí còn quota.
    Đây là cách tiếp cận "lọc theo điểm" phổ biến trước khi có matching ổn định;
    nó nhanh nhưng luôn để lại blocking pair.

    Args:
        profile: Hồ sơ ưu tiên đã tiền xử lý.
        weight_e: Trọng số điểm ``E``.
        weight_c: Trọng số điểm ``C``.
    """

    def __init__(
        self,
        profile: PreferenceProfile,
        *,
        weight_e: float = 0.5,
        weight_c: float = 0.5,
    ) -> None:
        self.profile = profile
        self.weight_e = weight_e
        self.weight_c = weight_c

    def solve(self) -> MatchingResult:
        """Chạy gán tham lam theo điểm trộn.

        Returns:
            ``MatchingResult`` với ``optimal="greedy"``.
        """

        profile = self.profile
        pairs = list(profile.feasible_pairs())

        def blended(pair: tuple[Id, Id]) -> float:
            candidate, job = pair
            e_value = profile.e_scores.get(candidate, {}).get(job, 0.0)
            c_value = profile.c_scores.get(candidate, {}).get(job, 0.0)
            return self.weight_e * e_value + self.weight_c * c_value

        pairs.sort(key=lambda pair: (-blended(pair), str(pair[0]), str(pair[1])))

        matches: dict[Id, Id] = {}
        job_assignments: dict[Id, list[Id]] = {job: [] for job in profile.job_ids}

        for candidate, job in pairs:
            if candidate in matches:
                continue
            if len(job_assignments[job]) >= profile.quotas.get(job, 0):
                continue
            matches[candidate] = job
            job_assignments[job].append(candidate)

        ordered_matches = {
            candidate: matches[candidate]
            for candidate in profile.candidate_ids
            if candidate in matches
        }
        unmatched_candidates = [
            candidate
            for candidate in profile.candidate_ids
            if candidate not in matches
        ]
        unmatched_jobs = [
            job
            for job in profile.job_ids
            if len(job_assignments[job]) < profile.quotas.get(job, 0)
        ]

        return MatchingResult(
            matches=ordered_matches,
            job_assignments=job_assignments,
            unmatched_candidates=unmatched_candidates,
            unmatched_jobs=unmatched_jobs,
            proposals=len(pairs),
            optimal="greedy",
            is_stable=False,
            metadata={"operations": len(pairs)},
        )


@dataclass
class BenchmarkResult:
    """Kết quả của một cấu hình thuật toán trên cùng một thị trường."""

    name: str
    matched_pairs: int
    coverage: float
    operations: int
    blocking_count: int
    blocking_rate: float
    average_e: float
    average_c: float
    fill_rate: float
    elapsed_ms: float

    def as_dict(self) -> dict[str, Any]:
        """Trả về ``dict`` tiện xuất JSON."""

        return {
            "name": self.name,
            "matched_pairs": self.matched_pairs,
            "coverage": round(self.coverage, 4),
            "operations": self.operations,
            "blocking_count": self.blocking_count,
            "blocking_rate": round(self.blocking_rate, 6),
            "average_e": round(self.average_e, 4),
            "average_c": round(self.average_c, 4),
            "fill_rate": round(self.fill_rate, 4),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


def _benchmark_one(
    profile: PreferenceProfile,
    result: MatchingResult,
    name: str,
    elapsed_ms: float,
) -> BenchmarkResult:
    """Gộp kết quả matching và chỉ số đánh giá thành một dòng benchmark."""

    report: EvaluationReport = evaluate(profile, result)
    return BenchmarkResult(
        name=name,
        matched_pairs=len(result.matches),
        coverage=report.satisfaction.coverage,
        operations=result.proposals,
        blocking_count=report.blocking.count,
        blocking_rate=report.blocking.rate,
        average_e=report.satisfaction.average_e,
        average_c=report.satisfaction.average_c,
        fill_rate=report.fill_rate.overall,
        elapsed_ms=elapsed_ms,
    )


def run_profile_benchmark(
    profile: PreferenceProfile,
    *,
    weight_e: float = 0.5,
    weight_c: float = 0.5,
) -> list[BenchmarkResult]:
    """Chạy ba cấu hình (C-optimal, J-optimal, Greedy) trên cùng một thị trường.

    Args:
        profile: Hồ sơ ưu tiên đã tiền xử lý.
        weight_e: Trọng số điểm ``E`` cho baseline.
        weight_c: Trọng số điểm ``C`` cho baseline.

    Returns:
        Danh sách ``BenchmarkResult`` theo thứ tự C-optimal, J-optimal, Greedy.
    """

    results: list[BenchmarkResult] = []

    start = time.perf_counter()
    candidate_result = GaleShapleyMatching(profile, optimal="candidate").solve()
    candidate_ms = (time.perf_counter() - start) * 1000.0
    results.append(
        _benchmark_one(profile, candidate_result, "GaleShapley (C-optimal)", candidate_ms)
    )

    start = time.perf_counter()
    employer_result = GaleShapleyMatching(profile, optimal="employer").solve()
    employer_ms = (time.perf_counter() - start) * 1000.0
    results.append(
        _benchmark_one(profile, employer_result, "GaleShapley (J-optimal)", employer_ms)
    )

    start = time.perf_counter()
    greedy_result = GreedyBaselineMatcher(
        profile, weight_e=weight_e, weight_c=weight_c
    ).solve()
    greedy_ms = (time.perf_counter() - start) * 1000.0
    results.append(
        _benchmark_one(profile, greedy_result, "Greedy Baseline", greedy_ms)
    )

    return results


def run_benchmark(
    n_candidates: int = 1000,
    n_jobs: int = 200,
    *,
    avg_quota: int = 5,
    density: float = 0.4,
    seed: int = 0,
    max_preferences: int = 40,
    max_employer_preferences: int | None = None,
) -> list[BenchmarkResult]:
    """Sinh thị trường mô phỏng rồi chạy :func:`run_profile_benchmark`.

    Args:
        n_candidates: Số ứng viên.
        n_jobs: Số vị trí.
        avg_quota: Hạn mức trung bình mỗi vị trí.
        density: Mật độ cạnh acceptable.
        seed: Hạt giống.
        max_preferences: ``L`` - giới hạn danh sách ưu tiên ứng viên.
        max_employer_preferences: ``M`` - giới hạn danh sách ưu tiên nhà tuyển dụng.

    Returns:
        Danh sách ``BenchmarkResult``.
    """

    e_scores, c_scores, quotas = generate_synthetic_market(
        n_candidates, n_jobs, avg_quota=avg_quota, density=density, seed=seed
    )
    profile = Preprocessor(
        PreprocessConfig(
            max_candidate_preferences=max_preferences,
            max_employer_preferences=max_employer_preferences,
        )
    ).build_profile(e_scores, c_scores, quotas)
    return run_profile_benchmark(profile)


def to_markdown_table(results: list[BenchmarkResult]) -> str:
    """Render danh sách kết quả benchmark thành bảng Markdown.

    Args:
        results: Kết quả từ :func:`run_benchmark`.

    Returns:
        Chuỗi Markdown sẵn sàng dán vào README.
    """

    header = (
        "| Cấu hình | Cặp ghép | Coverage | Số phép tính | Blocking pairs "
        "| Tỷ lệ blocking | Điểm E TB | Điểm C TB | Fill rate | Thời gian (ms) |"
    )
    separator = (
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [header, separator]

    for item in results:
        lines.append(
            "| {name} | {matched} | {coverage:.2%} | {ops} | {blocking} "
            "| {rate:.2%} | {ae:.4f} | {ac:.4f} | {fill:.2%} | {ms:.2f} |".format(
                name=item.name,
                matched=item.matched_pairs,
                coverage=item.coverage,
                ops=item.operations,
                blocking=item.blocking_count,
                rate=item.blocking_rate,
                ae=item.average_e,
                ac=item.average_c,
                fill=item.fill_rate,
                ms=item.elapsed_ms,
            )
        )

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Khai báo tham số dòng lệnh cho script benchmark."""

    parser = argparse.ArgumentParser(
        description="Benchmark Gale-Shapley Matching vs Greedy Baseline."
    )
    parser.add_argument("--candidates", type=int, default=1000, help="Số ứng viên.")
    parser.add_argument("--jobs", type=int, default=200, help="Số vị trí tuyển dụng.")
    parser.add_argument("--avg-quota", type=int, default=5, help="Quota trung bình.")
    parser.add_argument("--density", type=float, default=0.4, help="Mật độ cạnh 0..1.")
    parser.add_argument("--seed", type=int, default=0, help="Hạt giống ngẫu nhiên.")
    parser.add_argument(
        "--max-preferences",
        type=int,
        default=40,
        help="L - số nguyện vọng tối đa mỗi ứng viên.",
    )
    parser.add_argument(
        "--max-employer-preferences",
        type=int,
        default=None,
        help="M - số ứng viên tối đa mỗi vị trí giữ (mặc định không cắt).",
    )
    parser.add_argument("--json", action="store_true", help="In kết quả dạng JSON.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entrypoint dòng lệnh: chạy benchmark và in bảng Markdown."""

    args = parse_args(argv)

    if args.candidates <= 0 or args.jobs <= 0:
        print("Lỗi: --candidates và --jobs phải lớn hơn 0.", flush=True)
        return 2
    if not 0.0 < args.density <= 1.0:
        print("Lỗi: --density phải nằm trong (0, 1].", flush=True)
        return 2

    results = run_benchmark(
        n_candidates=args.candidates,
        n_jobs=args.jobs,
        avg_quota=args.avg_quota,
        density=args.density,
        seed=args.seed,
        max_preferences=args.max_preferences,
        max_employer_preferences=args.max_employer_preferences,
    )

    if args.json:
        import json

        print(json.dumps([item.as_dict() for item in results], ensure_ascii=False, indent=2))
    else:
        print(
            f"=== Benchmark: N={args.candidates} ứng viên, M={args.jobs} vị trí, "
            f"quota TB={args.avg_quota}, density={args.density}, L={args.max_preferences} ==="
        )
        print(to_markdown_table(results))

    stable_ok = all(
        item.blocking_count == 0
        for item in results
        if item.name.startswith("GaleShapley")
    )
    print(f"\nKiểm chứng ổn định (blocking pairs = 0 với Gale-Shapley): {stable_ok}")
    return 0 if stable_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())