#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm thử engine AVORA Matching.

Bao phủ:

* Tính ổn định (blocking pair = 0) trên nhiều thị trường, cả hai chế độ.
* Candidate-optimality và Employer-optimality (Gale & Shapley 1962; Roth 1984).
* Hard Constraints và Truncated Preference Lists.
* Thừa cầu (oversupply) / thừa cung (undersupply) quota.
* Phá hòa tất định (deterministic tie-breaking).
* Top-K recommendation và Iterative Clearance Rounds.
* Chỉ số đánh giá và baseline tham lam.

Chạy: ``python -m pytest tests -q`` hoặc ``python -m unittest discover -s tests``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.core.matching_engine import GaleShapleyMatching
from src.core.postprocessor import (
    generate_top_k,
    iterative_clearance_rounds,
    normalized_surplus_scores,
)
from src.core.preprocessor import (
    PreprocessConfig,
    Preprocessor,
    load_market_from_csv,
    normalize_score_matrix,
    truncate_preferences,
)
from src.evaluation.benchmark import (
    GreedyBaselineMatcher,
    generate_synthetic_market,
    run_benchmark,
    to_markdown_table,
)
from src.evaluation.metrics import (
    average_satisfaction,
    calculate_blocking_pairs,
    evaluate,
    fill_rate,
    find_blocking_pairs,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "scripts" / "data"


def build_profile(
    e_scores: dict,
    c_scores: dict,
    quotas: dict,
    *,
    config: PreprocessConfig | None = None,
    feasible_pairs: set | None = None,
):
    """Tiện ích: dựng ``PreferenceProfile`` với cấu hình mặc định."""

    preprocessor = Preprocessor(config or PreprocessConfig())
    return preprocessor.build_profile(
        e_scores, c_scores, quotas, feasible_pairs=feasible_pairs
    )


# Thị trường 2x2 có HAI matching ổn định -> phân biệt C-optimal vs J-optimal.
TWO_BY_TWO_E = {
    "c1": {"j1": 0.3, "j2": 0.9},
    "c2": {"j1": 0.9, "j2": 0.3},
}
TWO_BY_TWO_C = {
    "c1": {"j1": 0.9, "j2": 0.3},
    "c2": {"j1": 0.3, "j2": 0.9},
}
TWO_BY_TWO_QUOTAS = {"j1": 1, "j2": 1}

# Thị trường khiến Greedy Baseline tạo ra blocking pair, còn GS thì ổn định.
GREEDY_TRAP_E = {
    "c1": {"j1": 0.9, "j2": 0.05},
    "c2": {"j2": 0.9},
    "c3": {"j1": 0.5},
}
GREEDY_TRAP_C = {
    "c1": {"j1": 0.1, "j2": 0.05},
    "c2": {"j2": 0.9},
    "c3": {"j1": 1.0},
}
GREEDY_TRAP_QUOTAS = {"j1": 1, "j2": 1}


class TestPreprocessing(unittest.TestCase):
    """Giai đoạn 1 & 2 - tiền xử lý."""

    def test_normalize_sparse_and_dense_match(self) -> None:
        dense = normalize_score_matrix(
            [[0.1, 0.2], [0.3, 0.4]],
            candidate_ids=["c1", "c2"],
            job_ids=["j1", "j2"],
        )
        self.assertEqual(
            dense, {"c1": {"j1": 0.1, "j2": 0.2}, "c2": {"j1": 0.3, "j2": 0.4}}
        )

    def test_normalize_dense_wrong_width_raises(self) -> None:
        with self.assertRaises(ValueError):
            normalize_score_matrix([[0.1, 0.2]], candidate_ids=["c1"], job_ids=["j1"])

    def test_hard_constraints_drop_zero_scores(self) -> None:
        e = {"c1": {"j1": 0.0, "j2": 0.5}}
        c = {"c1": {"j1": 0.9, "j2": 0.5}}
        profile = build_profile(e, c, {"j1": 1, "j2": 1})
        self.assertEqual(profile.candidate_preferences["c1"], ("j2",))

    def test_hard_constraints_restrict_feasible_pairs(self) -> None:
        e = {"c1": {"j1": 0.5, "j2": 0.5}, "c2": {"j1": 0.5, "j2": 0.5}}
        c = {"c1": {"j1": 0.5, "j2": 0.5}, "c2": {"j1": 0.5, "j2": 0.5}}
        profile = build_profile(
            e, c, {"j1": 1, "j2": 1}, feasible_pairs={("c1", "j1"), ("c2", "j2")}
        )
        self.assertEqual(
            profile.feasible_pairs(), {("c1", "j1"), ("c2", "j2")}
        )
        self.assertEqual(profile.candidate_preferences["c1"], ("j1",))
        self.assertEqual(profile.candidate_preferences["c2"], ("j2",))

    def test_truncation_keeps_top_l(self) -> None:
        scores = {
            "c1": {f"j{i}": float(i) for i in range(1, 6)}  # j1=1 ... j5=5
        }
        truncated = truncate_preferences(scores, 3, by="candidate", seed=0)
        self.assertEqual(set(truncated["c1"]), {"j3", "j4", "j5"})

        profile = build_profile(
            {"c1": {f"j{i}": 0.5 for i in range(1, 6)}},
            scores,
            {f"j{i}": 1 for i in range(1, 6)},
            config=PreprocessConfig(max_candidate_preferences=3),
        )
        self.assertEqual(len(profile.candidate_preferences["c1"]), 3)
        self.assertEqual(profile.candidate_preferences["c1"], ("j5", "j4", "j3"))

    def test_employer_side_truncation(self) -> None:
        e = {"c1": {"j1": 0.1}, "c2": {"j1": 0.9}, "c3": {"j1": 0.5}}
        c = {"c1": {"j1": 0.1}, "c2": {"j1": 0.9}, "c3": {"j1": 0.5}}
        profile = build_profile(
            e,
            c,
            {"j1": 1},
            config=PreprocessConfig(max_employer_preferences=2),
        )
        # Chỉ c2 và c3 còn acceptable ở phía vị trí j1.
        self.assertEqual(profile.feasible_pairs(), {("c2", "j1"), ("c3", "j1")})

    def test_negative_quota_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_profile({"c1": {"j1": 0.5}}, {"c1": {"j1": 0.5}}, {"j1": -1})


class TestMatchingEngine(unittest.TestCase):
    """Giai đoạn 3 - Gale-Shapley Deferred Acceptance."""

    def test_invalid_optimal_mode_raises(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        with self.assertRaises(ValueError):
            GaleShapleyMatching(profile, optimal="random")

    def test_rank_lookup_table(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        engine = GaleShapleyMatching(profile, optimal="candidate")
        # j1 thích c2 (0.9) hơn c1 (0.3) -> hạng 0 và 1.
        self.assertEqual(engine.job_rank("j1", "c2"), 0)
        self.assertEqual(engine.job_rank("j1", "c1"), 1)
        self.assertEqual(engine.candidate_rank("c1", "j1"), 0)
        self.assertEqual(engine.candidate_rank("c1", "j2"), 1)

    def test_candidate_proposing_is_c_optimal(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        self.assertEqual(result.matches, {"c1": "j1", "c2": "j2"})
        self.assertEqual(calculate_blocking_pairs(profile, result).count, 0)

    def test_employer_proposing_is_j_optimal(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        result = GaleShapleyMatching(profile, optimal="employer").solve()
        self.assertEqual(result.matches, {"c1": "j2", "c2": "j1"})
        self.assertEqual(calculate_blocking_pairs(profile, result).count, 0)

    def test_c_optimal_and_j_optimal_are_dual(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        c_opt = GaleShapleyMatching(profile, optimal="candidate").solve()
        j_opt = GaleShapleyMatching(profile, optimal="employer").solve()

        c_satisfaction_best = average_satisfaction(profile, c_opt)
        c_satisfaction_worst = average_satisfaction(profile, j_opt)

        # C-optimal cho điểm C cao hơn; J-optimal cho điểm E cao hơn.
        self.assertGreater(c_satisfaction_best.average_c, c_satisfaction_worst.average_c)
        self.assertGreater(c_satisfaction_worst.average_e, c_satisfaction_best.average_e)

    def test_oversupply_fills_every_slot(self) -> None:
        e = {"c1": {"j1": 0.9}, "c2": {"j1": 0.5}, "c3": {"j1": 0.1}}
        c = {"c1": {"j1": 0.9}, "c2": {"j1": 0.5}, "c3": {"j1": 0.1}}
        profile = build_profile(e, c, {"j1": 1})
        result = GaleShapleyMatching(profile, optimal="candidate").solve()

        self.assertEqual(result.matches, {"c1": "j1"})
        self.assertEqual(result.job_assignments["j1"], ["c1"])
        self.assertEqual(result.unmatched_jobs, [])
        self.assertEqual(sorted(result.unmatched_candidates), ["c2", "c3"])
        self.assertEqual(fill_rate(profile, result).overall, 1.0)
        self.assertAlmostEqual(average_satisfaction(profile, result).coverage, 1 / 3)

    def test_undersupply_matches_all_candidates(self) -> None:
        e = {"c1": {"j1": 0.5, "j2": 0.9, "j3": 0.2}}
        c = {"c1": {"j1": 0.5, "j2": 0.9, "j3": 0.2}}
        profile = build_profile(e, c, {"j1": 2, "j2": 2, "j3": 2})
        result = GaleShapleyMatching(profile, optimal="candidate").solve()

        self.assertEqual(result.matches, {"c1": "j2"})
        self.assertEqual(sorted(result.unmatched_jobs), ["j1", "j2", "j3"])
        self.assertLess(fill_rate(profile, result).overall, 1.0)

    def test_quota_larger_than_one_multi_hire(self) -> None:
        e = {
            "c1": {"j1": 0.9},
            "c2": {"j1": 0.8},
            "c3": {"j1": 0.7},
        }
        c = {
            "c1": {"j1": 0.9},
            "c2": {"j1": 0.8},
            "c3": {"j1": 0.7},
        }
        profile = build_profile(e, c, {"j1": 2})
        result = GaleShapleyMatching(profile, optimal="candidate").solve()

        self.assertEqual(result.job_assignments["j1"], ["c1", "c2"])
        self.assertEqual(result.job_fill("j1"), 2)
        self.assertEqual(result.unmatched_candidates, ["c3"])

    def test_ties_are_broken_deterministically(self) -> None:
        e = {c: {j: 0.5 for j in ("j1", "j2", "j3")} for c in ("c1", "c2", "c3")}
        c_scores = {c: {j: 0.5 for j in ("j1", "j2", "j3")} for c in ("c1", "c2", "c3")}
        quotas = {"j1": 1, "j2": 1, "j3": 1}

        first = build_profile(e, c_scores, quotas)
        second = build_profile(e, c_scores, quotas)
        self.assertEqual(first.candidate_preferences, second.candidate_preferences)

        result_a = GaleShapleyMatching(first, optimal="candidate").solve()
        result_b = GaleShapleyMatching(second, optimal="candidate").solve()
        self.assertEqual(result_a.matches, result_b.matches)
        self.assertEqual(calculate_blocking_pairs(first, result_a).count, 0)

    def test_zero_quota_job_never_matches(self) -> None:
        e = {"c1": {"j1": 0.9, "j2": 0.8}}
        c = {"c1": {"j1": 0.9, "j2": 0.8}}
        profile = build_profile(e, c, {"j1": 0, "j2": 1})
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        self.assertEqual(result.matches, {"c1": "j2"})

    def test_stability_on_random_markets_both_modes(self) -> None:
        e, c, quotas = generate_synthetic_market(
            250, 60, avg_quota=4, density=0.35, seed=11
        )
        profile = build_profile(
            e, c, quotas, config=PreprocessConfig(max_candidate_preferences=30)
        )
        for mode in ("candidate", "employer"):
            with self.subTest(mode=mode):
                result = GaleShapleyMatching(profile, optimal=mode).solve()
                self.assertEqual(find_blocking_pairs(profile, result), [])


class TestPostprocessing(unittest.TestCase):
    """Giai đoạn 4 - Top-K recommendation."""

    def test_generate_top_k_primary_first_no_duplicates(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        recommendations = generate_top_k(profile, result, k=2)

        c1_recs = recommendations["c1"]
        self.assertEqual([rec.position for rec in c1_recs], [1, 2])
        self.assertEqual(c1_recs[0].job_id, "j1")
        self.assertTrue(c1_recs[0].is_primary)
        self.assertEqual(len({rec.job_id for rec in c1_recs}), len(c1_recs))

    def test_generate_top_k_caps_at_available_jobs(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        recommendations = generate_top_k(profile, result, k=10)
        # Mỗi ứng viên chỉ có 2 công việc khả thi.
        self.assertEqual(len(recommendations["c1"]), 2)

    def test_generate_top_k_zero_returns_empty(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        recommendations = generate_top_k(profile, result, k=0)
        self.assertTrue(all(items == [] for items in recommendations.values()))

    def test_normalized_surplus_in_unit_range(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        surplus = normalized_surplus_scores(profile)
        self.assertTrue(surplus)
        for value in surplus.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_iterative_clearance_produces_distinct_recommendations(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        recommendations = iterative_clearance_rounds(profile, k=2)

        c1_jobs = [rec.job_id for rec in recommendations["c1"]]
        self.assertEqual(len(c1_jobs), 2)
        self.assertEqual(len(set(c1_jobs)), 2)
        self.assertEqual(recommendations["c1"][0].job_id, "j1")
        self.assertTrue(recommendations["c1"][0].is_primary)
        self.assertFalse(recommendations["c1"][1].is_primary)

    def test_iterative_clearance_employer_mode(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        recommendations = iterative_clearance_rounds(profile, k=2, optimal="employer")
        c1_jobs = [rec.job_id for rec in recommendations["c1"]]
        self.assertEqual(c1_jobs[0], "j2")  # J-optimal round đầu.
        self.assertEqual(len(set(c1_jobs)), 2)


class TestMetrics(unittest.TestCase):
    """Blocking pairs, satisfaction, fill rate."""

    def test_evaluate_reports_stable_result(self) -> None:
        profile = build_profile(TWO_BY_TWO_E, TWO_BY_TWO_C, TWO_BY_TWO_QUOTAS)
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        report = evaluate(profile, result)

        self.assertTrue(report.blocking.is_stable)
        self.assertEqual(report.blocking.count, 0)
        self.assertEqual(report.blocking.rate, 0.0)
        self.assertAlmostEqual(report.satisfaction.average_c, 0.9)
        self.assertAlmostEqual(report.satisfaction.average_e, 0.3)
        self.assertEqual(report.satisfaction.coverage, 1.0)
        self.assertEqual(report.fill_rate.overall, 1.0)

    def test_fill_rate_by_job(self) -> None:
        e = {"c1": {"j1": 0.9}, "c2": {"j1": 0.8}, "c3": {"j1": 0.7}}
        c = {"c1": {"j1": 0.9}, "c2": {"j1": 0.8}, "c3": {"j1": 0.7}}
        profile = build_profile(e, c, {"j1": 3})
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        report = fill_rate(profile, result)
        self.assertEqual(report.filled_slots, 3)
        self.assertEqual(report.total_quota, 3)
        self.assertEqual(report.by_job["j1"], 1.0)

    def test_greedy_baseline_is_unstable_but_gs_is_stable(self) -> None:
        profile = build_profile(GREEDY_TRAP_E, GREEDY_TRAP_C, GREEDY_TRAP_QUOTAS)

        gs_result = GaleShapleyMatching(profile, optimal="candidate").solve()
        gs_blocking = calculate_blocking_pairs(profile, gs_result)
        self.assertEqual(gs_blocking.count, 0)

        greedy_result = GreedyBaselineMatcher(profile).solve()
        greedy_blocking = calculate_blocking_pairs(profile, greedy_result)
        self.assertGreaterEqual(greedy_blocking.count, 1)
        self.assertFalse(greedy_blocking.is_stable)


class TestBenchmark(unittest.TestCase):
    """Benchmark end-to-end và render bảng Markdown."""

    def test_run_benchmark_produces_three_configs(self) -> None:
        results = run_benchmark(
            n_candidates=120, n_jobs=25, avg_quota=4, density=0.4, seed=5
        )
        names = [item.name for item in results]
        self.assertEqual(
            names,
            ["GaleShapley (C-optimal)", "GaleShapley (J-optimal)", "Greedy Baseline"],
        )
        for item in results:
            if item.name.startswith("GaleShapley"):
                self.assertEqual(item.blocking_count, 0)
            self.assertGreaterEqual(item.fill_rate, 0.0)
            self.assertLessEqual(item.fill_rate, 1.0)

    def test_markdown_table_contains_all_rows(self) -> None:
        results = run_benchmark(n_candidates=40, n_jobs=10, seed=2)
        table = to_markdown_table(results)
        self.assertIn("| Cấu hình |", table)
        self.assertIn("GaleShapley (C-optimal)", table)
        self.assertIn("Greedy Baseline", table)
        self.assertEqual(len(table.splitlines()), 5)  # header + separator + 3 dòng


class TestCsvAdapter(unittest.TestCase):
    """Adapter đọc dữ liệu CSV hiện có của repo."""

    def setUp(self) -> None:
        self.candidates_csv = DATA_DIR / "candidates.csv"
        self.jobs_csv = DATA_DIR / "job_descriptions.csv"
        if not self.candidates_csv.is_file() or not self.jobs_csv.is_file():
            self.skipTest(f"Không tìm thấy dữ liệu CSV trong {DATA_DIR}")

    def test_load_market_from_csv_builds_stable_market(self) -> None:
        e_scores, c_scores, quotas, feasible = load_market_from_csv(
            self.candidates_csv,
            self.jobs_csv,
            default_quota=2,
            candidate_limit=200,
            job_limit=50,
        )
        self.assertTrue(e_scores)
        self.assertTrue(quotas)
        self.assertTrue(feasible)

        profile = build_profile(e_scores, c_scores, quotas, feasible_pairs=feasible)
        result = GaleShapleyMatching(profile, optimal="candidate").solve()
        self.assertEqual(calculate_blocking_pairs(profile, result).count, 0)

    def test_feasibility_respects_disability_group(self) -> None:
        e_scores, c_scores, quotas, feasible = load_market_from_csv(
            self.candidates_csv, self.jobs_csv, candidate_limit=100, job_limit=40
        )
        profile = build_profile(e_scores, c_scores, quotas, feasible_pairs=feasible)
        # Mọi cặp acceptable phải nằm trong tập feasible (trùng nhóm khuyết tật).
        self.assertTrue(profile.feasible_pairs().issubset(feasible))


if __name__ == "__main__":
    unittest.main(verbosity=2)
