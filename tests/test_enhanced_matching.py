#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Kiểm thử toàn diện cho hệ thống nâng cấp AVORA Matching (Mục 6.1 - 6.8).

Bao phủ các khía cạnh:
1. Mô hình hồ sơ chuẩn hóa (CandidateProfile, JobProfile, ExecutionAudit - Mục 6.1).
2. Công thức chấm điểm thành phần E & C và thang ưu tiên (0.5, 0.3, 0.2 - Mục 6.2).
3. Bộ lọc điều kiện bắt buộc (Hard Constraints & Direct Track - Mục 6.1, 6.7).
4. Phá hòa tất định theo điểm kỹ năng rồi mã ứng viên (Deterministic Tie-breaking - Mục 6.4).
5. Sinh giải thích gợi ý và phân rã điểm thành phần (Explainability & Breakdown - Mục 6.1).
6. Bộ chỉ số đánh giá nâng cao & so sánh Baseline (Rank, Coverage, Congestion, Recall, nDCG, Fairness - Mục 6.5, 6.6).
7. Tích hợp FastAPI endpoint /matching/stable với hồ sơ nâng cấp.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app
from src.core.profile_models import CandidateProfile, JobProfile, ExecutionAudit
from src.core.scoring import (
    EmployerWeightsConfig,
    calculate_cosine_similarity,
    calculate_token_jaccard,
    calculate_skill_similarity,
    calculate_work_mode_hours_match,
    calculate_nice_to_have_access_ratio,
    calculate_salary_overlap,
    calculate_experience_score,
    calculate_location_match,
    check_hard_constraints,
    calculate_composite_e_score,
    calculate_composite_c_score,
    explain_match,
)
from src.core.preprocessor import (
    PreprocessConfig,
    Preprocessor,
    build_profile_from_profiles,
)
from src.core.matching_engine import GaleShapleyMatching
from src.core.postprocessor import generate_top_k
from src.evaluation.metrics import (
    average_ranks,
    coverage_metrics,
    conflict_metrics,
    calculate_vector_recall,
    calculate_precision_at_k,
    calculate_ndcg_at_k,
    audit_fairness,
    evaluate,
)
from src.evaluation.benchmark import (
    run_profile_benchmark,
    to_evaluation_comparison_table,
)


class TestProfileModelsAndPriorityWeights(unittest.TestCase):
    """Kiểm thử mô hình dữ liệu hồ sơ và thang ưu tiên ứng viên (Mục 6.1, 6.2)."""

    def test_priority_weights_ladder(self) -> None:
        """Kiểm tra quy thứ tự ưu tiên 3 yếu tố thành thang 0.5 / 0.3 / 0.2."""
        cand = CandidateProfile(
            candidate_id="c1",
            priority_ranking=["distance_or_remote", "income", "career_path"],
        )
        weights = cand.get_priority_weights()
        self.assertEqual(weights["distance_or_remote"], 0.5)
        self.assertEqual(weights["income"], 0.3)
        self.assertEqual(weights["career_path"], 0.2)

    def test_priority_weights_default_fallback(self) -> None:
        """Kiểm tra trọng số mặc định chung khi thiếu hoặc sai thứ tự ưu tiên."""
        cand_empty = CandidateProfile(
            candidate_id="c2",
            priority_ranking=[],
        )
        w_empty = cand_empty.get_priority_weights()
        self.assertEqual(w_empty["income"], 0.35)
        self.assertEqual(w_empty["distance_or_remote"], 0.30)
        self.assertEqual(w_empty["career_path"], 0.35)

        # Danh sách lặp hoặc không đủ 3 yếu tố
        cand_invalid = CandidateProfile(
            candidate_id="c3",
            priority_ranking=["income", "income"],
        )
        w_invalid = cand_invalid.get_priority_weights()
        self.assertEqual(w_invalid["income"], 0.35)

    def test_salary_range_normalization(self) -> None:
        """Kiểm tra trích xuất khoảng lương an toàn."""
        cand = CandidateProfile(
            candidate_id="c4",
            expected_salary_min=15.0,
            expected_salary_max=20.0,
        )
        self.assertEqual(cand.get_salary_range(), (15.0, 20.0))

        job = JobProfile(
            job_id="j1",
            salary_min=12.0,
            salary_max=18.0,
        )
        self.assertEqual(job.get_salary_range(), (12.0, 18.0))

    def test_execution_audit_metadata(self) -> None:
        """Kiểm tra đối tượng ExecutionAudit lưu vết kiểm toán (Mục 6.4)."""
        audit = ExecutionAudit(
            run_id="run_12345",
            timestamp="2026-10-13T12:00:00Z",
            seed=42,
            weights_config={"skill": 0.35, "work_mode_hours": 0.20},
            data_version="v2.1",
        )
        audit_dict = audit.as_dict()
        self.assertEqual(audit_dict["run_id"], "run_12345")
        self.assertEqual(audit_dict["seed"], 42)
        self.assertEqual(audit_dict["data_version"], "v2.1")


class TestScoringComponents(unittest.TestCase):
    """Kiểm thử công thức chấm điểm E và C (Mục 6.2)."""

    def test_employer_weights_validation(self) -> None:
        """Kiểm tra tổng trọng số E bắt buộc phải bằng 1.0."""
        valid_cfg = EmployerWeightsConfig()
        valid_cfg.validate()  # Không bắn exception

        invalid_cfg = EmployerWeightsConfig(skill=0.50)  # Tổng > 1.0
        with self.assertRaises(ValueError):
            invalid_cfg.validate()

    def test_skill_similarity_vector_and_text(self) -> None:
        """Kiểm tra tính điểm kỹ năng qua Cosine Similarity vector và từ khóa."""
        # Cosine similarity vector
        vec_a = [1.0, 0.0, 0.0]
        vec_b = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(calculate_cosine_similarity(vec_a, vec_b), 1.0)

        vec_c = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(calculate_cosine_similarity(vec_a, vec_c), 0.0)

        # Calculate skill similarity với float vector
        score_vec = calculate_skill_similarity([1.0, 2.0], [1.0, 2.0])
        self.assertAlmostEqual(score_vec, 1.0)

        # Calculate skill similarity với text
        score_text = calculate_skill_similarity("Python Fastapi PostgreSQL", "Python PostgreSQL")
        self.assertGreater(score_text, 0.5)

    def test_work_mode_and_hours_match(self) -> None:
        """Kiểm tra mức khớp thời gian và hình thức làm việc: 1.0 / 0.5 / 0.0."""
        c_full = CandidateProfile(candidate_id="c1", work_mode="remote", work_hours="full-time")
        j_full = JobProfile(job_id="j1", work_mode="remote", work_hours="full-time")
        self.assertEqual(calculate_work_mode_hours_match(c_full, j_full), 1.0)

        c_hybrid = CandidateProfile(candidate_id="c2", work_mode="hybrid", work_hours="full-time")
        j_onsite = JobProfile(job_id="j2", work_mode="on-site", work_hours="full-time")
        self.assertEqual(calculate_work_mode_hours_match(c_hybrid, j_onsite), 0.5)

        c_part = CandidateProfile(candidate_id="c3", work_mode="on-site", work_hours="part-time")
        j_full_onsite = JobProfile(job_id="j3", work_mode="on-site", work_hours="full-time")
        self.assertEqual(calculate_work_mode_hours_match(c_part, j_full_onsite), 0.5)

    def test_nice_to_have_access_ratio(self) -> None:
        """Kiểm tra tỷ lệ điều kiện tiếp cận nên có được đáp ứng."""
        cand = CandidateProfile(
            candidate_id="c1",
            nice_to_have_access_needs=["chieu_sang_tot", "phong_nghi_rieng"],
        )
        job_full = JobProfile(
            job_id="j1",
            access_support_provided=["chieu_sang_tot", "phong_nghi_rieng", "loi_xe_lan"],
        )
        self.assertEqual(calculate_nice_to_have_access_ratio(cand, job_full), 1.0)

        job_partial = JobProfile(
            job_id="j2",
            access_support_provided=["chieu_sang_tot"],
        )
        self.assertEqual(calculate_nice_to_have_access_ratio(cand, job_partial), 0.5)

        # Ứng viên không có nhu cầu thêm -> mặc định 1.0
        cand_none = CandidateProfile(candidate_id="c2", nice_to_have_access_needs=[])
        self.assertEqual(calculate_nice_to_have_access_ratio(cand_none, job_partial), 1.0)

    def test_salary_overlap(self) -> None:
        """Kiểm tra tính mức độ chồng lấn khoảng lương."""
        cand = CandidateProfile(candidate_id="c1", expected_salary_min=10, expected_salary_max=20)
        job_match = JobProfile(job_id="j1", salary_min=10, salary_max=20)
        self.assertEqual(calculate_salary_overlap(cand, job_match), 1.0)

        job_overlap = JobProfile(job_id="j2", salary_min=15, salary_max=25)
        # [10, 20] và [15, 25]: overlap = [15, 20] (5), union = [10, 25] (15) -> 5/15 = 1/3
        self.assertAlmostEqual(calculate_salary_overlap(cand, job_overlap), 5.0 / 15.0)

        job_no_overlap = JobProfile(job_id="j3", salary_min=25, salary_max=35)
        self.assertEqual(calculate_salary_overlap(cand, job_no_overlap), 0.0)

    def test_experience_and_location_score(self) -> None:
        """Kiểm tra điểm kinh nghiệm chuẩn hóa và khu vực."""
        cand = CandidateProfile(candidate_id="c1", experience_years=2.0, location="Hà Nội")
        job = JobProfile(job_id="j1", experience_required_years=4.0, location="Hà Nội")
        self.assertEqual(calculate_experience_score(cand, job), 0.5)
        self.assertEqual(calculate_location_match(cand, job), 1.0)

        job_diff = JobProfile(job_id="j2", experience_required_years=0.0, location="TP.HCM", work_mode="on-site")
        cand.work_mode = "on-site"
        self.assertEqual(calculate_experience_score(cand, job_diff), 1.0)
        self.assertEqual(calculate_location_match(cand, job_diff), 0.0)

    def test_composite_e_and_c_scores(self) -> None:
        """Kiểm tra tổng hợp điểm E (6 thành phần) và điểm C (3 thành phần cá nhân hóa)."""
        cand = CandidateProfile(
            candidate_id="c1",
            skills=[1.0, 0.0],
            nice_to_have_access_needs=["anh_sang"],
            expected_salary_min=10,
            expected_salary_max=15,
            work_mode="remote",
            work_hours="full-time",
            location="Hà Nội",
            experience_years=3,
            priority_ranking=["income", "distance_or_remote", "career_path"],
            career_orientation="AI Engineer",
        )
        job = JobProfile(
            job_id="j1",
            skills_required=[1.0, 0.0],
            access_support_provided=["anh_sang"],
            salary_min=10,
            salary_max=15,
            work_mode="remote",
            work_hours="full-time",
            location="Hà Nội",
            experience_required_years=3,
            career_domain="AI Engineer",
        )

        score_e, breakdown_e = calculate_composite_e_score(cand, job)
        self.assertAlmostEqual(score_e, 1.0)
        self.assertEqual(breakdown_e["skill"], 1.0)
        self.assertEqual(breakdown_e["work_mode_hours"], 1.0)
        self.assertEqual(breakdown_e["nice_to_have_access"], 1.0)
        self.assertEqual(breakdown_e["salary"], 1.0)
        self.assertEqual(breakdown_e["experience"], 1.0)
        self.assertEqual(breakdown_e["location"], 1.0)

        score_c, breakdown_c = calculate_composite_c_score(cand, job)
        self.assertAlmostEqual(score_c, 1.0)
        self.assertEqual(len(breakdown_c), 3)
        self.assertEqual(breakdown_c["income"], 1.0)
        self.assertEqual(breakdown_c["distance_or_remote"], 1.0)
        self.assertEqual(breakdown_c["career_path"], 1.0)


class TestHardConstraintsAndDirectTrack(unittest.TestCase):
    """Kiểm thử bộ lọc điều kiện bắt buộc và cơ chế Direct Track (Mục 6.1, 6.7)."""

    def test_essential_access_needs_violation(self) -> None:
        """Loại cặp vi phạm nhu cầu tiếp cận thiết yếu khi làm việc tại chỗ."""
        cand = CandidateProfile(
            candidate_id="c1",
            work_mode="on-site",
            essential_access_needs=["loi_xe_lan", "man_hinh_chu_lon"],
        )
        job = JobProfile(
            job_id="j1",
            work_mode="on-site",
            access_support_provided=["man_hinh_chu_lon"],  # Thiếu loi_xe_lan
        )
        is_feasible, reason = check_hard_constraints(cand, job)
        self.assertFalse(is_feasible)
        self.assertIn("loi_xe_lan", reason)

    def test_essential_access_needs_satisfied(self) -> None:
        """Chấp nhận cặp khi công ty đáp ứng đủ nhu cầu thiết yếu."""
        cand = CandidateProfile(
            candidate_id="c1",
            work_mode="on-site",
            essential_access_needs=["loi_xe_lan"],
        )
        job = JobProfile(
            job_id="j1",
            work_mode="on-site",
            access_support_provided=["loi_xe_lan", "may_tro_thinh"],
        )
        is_feasible, _ = check_hard_constraints(cand, job)
        self.assertTrue(is_feasible)

    def test_remote_work_mode_conflict(self) -> None:
        """Ứng viên chỉ làm remote nhưng công việc bắt buộc 100% on-site -> loại."""
        cand = CandidateProfile(candidate_id="c1", work_mode="remote")
        job = JobProfile(job_id="j1", work_mode="on-site")
        is_feasible, reason = check_hard_constraints(cand, job)
        self.assertFalse(is_feasible)
        self.assertIn("Remote", reason)

    def test_direct_track_exclusion(self) -> None:
        """Ứng viên Direct Track (đã được nhận riêng) bị loại khỏi pool."""
        cand = CandidateProfile(candidate_id="c1", direct_track=True)
        job = JobProfile(job_id="j1")
        is_feasible, reason = check_hard_constraints(cand, job)
        self.assertFalse(is_feasible)
        self.assertIn("Direct Track", reason)


class TestDeterministicTieBreaking(unittest.TestCase):
    """Kiểm thử phá hòa tất định theo điểm kỹ năng rồi mã ứng viên (Mục 6.4)."""

    def test_tie_breaking_order(self) -> None:
        """Khi 2 ứng viên bằng điểm E, ưu tiên: điểm kỹ năng cao hơn, rồi mã ID."""
        # c1 và c2 đều có tổng E như nhau từ job j1, nhưng c1 có skill cao hơn c2
        # Giả lập bằng build_profile với cấu hình skill tie break
        preprocessor = Preprocessor(PreprocessConfig(use_skill_tie_break=True))

        e_scores = {
            "c1": {"j1": 0.8},
            "c2": {"j1": 0.8},
            "c3": {"j1": 0.8},
        }
        skill_scores = {
            "c1": {"j1": 0.9},
            "c2": {"j1": 0.7},
            "c3": {"j1": 0.9},
        }
        c_scores = {
            "c1": {"j1": 0.8},
            "c2": {"j1": 0.8},
            "c3": {"j1": 0.8},
        }
        quotas = {"j1": 2}

        profile = preprocessor.build_profile(
            e_scores=e_scores,
            c_scores=c_scores,
            quotas=quotas,
            skill_scores=skill_scores,
        )

        job_pref = profile.job_preferences["j1"]
        # Thứ tự đúng: c1 (skill 0.9), c3 (skill 0.9, ID sau c1), c2 (skill 0.7)
        self.assertEqual(job_pref, ("c1", "c3", "c2"))

    def test_tie_breaking_in_full_pipeline(self) -> None:
        """Kiểm thử phá hòa qua pipeline build_profile_from_profiles."""
        job = JobProfile(
            job_id="j1",
            quota=1,
            skills_required=[1.0, 0.0],
            salary_min=10,
            salary_max=20,
        )
        # c1 có kỹ năng khớp hoàn toàn, nhưng lương lệch nhẹ
        # c2 có kỹ năng khớp 1 phần, nhưng các thành phần khác bù để E ngang nhau
        cand_a = CandidateProfile(
            candidate_id="cand_A",
            skills=[1.0, 0.0],  # Skill = 1.0
            expected_salary_min=10,
            expected_salary_max=20,
        )
        cand_b = CandidateProfile(
            candidate_id="cand_B",
            skills=[0.5, 0.5],  # Skill thấp hơn
            expected_salary_min=10,
            expected_salary_max=20,
        )

        profile = build_profile_from_profiles([cand_a, cand_b], [job])
        # cand_A có skill_score cao hơn nên phải đứng đầu trong nguyện vọng của j1
        self.assertEqual(profile.job_preferences["j1"][0], "cand_A")


class TestExplainabilityAndPostprocessing(unittest.TestCase):
    """Kiểm thử sinh giải thích và Top-K Recommendation (Mục 6.1, 6.7)."""

    def test_explain_match_content(self) -> None:
        """Kiểm tra chuỗi giải thích tiếng Việt chứa chi tiết thành phần."""
        cand = CandidateProfile(
            candidate_id="c1",
            skills=[1.0, 0.0],
            nice_to_have_access_needs=["anh_sang"],
        )
        job = JobProfile(
            job_id="j1",
            skills_required=[1.0, 0.0],
            access_support_provided=["anh_sang"],
        )
        score_e, b_e = calculate_composite_e_score(cand, job)
        score_c, b_c = calculate_composite_c_score(cand, job)

        explanation = explain_match(cand, job, score_e, b_e, score_c, b_c)
        self.assertIn("Kỹ năng chuyên môn", explanation)
        self.assertIn("tiếp cận", explanation)

    def test_top_k_recommendations_with_breakdown(self) -> None:
        """Top-K recommendation và danh sách dự phòng mang theo giải thích và breakdown."""
        cand = CandidateProfile(
            candidate_id="c1",
            skills=[1.0, 0.0],
            expected_salary_min=10,
            expected_salary_max=20,
        )
        job1 = JobProfile(job_id="j1", quota=1, skills_required=[1.0, 0.0], salary_min=10, salary_max=20)
        job2 = JobProfile(job_id="j2", quota=1, skills_required=[0.9, 0.1], salary_min=10, salary_max=20)

        profile = build_profile_from_profiles([cand], [job1, job2])
        engine = GaleShapleyMatching(profile, optimal="candidate")
        result = engine.solve()

        top_k = generate_top_k(profile, result, k=2)
        recs = top_k["c1"]
        self.assertGreaterEqual(len(recs), 1)
        first_rec = recs[0]
        self.assertIn("e_breakdown", first_rec.breakdown)
        self.assertIn("skill", first_rec.breakdown["e_breakdown"])


class TestEvaluationMetricsAndFairness(unittest.TestCase):
    """Kiểm thử bộ chỉ số đánh giá nâng cao (Mục 6.5) và kiểm toán công bằng (Mục 6.6)."""

    def test_average_ranks(self) -> None:
        """Kiểm tra tính thứ bậc trung bình (1-based rank)."""
        job = JobProfile(job_id="j1", quota=1, skills_required=[1.0, 0.0])
        cand = CandidateProfile(candidate_id="c1", skills=[1.0, 0.0])
        profile = build_profile_from_profiles([cand], [job])
        result = GaleShapleyMatching(profile).solve()

        rank_rep = average_ranks(profile, result)
        self.assertEqual(rank_rep.average_candidate_rank, 1.0)
        self.assertEqual(rank_rep.average_job_rank, 1.0)

    def test_coverage_and_conflict_metrics(self) -> None:
        """Kiểm tra tỷ lệ chưa ghép và tỷ lệ xung đột tắc nghẽn (Congestion Rate)."""
        c1 = CandidateProfile(candidate_id="c1", skills=[1.0, 0.0])
        c2 = CandidateProfile(candidate_id="c2", skills=[0.8, 0.2])
        j1 = JobProfile(job_id="j1", quota=1, skills_required=[1.0, 0.0])

        profile = build_profile_from_profiles([c1, c2], [j1])
        result = GaleShapleyMatching(profile).solve()

        cov = coverage_metrics(profile, result)
        self.assertEqual(cov.total_candidates, 2)
        self.assertEqual(cov.unmatched_candidate_count, 1)
        self.assertEqual(cov.unmatched_candidate_rate, 0.5)

        conf = conflict_metrics(result)
        # Trong Gale-Shapley mỗi người chỉ ghép tối đa 1 nơi -> congestion = 0
        self.assertEqual(conf.congestion_rate, 0.0)

    def test_vector_recall_and_ranking_quality(self) -> None:
        """Kiểm tra độ bao phủ Recall Top-N của pgvector và Precision@K, nDCG@K."""
        ann_results = ["c1", "c2", "c3"]
        brute_force = ["c1", "c2", "c4"]
        recall = calculate_vector_recall(ann_results, brute_force)
        # Trùng 2 / 3 -> 0.6667
        self.assertAlmostEqual(recall, 2.0 / 3.0)

        # Precision@K và nDCG@K
        retrieved = ["c1", "c2", "c3", "c4"]
        ground_truth = {"c1": 3.0, "c3": 2.0}
        p_at_2 = calculate_precision_at_k(retrieved, ground_truth, k=2)
        self.assertEqual(p_at_2, 0.5)  # c1 relevant, c2 không

        ndcg = calculate_ndcg_at_k(retrieved, ground_truth, k=3)
        self.assertGreater(ndcg, 0.0)
        self.assertLessEqual(ndcg, 1.0)

    def test_fairness_audit_four_fifths_rule(self) -> None:
        """Kiểm tra kiểm toán công bằng theo Four-Fifths Rule (Mục 6.6)."""
        # Nhóm vận động: 8/10 được ghép (0.80), Nhóm thị giác: 7/10 (0.70)
        # Tỷ số: 0.70 / 0.80 = 0.875 >= 0.80 -> Hợp lệ
        matches_fair = {f"c_vd_{i}": "j1" for i in range(8)}
        matches_fair.update({f"c_tg_{i}": "j2" for i in range(7)})
        groups = {f"c_vd_{i}": "van_dong" for i in range(10)}
        groups.update({f"c_tg_{i}": "thi_giac" for i in range(10)})

        report_fair = audit_fairness(matches_fair, groups)
        self.assertTrue(report_fair.is_fair)
        self.assertAlmostEqual(report_fair.disparate_impact_ratio, 0.875)

        # Trường hợp vi phạm: nhóm thị giác chỉ được ghép 1/10 (0.10) so với 8/10 (0.80)
        matches_unfair = {f"c_vd_{i}": "j1" for i in range(8)}
        matches_unfair.update({"c_tg_0": "j2"})
        report_unfair = audit_fairness(matches_unfair, groups)
        self.assertFalse(report_unfair.is_fair)
        self.assertLess(report_unfair.disparate_impact_ratio, 0.80)

    def test_benchmark_comparison_table_generation(self) -> None:
        """Kiểm tra sinh bảng so sánh Mục 6.5 trực diện với Greedy Baseline."""
        c1 = CandidateProfile(candidate_id="c1", skills=[1.0, 0.0])
        j1 = JobProfile(job_id="j1", quota=1, skills_required=[1.0, 0.0])
        profile = build_profile_from_profiles([c1], [j1])

        benchmark_results = run_profile_benchmark(profile)
        self.assertGreaterEqual(len(benchmark_results), 2)

        table_md = to_evaluation_comparison_table(benchmark_results)
        self.assertIn("| Chỉ số đánh giá (Mục 6.5) |", table_md)
        self.assertIn("Gale-Shapley (C-optimal)", table_md)
        self.assertIn("Baseline (Sắp xếp theo điểm)", table_md)


class TestFastAPIStableMatchingEndpoint(unittest.TestCase):
    """Kiểm thử API /matching/stable với dữ liệu hồ sơ nâng cấp (Mục 6.1 - 6.5)."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_api_composite_profiles_matching(self) -> None:
        """Gửi request chứa các trường nâng cấp và kiểm tra response giải thích & breakdown."""
        payload = {
            "candidates": [
                {
                    "candidate_id": "cand_01",
                    "skills": [0.9, 0.8, 0.1],
                    "essential_access_needs": ["loi_xe_lan"],
                    "nice_to_have_access_needs": ["anh_sang_tot"],
                    "expected_salary_min": 12.0,
                    "expected_salary_max": 18.0,
                    "work_mode": "on-site",
                    "work_hours": "full-time",
                    "location": "Hà Nội",
                    "priority_ranking": ["income", "distance_or_remote", "career_path"],
                    "experience_years": 2.5,
                },
                {
                    "candidate_id": "cand_02",
                    "skills": [0.3, 0.2, 0.9],
                    "work_mode": "remote",
                    "work_hours": "full-time",
                    "location": "Đà Nẵng",
                },
            ],
            "jobs": [
                {
                    "job_id": "job_01",
                    "title": "Backend Python Developer",
                    "quota": 1,
                    "skills_required": [0.9, 0.8, 0.1],
                    "access_support_provided": ["loi_xe_lan", "anh_sang_tot"],
                    "salary_min": 15.0,
                    "salary_max": 20.0,
                    "work_mode": "on-site",
                    "work_hours": "full-time",
                    "location": "Hà Nội",
                    "experience_required_years": 2.0,
                }
            ],
            "proposing_side": "candidate",
            "top_k": 2,
        }

        resp = self.client.post("/matching/stable", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Kiểm tra tính ổn định
        metrics = data["metrics"]
        self.assertTrue(metrics["is_stable"])
        self.assertEqual(metrics["blocking_pairs_count"], 0)

        # cand_01 phải được ghép với job_01
        self.assertEqual(data["matches"]["cand_01"], "job_01")
        self.assertEqual(len(data["matched_pairs"]), 1)
        matched_pair = data["matched_pairs"][0]
        self.assertEqual(matched_pair["candidate_id"], "cand_01")
        self.assertEqual(matched_pair["job_id"], "job_01")

        # Phải có giải thích và phân rã E / C
        self.assertIn("explanation", matched_pair)
        self.assertTrue(len(matched_pair["explanation"]) > 0)
        self.assertIn("e_breakdown", matched_pair)
        self.assertIn("c_breakdown", matched_pair)
        self.assertIn("skill", matched_pair["e_breakdown"])

        # Top-K recommendations
        self.assertIn("cand_01", data["recommendations"])
        cand_recs = data["recommendations"]["cand_01"]
        self.assertGreaterEqual(len(cand_recs), 1)
        self.assertIn("explanation", cand_recs[0])

        # Metrics mở rộng
        metrics = data["metrics"]
        self.assertIn("average_candidate_rank", metrics)
        self.assertIn("average_job_rank", metrics)
        self.assertIn("unmatched_candidate_rate", metrics)
        self.assertEqual(metrics["congestion_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
