#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bộ kiểm thử hồi quy và vét cạn cho 5 lỗi đã sửa (Regression Tests).

Bao gồm:
1. Lỗi 1: Chế độ employer-proposing không ghép ứng viên vào việc họ không chấp nhận
   (Kiểm tra vét cạn trên nhiều hạt giống và danh sách bị cắt max_candidate_preferences).
2. Lỗi 2: Quota = 0 không bị ép thành 1, bỏ qua việc có quota 0 khi dựng danh sách.
3. Lỗi 3: Direct Track trừ đúng số slot của việc tuyển trực tiếp (không tuyển vượt).
4. Lỗi 4: is_stable ban đầu là None, chỉ đặt True sau khi gọi find_blocking_pairs / verify_stability.
5. Endpoint /matching/stable hoạt động chính xác với cả 2 chế độ và kiểm tra hồi quy.
"""

import unittest
from fastapi.testclient import TestClient

from app.main import app
from src.core.matching_engine import GaleShapleyMatching, MatchingResult
from src.core.preprocessor import (
    PreferenceProfile,
    PreprocessConfig,
    Preprocessor,
    build_profile_from_profiles,
)
from src.core.profile_models import CandidateProfile, JobProfile
from src.evaluation.benchmark import generate_synthetic_market
from src.evaluation.metrics import find_blocking_pairs


def build_profile(
    e_scores,
    c_scores,
    quotas,
    config: PreprocessConfig | None = None,
    **kwargs,
) -> PreferenceProfile:
    preprocessor = Preprocessor(config or PreprocessConfig())
    return preprocessor.build_profile(e_scores, c_scores, quotas, **kwargs)


class TestExhaustiveEmployerProposingRegression(unittest.TestCase):
    """Kiểm tra vét cạn chế độ Employer-Proposing (Lỗi 1)."""

    def test_exhaustive_employer_proposing_no_unacceptable_assignments(self) -> None:
        """Kiểm tra vét cạn trên nhiều thị trường với max_candidate_preferences nhỏ:

        Đảm bảo không bao giờ có ứng viên bị gán vào công việc không nằm trong
        danh sách nguyện vọng của họ, cho cả 2 chế độ candidate và employer.
        """
        for seed in (42, 101, 2024, 9999):
            for max_cand_prefs in (1, 2, 3, 5):
                e, c, quotas = generate_synthetic_market(
                    n_candidates=150,
                    n_jobs=30,
                    avg_quota=3,
                    density=0.4,
                    seed=seed,
                )
                profile = build_profile(
                    e,
                    c,
                    quotas,
                    config=PreprocessConfig(
                        max_candidate_preferences=max_cand_prefs,
                        tie_break_seed=seed,
                    ),
                )

                for mode in ("candidate", "employer"):
                    with self.subTest(seed=seed, max_prefs=max_cand_prefs, mode=mode):
                        engine = GaleShapleyMatching(profile, optimal=mode)
                        result = engine.solve()

                        # 1. Mọi ứng viên có việc phải có công việc đó trong preference list
                        for cand, job in result.matches.items():
                            cand_prefs = profile.candidate_preferences.get(cand, ())
                            self.assertIn(
                                job,
                                cand_prefs,
                                f"Lỗi 1 vi phạm: Ứng viên {cand} bị gán vào việc {job} "
                                f"không có trong nguyện vọng ở chế độ {mode}!",
                            )

                            job_prefs = profile.job_preferences.get(job, ())
                            self.assertIn(
                                cand,
                                job_prefs,
                                f"Ứng viên {cand} không có trong danh sách của {job}!",
                            )

                        # 2. Không vi phạm quota
                        for job, assigned in result.job_assignments.items():
                            self.assertLessEqual(
                                len(assigned),
                                profile.quotas.get(job, 0),
                                f"Vị trí {job} bị vượt quota ở chế độ {mode}!",
                            )

                        # 3. Không có blocking pairs
                        bps = find_blocking_pairs(profile, result)
                        self.assertEqual(
                            bps,
                            [],
                            f"Xuất hiện blocking pairs ở chế độ {mode}: {bps}",
                        )


class TestZeroQuotaRegression(unittest.TestCase):
    """Kiểm thử xử lý Quota = 0 (Lỗi 2)."""

    def test_quota_zero_not_forced_to_one(self) -> None:
        """Việc có quota = 0 không nhận bất kỳ ứng viên nào."""
        c1 = CandidateProfile(
            candidate_id="c1",
            skills=[1.0, 0.0],
            expected_salary_min=10,
            expected_salary_max=20,
            work_mode="remote",
        )
        c2 = CandidateProfile(
            candidate_id="c2",
            skills=[0.9, 0.1],
            expected_salary_min=10,
            expected_salary_max=20,
            work_mode="remote",
        )
        j_zero = JobProfile(
            job_id="j_zero",
            quota=0,
            skills_required=[1.0, 0.0],
            salary_min=10,
            salary_max=20,
            work_mode="remote",
        )
        j_open = JobProfile(
            job_id="j_open",
            quota=1,
            skills_required=[1.0, 0.0],
            salary_min=10,
            salary_max=20,
            work_mode="remote",
        )

        profile = build_profile_from_profiles([c1, c2], [j_zero, j_open])

        # Quota của j_zero phải là 0, không bị ép thành 1
        self.assertEqual(profile.quotas["j_zero"], 0)
        self.assertEqual(profile.quotas["j_open"], 1)

        # j_zero không xuất hiện trong nguyện vọng của ứng viên
        for cand_id in ("c1", "c2"):
            self.assertNotIn("j_zero", profile.candidate_preferences.get(cand_id, ()))

        # Chạy matching cho cả 2 chế độ
        for mode in ("candidate", "employer"):
            with self.subTest(mode=mode):
                result = GaleShapleyMatching(profile, optimal=mode).solve()
                # j_zero không được nhận ai
                self.assertEqual(result.job_assignments.get("j_zero", []), [])
                self.assertNotIn("j_zero", result.matches.values())
                # j_open nhận đúng 1 người
                self.assertEqual(len(result.job_assignments.get("j_open", [])), 1)


class TestDirectTrackQuotaDeduction(unittest.TestCase):
    """Kiểm thử Direct Track trừ đúng slot của việc (Lỗi 3)."""

    def test_direct_track_deducts_quota_prevents_overhiring(self) -> None:
        """Sau khi tuyển trực tiếp 1 người vào việc có 1 chỗ, batch không nhận thêm."""
        # Ứng viên trực tiếp đã vào j1
        cand_dt = CandidateProfile(
            candidate_id="cand_dt",
            direct_track=True,
            hired_job_id="j1",
            skills=[1.0, 0.0],
        )
        # Ứng viên bình thường cũng muốn j1
        cand_batch = CandidateProfile(
            candidate_id="cand_batch",
            direct_track=False,
            skills=[0.9, 0.1],
            expected_salary_min=10,
            expected_salary_max=20,
            work_mode="remote",
        )
        # j1 ban đầu có quota 1
        j1 = JobProfile(
            job_id="j1",
            quota=1,
            skills_required=[1.0, 0.0],
            salary_min=10,
            salary_max=20,
            work_mode="remote",
        )

        profile = build_profile_from_profiles([cand_dt, cand_batch], [j1])

        # Quota còn lại của j1 trong profile phải là 0
        self.assertEqual(profile.quotas["j1"], 0)

        # Batch matching không được ghép cand_batch vào j1
        for mode in ("candidate", "employer"):
            with self.subTest(mode=mode):
                result = GaleShapleyMatching(profile, optimal=mode).solve()
                self.assertEqual(result.job_assignments.get("j1", []), [])
                self.assertNotIn("cand_batch", result.matches)

    def test_filled_slots_field_on_job_profile(self) -> None:
        """filled_slots trên JobProfile trừ trực tiếp vào quota khả dụng."""
        cand = CandidateProfile(
            candidate_id="c1",
            skills=[1.0, 0.0],
            expected_salary_min=10,
            expected_salary_max=20,
            work_mode="remote",
        )
        j1 = JobProfile(
            job_id="j1",
            quota=2,
            filled_slots=2,  # Đã tuyển đủ 2 slot trước đó
            skills_required=[1.0, 0.0],
            salary_min=10,
            salary_max=20,
            work_mode="remote",
        )

        self.assertEqual(j1.remaining_quota, 0)
        profile = build_profile_from_profiles([cand], [j1])
        self.assertEqual(profile.quotas["j1"], 0)

        result = GaleShapleyMatching(profile).solve()
        self.assertEqual(result.job_assignments.get("j1", []), [])


class TestIsStableVerification(unittest.TestCase):
    """Kiểm thử cờ is_stable trong MatchingResult (Lỗi 4)."""

    def test_is_stable_is_none_before_verification(self) -> None:
        """is_stable ban đầu là None, sau khi gọi verify_stability hoặc find_blocking_pairs mới là True."""
        e, c, quotas = generate_synthetic_market(10, 3, avg_quota=2, seed=42)
        profile = build_profile(e, c, quotas)

        result = GaleShapleyMatching(profile).solve()
        # Ban đầu kết quả có is_stable là None
        self.assertIsNone(result.is_stable)

        # Sau khi verify_stability
        is_stable = result.verify_stability(profile)
        self.assertTrue(is_stable)
        self.assertTrue(result.is_stable)

    def test_find_blocking_pairs_sets_is_stable_flag(self) -> None:
        """find_blocking_pairs tự động cập nhật cờ is_stable trên MatchingResult."""
        e, c, quotas = generate_synthetic_market(10, 3, avg_quota=2, seed=42)
        profile = build_profile(e, c, quotas)

        result = GaleShapleyMatching(profile).solve()
        self.assertIsNone(result.is_stable)

        find_blocking_pairs(profile, result)
        self.assertTrue(result.is_stable)


class TestApiEmployerProposingRegression(unittest.TestCase):
    """Kiểm thử endpoint API /matching/stable ở chế độ optimal=employer."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_api_employer_mode_stability_and_validity(self) -> None:
        """Gọi API /matching/stable với optimal=employer và kiểm tra kết quả."""
        payload = {
            "candidates": [
                {
                    "id": "c1",
                    "skills": [1.0, 0.0],
                    "work_mode": "remote",
                    "expected_salary_min": 10.0,
                    "expected_salary_max": 20.0,
                },
                {
                    "id": "c2",
                    "skills": [0.0, 1.0],
                    "work_mode": "remote",
                    "expected_salary_min": 10.0,
                    "expected_salary_max": 20.0,
                },
            ],
            "jobs": [
                {
                    "id": "j1",
                    "quota": 1,
                    "skills_required": [1.0, 0.0],
                    "work_mode": "remote",
                    "salary_min": 10.0,
                    "salary_max": 20.0,
                },
                {
                    "id": "j2",
                    "quota": 1,
                    "skills_required": [0.0, 1.0],
                    "work_mode": "remote",
                    "salary_min": 10.0,
                    "salary_max": 20.0,
                },
            ],
            "optimal": "employer",
            "top_k": 2,
        }
        resp = self.client.post("/matching/stable", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["metrics"]["is_stable"])
        self.assertEqual(data["metrics"]["blocking_pairs_count"], 0)
        self.assertEqual(data["matches"]["c1"], "j1")
        self.assertEqual(data["matches"]["c2"], "j2")


if __name__ == "__main__":
    unittest.main()
