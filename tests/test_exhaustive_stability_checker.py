#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bộ kiểm tra tính ổn định vét cạn độc lập (Independent Exhaustive Stability Checker).

Mục đích:
    Cung cấp lớp kiểm thử thứ hai (second verification layer) hoàn toàn độc lập
    với engine và hàm find_blocking_pairs nội bộ của dự án.
    Kiểm tra tính đúng đắn từ nguyên lý toán học gốc (First-principles Many-to-One Stability).

Định nghĩa toán học của Cặp gây bất ổn (Blocking Pair) trong Many-to-One Matching:
    Cho thị trường hai phía gồm tập ứng viên C và tập công việc J với hạn mức q_j.
    Một kết quả ghép cặp mu = (matches, job_assignments) có cặp (c, j) là blocking pair
    khi và chỉ khi thỏa mãn đồng thời hai điều kiện:
    1) Ứng viên c xem j là chấp nhận được (j in pref(c)) và c thích j hơn đối tác hiện tại mu(c):
       - Nếu c chưa có việc (mu(c) is None): c luôn muốn làm việc tại j hơn là thất nghiệp.
       - Nếu c đã có việc (mu(c) = j_curr): j nằm trước j_curr trong pref(c).
    2) Công việc j xem c là chấp nhận được (c in pref(j)) và:
       - Trường hợp 2a: j còn chỗ trống (|mu(j)| < q_j).
       - Trường hợp 2b: j đã đủ người (|mu(j)| >= q_j), nhưng tồn tại ít nhất một người c' in mu(j)
         mà j thích c hơn c' (tức c nằm trước c' trong pref(j)).
"""

from __future__ import annotations

import unittest
from typing import Any

from src.core.matching_engine import GaleShapleyMatching
from src.core.preprocessor import (
    PreferenceProfile,
    PreprocessConfig,
    Preprocessor,
    build_profile_from_profiles,
)
from src.core.profile_models import CandidateProfile, JobProfile
from src.evaluation.benchmark import generate_synthetic_market
from src.evaluation.metrics import find_blocking_pairs as project_find_blocking_pairs


def independent_find_blocking_pairs(
    candidate_preferences: dict[str, list[str]],
    job_preferences: dict[str, list[str]],
    job_quotas: dict[str, int],
    matches: dict[str, str],
    job_assignments: dict[str, list[str]],
) -> list[dict[str, str]]:
    """Hàm kiểm tra vét cạn độc lập 100%, không dùng mã nguồn của engine hay metrics nội bộ.

    Args:
        candidate_preferences: {candidate_id: [job_id, ...]} theo thứ tự giảm dần.
        job_preferences: {job_id: [candidate_id, ...]} theo thứ tự giảm dần.
        job_quotas: {job_id: quota} (quota >= 0).
        matches: {candidate_id: job_id} kết quả phân bổ.
        job_assignments: {job_id: [candidate_id, ...]} danh sách ứng viên nhận vào mỗi job.

    Returns:
        Danh sách các cặp bất ổn tìm thấy: [{"candidate_id": c, "job_id": j, "reason": ...}].
    """
    blocking_pairs: list[dict[str, str]] = []

    for cand_id, cand_prefs in candidate_preferences.items():
        curr_job = matches.get(cand_id)

        # Xác định danh sách công việc mà ứng viên cand_id nghiêm ngặt thích hơn công việc hiện tại
        preferred_jobs: list[str] = []
        if curr_job is None:
            # Ứng viên chưa được phân bổ -> thích mọi công việc chấp nhận được trong danh sách
            preferred_jobs = list(cand_prefs)
        else:
            if curr_job in cand_prefs:
                curr_idx = cand_prefs.index(curr_job)
                preferred_jobs = cand_prefs[:curr_idx]
            else:
                # Nếu curr_job không có trong cand_prefs (vi phạm cá nhân), thích tất cả
                preferred_jobs = list(cand_prefs)

        for job_id in preferred_jobs:
            job_pref_list = job_preferences.get(job_id, [])
            if cand_id not in job_pref_list:
                # Nhà tuyển dụng job_id không chấp nhận cand_id -> không thể tạo blocking pair
                continue

            cand_rank_in_job = job_pref_list.index(cand_id)
            assigned_cands = job_assignments.get(job_id, [])
            quota = job_quotas.get(job_id, 0)

            # Trường hợp 2a: Vị trí còn chỗ trống
            if len(assigned_cands) < quota:
                blocking_pairs.append({
                    "candidate_id": cand_id,
                    "job_id": job_id,
                    "reason": f"Vị trí {job_id} còn slot trống ({len(assigned_cands)}/{quota}) và cả hai đều muốn nhau",
                })
                continue

            # Trường hợp 2b: Vị trí đã đủ người nhưng có người kém hơn cand_id
            for assigned in assigned_cands:
                if assigned in job_pref_list:
                    assigned_rank = job_pref_list.index(assigned)
                    if cand_rank_in_job < assigned_rank:
                        blocking_pairs.append({
                            "candidate_id": cand_id,
                            "job_id": job_id,
                            "reason": f"Vị trí {job_id} thích {cand_id} hơn {assigned}",
                        })
                        break
                else:
                    # Ứng viên đang giữ chỗ không có trong preference của job
                    blocking_pairs.append({
                        "candidate_id": cand_id,
                        "job_id": job_id,
                        "reason": f"Vị trí {job_id} đang chứa ứng viên không hợp lệ {assigned}",
                    })
                    break

    return blocking_pairs


def independent_verify_individual_rationality(
    candidate_preferences: dict[str, list[str]],
    job_preferences: dict[str, list[str]],
    job_quotas: dict[str, int],
    matches: dict[str, str],
    job_assignments: dict[str, list[str]],
) -> list[str]:
    """Kiểm tra tính hợp lý cá nhân (Individual Rationality) và toàn vẹn quota.

    Returns:
        Danh sách thông báo vi phạm (nếu có). Trả về [] nếu hợp lệ hoàn toàn.
    """
    violations: list[str] = []

    # 1. Ứng viên chỉ được ghép vào công việc họ chấp nhận
    for c_id, j_id in matches.items():
        if j_id not in candidate_preferences.get(c_id, []):
            violations.append(
                f"Ứng viên {c_id} bị gán vào {j_id} không có trong danh sách nguyện vọng!"
            )
        if c_id not in job_preferences.get(j_id, []):
            violations.append(
                f"Vị trí {j_id} bị gán ứng viên {c_id} không có trong danh sách NTD chấp nhận!"
            )

    # 2. Quota không bị vượt quá
    for j_id, assigned in job_assignments.items():
        quota = job_quotas.get(j_id, 0)
        if len(assigned) > quota:
            violations.append(
                f"Vị trí {j_id} nhận {len(assigned)} người, vượt quá quota {quota}!"
            )

    # 3. Tính nhất quán hai chiều giữa matches và job_assignments
    for c_id, j_id in matches.items():
        if c_id not in job_assignments.get(j_id, []):
            violations.append(
                f"Mâu thuẫn: {c_id} trỏ tới {j_id} trong matches nhưng không có trong job_assignments[{j_id}]!"
            )

    for j_id, c_list in job_assignments.items():
        for c_id in c_list:
            if matches.get(c_id) != j_id:
                violations.append(
                    f"Mâu thuẫn: {c_id} nằm trong job_assignments[{j_id}] nhưng matches[{c_id}]={matches.get(c_id)}!"
                )

    return violations


class TestIndependentExhaustiveStabilityChecker(unittest.TestCase):
    """Kiểm thử tính đúng đắn của bộ kiểm tra độc lập và đối chiếu hai lớp."""

    def test_checker_detects_intentional_blocking_pair(self) -> None:
        """Kiểm tra độc lập bắt chính xác blocking pair khi cố ý hoán đổi kết quả phân bổ."""
        # Thị trường: C1 thích J1 > J2; C2 thích J1 > J2
        # J1 thích C1 > C2; J2 thích C1 > C2. Quota mỗi việc = 1.
        cand_prefs = {
            "C1": ["J1", "J2"],
            "C2": ["J1", "J2"],
        }
        job_prefs = {
            "J1": ["C1", "C2"],
            "J2": ["C1", "C2"],
        }
        quotas = {"J1": 1, "J2": 1}

        # Ghép cặp cố tình không ổn định: C1 vào J2, C2 vào J1
        unstable_matches = {"C1": "J2", "C2": "J1"}
        unstable_assignments = {"J1": ["C2"], "J2": ["C1"]}

        bps = independent_find_blocking_pairs(
            candidate_preferences=cand_prefs,
            job_preferences=job_prefs,
            job_quotas=quotas,
            matches=unstable_matches,
            job_assignments=unstable_assignments,
        )

        # C1 thích J1 hơn J2, và J1 thích C1 hơn C2 -> (C1, J1) phải là blocking pair!
        self.assertGreaterEqual(len(bps), 1)
        found = any(bp["candidate_id"] == "C1" and bp["job_id"] == "J1" for bp in bps)
        self.assertTrue(found, "Independent checker phải phát hiện cặp (C1, J1) là blocking pair!")

    def test_checker_detects_vacancy_blocking_pair(self) -> None:
        """Kiểm tra độc lập phát hiện blocking pair khi job còn chỗ trống mà ứng viên thích job đó."""
        cand_prefs = {"C1": ["J1"]}
        job_prefs = {"J1": ["C1"]}
        quotas = {"J1": 1}

        # C1 chưa có việc, J1 còn trống
        empty_matches: dict[str, str] = {}
        empty_assignments: dict[str, list[str]] = {"J1": []}

        bps = independent_find_blocking_pairs(
            candidate_preferences=cand_prefs,
            job_preferences=job_prefs,
            job_quotas=quotas,
            matches=empty_matches,
            job_assignments=empty_assignments,
        )

        self.assertEqual(len(bps), 1)
        self.assertEqual(bps[0]["candidate_id"], "C1")
        self.assertEqual(bps[0]["job_id"], "J1")

    def test_dual_layer_verification_on_gale_shapley_synthetic_markets(self) -> None:
        """Đối chiếu hai lớp độc lập trên thị trường ngẫu nhiên đa dạng hạt giống:

        Lớp 1: find_blocking_pairs nội bộ của dự án.
        Lớp 2: independent_find_blocking_pairs độc lập từ nguyên lý gốc.
        Cả hai lớp phải đồng thuận tuyệt đối: 0 blocking pairs và 0 vi phạm tính hợp lý cá nhân.
        """
        seeds = [42, 101, 777, 2024, 9999]
        modes = ["candidate", "employer"]

        for seed in seeds:
            e, c, quotas = generate_synthetic_market(
                n_candidates=80,
                n_jobs=20,
                avg_quota=3,
                density=0.5,
                seed=seed,
            )
            preprocessor = Preprocessor(PreprocessConfig(tie_break_seed=seed))
            profile = preprocessor.build_profile(e, c, quotas)

            cand_prefs_dict = {k: list(v) for k, v in profile.candidate_preferences.items()}
            job_prefs_dict = {k: list(v) for k, v in profile.job_preferences.items()}

            for mode in modes:
                with self.subTest(seed=seed, mode=mode):
                    engine = GaleShapleyMatching(profile, optimal=mode)
                    result = engine.solve()

                    # Lớp 1: Hàm của dự án
                    project_bps = project_find_blocking_pairs(profile, result)
                    self.assertEqual(
                        len(project_bps),
                        0,
                        f"Lớp 1 (Project) phát hiện blocking pair ở seed={seed}, mode={mode}",
                    )

                    # Lớp 2: Hàm độc lập từ nguyên lý gốc
                    independent_bps = independent_find_blocking_pairs(
                        candidate_preferences=cand_prefs_dict,
                        job_preferences=job_prefs_dict,
                        job_quotas=profile.quotas,
                        matches=result.matches,
                        job_assignments=result.job_assignments,
                    )
                    self.assertEqual(
                        len(independent_bps),
                        0,
                        f"Lớp 2 (Độc lập) phát hiện blocking pair ở seed={seed}, mode={mode}: {independent_bps}",
                    )

                    # Kiểm tra tính hợp lý cá nhân và quota
                    ir_violations = independent_verify_individual_rationality(
                        candidate_preferences=cand_prefs_dict,
                        job_preferences=job_prefs_dict,
                        job_quotas=profile.quotas,
                        matches=result.matches,
                        job_assignments=result.job_assignments,
                    )
                    self.assertEqual(
                        ir_violations,
                        [],
                        f"Phát hiện vi phạm tính hợp lý cá nhân: {ir_violations}",
                    )

    def test_independent_checker_confirms_greedy_instability(self) -> None:
        """Hàm kiểm tra độc lập xác nhận Baseline Greedy tạo ra blocking pairs thực sự."""
        seed = 42
        e, c, quotas = generate_synthetic_market(
            n_candidates=60,
            n_jobs=15,
            avg_quota=2,
            seed=seed,
        )
        preprocessor = Preprocessor(PreprocessConfig(tie_break_seed=seed))
        profile = preprocessor.build_profile(e, c, quotas)

        # Chạy thuật toán Greedy Baseline (sắp xếp ứng viên theo điểm cao nhất)
        from src.evaluation.benchmark import GreedyBaselineMatcher
        greedy_matcher = GreedyBaselineMatcher(profile)
        greedy_result = greedy_matcher.solve()

        cand_prefs_dict = {k: list(v) for k, v in profile.candidate_preferences.items()}
        job_prefs_dict = {k: list(v) for k, v in profile.job_preferences.items()}

        greedy_bps = independent_find_blocking_pairs(
            candidate_preferences=cand_prefs_dict,
            job_preferences=job_prefs_dict,
            job_quotas=profile.quotas,
            matches=greedy_result.matches,
            job_assignments=greedy_result.job_assignments,
        )

        # Greedy phải có blocking pairs (> 0)
        self.assertGreater(
            len(greedy_bps),
            0,
            "Independent checker phải chứng minh Greedy tạo ra blocking pairs!",
        )


if __name__ == "__main__":
    unittest.main()
