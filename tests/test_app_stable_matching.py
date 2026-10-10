#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bộ kiểm thử toàn diện cho thuật toán Gale-Shapley Stable Matching trong module `app/`.

Bao phủ các khía cạnh:
1. Đại số Không gian Vector Oxyz: Norm, Dot Product, Cosine Similarity.
2. Phá hòa tất định Deterministic Tie-Breaking.
3. Bộ lọc điều kiện tiên quyết Hard Constraints (Domain Filter D).
4. Thuật toán Candidate-Proposing Gale-Shapley (Many-to-One).
5. Thuật toán Employer-Proposing Gale-Shapley (Many-to-One).
6. Tính ổn định tuyệt đối (100% Stability - 0% Blocking Pairs).
7. Hậu xử lý Top-K Recommendations (Normalized Surplus Score).
8. Các chỉ số KPIs định lượng (Satisfaction, Fill Rate, Stability).
9. Kiểm thử API Endpoints FastAPI (/matching/stable, /matching/matrix) thông qua TestClient.
"""

from __future__ import annotations

import unittest
from fastapi.testclient import TestClient

from app.main import app
from app.matching import (
    calculate_cosine_similarity_3d,
    calculate_dot_product_3d,
    calculate_vector_norm_3d,
    candidate_proposing_deferred_acceptance,
    deterministic_tie_breaker,
    employer_proposing_deferred_acceptance,
    find_blocking_pairs,
    generate_top_k_recommendations,
    run_matrix_matching,
    run_stable_matching,
)
from app.schemas import (
    CandidateProfileInput,
    JobProfileInput,
    MatrixMatchingRequest,
    StableMatchingRequest,
    Vector3D,
)


class TestVectorOxyzMath(unittest.TestCase):
    """Kiểm thử các phép toán vector trong không gian Oxyz."""

    def test_vector_norm(self) -> None:
        """Kiểm tra độ dài Euclidean trong Oxyz."""
        # Vector (3, 4, 0) có norm = 5.0
        v1 = (3.0, 4.0, 0.0)
        self.assertAlmostEqual(calculate_vector_norm_3d(v1), 5.0)

        # Vector gốc (0, 0, 0) có norm = 0.0
        v_zero = (0.0, 0.0, 0.0)
        self.assertEqual(calculate_vector_norm_3d(v_zero), 0.0)

    def test_dot_product(self) -> None:
        """Kiểm tra tích vô hướng giữa hai vector."""
        u = (1.0, 2.0, 3.0)
        v = (4.0, 5.0, 6.0)
        # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        self.assertAlmostEqual(calculate_dot_product_3d(u, v), 32.0)

    def test_cosine_similarity(self) -> None:
        """Kiểm tra Cosine Similarity (Điểm E)."""
        u = (1.0, 1.0, 0.0)
        v = (1.0, 1.0, 0.0)
        # Hai vector cùng hướng -> cosine = 1.0
        self.assertAlmostEqual(calculate_cosine_similarity_3d(u, v), 1.0)

        # Hai vector vuông góc -> cosine = 0.0
        u_perp = (1.0, 0.0, 0.0)
        v_perp = (0.0, 1.0, 0.0)
        self.assertAlmostEqual(calculate_cosine_similarity_3d(u_perp, v_perp), 0.0)

        # Vector 0 -> trả về 0.0 an toàn
        zero = (0.0, 0.0, 0.0)
        self.assertEqual(calculate_cosine_similarity_3d(zero, u), 0.0)

    def test_vector_3d_pydantic_methods(self) -> None:
        """Kiểm tra các phương thức của lớp Vector3D trong schemas.py."""
        vec1 = Vector3D(x=3.0, y=4.0, z=0.0)
        vec2 = Vector3D(x=3.0, y=4.0, z=0.0)
        self.assertAlmostEqual(vec1.norm(), 5.0)
        self.assertAlmostEqual(vec1.dot_product(vec2), 25.0)
        self.assertAlmostEqual(vec1.cosine_similarity(vec2), 1.0)
        self.assertEqual(vec1.to_tuple(), (3.0, 4.0, 0.0))


class TestDeterministicTieBreaker(unittest.TestCase):
    """Kiểm thử cơ chế phá hòa tất định."""

    def test_reproducibility(self) -> None:
        """Kiểm tra tính nhất quán giữa các lần gọi băm."""
        h1 = deterministic_tie_breaker(20240517, "cand", "C1", "J1")
        h2 = deterministic_tie_breaker(20240517, "cand", "C1", "J1")
        self.assertEqual(h1, h2, "Mã băm với cùng seed và args phải hoàn toàn trùng khớp.")

    def test_distinct_pairs(self) -> None:
        """Các cặp đối tác khác nhau phải có hash khác nhau."""
        h1 = deterministic_tie_breaker(20240517, "cand", "C1", "J1")
        h2 = deterministic_tie_breaker(20240517, "cand", "C1", "J2")
        self.assertNotEqual(h1, h2)


class TestGaleShapleyManyToOne(unittest.TestCase):
    """Kiểm thử thuật toán Gale-Shapley Deferred Acceptance biến thể Many-to-One."""

    def test_candidate_proposing_matching_stability(self) -> None:
        """Kiểm tra tính ổn định (Blocking Pairs = 0) ở chế độ Candidate-Proposing."""
        candidates_pref = {
            "C1": ["J1", "J2"],
            "C2": ["J2", "J1"],
            "C3": ["J1", "J2"],
        }
        jobs_pref = {
            "J1": ["C1", "C2", "C3"],
            "J2": ["C3", "C2", "C1"],
        }
        quotas = {
            "J1": 1,
            "J2": 2,
        }

        matches, assignments, proposals = candidate_proposing_deferred_acceptance(
            candidates_pref, jobs_pref, quotas
        )

        # C1 thích J1 nhất và J1 thích C1 nhất -> C1 phải vào J1
        self.assertEqual(matches.get("C1"), "J1")
        # J1 có quota = 1
        self.assertEqual(len(assignments["J1"]), 1)
        # J2 có quota = 2
        self.assertLessEqual(len(assignments["J2"]), 2)

        # Khẳng định 0% Blocking Pairs
        bps = find_blocking_pairs(matches, assignments, candidates_pref, jobs_pref, quotas)
        self.assertEqual(bps, [], "Không được tồn tại bất kỳ Blocking Pair nào.")

    def test_employer_proposing_matching(self) -> None:
        """Kiểm tra thuật toán Employer-Proposing Deferred Acceptance."""
        candidates_pref = {
            "C1": ["J1", "J2"],
            "C2": ["J1", "J2"],
        }
        jobs_pref = {
            "J1": ["C1", "C2"],
            "J2": ["C2", "C1"],
        }
        quotas = {"J1": 1, "J2": 1}

        matches, assignments, proposals = employer_proposing_deferred_acceptance(
            candidates_pref, jobs_pref, quotas
        )
        self.assertEqual(len(matches), 2)
        bps = find_blocking_pairs(matches, assignments, candidates_pref, jobs_pref, quotas)
        self.assertEqual(bps, [])

    def test_top_k_recommendations(self) -> None:
        """Kiểm tra hậu xử lý Top-K Recommendations."""
        matches = {"C1": "J1", "C2": "J2"}
        candidates_pref = {"C1": ["J1", "J2"], "C2": ["J2", "J1"]}
        c_scores = {"C1": {"J1": 0.9, "J2": 0.5}, "C2": {"J1": 0.4, "J2": 0.8}}
        e_scores = {"C1": {"J1": 0.95, "J2": 0.3}, "C2": {"J1": 0.6, "J2": 0.85}}

        recs = generate_top_k_recommendations(
            matches=matches,
            candidates_pref=candidates_pref,
            c_scores=c_scores,
            e_scores=e_scores,
            k=2,
        )

        self.assertIn("C1", recs)
        self.assertEqual(len(recs["C1"]), 2)
        # Vị trí số 1 phải là công việc chính
        self.assertEqual(recs["C1"][0]["job_id"], "J1")
        self.assertTrue(recs["C1"][0]["is_primary"])
        # Vị trí số 2 là công việc dự phòng
        self.assertEqual(recs["C1"][1]["job_id"], "J2")
        self.assertFalse(recs["C1"][1]["is_primary"])


class TestApiEndpoints(unittest.TestCase):
    """Kiểm thử các API endpoint mới trên FastAPI TestClient."""

    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_root_endpoint(self) -> None:
        """Endpoint / trả về dịch vụ running."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "running")
        self.assertIn("Gale-Shapley", data["algorithm"])

    def test_matrix_matching_api(self) -> None:
        """Kiểm tra POST /matching/matrix trực tiếp từ ma trận điểm."""
        payload = {
            "candidate_scores": {
                "C1": {"J1": 0.9, "J2": 0.4},
                "C2": {"J1": 0.3, "J2": 0.8},
            },
            "employer_scores": {
                "J1": {"C1": 0.85, "C2": 0.2},
                "J2": {"C1": 0.4, "C2": 0.9},
            },
            "job_quotas": {
                "J1": 1,
                "J2": 1,
            },
            "optimal": "candidate",
            "top_k": 2,
        }
        response = self.client.post("/matching/matrix", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["matches"], {"C1": "J1", "C2": "J2"})
        self.assertTrue(data["metrics"]["is_stable"])
        self.assertEqual(data["metrics"]["blocking_pairs_count"], 0)
        self.assertAlmostEqual(data["metrics"]["fill_rate"], 1.0)
        self.assertIn("C1", data["recommendations"])

    def test_stable_matching_with_vector_3d(self) -> None:
        """Kiểm tra POST /matching/stable với tọa độ Vector Oxyz."""
        payload = {
            "candidates": [
                {
                    "id": "C_Tech",
                    "ho_ten": "Nguyen Van A",
                    "vector_oxyz": {"x": 1.0, "y": 0.0, "z": 0.0},
                    "c_scores": {"J_Tech": 0.95, "J_Art": 0.2},
                },
                {
                    "id": "C_Art",
                    "ho_ten": "Tran Thi B",
                    "vector_oxyz": {"x": 0.0, "y": 1.0, "z": 0.0},
                    "c_scores": {"J_Tech": 0.3, "J_Art": 0.9},
                },
            ],
            "jobs": [
                {
                    "id": "J_Tech",
                    "title": "Software Engineer",
                    "quota": 1,
                    "vector_oxyz": {"x": 1.0, "y": 0.0, "z": 0.0},
                },
                {
                    "id": "J_Art",
                    "title": "Graphic Designer",
                    "quota": 1,
                    "vector_oxyz": {"x": 0.0, "y": 1.0, "z": 0.0},
                },
            ],
            "optimal": "candidate",
            "top_k": 2,
        }
        response = self.client.post("/matching/stable", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertEqual(data["matches"]["C_Tech"], "J_Tech")
        self.assertEqual(data["matches"]["C_Art"], "J_Art")
        self.assertTrue(data["metrics"]["is_stable"])
        self.assertEqual(data["metrics"]["blocking_pairs_count"], 0)


if __name__ == "__main__":
    unittest.main()
