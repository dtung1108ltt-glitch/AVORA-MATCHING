#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module triển khai Thuật toán Ghép cặp Ổn định (Gale-Shapley Many-to-One Deferred Acceptance)
dành riêng cho dự án AVORA-MATCHING.

Hệ thống mô hình hóa thị trường tuyển dụng hai phía (Two-Sided Matching Market):
- Ứng viên (Candidates) tìm kiếm công việc phù hợp (ưu tiên theo Điểm C - Candidate Preference Score).
- Nhà tuyển dụng/Vị trí việc làm (Employers/Jobs) tìm kiếm ứng viên đạt chuẩn (ưu tiên theo Điểm E - Employer Preference Score
  được tính toán từ Cosine Similarity trong Không gian Vector Oxyz).
- Hạn mức tuyển dụng (Quotas q_j): Số lượng chỉ tiêu tối đa cho mỗi vị trí việc làm (Many-to-One Assignment).
- Bộ lọc điều kiện tiên quyết (Hard Constraints / Domain Filter D): Triệt tiêu điểm các cặp vi phạm về 0.
- Cơ chế giải quyết hòa điểm tất định (Deterministic Tie-Breaking).
- Hậu xử lý Top-K Recommendations (Normalized Surplus Score).
- Đánh giá định lượng tính ổn định (100% Stability - 0% Blocking Pairs) và các chỉ số KPIs.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import math
import time
from collections import deque
from typing import Any

import numba
import numpy as np

from app.schemas import (
    CandidateProfileInput,
    CandidateRecommendationResponse,
    JobProfileInput,
    MatchedPairResponse,
    MatrixMatchingRequest,
    StableMatchingMetricsResponse,
    StableMatchingRequest,
    StableMatchingResponse,
    Vector3D,
)
from src.core.profile_models import CandidateProfile, ExecutionAudit, JobProfile
from src.core.scoring import (
    EmployerWeightsConfig,
    calculate_composite_c_score,
    calculate_composite_e_score,
    check_hard_constraints,
    explain_match,
)



# ===========================================================================
# GIAI ĐOẠN 1: TÍNH TOÁN ĐẠI SỐ VECTOR Oxyz & TIỀN XỬ LÝ MA TRẬN ĐIỂM
# ===========================================================================

def calculate_vector_norm_3d(v: tuple[float, float, float]) -> float:
    """
    Tính độ dài Euclidean (Norm) của một vector trong không gian Oxyz: |v| = sqrt(x^2 + y^2 + z^2).

    Mục đích:
        Xác định độ lớn hình học của vector năng lực hoặc vector yêu cầu công việc,
        phục vụ phép chuẩn hóa khi tính Cosine Similarity.

    Tham số đầu vào:
        v (tuple[float, float, float]): Bộ ba tọa độ số thực (x, y, z) biểu diễn vector trong Oxyz.

    Kết quả trả về:
        float: Độ dài Euclidean của vector v. Trả về 0.0 nếu vector có tọa độ (0.0, 0.0, 0.0).
    """
    # Trích xuất 3 thành phần tọa độ x, y, z từ tuple
    x, y, z = v
    # Áp dụng công thức căn bậc hai tổng bình phương các tọa độ trong không gian Oxyz
    norm_val = math.sqrt(x * x + y * y + z * z)
    # Trả về kết quả độ dài
    return norm_val


def calculate_dot_product_3d(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> float:
    """
    Tính tích vô hướng (Dot Product) giữa hai vector u và v trong không gian Oxyz: u . v = x1*x2 + y1*y2 + z1*z2.

    Mục đích:
        Đo lường mức độ tương quan trực tiếp giữa vector năng lực ứng viên và vector yêu cầu tuyển dụng.

    Tham số đầu vào:
        u (tuple[float, float, float]): Vector năng lực ứng viên (x1, y1, z1).
        v (tuple[float, float, float]): Vector yêu cầu tuyển dụng (x2, y2, z2).

    Kết quả trả về:
        float: Giá trị tích vô hướng giữa hai vector.
    """
    # Nhân từng thành phần tọa độ tương ứng trên 3 trục Ox, Oy, Oz và cộng tổng
    dot = u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
    # Trả về kết quả tích vô hướng
    return dot


def calculate_cosine_similarity_3d(
    u: tuple[float, float, float],
    v: tuple[float, float, float],
) -> float:
    """
    Tính Cosine Similarity giữa vector năng lực ứng viên và vector yêu cầu tuyển dụng trong không gian Oxyz:
    E = cos(u, v) = (u . v) / (|u| * |v|).

    Mục đích:
        Đo lường góc giữa hai vector trong không gian Oxyz để xác định Điểm E (Employer Preference Score).
        Góc càng nhỏ thì cosin càng tiệm cận 1.0, biểu thị mức độ phù hợp kỹ năng càng cao.

    Tham số đầu vào:
        u (tuple[float, float, float]): Tọa độ vector năng lực ứng viên (x1, y1, z1).
        v (tuple[float, float, float]): Tọa độ vector yêu cầu công việc (x2, y2, z2).

    Kết quả trả về:
        float: Điểm tương đồng cosin trong đoạn [0.0, 1.0]. Nếu góc tù hoặc vector suy biến, trả về 0.0.
    """
    # Bước 1: Tính độ dài vector năng lực ứng viên u
    norm_u = calculate_vector_norm_3d(u)
    # Bước 2: Tính độ dài vector yêu cầu tuyển dụng v
    norm_v = calculate_vector_norm_3d(v)

    # Bước 3: Kiểm tra nếu một trong hai vector có độ dài bằng 0 (vector suy biến)
    if norm_u == 0.0 or norm_v == 0.0:
        # Triệt tiêu điểm về 0.0 vì không đủ thông tin vector để so khớp
        return 0.0

    # Bước 4: Tính tích vô hướng u . v
    dot_val = calculate_dot_product_3d(u, v)

    # Bước 5: Áp dụng công thức Cosine Similarity
    cosine = dot_val / (norm_u * norm_v)

    # Bước 6: Giới hạn trong khoảng chuẩn [0.0, 1.0] (các góc âm hoặc không phù hợp coi như điểm sàn 0.0)
    score = max(0.0, min(1.0, cosine))
    return score


def deterministic_tie_breaker(seed: int, *identifiers: object) -> int:
    """
    Tạo số nguyên băm tất định (Deterministic Hash) 64-bit dùng cho cơ chế phá hòa (Tie-Breaking).

    Mục đích:
        Giải quyết hiện tượng hòa điểm (Tie-Breaking) khi nhiều ứng viên có cùng Điểm E hoặc
        nhiều công việc có cùng Điểm C. Bằng cách dùng hàm băm cryptographic cố định seed,
        thứ tự phá hòa luôn nhất quán và có thể tái lập 100% giữa các lần thực thi.

    Tham số đầu vào:
        seed (int): Hạt giống số học cấu hình của hệ thống.
        *identifiers (object): Các đối tượng định danh cần băm (ví dụ: ('cand_proposing', candidate_id, job_id)).

    Kết quả trả về:
        int: Số nguyên không âm 64-bit dùng làm khóa phụ ổn định khi sắp xếp danh sách ưu tiên.
    """
    # Ghép hạt giống seed và các định danh thành chuỗi duy nhất ngăn cách bởi ký tự phân tách '|'
    payload = "|".join([str(seed), *(str(item) for item in identifiers)])
    # Sinh mã băm 8 byte (64 bit) bằng thuật toán Blake2b
    digest_bytes = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    # Chuyển đổi chuỗi byte sang số nguyên lớn không dấu theo thứ tự big-endian
    hash_int = int.from_bytes(digest_bytes, "big")
    return hash_int


def apply_hard_constraints_filter(
    e_scores: dict[str, dict[str, float]],
    c_scores: dict[str, dict[str, float]],
    min_e_score: float = 0.0,
    min_c_score: float = 0.0,
    require_positive: bool = True,
    feasible_pairs: set[tuple[str, str]] | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """
    Áp dụng bộ lọc điều kiện bắt buộc (Hard Constraints / Domain Filter D).

    Mục đích:
        Thiết lập Tập xác định D chứa các điều kiện tiên quyết (khoảng cách địa lý, lương sàn, nhóm khuyết tật).
        Nếu một cặp (c, j) không thuộc D hoặc có điểm thấp hơn ngưỡng sàn, điểm sẽ bị triệt tiêu bằng 0
        thông qua Hàm số gián đoạn (Step/Indicator Function), loại bỏ cặp này khỏi bài toán phân bổ ngay từ đầu.

    Tham số đầu vào:
        e_scores (dict[str, dict[str, float]]): Ma trận điểm E thô theo cấu trúc {candidate_id: {job_id: score}}.
        c_scores (dict[str, dict[str, float]]): Ma trận điểm C thô theo cấu trúc {candidate_id: {job_id: score}}.
        min_e_score (float, mặc định 0.0): Ngưỡng điểm E tối thiểu bắt buộc.
        min_c_score (float, mặc định 0.0): Ngưỡng điểm C tối thiểu bắt buộc.
        require_positive (bool, mặc định True): Yêu cầu cả hai điểm E và C phải lớn hơn 0.
        feasible_pairs (set[tuple[str, str]] | None, mặc định None): Tập hợp các cặp (candidate, job) hợp lệ từ bộ lọc miền.

    Kết quả trả về:
        tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
            Cặp ma trận đã lọc (filtered_e_scores, filtered_c_scores) chỉ chứa các cặp hoàn toàn hợp lệ.
    """
    # Khởi tạo ma trận điểm E và C đã được làm sạch
    filtered_e: dict[str, dict[str, float]] = {}
    filtered_c: dict[str, dict[str, float]] = {}

    # Lấy toàn bộ danh sách ID ứng viên từ cả hai ma trận điểm
    all_candidates = set(e_scores.keys()) | set(c_scores.keys())

    for c_id in sorted(all_candidates):
        # Lấy dòng điểm tương ứng của ứng viên c_id
        e_row = e_scores.get(c_id, {})
        c_row = c_scores.get(c_id, {})

        valid_e_row: dict[str, float] = {}
        valid_c_row: dict[str, float] = {}

        # Duyệt qua các công việc xuất hiện ở cả hai phía đánh giá
        common_jobs = set(e_row.keys()) & set(c_row.keys())
        for j_id in sorted(common_jobs):
            e_val = float(e_row[j_id])
            c_val = float(c_row[j_id])

            # Kiểm tra ràng buộc tập khả thi D nếu có thiết lập
            if feasible_pairs is not None and (c_id, j_id) not in feasible_pairs:
                continue

            # Kiểm tra ràng buộc điểm dương
            if require_positive and (e_val <= 0.0 or c_val <= 0.0):
                continue

            # Kiểm tra ngưỡng sàn điểm tối thiểu
            if e_val < min_e_score or c_val < min_c_score:
                continue

            # Thỏa mãn mọi điều kiện bắt buộc -> đưa vào ma trận hợp lệ
            valid_e_row[j_id] = e_val
            valid_c_row[j_id] = c_val

        filtered_e[c_id] = valid_e_row
        filtered_c[c_id] = valid_c_row

    return filtered_e, filtered_c


# ===========================================================================
# GIAI ĐOẠN 2: KHỞI TẠO DANH SÁCH ƯU TIÊN VÀ BẢNG TRA CỨU RANK TABLE O(1)
# ===========================================================================

def build_preference_lists(
    candidate_scores: dict[str, dict[str, float]],
    employer_scores: dict[str, dict[str, float]],
    max_c_prefs: int | None = None,
    max_e_prefs: int | None = None,
    seed: int = 20240517,
    skill_scores: dict[str, dict[str, float]] | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Chuyển đổi ma trận điểm C và E thành Danh sách Ưu tiên tuyến tính (Preference Lists)
    kèm cơ chế cắt ngắn (Truncated Preference Lists) và phá hòa tất định theo Mục 6.4.

    Mục đích:
        Sắp xếp các công việc theo thứ tự điểm C giảm dần cho từng ứng viên,
        và sắp xếp các ứng viên theo thứ tự điểm E giảm dần cho từng nhà tuyển dụng/công việc.
        Áp dụng Deterministic Tie-Breaking (Mục 6.4: điểm kỹ năng cao hơn, rồi mã ứng viên)
        để giải quyết các trường hợp bằng điểm E.

    Tham số đầu vào:
        candidate_scores (dict[str, dict[str, float]]): Điểm C theo cấu trúc {candidate_id: {job_id: score}}.
        employer_scores (dict[str, dict[str, float]]): Điểm E theo cấu trúc {job_id: {candidate_id: score}}.
        max_c_prefs (int | None, mặc định None): Giới hạn L nguyện vọng tối đa của ứng viên (Truncation).
        max_e_prefs (int | None, mặc định None): Giới hạn M ứng viên tối đa của công việc (Truncation).
        seed (int, mặc định 20240517): Hạt giống phá hòa tất định.
        skill_scores (dict[str, dict[str, float]] | None): Điểm kỹ năng để phá hòa theo Mục 6.4.

    Kết quả trả về:
        tuple[dict[str, list[str]], dict[str, list[str]]]:
            - candidates_pref: {candidate_id: [danh sách job_id xếp theo thứ tự ưu tiên giảm dần]}
            - jobs_pref: {job_id: [danh sách candidate_id xếp theo thứ tự ưu tiên giảm dần]}
    """
    candidates_pref: dict[str, list[str]] = {}

    # Bước 1: Xây dựng Preference Lists cho phía Ứng viên (theo điểm C)
    for c_id, jobs_dict in candidate_scores.items():
        # Lọc các công việc có điểm hợp lệ (> 0)
        valid_items = [(j_id, score) for j_id, score in jobs_dict.items() if score > 0.0]

        # Sắp xếp theo: 1) Điểm C giảm dần (-score), 2) Hàm băm phá hòa tất định, 3) ID công việc
        valid_items.sort(
            key=lambda item: (
                -item[1],
                deterministic_tie_breaker(seed, "cand_pref", c_id, item[0]),
                str(item[0]),
            )
        )

        # Lấy danh sách ID các công việc sau khi sắp xếp
        sorted_job_ids = [j_id for j_id, _ in valid_items]

        # Áp dụng Truncated Preference Lists nếu có cấu hình max_c_prefs
        if max_c_prefs is not None and max_c_prefs > 0:
            sorted_job_ids = sorted_job_ids[:max_c_prefs]

        candidates_pref[c_id] = sorted_job_ids

    # Bước 2: Xây dựng Preference Lists cho phía Công việc/Nhà tuyển dụng (theo điểm E)
    jobs_pref: dict[str, list[str]] = {}
    for j_id, cands_dict in employer_scores.items():
        # Lọc các ứng viên có điểm E hợp lệ (> 0)
        valid_items = [(c_id, score) for c_id, score in cands_dict.items() if score > 0.0]

        # Sắp xếp theo Mục 6.4:
        # Nếu có skill_scores: ưu tiên điểm E giảm dần, rồi điểm kỹ năng cao hơn, rồi mã ứng viên
        if skill_scores is not None:
            valid_items.sort(
                key=lambda item: (
                    -item[1],
                    -float(skill_scores.get(item[0], {}).get(j_id, 0.0)),
                    str(item[0]),
                )
            )
        else:
            # Fallback hàm băm tất định
            valid_items.sort(
                key=lambda item: (
                    -item[1],
                    deterministic_tie_breaker(seed, "job_pref", j_id, item[0]),
                    str(item[0]),
                )
            )

        # Lấy danh sách ID các ứng viên sau khi sắp xếp
        sorted_cand_ids = [c_id for c_id, _ in valid_items]

        # Áp dụng Truncated Preference Lists nếu có cấu hình max_e_prefs
        if max_e_prefs is not None and max_e_prefs > 0:
            sorted_cand_ids = sorted_cand_ids[:max_e_prefs]

        jobs_pref[j_id] = sorted_cand_ids


    return candidates_pref, jobs_pref


def build_rank_lookup_table(preferences: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """
    Xây dựng Bảng tra cứu thứ hạng nghịch đảo (Rank Lookup Table) đạt độ phức tạp O(1).

    Mục đích:
        Thay vì duyệt tuyến tính O(N) để tìm xem đối tác mới có được ưu tiên hơn đối tác cũ hay không,
        bảng này ánh xạ trực tiếp rank_table[entity_id][partner_id] = thứ_hạng (0 là tốt nhất),
        giúp việc so sánh và loại bỏ ứng viên trong Gale-Shapley diễn ra trong thời gian O(1).

    Tham số đầu vào:
        preferences (dict[str, list[str]]): Danh sách ưu tiên tuyến tính của các đối tượng.

    Kết quả trả về:
        dict[str, dict[str, int]]: Bảng tra cứu thứ bậc dạng {entity_id: {partner_id: rank_index}}.
    """
    # Khởi tạo bảng tra cứu rỗng
    lookup_table: dict[str, dict[str, int]] = {}

    for entity_id, partner_list in preferences.items():
        # Ánh xạ từng đối tác với chỉ số index của họ trong danh sách ưu tiên (0 = số 1 mong muốn)
        lookup_table[entity_id] = {partner_id: rank for rank, partner_id in enumerate(partner_list)}

    return lookup_table


# ===========================================================================
# GIAI ĐOẠN 3: LÕI THUẬT TOÁN GALE-SHAPLEY DEFERRED ACCEPTANCE (MANY-TO-ONE)
# ===========================================================================


@numba.njit(cache=True)
def _app_gs_candidate_proposing_numba(
    cand_prefs_arr: np.ndarray,
    cand_pref_lens: np.ndarray,
    job_rank_arr: np.ndarray,
    quotas_arr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Vòng lặp while Gale-Shapley Candidate-Proposing biên dịch mã máy Numba trên mảng 2 chiều int32[:, :]."""
    n_cands = cand_prefs_arr.shape[0]
    n_jobs = job_rank_arr.shape[0]

    max_quota = 1
    for j in range(n_jobs):
        if quotas_arr[j] > max_quota:
            max_quota = quotas_arr[j]

    held_cands = np.full((n_jobs, max_quota), -1, dtype=np.int32)
    held_ranks = np.full((n_jobs, max_quota), -1, dtype=np.int32)
    held_counts = np.zeros(n_jobs, dtype=np.int32)

    matches_arr = np.full(n_cands, -1, dtype=np.int32)
    next_proposal = np.zeros(n_cands, dtype=np.int32)

    max_q = max(16, n_cands * (cand_prefs_arr.shape[1] + 2) + 16)
    queue = np.empty(max_q, dtype=np.int32)
    head = 0
    tail = 0

    for c in range(n_cands):
        if cand_pref_lens[c] > 0:
            queue[tail] = c
            tail += 1

    total_proposals = 0

    while head < tail:
        c = queue[head]
        head += 1

        p_index = next_proposal[c]
        if p_index >= cand_pref_lens[c]:
            continue

        j = cand_prefs_arr[c, p_index]
        next_proposal[c] = p_index + 1
        total_proposals += 1

        if j < 0:
            queue[tail] = c
            tail += 1
            continue

        quota = quotas_arr[j]
        if quota <= 0:
            queue[tail] = c
            tail += 1
            continue

        c_rank_in_j = job_rank_arr[j, c]
        if c_rank_in_j < 0:
            queue[tail] = c
            tail += 1
            continue

        cnt = held_counts[j]
        if cnt < quota:
            held_cands[j, cnt] = c
            held_ranks[j, cnt] = c_rank_in_j
            held_counts[j] = cnt + 1
            matches_arr[c] = j
            continue

        worst_slot = 0
        worst_rank = held_ranks[j, 0]
        for s in range(1, cnt):
            r = held_ranks[j, s]
            if r > worst_rank:
                worst_rank = r
                worst_slot = s

        if c_rank_in_j < worst_rank:
            worst_cand = held_cands[j, worst_slot]
            held_cands[j, worst_slot] = c
            held_ranks[j, worst_slot] = c_rank_in_j
            matches_arr[worst_cand] = -1
            matches_arr[c] = j
            queue[tail] = worst_cand
            tail += 1
        else:
            queue[tail] = c
            tail += 1

    return matches_arr, total_proposals


@numba.njit(cache=True)
def _app_gs_employer_proposing_numba(
    job_prefs_arr: np.ndarray,
    job_pref_lens: np.ndarray,
    cand_rank_arr: np.ndarray,
    quotas_arr: np.ndarray,
    initial_free_jobs: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Vòng lặp while Gale-Shapley Employer-Proposing biên dịch mã máy Numba trên mảng 2 chiều int32[:, :]."""
    n_cands = cand_rank_arr.shape[0]
    n_jobs = job_prefs_arr.shape[0]

    assigned_matrix = np.zeros((n_jobs, max(1, n_cands)), dtype=np.int32)
    assigned_counts = np.zeros(n_jobs, dtype=np.int32)
    held_by_candidate = np.full(n_cands, -1, dtype=np.int32)
    next_proposal_idx = np.zeros(n_jobs, dtype=np.int32)

    max_q = max(32, n_jobs * (job_prefs_arr.shape[1] * 2 + 4) + 32)
    queue = np.empty(max_q, dtype=np.int32)
    head = 0
    tail = 0

    for idx_j in range(initial_free_jobs.shape[0]):
        queue[tail] = initial_free_jobs[idx_j]
        tail += 1

    total_proposals = 0

    while head < tail:
        j = queue[head]
        head += 1

        quota = quotas_arr[j]
        if assigned_counts[j] >= quota or quota <= 0:
            continue

        p_idx = next_proposal_idx[j]
        if p_idx >= job_pref_lens[j]:
            continue

        candidate = job_prefs_arr[j, p_idx]
        next_proposal_idx[j] = p_idx + 1
        total_proposals += 1

        if candidate < 0:
            if assigned_counts[j] < quota and next_proposal_idx[j] < job_pref_lens[j]:
                queue[tail] = j
                tail += 1
            continue

        rank_new = cand_rank_arr[candidate, j]
        if rank_new < 0:
            if assigned_counts[j] < quota and next_proposal_idx[j] < job_pref_lens[j]:
                queue[tail] = j
                tail += 1
            continue

        current_job = held_by_candidate[candidate]
        if current_job < 0:
            assigned_matrix[j, candidate] = 1
            assigned_counts[j] += 1
            held_by_candidate[candidate] = j
        else:
            rank_curr = cand_rank_arr[candidate, current_job]
            if rank_curr < 0 or rank_new < rank_curr:
                if assigned_matrix[current_job, candidate] == 1:
                    assigned_matrix[current_job, candidate] = 0
                    assigned_counts[current_job] -= 1
                queue[tail] = current_job
                tail += 1

                assigned_matrix[j, candidate] = 1
                assigned_counts[j] += 1
                held_by_candidate[candidate] = j

        if assigned_counts[j] < quota and next_proposal_idx[j] < job_pref_lens[j]:
            queue[tail] = j
            tail += 1

    return held_by_candidate, total_proposals


def candidate_proposing_deferred_acceptance(
    candidates_pref: dict[str, list[str]],
    jobs_pref: dict[str, list[str]],
    job_quotas: dict[str, int],
) -> tuple[dict[str, str], dict[str, list[str]], int]:
    """
    Thuật toán Gale-Shapley biến thể Many-to-One: Candidate-Proposing Deferred Acceptance (Hospital-Residents Problem).

    Mục đích:
        Giải bài toán phân bổ ổn định trên thị trường hai phía với phía Ứng viên là người chủ động gửi đề nghị.
        Đảm bảo 100% tính ổn định (không có blocking pair), đạt tính tối ưu cao nhất cho Ứng viên (C-optimal),
        và có tính chất chống khai gian (Strategy-Proof) đối với Ứng viên.
        Vòng lặp while được biên dịch sang mã máy bằng @numba.njit trên mảng 2 chiều int32[:, :].
    """
    job_rank_table = build_rank_lookup_table(jobs_pref)

    all_cands = sorted(
        set(candidates_pref.keys())
        | {c for j_list in jobs_pref.values() for c in j_list}
    )
    all_jobs = sorted(
        set(jobs_pref.keys())
        | set(job_quotas.keys())
        | {j for c_list in candidates_pref.values() for j in c_list}
    )

    if not all_cands or not all_jobs:
        return {}, {j: [] for j in jobs_pref}, 0

    cand_to_idx = {c: i for i, c in enumerate(all_cands)}
    job_to_idx = {j: i for i, j in enumerate(all_jobs)}
    n_cands = len(all_cands)
    n_jobs = len(all_jobs)

    max_c_len = max((len(candidates_pref.get(c, [])) for c in all_cands), default=0)
    cand_prefs_arr = np.full((n_cands, max(1, max_c_len)), -1, dtype=np.int32)
    cand_pref_lens = np.zeros(n_cands, dtype=np.int32)
    job_rank_arr = np.full((n_jobs, n_cands), -1, dtype=np.int32)
    quotas_arr = np.zeros(n_jobs, dtype=np.int32)

    for c_idx, c_id in enumerate(all_cands):
        c_list = candidates_pref.get(c_id, [])
        if c_list:
            cand_prefs_arr[c_idx, : len(c_list)] = [job_to_idx[j] for j in c_list]
            cand_pref_lens[c_idx] = len(c_list)

    for j_idx, j_id in enumerate(all_jobs):
        quotas_arr[j_idx] = max(0, int(job_quotas.get(j_id, 0)))
        for rank_pos, c_id in enumerate(jobs_pref.get(j_id, [])):
            c_idx = cand_to_idx.get(c_id)
            if c_idx is not None:
                job_rank_arr[j_idx, c_idx] = rank_pos

    matches_arr, total_proposals = _app_gs_candidate_proposing_numba(
        cand_prefs_arr,
        cand_pref_lens,
        job_rank_arr,
        quotas_arr,
    )

    matches: dict[str, str] = {}
    for c_idx, j_idx in enumerate(matches_arr):
        if j_idx >= 0:
            matches[all_cands[c_idx]] = all_jobs[int(j_idx)]

    job_assignments: dict[str, list[str]] = {j: [] for j in jobs_pref}
    for c_id, j_id in matches.items():
        if j_id in job_assignments:
            job_assignments[j_id].append(c_id)

    for j_id, c_list in job_assignments.items():
        rank_dict = job_rank_table.get(j_id, {})
        c_list.sort(key=lambda cand: rank_dict.get(cand, 999999))

    return matches, job_assignments, int(total_proposals)


def employer_proposing_deferred_acceptance(
    candidates_pref: dict[str, list[str]],
    jobs_pref: dict[str, list[str]],
    job_quotas: dict[str, int],
) -> tuple[dict[str, str], dict[str, list[str]], int]:
    """
    Thuật toán Gale-Shapley biến thể Many-to-One: Employer-Proposing Deferred Acceptance.

    Mục đích:
        Thực hiện ghép cặp ổn định với phía Nhà tuyển dụng/Công việc chủ động gửi đề nghị tới ứng viên.
        Kết quả đạt tối ưu cho phía nhà tuyển dụng (J-optimal / E-optimal), phù hợp mô hình B2B.
        Vòng lặp while được biên dịch sang mã máy bằng @numba.njit trên mảng 2 chiều int32[:, :].
    """
    all_cands = sorted(
        set(candidates_pref.keys())
        | {c for j_list in jobs_pref.values() for c in j_list}
    )
    all_jobs = sorted(
        set(jobs_pref.keys())
        | set(job_quotas.keys())
        | {j for c_list in candidates_pref.values() for j in c_list}
    )

    if not all_cands or not all_jobs:
        return {}, {j: [] for j in jobs_pref}, 0

    cand_to_idx = {c: i for i, c in enumerate(all_cands)}
    job_to_idx = {j: i for i, j in enumerate(all_jobs)}
    n_cands = len(all_cands)
    n_jobs = len(all_jobs)

    max_j_len = max((len(jobs_pref.get(j, [])) for j in all_jobs), default=0)
    job_prefs_arr = np.full((n_jobs, max(1, max_j_len)), -1, dtype=np.int32)
    job_pref_lens = np.zeros(n_jobs, dtype=np.int32)
    cand_rank_arr = np.full((n_cands, n_jobs), -1, dtype=np.int32)
    quotas_arr = np.zeros(n_jobs, dtype=np.int32)

    for c_idx, c_id in enumerate(all_cands):
        for rank_pos, j_id in enumerate(candidates_pref.get(c_id, [])):
            j_idx = job_to_idx.get(j_id)
            if j_idx is not None:
                cand_rank_arr[c_idx, j_idx] = rank_pos

    for j_idx, j_id in enumerate(all_jobs):
        quotas_arr[j_idx] = int(job_quotas.get(j_id, 0))
        j_list = jobs_pref.get(j_id, [])
        if j_list:
            job_prefs_arr[j_idx, : len(j_list)] = [cand_to_idx[c] for c in j_list]
            job_pref_lens[j_idx] = len(j_list)

    initial_free_jobs = np.array(
        [
            job_to_idx[j]
            for j in sorted(jobs_pref.keys())
            if jobs_pref.get(j) and job_quotas.get(j, 0) > 0
        ],
        dtype=np.int32,
    )

    held_by_cand_arr, total_proposals = _app_gs_employer_proposing_numba(
        job_prefs_arr,
        job_pref_lens,
        cand_rank_arr,
        quotas_arr,
        initial_free_jobs,
    )

    matches: dict[str, str] = {}
    assigned_by_job: dict[str, list[str]] = {j: [] for j in jobs_pref}
    for c_idx, j_idx in enumerate(held_by_cand_arr):
        if j_idx >= 0:
            c_id = all_cands[c_idx]
            j_id = all_jobs[int(j_idx)]
            matches[c_id] = j_id
            if j_id in assigned_by_job:
                assigned_by_job[j_id].append(c_id)

    job_assignments = {j: sorted(cands) for j, cands in assigned_by_job.items()}

    return matches, job_assignments, int(total_proposals)


# ===========================================================================
# KIỂM TRA TÍNH ỔN ĐỊNH VÀ PHÁT HIỆN CẶP BẤT ỔN (BLOCKING PAIRS)
# ===========================================================================

def find_blocking_pairs(
    matches: dict[str, str],
    job_assignments: dict[str, list[str]],
    candidates_pref: dict[str, list[str]],
    jobs_pref: dict[str, list[str]],
    job_quotas: dict[str, int],
) -> list[dict[str, str]]:
    """
    Kiểm tra và phát hiện tất cả các Cặp gây bất ổn (Blocking Pairs) trong kết quả ghép cặp.

    Mục đích:
        Chứng minh bằng toán học tính ổn định 100% của thuật toán Gale-Shapley (target 0 blocking pairs).
        Cặp (c, j) được gọi là blocking pair nếu thỏa mãn đồng thời:
        1) Ứng viên c thích công việc j hơn công việc hiện tại của mình (hoặc c chưa có việc nhưng thích j).
        2) Công việc j thích ứng viên c hơn ít nhất một ứng viên đang nhận HOẶC công việc j còn chỗ trống (dưới quota).

    Tham số đầu vào:
        matches (dict[str, str]): Ánh xạ kết quả ghép cặp {candidate_id: job_id}.
        job_assignments (dict[str, list[str]]): Danh sách ứng viên trúng tuyển của từng công việc.
        candidates_pref (dict[str, list[str]]): Danh sách ưu tiên của ứng viên.
        jobs_pref (dict[str, list[str]]): Danh sách ưu tiên của công việc.
        job_quotas (dict[str, int]): Chỉ tiêu tuyển dụng của từng công việc.

    Kết quả trả về:
        list[dict[str, str]]: Danh sách các cặp bất ổn tìm thấy (mỗi phần tử là {'candidate_id': c, 'job_id': j}).
                              Kỳ vọng kết quả luôn là rỗng [] đối với Gale-Shapley.
    """
    blocking_pairs: list[dict[str, str]] = []
    cand_rank_table = build_rank_lookup_table(candidates_pref)
    job_rank_table = build_rank_lookup_table(jobs_pref)

    for c_id, c_prefs in candidates_pref.items():
        curr_job = matches.get(c_id)
        curr_job_rank = cand_rank_table[c_id].get(curr_job, 999999) if curr_job else 999999

        for j_id in c_prefs:
            # Ứng viên c chỉ quan tâm đến các công việc mà họ thích hơn công việc hiện tại
            j_rank = cand_rank_table[c_id][j_id]
            if j_rank >= curr_job_rank:
                # Đã duyệt đến hoặc vượt qua công việc hiện tại -> không còn động cơ phá vỡ
                break

            # Kiểm tra xem công việc j có chấp nhận ứng viên c không
            if j_id not in job_rank_table or c_id not in job_rank_table[j_id]:
                continue

            c_rank_in_j = job_rank_table[j_id][c_id]
            assigned_cands = job_assignments.get(j_id, [])
            quota = job_quotas.get(j_id, 0)

            # Điều kiện 2a: Công việc j còn chỗ trống (chưa tuyển đủ quota)
            if len(assigned_cands) < quota:
                blocking_pairs.append({"candidate_id": c_id, "job_id": j_id})
                continue

            # Điều kiện 2b: Công việc j thích ứng viên c hơn ít nhất một người đang được nhận
            has_worse_cand = any(
                job_rank_table[j_id].get(cand, 999999) > c_rank_in_j
                for cand in assigned_cands
            )
            if has_worse_cand:
                blocking_pairs.append({"candidate_id": c_id, "job_id": j_id})

    return blocking_pairs


# ===========================================================================
# GIAI ĐOẠN 4: HẬU XỬ LÝ TOP-K RECOMMENDATIONS (NORMALIZED SURPLUS SCORE)
# ===========================================================================

def generate_top_k_recommendations(
    matches: dict[str, str],
    candidates_pref: dict[str, list[str]],
    c_scores: dict[str, dict[str, float]],
    e_scores: dict[str, dict[str, float]],
    k: int = 3,
    weight_c: float = 0.5,
    weight_e: float = 0.5,
) -> dict[str, list[dict[str, object]]]:
    """
    Tạo danh sách Top-K vị trí công việc gợi ý dự phòng (Backup Recommendations) cho mỗi ứng viên.

    Mục đích:
        Khắc phục hạn chế gán cặp duy nhất (One-to-One/Many-to-One) của Gale-Shapley, cung cấp thêm
        các vị trí việc làm dự phòng chất lượng cao theo Điểm thặng dư chuẩn hóa (Normalized Surplus Score).
        Cặp ghép chính thức từ Gale-Shapley luôn đứng ở vị trí số 1 (position=1, is_primary=True).

    Tham số đầu vào:
        matches (dict[str, str]): Kết quả gán cặp chính thức từ Gale-Shapley.
        candidates_pref (dict[str, list[str]]): Danh sách ưu tiên hợp lệ của ứng viên.
        c_scores (dict[str, dict[str, float]]): Ma trận điểm C {c: {j: score}}.
        e_scores (dict[str, dict[str, float]]): Ma trận điểm E {c: {j: score}}.
        k (int, mặc định 3): Số lượng vị trí tối đa gợi ý cho mỗi ứng viên.
        weight_c (float, mặc định 0.5): Trọng số của điểm hài lòng ứng viên.
        weight_e (float, mặc định 0.5): Trọng số của điểm chất lượng từ nhà tuyển dụng.

    Kết quả trả về:
        dict[str, list[dict[str, object]]]:
            Ánh xạ {candidate_id: [danh sách dict recommendation gồm candidate_id, job_id, position, score, is_primary]}.
    """
    recommendations: dict[str, list[dict[str, object]]] = {}

    for c_id, pref_jobs in candidates_pref.items():
        primary_job = matches.get(c_id)
        recs_for_cand: list[dict[str, object]] = []

        # Tính min-max để chuẩn hóa điểm C và E của ứng viên này
        cand_c_row = c_scores.get(c_id, {})
        cand_e_row = e_scores.get(c_id, {})

        c_values = [cand_c_row.get(j, 0.0) for j in pref_jobs]
        e_values = [cand_e_row.get(j, 0.0) for j in pref_jobs]

        max_c = max(c_values) if c_values else 1.0
        min_c = min(c_values) if c_values else 0.0
        span_c = max_c - min_c if max_c > min_c else 1.0

        max_e = max(e_values) if e_values else 1.0
        min_e = min(e_values) if e_values else 0.0
        span_e = max_e - min_e if max_e > min_e else 1.0

        # Nếu ứng viên có công việc ghép chính thức, đưa công việc này lên đầu tiên
        assigned_jobs_set = set()
        if primary_job:
            primary_c = cand_c_row.get(primary_job, 1.0)
            primary_e = cand_e_row.get(primary_job, 1.0)
            norm_c = (primary_c - min_c) / span_c if span_c > 0 else 1.0
            norm_e = (primary_e - min_e) / span_e if span_e > 0 else 1.0
            surplus = round(weight_c * norm_c + weight_e * norm_e, 4)

            recs_for_cand.append({
                "candidate_id": c_id,
                "job_id": primary_job,
                "position": 1,
                "score": max(0.01, surplus),
                "is_primary": True,
            })
            assigned_jobs_set.add(primary_job)

        # Tính điểm thặng dư cho các công việc còn lại trong danh sách nguyện vọng
        other_jobs_scored: list[tuple[str, float]] = []
        for j_id in pref_jobs:
            if j_id in assigned_jobs_set:
                continue
            raw_c = cand_c_row.get(j_id, 0.0)
            raw_e = cand_e_row.get(j_id, 0.0)
            norm_c = (raw_c - min_c) / span_c if span_c > 0 else 0.5
            norm_e = (raw_e - min_e) / span_e if span_e > 0 else 0.5
            surplus = weight_c * norm_c + weight_e * norm_e
            other_jobs_scored.append((j_id, surplus))

        # Sắp xếp các công việc dự phòng theo điểm thặng dư giảm dần
        other_jobs_scored.sort(key=lambda x: x[1], reverse=True)

        # Bổ sung các công việc dự phòng cho đủ top-k
        pos = len(recs_for_cand) + 1
        for j_id, score in other_jobs_scored:
            if len(recs_for_cand) >= k:
                break
            recs_for_cand.append({
                "candidate_id": c_id,
                "job_id": j_id,
                "position": pos,
                "score": round(score, 4),
                "is_primary": False,
            })
            pos += 1

        recommendations[c_id] = recs_for_cand

    return recommendations


# ===========================================================================
# CHỈ SỐ ĐO LƯỜNG HIỆU QUẢ KPIS & METRICS EVALUATION
# ===========================================================================

def calculate_matching_metrics(
    matches: dict[str, str],
    job_assignments: dict[str, list[str]],
    candidates_pref: dict[str, list[str]],
    jobs_pref: dict[str, list[str]],
    job_quotas: dict[str, int],
    c_scores: dict[str, dict[str, float]],
    e_scores: dict[str, dict[str, float]],
    proposals_count: int,
    all_candidate_ids: list[str],
    all_job_ids: list[str],
) -> StableMatchingMetricsResponse:
    """
    Tính toán các chỉ số định lượng đo lường hiệu quả (KPIs) của hệ thống ghép cặp ổn định.

    Mục đích:
        Báo cáo độ ổn định (Blocking Pairs), mức độ hài lòng hai phía (Satisfaction),
        và tỷ lệ lấp đầy hạn mức tuyển dụng (Fill Rate).

    Tham số đầu vào:
        matches (dict[str, str]): Kết quả ánh xạ ghép cặp {candidate_id: job_id}.
        job_assignments (dict[str, list[str]]): Danh sách ứng viên theo vị trí {job_id: [candidates]}.
        candidates_pref (dict[str, list[str]]): Danh sách ưu tiên của ứng viên.
        jobs_pref (dict[str, list[str]]): Danh sách ưu tiên của công việc.
        job_quotas (dict[str, int]): Hạn mức tuyển dụng cho từng công việc.
        c_scores (dict[str, dict[str, float]]): Điểm C.
        e_scores (dict[str, dict[str, float]]): Điểm E.
        proposals_count (int): Số lượt đề xuất.
        all_candidate_ids (list[str]): Toàn bộ ID ứng viên tham gia thị trường.
        all_job_ids (list[str]): Toàn bộ ID công việc trong thị trường.

    Kết quả trả về:
        StableMatchingMetricsResponse: Đối tượng chứa đầy đủ các chỉ số KPIs.
    """
    total_candidates = len(all_candidate_ids)
    total_jobs = len(all_job_ids)
    total_quota = sum(max(0, job_quotas.get(j, 0)) for j in all_job_ids)
    matched_pairs_count = len(matches)

    unmatched_candidates_count = total_candidates - matched_pairs_count
    unmatched_jobs_count = sum(
        1 for j in all_job_ids if len(job_assignments.get(j, [])) < job_quotas.get(j, 0)
    )

    # Kiểm tra Blocking Pairs
    blocking_pairs = find_blocking_pairs(
        matches=matches,
        job_assignments=job_assignments,
        candidates_pref=candidates_pref,
        jobs_pref=jobs_pref,
        job_quotas=job_quotas,
    )
    bp_count = len(blocking_pairs)
    total_possible_pairs = max(1, total_candidates * total_jobs)
    bp_rate = round(bp_count / total_possible_pairs, 4)
    is_stable = (bp_count == 0)

    # Tính Mức độ hài lòng trung bình phía Ứng viên (Average Candidate Satisfaction)
    cand_satisfaction_list: list[float] = []
    for c_id, j_id in matches.items():
        score = c_scores.get(c_id, {}).get(j_id, 0.0)
        cand_satisfaction_list.append(score)
    avg_c_sat = (
        round(sum(cand_satisfaction_list) / len(cand_satisfaction_list), 4)
        if cand_satisfaction_list
        else 0.0
    )

    # Tính Mức độ hài lòng trung bình phía Nhà tuyển dụng (Average Employer Satisfaction)
    emp_satisfaction_list: list[float] = []
    for c_id, j_id in matches.items():
        score = e_scores.get(c_id, {}).get(j_id, 0.0)
        emp_satisfaction_list.append(score)
    avg_e_sat = (
        round(sum(emp_satisfaction_list) / len(emp_satisfaction_list), 4)
        if emp_satisfaction_list
        else 0.0
    )

    # Tính Tỷ lệ lấp đầy vị trí tuyển dụng (Fill Rate)
    fill_rate = round(matched_pairs_count / total_quota, 4) if total_quota > 0 else 0.0

    # Tính thứ bậc trung bình hai phía và tỷ lệ chưa ghép theo Mục 6.5
    cand_rank_list: list[int] = []
    job_rank_list: list[int] = []
    for c_id, j_id in matches.items():
        c_p = candidates_pref.get(c_id, [])
        if j_id in c_p:
            cand_rank_list.append(c_p.index(j_id) + 1)
        j_p = jobs_pref.get(j_id, [])
        if c_id in j_p:
            job_rank_list.append(j_p.index(c_id) + 1)

    avg_c_rank = round(sum(cand_rank_list) / len(cand_rank_list), 2) if cand_rank_list else 0.0
    avg_j_rank = round(sum(job_rank_list) / len(job_rank_list), 2) if job_rank_list else 0.0
    unmatched_cand_rate = round(unmatched_candidates_count / total_candidates, 4) if total_candidates > 0 else 0.0
    congestion_rate = 0.0  # Gale-Shapley đảm bảo độc quyền, 0% trùng lặp

    return StableMatchingMetricsResponse(
        total_candidates=total_candidates,
        total_jobs=total_jobs,
        total_quota=total_quota,
        matched_pairs_count=matched_pairs_count,
        unmatched_candidates_count=unmatched_candidates_count,
        unmatched_jobs_count=unmatched_jobs_count,
        blocking_pairs_count=bp_count,
        blocking_pair_rate=bp_rate,
        is_stable=is_stable,
        candidate_satisfaction_avg=avg_c_sat,
        employer_satisfaction_avg=avg_e_sat,
        fill_rate=fill_rate,
        proposals_count=proposals_count,
        average_candidate_rank=avg_c_rank,
        average_job_rank=avg_j_rank,
        unmatched_candidate_rate=unmatched_cand_rate,
        congestion_rate=congestion_rate,
    )



# ===========================================================================
# CÁC HÀM ĐIỀU PHỐI CHÍNH TOÀN TRÌNH (PIPELINE EXECUTORS)
# ===========================================================================

def run_stable_matching(request: StableMatchingRequest) -> StableMatchingResponse:
    """
    Thực thi toàn bộ quy trình 4 giai đoạn của Thuật toán Gale-Shapley Stable Matching
    dựa trên payload `StableMatchingRequest`.

    Mục đích:
        Điều phối xuyên suốt từ tính toán vector Oxyz, áp dụng bộ lọc biên Hard Constraints,
        khởi tạo danh sách ưu tiên, giải bài toán Deferred Acceptance Many-to-One,
        sinh gợi ý Top-K dự phòng và đánh giá chỉ số hiệu quả KPIs.

    Tham số đầu vào:
        request (StableMatchingRequest): Dữ liệu đầu vào chứa danh sách candidates, jobs,
                                        chế độ optimal, các ngưỡng điểm và tham số thuật toán.

    Kết quả trả về:
        StableMatchingResponse: Đối tượng response hoàn chỉnh sẵn sàng trả về cho client.
    """
    start_time = time.perf_counter()

    candidate_ids = [c.id for c in request.candidates]
    job_ids = [j.id for j in request.jobs]

    # Kiểm tra tính hợp lệ của Direct Track và tính quota thực tế
    valid_job_ids = {j.id for j in request.jobs if j.id} | {getattr(j, "job_id", None) for j in request.jobs if getattr(j, "job_id", None)}
    valid_job_ids.discard(None)

    direct_hired_counts: dict[str, int] = {}
    for c in request.candidates:
        c_id = c.id or getattr(c, "candidate_id", "unknown")
        if c.direct_track:
            hired_job = getattr(c, "hired_job_id", None)
            if not hired_job or not str(hired_job).strip():
                raise ValueError(
                    f"Ứng viên '{c_id}' có direct_track=True nhưng thiếu hired_job_id "
                    "(bắt buộc phải chỉ định vị trí đã trúng tuyển)."
                )
            if hired_job not in valid_job_ids:
                raise ValueError(
                    f"Ứng viên '{c_id}' Direct Track trỏ tới vị trí không tồn tại: '{hired_job}'. "
                    f"Danh sách vị trí hợp lệ: {sorted(valid_job_ids)}"
                )
            direct_hired_counts[hired_job] = direct_hired_counts.get(hired_job, 0) + 1

    # Tính quota thực tế còn lại cho từng vị trí
    quotas = {}
    for j in request.jobs:
        j_id = j.id or getattr(j, "job_id", "")
        direct_count = direct_hired_counts.get(j_id, 0)
        filled = getattr(j, "filled_slots", 0)
        total_pre_filled = direct_count + filled
        if total_pre_filled > int(j.quota):
            raise ValueError(
                f"Vị trí '{j_id}' có số lượng tuyển trực tiếp ({direct_count}) "
                f"cộng số chỗ đã lấp ngoài hệ thống ({filled}) = {total_pre_filled} "
                f"vượt quá hạn mức tuyển dụng (quota={j.quota})."
            )
        quotas[j_id] = int(j.quota) - total_pre_filled

    # -----------------------------------------------------------------------
    # GIAI ĐOẠN 1: Tính toán ma trận Điểm E và Điểm C
    # -----------------------------------------------------------------------
    raw_e_matrix: dict[str, dict[str, float]] = {c_id: {} for c_id in candidate_ids}
    raw_c_matrix: dict[str, dict[str, float]] = {c_id: {} for c_id in candidate_ids}

    # Lưu trữ thông tin phân rã điểm và giải thích (Mục 6.1 bước 7)
    explanations_map: dict[tuple[str, str], str] = {}
    e_breakdown_map: dict[tuple[str, str], dict[str, float]] = {}
    c_breakdown_map: dict[tuple[str, str], dict[str, float]] = {}
    skill_scores_map: dict[str, dict[str, float]] = {c_id: {} for c_id in candidate_ids}

    # Map tra cứu nhanh ứng viên và công việc
    cand_map = {c.id: c for c in request.candidates}
    job_map = {j.id: j for j in request.jobs}

    for c in request.candidates:
        c_id = c.id
        for j in request.jobs:
            j_id = j.id

            if c.direct_track:
                raw_e_matrix[c_id][j_id] = 0.0
                raw_c_matrix[c_id][j_id] = 0.0
                continue

            if quotas.get(j_id, 0) <= 0:
                raw_e_matrix[c_id][j_id] = 0.0
                raw_c_matrix[c_id][j_id] = 0.0
                continue

            has_composite = (
                c.skills is not None
                or c.essential_access_needs is not None
                or c.nice_to_have_access_needs is not None
                or c.priority_ranking is not None
                or j.skills_required is not None
                or j.access_support_provided is not None
            )

            if has_composite:
                cand_prof = CandidateProfile(
                    candidate_id=c.id,
                    skills=c.skills or [],
                    essential_access_needs=c.essential_access_needs or [],
                    nice_to_have_access_needs=c.nice_to_have_access_needs or [],
                    expected_salary_min=c.expected_salary_min or (c.expected_salary or 0.0),
                    expected_salary_max=c.expected_salary_max or (c.expected_salary or 0.0),
                    work_mode=c.work_mode or "on-site",
                    work_hours=c.work_hours or "full-time",
                    location=c.location or "",
                    priority_ranking=c.priority_ranking or ["income", "distance_or_remote", "career_path"],
                    experience_years=c.experience_years or 0.0,
                    career_orientation=c.career_orientation or "",
                    disability_group=c.nhom_khuyet_tat,
                    direct_track=c.direct_track,
                    hired_job_id=getattr(c, "hired_job_id", None),
                )
                job_prof = JobProfile(
                    job_id=j.id,
                    title=j.title or "",
                    quota=j.quota,
                    filled_slots=getattr(j, "filled_slots", 0),
                    skills_required=j.skills_required or [],
                    access_support_provided=j.access_support_provided or [],
                    salary_min=j.salary_min or (j.salary_offered or 0.0),
                    salary_max=j.salary_max or (j.salary_offered or 0.0),
                    work_mode=j.work_mode or "on-site",
                    work_hours=j.work_hours or "full-time",
                    location=j.location or "",
                    experience_required_years=j.experience_required_years or 0.0,
                    career_domain=j.career_domain or "",
                )
                feasible, reason = check_hard_constraints(cand_prof, job_prof)
                if not feasible:
                    e_val = 0.0
                    c_val = 0.0
                else:
                    e_val, e_bk = calculate_composite_e_score(cand_prof, job_prof)
                    c_val, c_bk = calculate_composite_c_score(cand_prof, job_prof)
                    exp = explain_match(cand_prof, job_prof, e_val, e_bk, c_val, c_bk)
                    explanations_map[(c_id, j_id)] = exp
                    e_breakdown_map[(c_id, j_id)] = e_bk
                    c_breakdown_map[(c_id, j_id)] = c_bk
                    skill_scores_map[c_id][j_id] = e_bk.get("skill", 0.0)
            elif c.vector_oxyz and j.vector_oxyz:
                # Tính Cosine Similarity từ tọa độ Không gian Vector Oxyz: u=(x1, y1, z1) và v=(x2, y2, z2)
                e_val = calculate_cosine_similarity_3d(
                    c.vector_oxyz.to_tuple(),
                    j.vector_oxyz.to_tuple(),
                )
                if c.c_scores and j_id in c.c_scores:
                    c_val = float(c.c_scores[j_id])
                elif c.preferences and j_id in c.preferences:
                    rank = c.preferences.index(j_id)
                    c_val = max(0.1, 1.0 - (rank * 0.1))
                else:
                    c_val = 0.5
            elif j.e_scores and c_id in j.e_scores:
                e_val = float(j.e_scores[c_id])
                if c.c_scores and j_id in c.c_scores:
                    c_val = float(c.c_scores[j_id])
                elif c.preferences and j_id in c.preferences:
                    rank = c.preferences.index(j_id)
                    c_val = max(0.1, 1.0 - (rank * 0.1))
                else:
                    c_val = 0.5
            else:
                # Mặc định trung tính nếu không có thông tin
                e_val = 0.5
                c_val = 0.5

            raw_e_matrix[c_id][j_id] = e_val
            raw_c_matrix[c_id][j_id] = c_val

    # -----------------------------------------------------------------------
    # GIAI ĐOẠN 1B: Áp dụng Hard Constraints & Domain Filter D
    # -----------------------------------------------------------------------
    filtered_e, filtered_c = apply_hard_constraints_filter(
        e_scores=raw_e_matrix,
        c_scores=raw_c_matrix,
        min_e_score=request.min_e_score,
        min_c_score=request.min_c_score,
        require_positive=True,
    )

    # -----------------------------------------------------------------------
    # GIAI ĐOẠN 2: Khởi tạo Danh sách Ưu tiên (Preference Lists)
    # -----------------------------------------------------------------------
    # Đảo ma trận filtered_e từ {c: {j: score}} sang {j: {c: score}} cho phía Job
    job_e_matrix: dict[str, dict[str, float]] = {j_id: {} for j_id in job_ids}
    for c_id, j_dict in filtered_e.items():
        for j_id, score in j_dict.items():
            job_e_matrix[j_id][c_id] = score

    candidates_pref, jobs_pref = build_preference_lists(
        candidate_scores=filtered_c,
        employer_scores=job_e_matrix,
        seed=request.tie_break_seed,
        skill_scores=skill_scores_map if any(skill_scores_map.values()) else None,
    )

    # -----------------------------------------------------------------------
    # GIAI ĐOẠN 3: Thực thi Gale-Shapley Deferred Acceptance
    # -----------------------------------------------------------------------
    if request.optimal == "candidate":
        # Áp dụng Candidate-Proposing Deferred Acceptance (mặc định C-optimal)
        matches, job_assignments, proposals = candidate_proposing_deferred_acceptance(
            candidates_pref=candidates_pref,
            jobs_pref=jobs_pref,
            job_quotas=quotas,
        )
    else:
        # Áp dụng Employer-Proposing Deferred Acceptance (J-optimal)
        matches, job_assignments, proposals = employer_proposing_deferred_acceptance(
            candidates_pref=candidates_pref,
            jobs_pref=jobs_pref,
            job_quotas=quotas,
        )

    # -----------------------------------------------------------------------
    # GIAI ĐOẠN 4: Hậu xử lý Top-K Recommendations
    # -----------------------------------------------------------------------
    recommendations_raw = generate_top_k_recommendations(
        matches=matches,
        candidates_pref=candidates_pref,
        c_scores=filtered_c,
        e_scores=filtered_e,
        k=request.top_k,
    )

    # Chuyển đổi sang schema pydantic
    recommendations_formatted: dict[str, list[CandidateRecommendationResponse]] = {}
    for c_id, rec_list in recommendations_raw.items():
        recommendations_formatted[c_id] = [
            CandidateRecommendationResponse(
                candidate_id=str(r["candidate_id"]),
                job_id=str(r["job_id"]),
                position=int(r["position"]),
                score=float(r["score"]),
                is_primary=bool(r["is_primary"]),
                explanation=explanations_map.get((str(r["candidate_id"]), str(r["job_id"]))),
                e_breakdown=e_breakdown_map.get((str(r["candidate_id"]), str(r["job_id"]))),
                c_breakdown=c_breakdown_map.get((str(r["candidate_id"]), str(r["job_id"]))),
            )
            for r in rec_list
        ]

    # Tính toán các chỉ số KPIs
    metrics = calculate_matching_metrics(
        matches=matches,
        job_assignments=job_assignments,
        candidates_pref=candidates_pref,
        jobs_pref=jobs_pref,
        job_quotas=quotas,
        c_scores=filtered_c,
        e_scores=filtered_e,
        proposals_count=proposals,
        all_candidate_ids=candidate_ids,
        all_job_ids=job_ids,
    )

    # Dựng danh sách MatchedPairResponse
    cand_rank_table = build_rank_lookup_table(candidates_pref)
    job_rank_table = build_rank_lookup_table(jobs_pref)
    matched_pairs: list[MatchedPairResponse] = []

    for c_id in sorted(matches.keys()):
        j_id = matches[c_id]
        matched_pairs.append(
            MatchedPairResponse(
                candidate_id=c_id,
                job_id=j_id,
                c_score=round(filtered_c.get(c_id, {}).get(j_id, 0.0), 4),
                e_score=round(filtered_e.get(c_id, {}).get(j_id, 0.0), 4),
                candidate_rank=(cand_rank_table[c_id].get(j_id, -1) + 1) if c_id in cand_rank_table else None,
                job_rank=(job_rank_table[j_id].get(c_id, -1) + 1) if j_id in job_rank_table else None,
                explanation=explanations_map.get((c_id, j_id)),
                e_breakdown=e_breakdown_map.get((c_id, j_id)),
                c_breakdown=c_breakdown_map.get((c_id, j_id)),
            )
        )


    unmatched_candidates = [c_id for c_id in candidate_ids if c_id not in matches]
    unmatched_jobs = [j_id for j_id in job_ids if len(job_assignments.get(j_id, [])) < quotas.get(j_id, 0)]

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return StableMatchingResponse(
        status="success",
        optimal=request.optimal,
        matches=matches,
        matched_pairs=matched_pairs,
        job_assignments=job_assignments,
        unmatched_candidates=unmatched_candidates,
        unmatched_jobs=unmatched_jobs,
        recommendations=recommendations_formatted,
        metrics=metrics,
        execution_time_ms=duration_ms,
    )


def run_matrix_matching(request: MatrixMatchingRequest) -> StableMatchingResponse:
    """
    Thực thi bài toán Stable Matching từ ma trận điểm thô C và E (theo mục 7 của Báo cáo).

    Mục đích:
        Cung cấp API nhanh gọn cho phép nhận trực tiếp ma trận điểm số và quotas tuyển dụng,
        sau đó chạy Gale-Shapley Many-to-One Deferred Acceptance.

    Tham số đầu vào:
        request (MatrixMatchingRequest): Payload chứa candidate_scores, employer_scores,
                                         job_quotas và các tùy chọn thuật toán.

    Kết quả trả về:
        StableMatchingResponse: Kết quả gán cặp, danh sách trúng tuyển và các chỉ số KPIs.
    """
    start_time = time.perf_counter()

    # 1. Chuyển đổi ma trận điểm C thành danh sách ưu tiên của Candidate
    # Sắp xếp các job theo điểm C giảm dần (bỏ qua điểm <= 0 - Hard Constraints)
    candidates_pref: dict[str, list[str]] = {}
    for c_id, j_scores in request.candidate_scores.items():
        sorted_jobs = [
            j for j, score in sorted(
                j_scores.items(),
                key=lambda x: (
                    -x[1],
                    deterministic_tie_breaker(request.tie_break_seed, "cand", c_id, x[0]),
                    str(x[0]),
                )
            )
            if score > 0.0
        ]
        candidates_pref[c_id] = sorted_jobs

    # 2. Chuyển đổi ma trận điểm E thành danh sách ưu tiên của Employer/Job
    # Sắp xếp các candidate theo điểm E giảm dần (bỏ qua điểm <= 0)
    jobs_pref: dict[str, list[str]] = {}
    for j_id, c_scores in request.employer_scores.items():
        sorted_cands = [
            c for c, score in sorted(
                c_scores.items(),
                key=lambda x: (
                    -x[1],
                    deterministic_tie_breaker(request.tie_break_seed, "job", j_id, x[0]),
                    str(x[0]),
                )
            )
            if score > 0.0
        ]
        jobs_pref[j_id] = sorted_cands

    # 3. Giải bài toán theo cơ chế Deferred Acceptance
    if request.optimal == "candidate":
        matches, job_assignments, proposals = candidate_proposing_deferred_acceptance(
            candidates_pref=candidates_pref,
            jobs_pref=jobs_pref,
            job_quotas=request.job_quotas,
        )
    else:
        matches, job_assignments, proposals = employer_proposing_deferred_acceptance(
            candidates_pref=candidates_pref,
            jobs_pref=jobs_pref,
            job_quotas=request.job_quotas,
        )

    # 4. Hậu xử lý Top-K Recommendations
    # Chuẩn bị ma trận E theo chỉ mục [candidate][job]
    all_cand_ids = sorted(list(request.candidate_scores.keys()))
    all_job_ids = sorted(list(request.job_quotas.keys()))

    c_indexed_e: dict[str, dict[str, float]] = {c: {} for c in all_cand_ids}
    for j_id, c_dict in request.employer_scores.items():
        for c_id, score in c_dict.items():
            if c_id in c_indexed_e:
                c_indexed_e[c_id][j_id] = float(score)

    recommendations_raw = generate_top_k_recommendations(
        matches=matches,
        candidates_pref=candidates_pref,
        c_scores=request.candidate_scores,
        e_scores=c_indexed_e,
        k=request.top_k,
    )

    recommendations_formatted: dict[str, list[CandidateRecommendationResponse]] = {}
    for c_id, rec_list in recommendations_raw.items():
        recommendations_formatted[c_id] = [
            CandidateRecommendationResponse(
                candidate_id=str(r["candidate_id"]),
                job_id=str(r["job_id"]),
                position=int(r["position"]),
                score=float(r["score"]),
                is_primary=bool(r["is_primary"]),
            )
            for r in rec_list
        ]

    # 5. Tính toán chỉ số KPIs
    metrics = calculate_matching_metrics(
        matches=matches,
        job_assignments=job_assignments,
        candidates_pref=candidates_pref,
        jobs_pref=jobs_pref,
        job_quotas=request.job_quotas,
        c_scores=request.candidate_scores,
        e_scores=c_indexed_e,
        proposals_count=proposals,
        all_candidate_ids=all_cand_ids,
        all_job_ids=all_job_ids,
    )

    # Dựng matched pairs
    cand_rank_table = build_rank_lookup_table(candidates_pref)
    job_rank_table = build_rank_lookup_table(jobs_pref)
    matched_pairs: list[MatchedPairResponse] = []

    for c_id in sorted(matches.keys()):
        j_id = matches[c_id]
        matched_pairs.append(
            MatchedPairResponse(
                candidate_id=c_id,
                job_id=j_id,
                c_score=round(request.candidate_scores.get(c_id, {}).get(j_id, 0.0), 4),
                e_score=round(request.employer_scores.get(j_id, {}).get(c_id, 0.0), 4),
                candidate_rank=(cand_rank_table[c_id].get(j_id, -1) + 1) if c_id in cand_rank_table else None,
                job_rank=(job_rank_table[j_id].get(c_id, -1) + 1) if j_id in job_rank_table else None,
            )
        )

    unmatched_candidates = [c_id for c_id in all_cand_ids if c_id not in matches]
    unmatched_jobs = [
        j_id for j_id in all_job_ids
        if len(job_assignments.get(j_id, [])) < request.job_quotas.get(j_id, 0)
    ]

    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return StableMatchingResponse(
        status="success",
        optimal=request.optimal,
        matches=matches,
        matched_pairs=matched_pairs,
        job_assignments=job_assignments,
        unmatched_candidates=unmatched_candidates,
        unmatched_jobs=unmatched_jobs,
        recommendations=recommendations_formatted,
        metrics=metrics,
        execution_time_ms=duration_ms,
    )
