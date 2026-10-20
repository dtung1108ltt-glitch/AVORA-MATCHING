#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Module tính toán điểm ưu tiên E và C theo công thức chuẩn hóa Mục 6.2.

Đặc tả công thức:
- Điểm E (Nhà tuyển dụng):
  * Kỹ năng: Cosine similarity embedding (Trọng số 0.35)
  * Thời gian, hình thức làm việc: 1 / 0.5 / 0 theo mức khớp (Trọng số 0.20)
  * Điều kiện tiếp cận "nên có": Tỷ lệ được đáp ứng (Trọng số 0.20)
  * Lương: Mức chồng lấn hai khoảng lương (Trọng số 0.10)
  * Kinh nghiệm: Chuẩn hóa theo mức yêu cầu (Trọng số 0.10)
  * Khu vực: Cùng khu vực hay không (Trọng số 0.05)

- Điểm C (Ứng viên):
  * Mức thu nhập: Lương việc so với lương mong muốn (theo thứ tự tự xếp)
  * Khoảng cách hoặc remote: Khớp hình thức và khu vực (theo thứ tự tự xếp)
  * Định hướng nghề nghiệp: Độ gần với mục tiêu nghề (theo thứ tự tự xếp)
  Trọng số C: Thứ nhất 0.5, thứ hai 0.3, thứ ba 0.2. Thiếu dữ liệu thì mặc định 0.35, 0.30, 0.35.

- Lọc cứng (Hard Constraints - Mục 6.1):
  * Loại cặp vi phạm nhu cầu tiếp cận thiết yếu (essential access needs).
  * Loại ứng viên đã được nhận qua Direct Track.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numba
import numpy as np

from src.core.profile_models import CandidateProfile, JobProfile


@dataclass(slots=True)
class EmployerWeightsConfig:
    """Cấu hình trọng số điểm E theo Mục 6.2."""

    skill: float = 0.35
    work_mode_hours: float = 0.20
    nice_to_have_access: float = 0.20
    salary: float = 0.10
    experience: float = 0.10
    location: float = 0.05

    def validate(self) -> None:
        total = (
            self.skill
            + self.work_mode_hours
            + self.nice_to_have_access
            + self.salary
            + self.experience
            + self.location
        )
        if not math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError(f"Tổng trọng số điểm E phải bằng 1.0 (nhận {total}).")


@numba.njit(fastmath=True, cache=True)
def _cosine_similarity_numba(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Tính Cosine similarity biên dịch mã máy Numba trên mảng 1D float64."""
    n = vec_a.shape[0]
    if n == 0 or n != vec_b.shape[0]:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(n):
        a = vec_a[i]
        b = vec_b[i]
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    val = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


@numba.njit(parallel=True, fastmath=True, cache=True)
def compute_cosine_similarity_matrix_numba(
    cand_vectors: np.ndarray,
    job_vectors: np.ndarray,
) -> np.ndarray:
    """Tính ma trận Cosine Similarity (N x M) song song bằng Numba JIT."""
    n_cands = cand_vectors.shape[0]
    n_jobs = job_vectors.shape[0]
    dim = cand_vectors.shape[1]
    out = np.zeros((n_cands, n_jobs), dtype=np.float64)

    for i in numba.prange(n_cands):
        norm_a = 0.0
        for d in range(dim):
            va = cand_vectors[i, d]
            norm_a += va * va
        if norm_a <= 0.0:
            continue
        inv_norm_a = 1.0 / math.sqrt(norm_a)

        for j in range(n_jobs):
            dot = 0.0
            norm_b = 0.0
            for d in range(dim):
                va = cand_vectors[i, d]
                vb = job_vectors[j, d]
                dot += va * vb
                norm_b += vb * vb
            if norm_b <= 0.0:
                out[i, j] = 0.0
            else:
                val = dot * inv_norm_a / math.sqrt(norm_b)
                if val < 0.0:
                    val = 0.0
                elif val > 1.0:
                    val = 1.0
                out[i, j] = val
    return out


@numba.njit(parallel=True, fastmath=True, cache=True)
def compute_composite_score_matrices_numba(
    s_skill: np.ndarray,
    s_work: np.ndarray,
    s_access: np.ndarray,
    s_salary: np.ndarray,
    s_exp: np.ndarray,
    s_loc: np.ndarray,
    s_income: np.ndarray,
    s_distance: np.ndarray,
    s_career: np.ndarray,
    e_weights: np.ndarray,
    c_weights: np.ndarray,
    feasible_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Tính đồng thời ma trận điểm tổng hợp E[N, M] và C[N, M] trên mảng numpy.ndarray.

    Biên dịch sang mã máy đa luồng bằng ``@numba.njit(parallel=True, fastmath=True)``
    và giữ nguyên 100% công thức toán học cùng trọng số ở Mục 6.2.
    """
    n_cands, n_jobs = s_skill.shape
    e_mat = np.zeros((n_cands, n_jobs), dtype=np.float64)
    c_mat = np.zeros((n_cands, n_jobs), dtype=np.float64)

    w_skill = e_weights[0]
    w_work = e_weights[1]
    w_access = e_weights[2]
    w_salary = e_weights[3]
    w_exp = e_weights[4]
    w_loc = e_weights[5]

    for i in numba.prange(n_cands):
        w_inc = c_weights[i, 0]
        w_dist = c_weights[i, 1]
        w_car = c_weights[i, 2]
        for j in range(n_jobs):
            if not feasible_mask[i, j]:
                e_mat[i, j] = 0.0
                c_mat[i, j] = 0.0
                continue

            total_e = (
                w_skill * s_skill[i, j]
                + w_work * s_work[i, j]
                + w_access * s_access[i, j]
                + w_salary * s_salary[i, j]
                + w_exp * s_exp[i, j]
                + w_loc * s_loc[i, j]
            )
            rounded_e = np.round(total_e, 6)
            if rounded_e < 0.001:
                rounded_e = 0.001
            elif rounded_e > 1.0:
                rounded_e = 1.0
            e_mat[i, j] = rounded_e

            total_c = (
                w_inc * s_income[i, j]
                + w_dist * s_distance[i, j]
                + w_car * s_career[i, j]
            )
            rounded_c = np.round(total_c, 6)
            if rounded_c < 0.001:
                rounded_c = 0.001
            elif rounded_c > 1.0:
                rounded_c = 1.0
            c_mat[i, j] = rounded_c

    return e_mat, c_mat


def calculate_cosine_similarity(vec_a: list[float] | np.ndarray, vec_b: list[float] | np.ndarray) -> float:
    """Tính Cosine similarity giữa hai vector float."""
    if isinstance(vec_a, np.ndarray) and isinstance(vec_b, np.ndarray):
        if vec_a.size == 0 or vec_a.shape != vec_b.shape:
            return 0.0
        return float(_cosine_similarity_numba(vec_a, vec_b))
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    if len(vec_a) >= 64:
        arr_a = np.asarray(vec_a, dtype=np.float64)
        arr_b = np.asarray(vec_b, dtype=np.float64)
        return float(_cosine_similarity_numba(arr_a, arr_b))
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    val = dot / (norm_a * norm_b)
    return max(0.0, min(1.0, val))


def calculate_token_jaccard(text_a: str, text_b: str) -> float:
    """Tính Jaccard similarity giữa hai chuỗi văn bản hoặc từ khóa."""
    tokens_a = {t.strip().lower() for t in text_a.replace(",", " ").split() if t.strip()}
    tokens_b = {t.strip().lower() for t in text_b.replace(",", " ").split() if t.strip()}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def calculate_skill_similarity(
    candidate_skills: list[float] | str | list[str],
    job_skills_required: list[float] | str | list[str],
) -> float:
    """Tính điểm khớp kỹ năng [0.0, 1.0].

    Hỗ trợ vector embedding hoặc chuỗi văn bản/danh sách từ khóa.
    """
    if isinstance(candidate_skills, list) and isinstance(job_skills_required, list):
        if (
            candidate_skills
            and job_skills_required
            and isinstance(candidate_skills[0], (int, float))
            and isinstance(job_skills_required[0], (int, float))
        ):
            return calculate_cosine_similarity(
                [float(x) for x in candidate_skills],
                [float(x) for x in job_skills_required],
            )
        # Danh sách chuỗi từ khóa
        str_a = " ".join(str(s) for s in candidate_skills)
        str_b = " ".join(str(s) for s in job_skills_required)
        return calculate_token_jaccard(str_a, str_b)

    if isinstance(candidate_skills, str) and isinstance(job_skills_required, str):
        if not candidate_skills.strip() or not job_skills_required.strip():
            return 0.0
        if candidate_skills.strip().lower() == job_skills_required.strip().lower():
            return 1.0
        return calculate_token_jaccard(candidate_skills, job_skills_required)

    return 0.0


def calculate_work_mode_hours_match(
    candidate: CandidateProfile,
    job: JobProfile,
) -> float:
    """Tính mức khớp thời gian và hình thức làm việc: 1.0 / 0.5 / 0.0 theo Mục 6.2."""
    cand_mode = candidate.work_mode.strip().lower()
    job_mode = job.work_mode.strip().lower()

    cand_hours = candidate.work_hours.strip().lower()
    job_hours = job.work_hours.strip().lower()

    # Đánh giá hình thức làm việc
    if job_mode == "remote" or cand_mode == job_mode:
        mode_score = 1.0
    elif (cand_mode == "hybrid" and job_mode in ("remote", "on-site")) or (
        job_mode == "hybrid" and cand_mode in ("remote", "on-site")
    ):
        mode_score = 0.5
    else:
        mode_score = 0.0

    # Đánh giá thời gian làm việc
    if (
        job_hours in ("flexible", "linh_hoat")
        or cand_hours in ("flexible", "linh_hoat")
        or cand_hours == job_hours
    ):
        hours_score = 1.0
    elif (cand_hours == "part-time" and job_hours == "full-time") or (
        cand_hours == "full-time" and job_hours == "part-time"
    ):
        hours_score = 0.0
    else:
        hours_score = 0.5

    # Tổng hợp mức khớp: 1.0 / 0.5 / 0.0
    avg = (mode_score + hours_score) / 2.0
    if avg >= 0.99:
        return 1.0
    if avg >= 0.25:
        return 0.5
    return 0.0


def calculate_nice_to_have_access_ratio(
    candidate: CandidateProfile,
    job: JobProfile,
) -> float:
    """Tính tỷ lệ điều kiện tiếp cận 'nên có' được đáp ứng [0.0, 1.0] theo Mục 6.2."""
    if not candidate.nice_to_have_access_needs:
        return 1.0  # Không có đòi hỏi thêm -> xem như đáp ứng trọn vẹn

    cand_needs = {n.strip().lower() for n in candidate.nice_to_have_access_needs if n.strip()}
    if not cand_needs:
        return 1.0

    job_supports = {s.strip().lower() for s in job.access_support_provided if s.strip()}
    matched = cand_needs & job_supports
    return len(matched) / len(cand_needs)


def calculate_salary_overlap(
    candidate: CandidateProfile,
    job: JobProfile,
) -> float:
    """Tính mức chồng lấn hai khoảng lương [0.0, 1.0] theo Mục 6.2."""
    c_min, c_max = candidate.get_salary_range()
    j_min, j_max = job.get_salary_range()

    if c_max == 0.0 and j_max == 0.0:
        return 1.0
    if c_max == 0.0:
        return 1.0 if j_max >= c_min else 0.5
    if j_max == 0.0:
        return 0.5

    overlap = max(0.0, min(c_max, j_max) - max(c_min, j_min))
    union = max(c_max, j_max) - min(c_min, j_min)

    if union <= 0.0:
        return 1.0
    return max(0.0, min(1.0, overlap / union))


def calculate_experience_score(
    candidate: CandidateProfile,
    job: JobProfile,
) -> float:
    """Tính điểm kinh nghiệm chuẩn hóa theo mức yêu cầu [0.0, 1.0] theo Mục 6.2."""
    req_exp = max(0.0, float(job.experience_required_years))
    if req_exp <= 0.0:
        return 1.0
    cand_exp = max(0.0, float(candidate.experience_years))
    return max(0.0, min(1.0, cand_exp / req_exp))


def calculate_location_match(
    candidate: CandidateProfile,
    job: JobProfile,
) -> float:
    """Tính điểm khu vực: 1.0 nếu cùng khu vực hoặc làm remote, 0.0 nếu khác theo Mục 6.2."""
    if (
        job.work_mode.strip().lower() == "remote"
        or candidate.work_mode.strip().lower() == "remote"
    ):
        return 1.0
    cand_loc = candidate.location.strip().lower()
    job_loc = job.location.strip().lower()
    if not cand_loc or not job_loc:
        return 0.5
    return 1.0 if cand_loc == job_loc else 0.0


def check_hard_constraints(
    candidate: CandidateProfile,
    job: JobProfile,
) -> tuple[bool, str]:
    """Kiểm tra bộ lọc điều kiện bắt buộc (Hard Constraints) theo Mục 6.1 (bước 2):

    1. Nhu cầu tiếp cận thiết yếu (essential_access_needs):
       Nếu doanh nghiệp không cung cấp điều kiện tiếp cận thiết yếu mà ứng viên bắt buộc cần,
       cặp (c, j) bị loại ngay lập tức.
    2. Hình thức làm việc không thể tương thích (ví dụ ứng viên chỉ làm được remote nhưng job yêu cầu on-site).
    3. Ứng viên Direct Track (đã được nhận qua kênh trực tiếp) -> loại khỏi pool.

    Returns:
        tuple[bool, str]: (is_feasible, reason)
    """
    if candidate.direct_track:
        return False, "Ứng viên đã được phân bổ qua kênh Direct Track."

    # Ràng buộc hình thức làm việc bất khả thi
    cand_mode = candidate.work_mode.strip().lower()
    job_mode = job.work_mode.strip().lower()
    if cand_mode == "remote" and job_mode == "on-site":
        return (
            False,
            "Ứng viên yêu cầu làm việc từ xa (Remote) nhưng công việc yêu cầu 100% tại chỗ (On-site).",
        )

    # Kiểm tra nhu cầu tiếp cận thiết yếu
    if candidate.essential_access_needs:
        cand_essentials = {
            n.strip().lower() for n in candidate.essential_access_needs if n.strip()
        }
        job_supports = {
            s.strip().lower() for s in job.access_support_provided if s.strip()
        }
        if job_mode != "remote":
            missing_essentials = cand_essentials - job_supports
            if missing_essentials:
                missing_str = ", ".join(sorted(missing_essentials))
                return (
                    False,
                    f"Thiếu điều kiện tiếp cận thiết yếu bắt buộc: {missing_str}.",
                )

    return True, "Thỏa mãn mọi điều kiện bắt buộc."


def calculate_composite_e_score(
    candidate: CandidateProfile,
    job: JobProfile,
    weights: EmployerWeightsConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """Tính Điểm E (Nhà tuyển dụng) kèm bảng phân rã từng thành phần theo Mục 6.2:

    E = 0.35 * Kỹ năng
      + 0.20 * Thời gian, hình thức làm việc
      + 0.20 * Điều kiện tiếp cận "nên có"
      + 0.10 * Lương
      + 0.10 * Kinh nghiệm
      + 0.05 * Khu vực

    Returns:
        tuple[float, dict[str, float]]: (total_e_score, component_breakdown)
    """
    w = weights or EmployerWeightsConfig()

    s_skill = calculate_skill_similarity(candidate.skills, job.skills_required)
    s_work = calculate_work_mode_hours_match(candidate, job)
    s_access = calculate_nice_to_have_access_ratio(candidate, job)
    s_salary = calculate_salary_overlap(candidate, job)
    s_exp = calculate_experience_score(candidate, job)
    s_loc = calculate_location_match(candidate, job)

    total = (
        w.skill * s_skill
        + w.work_mode_hours * s_work
        + w.nice_to_have_access * s_access
        + w.salary * s_salary
        + w.experience * s_exp
        + w.location * s_loc
    )
    total_score = max(0.001, min(1.0, round(total, 6)))

    breakdown = {
        "skill": round(s_skill, 4),
        "work_mode_hours": round(s_work, 4),
        "nice_to_have_access": round(s_access, 4),
        "salary": round(s_salary, 4),
        "experience": round(s_exp, 4),
        "location": round(s_loc, 4),
    }
    return total_score, breakdown


def calculate_composite_c_score(
    candidate: CandidateProfile,
    job: JobProfile,
) -> tuple[float, dict[str, float]]:
    """Tính Điểm C (Ứng viên) theo Mục 6.2:

    3 thành phần:
    1. Mức thu nhập: Lương việc so với lương mong muốn
    2. Khoảng cách hoặc remote: Khớp hình thức và khu vực
    3. Định hướng nghề nghiệp: Độ gần với mục tiêu nghề

    Trọng số theo thứ tự ứng viên tự xếp:
    - Thứ nhất: 0.5
    - Thứ hai: 0.3
    - Thứ ba: 0.2
    Thiếu dữ liệu: 0.35, 0.30, 0.35.

    Returns:
        tuple[float, dict[str, float]]: (total_c_score, component_breakdown)
    """
    # 1. Thành phần Mức thu nhập
    c_min, _ = candidate.get_salary_range()
    _, j_max = job.get_salary_range()
    if c_min <= 0.0:
        s_income = 1.0
    else:
        s_income = min(1.0, j_max / c_min) if j_max > 0 else 0.5

    # 2. Thành phần Khoảng cách hoặc remote
    s_distance = calculate_location_match(candidate, job)

    # 3. Thành phần Định hướng nghề nghiệp
    if candidate.career_orientation and job.career_domain:
        if isinstance(candidate.career_orientation, list) and isinstance(
            job.career_domain, list
        ):
            s_career = calculate_cosine_similarity(
                candidate.career_orientation, job.career_domain
            )
        else:
            s_career = calculate_skill_similarity(
                candidate.career_orientation, job.career_domain
            )
    else:
        s_career = calculate_skill_similarity(candidate.skills, job.skills_required)

    weights_dict = candidate.get_priority_weights()
    w_income = weights_dict.get("income", 0.35)
    w_distance = weights_dict.get("distance_or_remote", 0.30)
    w_career = weights_dict.get("career_path", 0.35)

    total = w_income * s_income + w_distance * s_distance + w_career * s_career
    total_score = max(0.001, min(1.0, round(total, 6)))

    breakdown = {
        "income": round(s_income, 4),
        "distance_or_remote": round(s_distance, 4),
        "career_path": round(s_career, 4),
    }
    return total_score, breakdown


def compute_batch_composite_scores_matrix(
    candidates: Sequence[CandidateProfile],
    jobs: Sequence[JobProfile],
    weights: EmployerWeightsConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính toán ma trận điểm tổng hợp E[N, M] và C[N, M] bằng Numba JIT trên mảng numpy.ndarray."""
    w = weights or EmployerWeightsConfig()
    n_cands = len(candidates)
    n_jobs = len(jobs)

    s_skill = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_work = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_access = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_salary = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_exp = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_loc = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_income = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_distance = np.zeros((n_cands, n_jobs), dtype=np.float64)
    s_career = np.zeros((n_cands, n_jobs), dtype=np.float64)
    feasible_mask = np.zeros((n_cands, n_jobs), dtype=np.bool_)

    e_weights = np.array(
        [
            w.skill,
            w.work_mode_hours,
            w.nice_to_have_access,
            w.salary,
            w.experience,
            w.location,
        ],
        dtype=np.float64,
    )
    c_weights = np.zeros((n_cands, 3), dtype=np.float64)

    for i, cand in enumerate(candidates):
        cw = cand.get_priority_weights()
        c_weights[i, 0] = cw.get("income", 0.35)
        c_weights[i, 1] = cw.get("distance_or_remote", 0.30)
        c_weights[i, 2] = cw.get("career_path", 0.35)
        c_min, _ = cand.get_salary_range()

        for j, job in enumerate(jobs):
            feasible, _ = check_hard_constraints(cand, job)
            if not feasible:
                continue
            feasible_mask[i, j] = True
            skill_val = calculate_skill_similarity(cand.skills, job.skills_required)
            loc_val = calculate_location_match(cand, job)

            s_skill[i, j] = skill_val
            s_work[i, j] = calculate_work_mode_hours_match(cand, job)
            s_access[i, j] = calculate_nice_to_have_access_ratio(cand, job)
            s_salary[i, j] = calculate_salary_overlap(cand, job)
            s_exp[i, j] = calculate_experience_score(cand, job)
            s_loc[i, j] = loc_val

            _, j_max = job.get_salary_range()
            if c_min <= 0.0:
                s_income[i, j] = 1.0
            else:
                s_income[i, j] = min(1.0, j_max / c_min) if j_max > 0 else 0.5
            s_distance[i, j] = loc_val

            if cand.career_orientation and job.career_domain:
                if isinstance(cand.career_orientation, list) and isinstance(
                    job.career_domain, list
                ):
                    s_career[i, j] = calculate_cosine_similarity(
                        cand.career_orientation, job.career_domain
                    )
                else:
                    s_career[i, j] = calculate_skill_similarity(
                        cand.career_orientation, job.career_domain
                    )
            else:
                s_career[i, j] = skill_val

    e_mat, c_mat = compute_composite_score_matrices_numba(
        s_skill,
        s_work,
        s_access,
        s_salary,
        s_exp,
        s_loc,
        s_income,
        s_distance,
        s_career,
        e_weights,
        c_weights,
        feasible_mask,
    )
    return e_mat, c_mat, s_skill


def explain_match(
    candidate: CandidateProfile,
    job: JobProfile,
    e_score: float,
    e_breakdown: dict[str, float],
    c_score: float,
    c_breakdown: dict[str, float],
) -> str:
    """Sinh chuỗi văn bản giải thích rõ ràng lý do ứng viên được gợi ý (Mục 6.1, bước 7)."""
    reasons = []

    # Lý do kỹ năng
    skill_val = e_breakdown.get("skill", 0.0)
    if skill_val >= 0.8:
        reasons.append(
            f"Kỹ năng chuyên môn đạt độ tương đồng xuất sắc ({skill_val * 100:.1f}%)"
        )
    elif skill_val >= 0.5:
        reasons.append(
            f"Kỹ năng chuyên môn phù hợp tốt với yêu cầu vị trí ({skill_val * 100:.1f}%)"
        )

    # Lý do điều kiện tiếp cận
    access_val = e_breakdown.get("nice_to_have_access", 1.0)
    if access_val >= 1.0 and candidate.nice_to_have_access_needs:
        reasons.append("Đáp ứng 100% các điều kiện hỗ trợ tiếp cận mong muốn")
    elif access_val > 0.5:
        reasons.append(
            f"Đáp ứng phần lớn ({access_val * 100:.0f}%) tiện nghi hỗ trợ tiếp cận"
        )

    # Lý do hình thức & địa điểm
    work_val = e_breakdown.get("work_mode_hours", 0.0)
    loc_val = e_breakdown.get("location", 0.0)
    if work_val == 1.0 and loc_val == 1.0:
        reasons.append(
            f"Hoàn toàn tương thích về hình thức ({job.work_mode.title()}) và địa điểm làm việc"
        )

    # Lý do mức lương
    salary_val = e_breakdown.get("salary", 0.0)
    if salary_val >= 0.5:
        reasons.append("Khoảng lương hai bên có mức độ đồng thuận cao")

    # Lý do nguyện vọng ứng viên
    if c_breakdown.get("career_path", 0.0) >= 0.7:
        reasons.append(
            "Phù hợp cao với định hướng phát triển nghề nghiệp của ứng viên"
        )

    if not reasons:
        reasons.append(
            f"Tổng hợp các chỉ số đáp ứng ngưỡng sàn đề xuất (E={e_score:.2f}, C={c_score:.2f})"
        )

    return "; ".join(reasons) + "."
