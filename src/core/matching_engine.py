#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Giai đoạn 3 - Core Matching Engine: Gale-Shapley Many-to-One DA.

Triển khai **Deferred Acceptance (DA)** của Gale & Shapley (1962) cho bài toán
**Hospital-Residents / College-Admissions** (Many-to-One), trong đó mỗi vị trí
tuyển dụng ``j`` có hạn mức ``q_j >= 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numba
import numpy as np

from src.core.preprocessor import Id, PreferenceProfile

__all__ = [
    "GaleShapleyMatching",
    "MatchingResult",
    "_gs_candidate_proposing_numba",
    "_gs_employer_proposing_numba",
]


@numba.njit(cache=True)
def _gs_candidate_proposing_numba(
    cand_prefs_arr: np.ndarray,
    cand_pref_lens: np.ndarray,
    job_rank_arr: np.ndarray,
    quotas_arr: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Vòng lặp while Gale-Shapley Candidate-Proposing biên dịch mã máy với @numba.njit.

    Tham số:
        cand_prefs_arr: int32[N, max_L] - bảng xếp hạng ưu tiên của ứng viên.
        cand_pref_lens: int32[N] - độ dài thực tế danh sách ưu tiên của từng ứng viên.
        job_rank_arr: int32[M, N] - bảng thứ hạng O(1) của ứng viên trong mắt vị trí (-1 = không chấp nhận).
        quotas_arr: int32[M] - hạn mức tuyển dụng của từng vị trí.

    Trả về:
        (matches_arr: int32[N], proposals: int)
    """
    n_cands = cand_prefs_arr.shape[0]
    n_jobs = job_rank_arr.shape[0]

    max_quota = 1
    for j in range(n_jobs):
        if quotas_arr[j] > max_quota:
            max_quota = quotas_arr[j]

    # Biểu diễn trạng thái ghép cặp phía công việc bằng mảng 2 chiều int32[:, :]
    held_cands = np.full((n_jobs, max_quota), -1, dtype=np.int32)
    held_ranks = np.full((n_jobs, max_quota), -1, dtype=np.int32)
    held_counts = np.zeros(n_jobs, dtype=np.int32)

    matches_arr = np.full(n_cands, -1, dtype=np.int32)
    next_proposal = np.zeros(n_cands, dtype=np.int32)

    # Hàng đợi FIFO tất định tương đương collections.deque
    max_q = max(16, n_cands * (cand_prefs_arr.shape[1] + 2) + 16)
    queue = np.empty(max_q, dtype=np.int32)
    head = 0
    tail = 0

    for c in range(n_cands):
        if cand_pref_lens[c] > 0:
            queue[tail] = c
            tail += 1

    proposals = 0

    while head < tail:
        c = queue[head]
        head += 1

        idx = next_proposal[c]
        if idx >= cand_pref_lens[c]:
            continue

        j = cand_prefs_arr[c, idx]
        next_proposal[c] = idx + 1
        proposals += 1

        quota = quotas_arr[j]
        if quota <= 0:
            queue[tail] = c
            tail += 1
            continue

        rank = job_rank_arr[j, c]
        if rank < 0:
            queue[tail] = c
            tail += 1
            continue

        cnt = held_counts[j]
        if cnt < quota:
            held_cands[j, cnt] = c
            held_ranks[j, cnt] = rank
            held_counts[j] = cnt + 1
            matches_arr[c] = j
            continue

        # Tìm ứng viên có thứ hạng kém nhất (rank lớn nhất) đang giữ ở vị trí j
        worst_slot = 0
        worst_rank = held_ranks[j, 0]
        for s in range(1, cnt):
            r = held_ranks[j, s]
            if r > worst_rank:
                worst_rank = r
                worst_slot = s

        if rank < worst_rank:
            worst_cand = held_cands[j, worst_slot]
            held_cands[j, worst_slot] = c
            held_ranks[j, worst_slot] = rank
            matches_arr[worst_cand] = -1
            matches_arr[c] = j
            queue[tail] = worst_cand
            tail += 1
        else:
            queue[tail] = c
            tail += 1

    return matches_arr, proposals


@numba.njit(cache=True)
def _gs_employer_proposing_numba(
    job_prefs_arr: np.ndarray,
    job_pref_lens: np.ndarray,
    cand_rank_arr: np.ndarray,
    quotas_arr: np.ndarray,
    initial_free_jobs: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Vòng lặp while Gale-Shapley Employer-Proposing biên dịch mã máy với @numba.njit.

    Tham số:
        job_prefs_arr: int32[M, max_M] - bảng xếp hạng ưu tiên của vị trí tuyển dụng.
        job_pref_lens: int32[M] - độ dài thực tế danh sách ưu tiên của từng vị trí.
        cand_rank_arr: int32[N, M] - bảng thứ hạng O(1) của vị trí trong mắt ứng viên (-1 = không chấp nhận).
        quotas_arr: int32[M] - hạn mức tuyển dụng của từng vị trí.
        initial_free_jobs: int32[K] - danh sách vị trí khởi tạo trong hàng đợi.

    Trả về:
        (candidate_holder: int32[N], proposals: int)
    """
    n_cands = cand_rank_arr.shape[0]
    n_jobs = job_prefs_arr.shape[0]

    # Biểu diễn trạng thái ghép cặp dưới dạng mảng 2 chiều int32[:, :]
    assigned_matrix = np.zeros((n_jobs, max(1, n_cands)), dtype=np.int32)
    assigned_counts = np.zeros(n_jobs, dtype=np.int32)
    candidate_holder = np.full(n_cands, -1, dtype=np.int32)
    next_proposal = np.zeros(n_jobs, dtype=np.int32)

    max_q = max(32, n_jobs * (job_prefs_arr.shape[1] * 2 + 4) + 32)
    queue = np.empty(max_q, dtype=np.int32)
    head = 0
    tail = 0

    for idx_j in range(initial_free_jobs.shape[0]):
        queue[tail] = initial_free_jobs[idx_j]
        tail += 1

    proposals = 0

    while head < tail:
        j = queue[head]
        head += 1

        quota = quotas_arr[j]
        if quota <= 0 or assigned_counts[j] >= quota:
            continue

        p_idx = next_proposal[j]
        if p_idx >= job_pref_lens[j]:
            continue

        cand = job_prefs_arr[j, p_idx]
        next_proposal[j] = p_idx + 1
        proposals += 1

        new_rank = cand_rank_arr[cand, j]
        if new_rank >= 0:
            current_job = candidate_holder[cand]
            if current_job < 0:
                assigned_matrix[j, cand] = 1
                assigned_counts[j] += 1
                candidate_holder[cand] = j
            else:
                current_rank = cand_rank_arr[cand, current_job]
                if current_rank < 0 or new_rank < current_rank:
                    if assigned_matrix[current_job, cand] == 1:
                        assigned_matrix[current_job, cand] = 0
                        assigned_counts[current_job] -= 1
                    queue[tail] = current_job
                    tail += 1
                    assigned_matrix[j, cand] = 1
                    assigned_counts[j] += 1
                    candidate_holder[cand] = j

        if assigned_counts[j] < quota and next_proposal[j] < job_pref_lens[j]:
            queue[tail] = j
            tail += 1

    return candidate_holder, proposals


@dataclass(slots=True)
class MatchingResult:
    """Kết quả ghép cặp trả về từ engine."""

    matches: dict[Id, Id]
    job_assignments: dict[Id, list[Id]]
    unmatched_candidates: list[Id]
    unmatched_jobs: list[Id]
    proposals: int
    optimal: str = "candidate"
    is_stable: bool | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    _profile: PreferenceProfile | None = field(default=None, repr=False)

    def verify_stability(self, profile: PreferenceProfile | None = None) -> bool:
        """Kiểm chứng tính ổn định bằng cách tìm blocking pairs."""
        from src.evaluation.metrics import find_blocking_pairs

        target_profile = profile or self._profile
        if target_profile is None:
            raise ValueError(
                "Cần cung cấp PreferenceProfile để kiểm chứng tính ổn định."
            )
        bps = find_blocking_pairs(target_profile, self)
        self.is_stable = len(bps) == 0
        return self.is_stable

    def matched_pair_count(self) -> int:
        """Số cặp đã ghép thành công."""

        return len(self.matches)

    def is_complete(self, total_candidates: int) -> bool:
        """``True`` nếu mọi ứng viên đều có việc (thị trường thừa cầu)."""

        return len(self.matches) == total_candidates

    def job_fill(self, job: Id) -> int:
        """Số ứng viên đang giữ ở một vị trí."""

        return len(self.job_assignments.get(job, ()))


class GaleShapleyMatching:
    """Engine Gale-Shapley Many-to-One Deferred Acceptance (tăng tốc bằng Numba JIT)."""

    _VALID_OPTIMAL = ("candidate", "employer")

    def __init__(
        self,
        profile: PreferenceProfile,
        *,
        optimal: str = "candidate",
        candidate_preferences: dict[Id, tuple[Id, ...]] | None = None,
        job_preferences: dict[Id, tuple[Id, ...]] | None = None,
        quotas: dict[Id, int] | None = None,
    ) -> None:
        if optimal not in self._VALID_OPTIMAL:
            raise ValueError(
                f"optimal phải thuộc {self._VALID_OPTIMAL}, nhận {optimal!r}."
            )

        self.profile = profile
        self.optimal = optimal
        self._custom_prefs = (
            candidate_preferences is not None or job_preferences is not None
        )
        self._candidate_preferences = (
            candidate_preferences
            if candidate_preferences is not None
            else profile.candidate_preferences
        )
        self._job_preferences = (
            job_preferences
            if job_preferences is not None
            else profile.job_preferences
        )
        self._quotas = dict(quotas) if quotas is not None else dict(profile.quotas)

        self._build_rank_tables()

    # -- Giai đoạn 2: Rank Lookup Table & Ma trận int32[:, :] --------------

    def _build_rank_tables(self) -> None:
        """Dựng bảng thứ hạng :math:`\\mathcal{O}(1)` dưới dạng mảng ``int32[:, :]``."""
        cand_ids = self.profile.candidate_ids
        job_ids = self.profile.job_ids
        n_cands = len(cand_ids)
        n_jobs = len(job_ids)

        self._quotas_arr = np.array(
            [max(0, int(self._quotas.get(j_id, 0))) for j_id in job_ids],
            dtype=np.int32,
        )

        cache_key = "_numba_rank_tables"
        if (
            not self._custom_prefs
            and isinstance(self.profile.metadata, dict)
            and cache_key in self.profile.metadata
        ):
            cached = self.profile.metadata[cache_key]
            (
                self._cand_to_idx,
                self._job_to_idx,
                self._cand_prefs_arr,
                self._cand_pref_lens,
                self._job_prefs_arr,
                self._job_pref_lens,
                self._job_rank_arr,
                self._cand_rank_arr,
            ) = cached  # type: ignore[misc]
            return

        cand_to_idx: dict[Id, int] = {c: i for i, c in enumerate(cand_ids)}
        job_to_idx: dict[Id, int] = {j: i for i, j in enumerate(job_ids)}
        self._cand_to_idx = cand_to_idx
        self._job_to_idx = job_to_idx

        max_c_len = max(
            (len(self._candidate_preferences.get(c, ())) for c in cand_ids),
            default=0,
        )
        max_j_len = max(
            (len(self._job_preferences.get(j, ())) for j in job_ids),
            default=0,
        )

        cand_prefs_arr = np.full((n_cands, max(1, max_c_len)), -1, dtype=np.int32)
        cand_pref_lens = np.zeros(n_cands, dtype=np.int32)
        job_prefs_arr = np.full((n_jobs, max(1, max_j_len)), -1, dtype=np.int32)
        job_pref_lens = np.zeros(n_jobs, dtype=np.int32)

        job_rank_arr = np.full((n_jobs, max(1, n_cands)), -1, dtype=np.int32)
        cand_rank_arr = np.full((n_cands, max(1, n_jobs)), -1, dtype=np.int32)

        max_dim = max(1, max_c_len, max_j_len)
        ranks_seq = np.arange(max_dim, dtype=np.int32)

        for c_idx, c_id in enumerate(cand_ids):
            prefs = self._candidate_preferences.get(c_id, ())
            if not prefs:
                continue
            row = [job_to_idx[j_id] for j_id in prefs if j_id in job_to_idx]
            k = len(row)
            if k:
                row_arr = np.asarray(row, dtype=np.int32)
                cand_prefs_arr[c_idx, :k] = row_arr
                cand_pref_lens[c_idx] = k
                cand_rank_arr[c_idx, row_arr] = ranks_seq[:k]

        for j_idx, j_id in enumerate(job_ids):
            prefs = self._job_preferences.get(j_id, ())
            if not prefs:
                continue
            row = [cand_to_idx[c_id] for c_id in prefs if c_id in cand_to_idx]
            k = len(row)
            if k:
                row_arr = np.asarray(row, dtype=np.int32)
                job_prefs_arr[j_idx, :k] = row_arr
                job_pref_lens[j_idx] = k
                job_rank_arr[j_idx, row_arr] = ranks_seq[:k]

        self._cand_prefs_arr = cand_prefs_arr
        self._cand_pref_lens = cand_pref_lens
        self._job_prefs_arr = job_prefs_arr
        self._job_pref_lens = job_pref_lens
        self._job_rank_arr = job_rank_arr
        self._cand_rank_arr = cand_rank_arr

        if not self._custom_prefs and isinstance(self.profile.metadata, dict):
            self.profile.metadata[cache_key] = (
                cand_to_idx,
                job_to_idx,
                cand_prefs_arr,
                cand_pref_lens,
                job_prefs_arr,
                job_pref_lens,
                job_rank_arr,
                cand_rank_arr,
            )

    def job_rank(self, job: Id, candidate: Id) -> int | None:
        """Tra cứu ``O(1)`` hạng của ứng viên trong mắt vị trí (``None`` nếu kém)."""
        j_idx = self._job_to_idx.get(job)
        c_idx = self._cand_to_idx.get(candidate)
        if j_idx is None or c_idx is None:
            return None
        r = int(self._job_rank_arr[j_idx, c_idx])
        return r if r >= 0 else None

    def candidate_rank(self, candidate: Id, job: Id) -> int | None:
        """Tra cứu ``O(1)`` hạng của vị trí trong mắt ứng viên."""
        c_idx = self._cand_to_idx.get(candidate)
        j_idx = self._job_to_idx.get(job)
        if c_idx is None or j_idx is None:
            return None
        r = int(self._cand_rank_arr[c_idx, j_idx])
        return r if r >= 0 else None

    # -- Giai đoạn 3: Deferred Acceptance (Numba JIT) ----------------------

    def solve(self) -> MatchingResult:
        """Chạy thuật toán Deferred Acceptance theo chế độ ``optimal``."""

        if self.optimal == "candidate":
            return self._solve_candidate_proposing()
        return self._solve_employer_proposing()

    def _solve_candidate_proposing(self) -> MatchingResult:
        """Candidate-Proposing DA (Many-to-One) - biên dịch mã máy Numba JIT."""
        cand_ids = self.profile.candidate_ids
        job_ids = self.profile.job_ids
        if not cand_ids or not job_ids:
            return self._assemble({}, 0, optimal="candidate")

        matches_arr, proposals = _gs_candidate_proposing_numba(
            self._cand_prefs_arr,
            self._cand_pref_lens,
            self._job_rank_arr,
            self._quotas_arr,
        )

        matches: dict[Id, Id] = {}
        for c_idx, j_idx in enumerate(matches_arr):
            if j_idx >= 0:
                matches[cand_ids[c_idx]] = job_ids[int(j_idx)]

        return self._assemble(matches, int(proposals), optimal="candidate")

    def _solve_employer_proposing(self) -> MatchingResult:
        """Employer-Proposing DA (Many-to-One) - biên dịch mã máy Numba JIT."""
        cand_ids = self.profile.candidate_ids
        job_ids = self.profile.job_ids
        if not cand_ids or not job_ids:
            return self._assemble({}, 0, optimal="employer")

        initial_free = np.flatnonzero(self._job_pref_lens > 0).astype(np.int32)

        holder_arr, proposals = _gs_employer_proposing_numba(
            self._job_prefs_arr,
            self._job_pref_lens,
            self._cand_rank_arr,
            self._quotas_arr,
            initial_free,
        )

        matches: dict[Id, Id] = {}
        for c_idx, j_idx in enumerate(holder_arr):
            if j_idx >= 0:
                matches[cand_ids[c_idx]] = job_ids[int(j_idx)]

        return self._assemble(matches, int(proposals), optimal="employer")

    # -- Kết xuất kết quả --------------------------------------------------

    def _assemble(
        self,
        matches: dict[Id, Id],
        proposals: int,
        *,
        optimal: str,
    ) -> MatchingResult:
        """Chuyển ``matches`` nội bộ thành ``MatchingResult`` bất biến theo thứ tự."""

        job_assignments: dict[Id, list[Id]] = {
            job: [] for job in self.profile.job_ids
        }
        for job, candidates in self._assignment_view(matches).items():
            job_assignments[job] = candidates

        unmatched_candidates = [
            candidate
            for candidate in self.profile.candidate_ids
            if candidate not in matches
        ]
        unmatched_jobs = [
            job
            for job in self.profile.job_ids
            if len(job_assignments[job]) < self._quotas.get(job, 0)
        ]

        ordered_matches = {
            candidate: matches[candidate]
            for candidate in self.profile.candidate_ids
            if candidate in matches
        }

        return MatchingResult(
            matches=ordered_matches,
            job_assignments=job_assignments,
            unmatched_candidates=unmatched_candidates,
            unmatched_jobs=unmatched_jobs,
            proposals=proposals,
            optimal=optimal,
            is_stable=None,
            metadata={
                "matched_pairs": len(ordered_matches),
                "total_quota": self.profile.total_quota(),
            },
            _profile=self.profile,
        )

    def _assignment_view(self, matches: dict[Id, Id]) -> dict[Id, list[Id]]:
        """Đảo ``matches`` thành ``{job: [candidate, ...]}`` theo thứ tự tuyên bố."""

        grouped: dict[Id, list[Id]] = {job: [] for job in self.profile.job_ids}
        for candidate, job in matches.items():
            grouped.setdefault(job, []).append(candidate)
        for job, candidates in grouped.items():
            j_idx = self._job_to_idx.get(job)
            if j_idx is not None and len(candidates) > 1:
                candidates.sort(
                    key=lambda cand: (
                        int(self._job_rank_arr[j_idx, self._cand_to_idx[cand]])
                        if cand in self._cand_to_idx
                        and self._job_rank_arr[j_idx, self._cand_to_idx[cand]] >= 0
                        else (1 << 30),
                        str(cand),
                    )
                )
        return grouped
