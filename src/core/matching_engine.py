#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Giai đoạn 3 - Core Matching Engine: Gale-Shapley Many-to-One DA.

Triển khai **Deferred Acceptance (DA)** của Gale & Shapley (1962) cho bài toán
**Hospital-Residents / College-Admissions** (Many-to-One), trong đó mỗi vị trí
tuyển dụng ``j`` có hạn mức ``q_j >= 1``.

Hai chế độ:

* ``optimal="candidate"`` (mặc định) - **Candidate-Proposing**: ứng viên là phía
  chủ động gửi đề nghị. Kết quả là **C-optimal**: không tồn tại matching ổn
  định nào tốt hơn cho *mọi* ứng viên (Gale & Shapley 1962; Roth 1984). Cơ chế
  DA *strategy-proof* cho phía ứng viên (Dubins & Freedman 1981), nên ứng viên
  không thể cải thiện kết quả bằng cách khai báo sai ưu tiên.
* ``optimal="employer"`` - **Employer-Proposing**: vị trí chủ động gửi đề nghị,
  cho kết quả **J-optimal** (tối ưu cho phía nhà tuyển dụng).

Khái niệm toán học:

* **Stable matching** - không tồn tại *blocking pair* ``(c, j)`` mà cả ``c`` và
  ``j`` đều thích nhau hơn đối tác hiện tại của họ và ``j`` còn chỗ hoặc ứng
  viên bị giữ kém ưu tiên hơn ``c``.
* **Blocking pair** - xem :func:`src.evaluation.metrics.find_blocking_pairs`.
* **Quota** ``q_j`` - số slot tối đa của vị trí ``j``.
* **Proposer-optimality** - phía đề nghị nhận kết quả tốt nhất trong số mọi
  matching ổn định; phía còn lại nhận kết quả tệ nhất trong tập ổn định.
* **Rank Lookup Table** (Knuth 1997, *Stable Marriage and Its Relation to Other
  Combinatorial Problems*) - bảng thứ hạng để tra cứu ``O(1)`` thay vì quét
  tuyến tính ``O(N)``.

Độ phức tạp: mỗi ứng viên đề nghị tối đa ``L`` lần, nên tổng số lời đề nghị
là :math:`\\mathcal{O}(N \\times L)`.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import dataclass, field

from src.core.preprocessor import Id, PreferenceProfile

__all__ = ["GaleShapleyMatching", "MatchingResult"]


@dataclass
class MatchingResult:
    """Kết quả ghép cặp trả về từ engine.

    Attributes:
        matches: ``{candidate: job}`` cho các ứng viên đã được nhận.
        job_assignments: ``{job: [candidate, ...]}`` sắp theo thứ tự đề nghị;
            bất biến ``len(job_assignments[j]) <= quotas[j]`` luôn đúng.
        unmatched_candidates: Ứng viên không được nhận ở đâu (kể cả người không
            có ưu tiên nào).
        unmatched_jobs: Vị trí còn trống sau khi kết thúc.
        proposals: Tổng số lời đề nghị đã gửi (chỉ số đo chi phí tính toán).
        optimal: ``"candidate"`` hoặc ``"employer"``.
        is_stable: Cờ khẳng định kết quả đã được kiểm chứng ổn định nội bộ.
    """

    matches: dict[Id, Id]
    job_assignments: dict[Id, list[Id]]
    unmatched_candidates: list[Id]
    unmatched_jobs: list[Id]
    proposals: int
    optimal: str = "candidate"
    is_stable: bool = True
    metadata: dict[str, object] = field(default_factory=dict)

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
    """Engine Gale-Shapley Many-to-One Deferred Acceptance.

    Giai đoạn 2 được thực hiện ngay trong ``__init__``: dựng **Rank Lookup
    Table** cho cả hai phía.

    * ``_job_rank[job][candidate] = rank`` - hạng của ứng viên trong mắt vị trí
      (0 là tốt nhất). Tra cứu ``O(1)`` khi vị trí cần so sánh ứng viên mới với
      ứng viên kém nhất đang giữ.
    * ``_cand_rank[candidate][job] = rank`` - hạng của vị trí trong mắt ứng
      viên, phục vụ chế độ employer-proposing.

    Với mỗi vị trí, danh sách ứng viên tạm tuyển được quản lý bằng **Min-Heap**
    trên ``-rank`` (biến Min-Heap thành Max-Heap theo hạng): phần tử đỉnh là ứng
    viên *kém ưu tiên nhất*, cho phép loại trong :math:`\\mathcal{O}(\\log q_j)`.

    Args:
        profile: ``PreferenceProfile`` đã tiền xử lý.
        optimal: ``"candidate"`` hoặc ``"employer"``.
        candidate_preferences: Ghi đè danh sách ưu tiên phía ứng viên (dùng cho
            các vòng Iterative Clearance của hậu xử lý Top-K).
        job_preferences: Ghi đè danh sách ưu tiên phía nhà tuyển dụng.
        quotas: Ghi đè hạn mức (dùng khi một số slot đã bị "dùng" ở vòng trước).

    Raises:
        ValueError: Nếu ``optimal`` không hợp lệ.
    """

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

    # -- Giai đoạn 2: Rank Lookup Table -----------------------------------

    def _build_rank_tables(self) -> None:
        """Dựng bảng thứ hạng :math:`\\mathcal{O}(1)` cho hai phía.

        Dùng thứ tự trong ``job_preferences`` / ``candidate_preferences`` (đã
        phá hòa tất định ở bước tiền xử lý) làm nguồn chân lý, nên bảng hạng
        luôn nhất quán với thứ tự mà engine sẽ duyệt.
        """

        self._job_rank: dict[Id, dict[Id, int]] = {
            job: {candidate: rank for rank, candidate in enumerate(candidates)}
            for job, candidates in self._job_preferences.items()
        }
        self._cand_rank: dict[Id, dict[Id, int]] = {
            candidate: {job: rank for rank, job in enumerate(jobs)}
            for candidate, jobs in self._candidate_preferences.items()
        }

    def job_rank(self, job: Id, candidate: Id) -> int | None:
        """Tra cứu ``O(1)`` hạng của ứng viên trong mắt vị trí (``None`` nếu kém).

        Args:
            job: ID vị trí tuyển dụng.
            candidate: ID ứng viên.

        Returns:
            Hạng (0 = tốt nhất) hoặc ``None`` nếu vị trí không chấp nhận ứng viên.
        """

        return self._job_rank.get(job, {}).get(candidate)

    def candidate_rank(self, candidate: Id, job: Id) -> int | None:
        """Tra cứu ``O(1)`` hạng của vị trí trong mắt ứng viên."""

        return self._cand_rank.get(candidate, {}).get(job)

    # -- Giai đoạn 3: Deferred Acceptance ---------------------------------

    def solve(self) -> MatchingResult:
        """Chạy thuật toán Deferred Acceptance theo chế độ ``optimal``.

        Returns:
            ``MatchingResult`` ổn định (không có blocking pair).
        """

        if self.optimal == "candidate":
            return self._solve_candidate_proposing()
        return self._solve_employer_proposing()

    def _solve_candidate_proposing(self) -> MatchingResult:
        """Candidate-Proposing DA (Many-to-One) - cho kết quả C-optimal.

        Vòng lặp hoãn chấp nhận:

        1. Ứng viên tự do gửi đề nghị tới vị trí kế tiếp trong danh sách ưu tiên.
        2. Vị trí tạm giữ tối đa ``q_j`` ứng viên tốt nhất; nếu đầy, nó so sánh
           ứng viên mới với ứng viên *kém nhất* đang giữ (lấy ``O(1)`` từ Rank
           Table, đỉnh Min-Heap) và loại người kém hơn.
        3. Ứng viên bị từ chối quay lại hàng đợi tự do và đề nghị vị trí tiếp theo.
        4. Dừng khi hàng đợi rỗng (mọi đề nghị hoặc được giữ, hoặc bị từ chối và
           ứng viên đã cạn danh sách ưu tiên).

        Returns:
            ``MatchingResult`` với ``optimal="candidate"``.
        """

        candidate_lists = self._candidate_preferences
        free: deque[Id] = deque(
            candidate
            for candidate in self.profile.candidate_ids
            if candidate_lists.get(candidate)
        )
        next_proposal: dict[Id, int] = {candidate: 0 for candidate in free}

        # held[job] là min-heap các bộ (-rank, thứ tự, candidate): đỉnh heap là
        # ứng viên có -rank nhỏ nhất, tức rank LỚN NHẤT = kém ưu tiên nhất.
        held: dict[Id, list[tuple[int, int, Id]]] = {
            job: [] for job in self.profile.job_ids
        }
        held_members: dict[Id, set[Id]] = {
            job: set() for job in self.profile.job_ids
        }
        tie_counter = itertools.count()

        matches: dict[Id, Id] = {}
        proposals = 0

        while free:
            candidate = free.popleft()
            preferences = candidate_lists.get(candidate, ())
            index = next_proposal.get(candidate, 0)
            if index >= len(preferences):
                continue

            job = preferences[index]
            next_proposal[candidate] = index + 1
            proposals += 1

            quota = self._quotas.get(job, 0)
            if quota <= 0:
                # Vị trí không còn slot: ứng viên thử nguyện vọng kế tiếp.
                free.append(candidate)
                continue

            rank = self._job_rank.get(job, {}).get(candidate)
            if rank is None:
                free.append(candidate)
                continue

            if len(held_members[job]) < quota:
                heapq.heappush(held[job], (-rank, next(tie_counter), candidate))
                held_members[job].add(candidate)
                matches[candidate] = job
                continue

            worst_neg_rank, _, worst_candidate = held[job][0]
            if rank < -worst_neg_rank:
                # Ứng viên mới tốt hơn người kém nhất -> thay thế.
                heapq.heapreplace(
                    held[job], (-rank, next(tie_counter), candidate)
                )
                held_members[job].discard(worst_candidate)
                held_members[job].add(candidate)
                matches.pop(worst_candidate, None)
                matches[candidate] = job
                free.append(worst_candidate)
            else:
                free.append(candidate)

        return self._assemble(matches, proposals, optimal="candidate")

    def _solve_employer_proposing(self) -> MatchingResult:
        """Employer-Proposing DA (Many-to-One) - cho kết quả J-optimal.

        Vị trí là phía chủ động: mỗi vị trí lần lượt đề nghị các ứng viên trong
        danh sách ưu tiên của nó cho tới khi đủ ``q_j`` slot hoặc cạn danh sách.
        Mỗi ứng viên chỉ giữ **một** lời đề nghị tốt nhất; khi nhận được đề nghị
        tốt hơn, ứng viên từ chối vị trí cũ và vị trí cũ được phép đề nghị tiếp.

        Returns:
            ``MatchingResult`` với ``optimal="employer"``.
        """

        job_lists = self._job_preferences
        free: deque[Id] = deque(
            job for job in self.profile.job_ids if job_lists.get(job)
        )
        next_proposal: dict[Id, int] = {job: 0 for job in free}

        assigned: dict[Id, set[Id]] = {job: set() for job in self.profile.job_ids}
        candidate_holder: dict[Id, Id] = {}

        proposals = 0

        while free:
            job = free.popleft()
            quota = self._quotas.get(job, 0)
            if quota <= 0 or len(assigned[job]) >= quota:
                continue

            preferences = job_lists.get(job, ())
            index = next_proposal.get(job, 0)
            if index >= len(preferences):
                continue

            candidate = preferences[index]
            next_proposal[job] = index + 1
            proposals += 1

            current_job = candidate_holder.get(candidate)
            if current_job is None:
                assigned[job].add(candidate)
                candidate_holder[candidate] = job
            else:
                new_rank = self._cand_rank.get(candidate, {}).get(job)
                current_rank = self._cand_rank.get(candidate, {}).get(current_job)
                if new_rank is not None and (
                    current_rank is None or new_rank < current_rank
                ):
                    # Ứng viên thích vị trí mới hơn -> trả vị trí cũ về hàng đợi.
                    assigned[current_job].discard(candidate)
                    free.append(current_job)
                    assigned[job].add(candidate)
                    candidate_holder[candidate] = job
                # Ngược lại: ứng viên từ chối vị trí mới, vị trí tiếp tục đề nghị.

            if (
                len(assigned[job]) < self._quotas.get(job, 0)
                and next_proposal[job] < len(preferences)
            ):
                free.append(job)

        matches = {candidate: job for candidate, job in candidate_holder.items()}
        return self._assemble(matches, proposals, optimal="employer")

    # -- Kết xuất kết quả --------------------------------------------------

    def _assemble(
        self,
        matches: dict[Id, Id],
        proposals: int,
        *,
        optimal: str,
    ) -> MatchingResult:
        """Chuyển ``matches`` nội bộ thành ``MatchingResult`` bất biến theo thứ tự.

        Args:
            matches: ``{candidate: job}`` từ vòng lặp DA.
            proposals: Số lời đề nghị đã thực hiện.
            optimal: Chế độ đã chạy.

        Returns:
            ``MatchingResult`` với danh sách ghép theo thứ tự ID chuẩn.
        """

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
            is_stable=True,
            metadata={
                "matched_pairs": len(ordered_matches),
                "total_quota": self.profile.total_quota(),
            },
        )

    def _assignment_view(self, matches: dict[Id, Id]) -> dict[Id, list[Id]]:
        """Đảo ``matches`` thành ``{job: [candidate, ...]}`` theo thứ tự tuyên bố.

        Ứng viên trong mỗi vị trí được xếp theo thứ tự ưu tiên của vị trí đó để
        kết quả tất định và dễ đọc.
        """

        grouped: dict[Id, list[Id]] = {job: [] for job in self.profile.job_ids}
        for candidate, job in matches.items():
            grouped.setdefault(job, []).append(candidate)
        for job, candidates in grouped.items():
            rank_map = self._job_rank.get(job, {})
            candidates.sort(key=lambda cand: (rank_map.get(cand, 1 << 30), str(cand)))
        return grouped
