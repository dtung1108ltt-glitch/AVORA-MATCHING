#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Giai đoạn 1 & 2 - Tiền xử lý và khởi tạo cấu trúc dữ liệu tối ưu.

Module này nhận dữ liệu thô của thị trường hai phía và biến nó thành một
``PreferenceProfile`` chuẩn mà engine Gale-Shapley có thể tiêu thụ trực tiếp.

Ma trận đầu vào (đều được đánh chỉ số ``[candidate][job]``):

* ``E[c][j]`` - **Employer Preference Score**: mức độ phù hợp kỹ năng của
  ứng viên ``c`` trong mắt vị trí ``j`` (điểm càng cao, ``j`` càng muốn ``c``).
* ``C[c][j]`` - **Candidate Preference Score**: mức độ yêu thích công việc
  ``j`` của ứng viên ``c`` (điểm càng cao, ``c`` càng muốn ``j``).
* ``Quota q_j`` - hạn mức tuyển dụng của vị trí ``j`` (Many-to-One).

Ba công việc của module:

1. **Chuẩn hóa** ma trận điểm (dense ``list[list[float]]`` hoặc thưa
   ``dict[candidate][job]``) về một cấu trúc thưa thống nhất.
2. **Lọc Hard Constraints**: loại bỏ vĩnh viễn các cặp vi phạm điều kiện bắt
   buộc (điểm bằng 0, không nằm trong tập cặp khả thi, dưới ngưỡng sàn).
3. **Truncated Preference Lists**: mỗi ứng viên chỉ giữ tối đa ``L`` công việc
   tiềm năng nhất, mỗi vị trí giữ tối đa ``M`` ứng viên tốt nhất - đưa độ phức
   tạp về :math:`\\mathcal{O}(N \\times L)`.

Thuật ngữ toán học:

* **Acceptable pair** - cặp ``(c, j)`` mà cả hai phía đều chấp nhận được
  (tồn tại trong danh sách ưu tiên của nhau sau khi lọc Hard Constraints).
* **Preference list** - dãy các đối tác tiềm năng xếp theo thứ tự ưu tiên
  giảm dần, phá hòa bằng hàm băm tất định (deterministic tie-breaking).
* **Truncation** - cắt bớt đuôi danh sách ưu tiên; cặp bị cắt coi như không
  chấp nhận được, đánh đổi tính tối ưu lấy tốc độ.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

#: Kiểu định danh dùng cho ứng viên và vị trí tuyển dụng.
Id: TypeAlias = Hashable

#: Ma trận điểm thưa, luôn đánh chỉ số ``[candidate][job]``.
ScoreMatrix: TypeAlias = dict[Id, dict[Id, float]]


def stable_hash(*parts: object, seed: int = 0) -> int:
    """Sinh số băm tất định, ổn định giữa các lần chạy và các process.

    Dùng để phá hòa (tie-breaking) khi hai cặp có cùng điểm. Nhờ dùng
    ``hashlib`` (không phải ``hash()`` nội suy), kết quả không phụ thuộc vào
    ``PYTHONHASHSEED`` nên matching luôn tái lập được.

    Args:
        *parts: Các thành phần tham gia băm (thường là ``(candidate_id,
            job_id)`` cùng một nhãn phân biệt hướng ưu tiên).
        seed: Hạt giống cấu hình, cho phép đổi thứ tự phá hòa theo nghiệp vụ.

    Returns:
        Số nguyên 64-bit dùng làm khóa phụ khi sắp xếp.
    """

    payload = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def normalize_score_matrix(
    raw: Mapping[Id, Mapping[Id, float]] | Sequence[Sequence[float]],
    *,
    candidate_ids: Iterable[Id] | None = None,
    job_ids: Iterable[Id] | None = None,
) -> ScoreMatrix:
    """Chuẩn hóa ma trận điểm thô về cấu trúc thưa ``dict[candidate][job]``.

    Hỗ trợ hai định dạng đầu vào:

    * **Thưa (sparse)**: ``{candidate_id: {job_id: score, ...}, ...}``.
      Cặp không có mặt được hiểu là *không chấp nhận được* (không phải điểm 0).
    * **Đặc (dense)**: ``[[score, ...], ...]`` với hàng theo ứng viên, cột theo
      vị trí; khi đó ``candidate_ids`` / ``job_ids`` cung cấp nhãn tương ứng.

    Args:
        raw: Ma trận điểm thô.
        candidate_ids: Nhãn ứng viên, bắt buộc khi ``raw`` là list-of-list và
            muốn dùng ID nghiệp vụ thay vì chỉ số 0..N-1.
        job_ids: Nhãn vị trí tuyển dụng, tương tự.

    Returns:
        Ma trận thưa ``dict[candidate][job] -> float``.

    Raises:
        TypeError: Nếu một hàng có kiểu không hợp lệ.
        ValueError: Nếu kích thước hàng không khớp số nhãn công việc.
    """

    if isinstance(raw, Mapping):
        matrix: ScoreMatrix = {}
        for candidate, row in raw.items():
            if not isinstance(row, Mapping):
                raise TypeError(
                    "Mỗi hàng của ma trận thưa phải là Mapping[job] -> score."
                )
            matrix[candidate] = {job: float(score) for job, score in row.items()}
        return matrix

    rows = list(raw)
    labels: list[Id] = (
        list(candidate_ids) if candidate_ids is not None else list(range(len(rows)))
    )
    if len(labels) != len(rows):
        raise ValueError(
            f"candidate_ids có {len(labels)} phần tử nhưng ma trận có {len(rows)} hàng."
        )

    if job_ids is not None:
        columns: list[Id] = list(job_ids)
    else:
        width = max((len(row) for row in rows), default=0)
        columns = list(range(width))

    matrix = {}
    for candidate, row in zip(labels, rows, strict=True):
        if len(row) != len(columns):
            raise ValueError(
                f"Hàng của {candidate!r} có {len(row)} cột, cần {len(columns)}."
            )
        matrix[candidate] = {
            job: float(score)
            for job, score in zip(columns, row, strict=True)
        }
    return matrix


def apply_hard_constraints(
    e_scores: ScoreMatrix,
    c_scores: ScoreMatrix,
    *,
    feasible_pairs: set[tuple[Id, Id]] | None = None,
    min_e_score: float = 0.0,
    min_c_score: float = 0.0,
    require_positive: bool = True,
) -> tuple[ScoreMatrix, ScoreMatrix]:
    """Loại bỏ hoàn toàn các cặp vi phạm điều kiện bắt buộc.

    Một cặp ``(c, j)`` bị loại nếu rơi vào bất kỳ trường hợp nào sau đây:

    * Không nằm trong ``feasible_pairs`` (ví dụ: khoảng cách địa lý vượt
      ngưỡng, lương không đạt sàn, sai nhóm kỹ năng bắt buộc).
    * Điểm ``E[c][j]`` hoặc ``C[c][j]`` không dương khi ``require_positive``.
    * Điểm thấp hơn ngưỡng sàn ``min_e_score`` / ``min_c_score``.

    Cặp bị loại sẽ không xuất hiện trong danh sách ưu tiên của bất kỳ phía nào,
    nên không thể trở thành blocking pair.

    Args:
        e_scores: Ma trận điểm nhà tuyển dụng.
        c_scores: Ma trận điểm ứng viên.
        feasible_pairs: Tập cặp ``(candidate, job)`` được phép; ``None`` nghĩa là
            mọi cặp có trong ma trận đều khả thi.
        min_e_score: Ngưỡng sàn cho điểm ``E``.
        min_c_score: Ngưỡng sàn cho điểm ``C``.
        require_positive: Yêu cầu cả hai điểm phải lớn hơn 0.

    Returns:
        Cặp ``(e_filtered, c_filtered)`` chỉ chứa các cặp khả thi, cùng tập ID
        hàng như đầu vào (nhưng mỗi hàng có thể rỗng).
    """

    e_out: ScoreMatrix = {}
    c_out: ScoreMatrix = {}

    for candidate, e_row in e_scores.items():
        c_row = c_scores.get(candidate, {})
        new_e: dict[Id, float] = {}
        new_c: dict[Id, float] = {}
        for job, e_value in e_row.items():
            if job not in c_row:
                continue
            c_value = c_row[job]
            if feasible_pairs is not None and (candidate, job) not in feasible_pairs:
                continue
            if require_positive and (e_value <= 0.0 or c_value <= 0.0):
                continue
            if e_value < min_e_score or c_value < min_c_score:
                continue
            new_e[job] = e_value
            new_c[job] = c_value
        e_out[candidate] = new_e
        c_out[candidate] = new_c

    # Giữ cả ứng viên chỉ xuất hiện trong c_scores nhưng không có trong e_scores.
    for candidate in c_scores:
        if candidate not in e_out:
            e_out[candidate] = {}
            c_out[candidate] = {}

    return e_out, c_out


def truncate_preferences(
    scores: ScoreMatrix,
    top_k: int | None,
    *,
    by: str = "candidate",
    seed: int = 0,
) -> ScoreMatrix:
    """Cắt ngắn danh sách ưu tiên (Truncated Preference Lists).

    Args:
        scores: Ma trận thưa ``[candidate][job]`` đã qua Hard Constraints.
        top_k: Số đối tác tối đa giữ lại cho mỗi hàng (``L`` hoặc ``M``).
            ``None`` hoặc ``<= 0`` nghĩa là không cắt.
        by: ``"candidate"`` để giới hạn mỗi ứng viên giữ tối đa ``top_k`` công
            việc; ``"job"`` để giới hạn mỗi vị trí giữ tối đa ``top_k`` ứng viên.
        seed: Hạt giống phá hòa khi sắp xếp.

    Returns:
        Ma trận đã cắt, cùng không gian khóa nhưng ít cặp hơn.

    Raises:
        ValueError: Nếu ``by`` không hợp lệ.
    """

    if top_k is None or top_k <= 0:
        return {candidate: dict(row) for candidate, row in scores.items()}

    if by == "candidate":
        out: ScoreMatrix = {}
        for candidate, row in scores.items():
            ordered = sorted(
                row.items(),
                key=lambda item: (
                    -item[1],
                    stable_hash("candidate", candidate, item[0], seed=seed),
                    str(item[0]),
                ),
            )
            out[candidate] = dict(ordered[:top_k])
        return out

    if by == "job":
        all_jobs: set[Id] = set()
        for row in scores.values():
            all_jobs.update(row)

        out = {candidate: {} for candidate in scores}
        for job in sorted(all_jobs, key=str):
            column = [
                (candidate, row[job])
                for candidate, row in scores.items()
                if job in row
            ]
            column.sort(
                key=lambda item: (
                    -item[1],
                    stable_hash("job", item[0], job, seed=seed),
                    str(item[0]),
                )
            )
            for candidate, score in column[:top_k]:
                out[candidate][job] = score
        return out

    raise ValueError("by phải là 'candidate' hoặc 'job'.")


@dataclass
class PreprocessConfig:
    """Tham số tiền xử lý.

    Attributes:
        max_candidate_preferences: ``L`` - số công việc tối đa mỗi ứng viên giữ
            lại. Khuyến nghị 30-50 để giữ độ phức tạp :math:`\\mathcal{O}(N
            \\times L)`.
        max_employer_preferences: ``M`` - số ứng viên tối đa mỗi vị trí giữ lại;
            ``None`` nghĩa là không cắt phía nhà tuyển dụng.
        min_e_score: Ngưỡng sàn điểm ``E``.
        min_c_score: Ngưỡng sàn điểm ``C``.
        require_positive_scores: Loại cặp có điểm ``<= 0``.
        tie_break_seed: Hạt giống phá hòa tất định.
    """

    max_candidate_preferences: int = 40
    max_employer_preferences: int | None = None
    min_e_score: float = 0.0
    min_c_score: float = 0.0
    require_positive_scores: bool = True
    tie_break_seed: int = 20_240_517


@dataclass
class PreferenceProfile:
    """Hồ sơ ưu tiên đã tiền xử lý - đầu vào của Gale-Shapley engine.

    Attributes:
        candidate_ids: Thứ tự chuẩn của toàn bộ ứng viên (kể cả người không có
            ưu tiên nào, họ sẽ luôn thất nghiệp).
        job_ids: Thứ tự chuẩn của toàn bộ vị trí tuyển dụng.
        e_scores: Điểm ``E[c][j]`` sau khi lọc (dùng cho satisfaction).
        c_scores: Điểm ``C[c][j]`` sau khi lọc.
        quotas: ``q_j`` - hạn mức của từng vị trí, giá trị nguyên không âm.
        candidate_preferences: ``{c: (j1, j2, ...)}`` đã sắp giảm dần theo ``C``
            và phá hòa tất định.
        job_preferences: ``{j: (c1, c2, ...)}`` đã sắp giảm dần theo ``E``.
        tie_break_seed: Hạt giống phá hòa đã dùng.
    """

    candidate_ids: tuple[Id, ...]
    job_ids: tuple[Id, ...]
    e_scores: ScoreMatrix
    c_scores: ScoreMatrix
    quotas: dict[Id, int]
    candidate_preferences: dict[Id, tuple[Id, ...]]
    job_preferences: dict[Id, tuple[Id, ...]]
    tie_break_seed: int = 20_240_517

    def total_quota(self) -> int:
        """Tổng hạn mức của toàn bộ vị trí tuyển dụng."""

        return sum(max(0, quota) for quota in self.quotas.values())

    def acceptable_jobs(self, candidate: Id) -> tuple[Id, ...]:
        """Danh sách công việc ứng viên chấp nhận được."""

        return self.candidate_preferences.get(candidate, ())

    def acceptable_candidates(self, job: Id) -> tuple[Id, ...]:
        """Danh sách ứng viên vị trí chấp nhận được."""

        return self.job_preferences.get(job, ())

    def feasible_pairs(self) -> set[tuple[Id, Id]]:
        """Tập cặp *acceptable* hai chiều (nền tảng để kiểm tra ổn định)."""

        pairs: set[tuple[Id, Id]] = set()
        job_sets = {
            job: set(candidates) for job, candidates in self.job_preferences.items()
        }
        for candidate, jobs in self.candidate_preferences.items():
            for job in jobs:
                if candidate in job_sets.get(job, ()):
                    pairs.add((candidate, job))
        return pairs

    def preference_count(self) -> int:
        """Tổng số cặp trong danh sách ưu tiên phía ứng viên."""

        return sum(len(jobs) for jobs in self.candidate_preferences.values())


class Preprocessor:
    """Bộ tiền xử lý thị trường hai phía thành ``PreferenceProfile``.

    Example:
        >>> pre = Preprocessor(PreprocessConfig(max_candidate_preferences=30))
        >>> profile = pre.build_profile(
        ...     e_scores={0: {0: 0.9, 1: 0.2}},   # E[candidate][job]
        ...     c_scores={0: {0: 0.4, 1: 0.8}},   # C[candidate][job]
        ...     quotas={0: 1, 1: 1},
        ... )
        >>> profile.candidate_preferences[0]
        (1, 0)
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()

    def build_profile(
        self,
        e_scores: Mapping[Id, Mapping[Id, float]] | Sequence[Sequence[float]],
        c_scores: Mapping[Id, Mapping[Id, float]] | Sequence[Sequence[float]],
        quotas: Mapping[Id, int],
        *,
        feasible_pairs: set[tuple[Id, Id]] | None = None,
        candidate_ids: Iterable[Id] | None = None,
        job_ids: Iterable[Id] | None = None,
    ) -> PreferenceProfile:
        """Chạy đủ pipeline Giai đoạn 1 và 2.

        Args:
            e_scores: Điểm ``E[c][j]`` (thưa hoặc đặc).
            c_scores: Điểm ``C[c][j]`` (thưa hoặc đặc).
            quotas: Hạn mức ``q_j`` theo từng vị trí.
            feasible_pairs: Tập cặp khả thi (Hard Constraints định sẵn).
            candidate_ids: Nhãn ứng viên khi đầu vào là ma trận đặc.
            job_ids: Nhãn công việc khi đầu vào là ma trận đặc.

        Returns:
            ``PreferenceProfile`` sẵn sàng cho ``GaleShapleyMatching``.

        Raises:
            ValueError: Nếu quota âm.
        """

        e_norm = normalize_score_matrix(
            e_scores, candidate_ids=candidate_ids, job_ids=job_ids
        )
        c_norm = normalize_score_matrix(
            c_scores, candidate_ids=candidate_ids, job_ids=job_ids
        )

        e_filtered, c_filtered = apply_hard_constraints(
            e_norm,
            c_norm,
            feasible_pairs=feasible_pairs,
            min_e_score=self.config.min_e_score,
            min_c_score=self.config.min_c_score,
            require_positive=self.config.require_positive_scores,
        )

        # Giai đoạn 1: Truncated Preference Lists.
        c_truncated = truncate_preferences(
            c_filtered,
            self.config.max_candidate_preferences,
            by="candidate",
            seed=self.config.tie_break_seed,
        )
        if self.config.max_employer_preferences is not None:
            e_truncated = truncate_preferences(
                e_filtered,
                self.config.max_employer_preferences,
                by="job",
                seed=self.config.tie_break_seed,
            )
            # Chỉ giữ cặp còn tồn tại ở CẢ HAI phía sau khi cắt.
            e_truncated, c_truncated = apply_hard_constraints(
                e_truncated,
                c_truncated,
                require_positive=False,
            )
        else:
            e_truncated = e_filtered

        # Xác định thứ tự ID chuẩn.
        if candidate_ids is None:
            candidate_universe = set(c_truncated) | set(e_truncated)
            ordered_candidates = tuple(sorted(candidate_universe, key=str))
        else:
            ordered_candidates = tuple(candidate_ids)

        quota_map = self._normalize_quotas(quotas)

        if job_ids is None:
            job_universe: set[Id] = set(quota_map)
            for row in c_truncated.values():
                job_universe.update(row)
            ordered_jobs = tuple(sorted(job_universe, key=str))
        else:
            ordered_jobs = tuple(job_ids)

        quotas_norm: dict[Id, int] = {
            job: max(0, int(quota_map.get(job, 0))) for job in ordered_jobs
        }

        seed = self.config.tie_break_seed
        candidate_preferences: dict[Id, tuple[Id, ...]] = {}
        for candidate in ordered_candidates:
            row = c_truncated.get(candidate, {})
            ordered = sorted(
                row.items(),
                key=lambda item: (
                    -item[1],
                    stable_hash("candidate", candidate, item[0], seed=seed),
                    str(item[0]),
                ),
            )
            candidate_preferences[candidate] = tuple(job for job, _ in ordered)

        job_preferences: dict[Id, tuple[Id, ...]] = {}
        for job in ordered_jobs:
            column: list[tuple[Id, float]] = [
                (candidate, e_truncated[candidate][job])
                for candidate in ordered_candidates
                if job in e_truncated.get(candidate, {})
            ]
            column.sort(
                key=lambda item: (
                    -item[1],
                    stable_hash("job", item[0], job, seed=seed),
                    str(item[0]),
                )
            )
            job_preferences[job] = tuple(candidate for candidate, _ in column)

        return PreferenceProfile(
            candidate_ids=ordered_candidates,
            job_ids=ordered_jobs,
            e_scores=e_truncated,
            c_scores=c_truncated,
            quotas=quotas_norm,
            candidate_preferences=candidate_preferences,
            job_preferences=job_preferences,
            tie_break_seed=seed,
        )

    @staticmethod
    def _normalize_quotas(quotas: Mapping[Id, int]) -> dict[Id, int]:
        """Chuẩn hóa quota về ``dict[job] -> int``.

        Args:
            quotas: Mapping quota theo từng vị trí.

        Returns:
            Mapping quota đã ép kiểu ``int``.

        Raises:
            TypeError: Nếu ``quotas`` không phải Mapping.
            ValueError: Nếu có quota âm.
        """

        if not isinstance(quotas, Mapping):
            raise TypeError("quotas phải là Mapping[job, int].")

        normalized = {job: int(value) for job, value in quotas.items()}
        negative = [job for job, value in normalized.items() if value < 0]
        if negative:
            raise ValueError(f"Quota không được âm: {negative}")
        return normalized


# ---------------------------------------------------------------------------
# Adapter kế thừa dữ liệu CSV hiện có của repo
# ---------------------------------------------------------------------------

CANDIDATE_NAME_COLUMN = "Họ và tên"
CANDIDATE_GROUP_COLUMN = "Nhóm khuyết tật"
JOB_GROUP_COLUMN = "Nhóm khuyết tật"
JOB_DESCRIPTION_COLUMN = "Mô tả công việc"


def load_market_from_csv(
    candidates_csv: str | Path,
    job_descriptions_csv: str | Path,
    *,
    default_quota: int = 5,
    seed: int = 42,
    candidate_limit: int | None = None,
    job_limit: int | None = None,
) -> tuple[ScoreMatrix, ScoreMatrix, dict[Id, int], set[tuple[Id, Id]]]:
    """Kế thừa dữ liệu CSV hiện có để tạo ma trận ``E``, ``C`` và quota.

    File ``candidates.csv`` có cột *Họ và tên*, *Nhóm khuyết tật*; file
    ``job_descriptions.csv`` có cột *Nhóm khuyết tật*, *Mô tả công việc*.

    Vì dữ liệu hiện tại **chưa có điểm số thực**, adapter sinh điểm mô phỏng
    tất định (deterministic) từ hàm băm để phục vụ benchmark; khi lên
    production, ``E`` sẽ đến từ pipeline embedding kỹ năng và ``C`` từ tín hiệu
    sở thích của ứng viên. Hard Constraint bắt buộc: nhóm khuyết tật của ứng
    viên phải trùng nhóm gắn với mô tả công việc.

    Args:
        candidates_csv: Đường dẫn ``candidates.csv``.
        job_descriptions_csv: Đường dẫn ``job_descriptions.csv``.
        default_quota: Hạn mức mặc định cho mỗi dòng mô tả công việc.
        seed: Hạt giống sinh điểm mô phỏng.
        candidate_limit: Giới hạn số ứng viên đọc vào (``None`` = tất cả).
        job_limit: Giới hạn số dòng mô tả công việc đọc vào.

    Returns:
        ``(e_scores, c_scores, quotas, feasible_pairs)`` sẵn sàng truyền vào
        ``Preprocessor.build_profile``.

    Raises:
        FileNotFoundError: Nếu thiếu file CSV.
        ValueError: Nếu thiếu cột bắt buộc.
    """

    candidates_path = Path(candidates_csv)
    jobs_path = Path(job_descriptions_csv)
    if not candidates_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file ứng viên: {candidates_path}")
    if not jobs_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file mô tả công việc: {jobs_path}")

    candidates: list[tuple[Id, str]] = []
    with candidates_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if CANDIDATE_GROUP_COLUMN not in headers:
            raise ValueError(
                f"CSV ứng viên thiếu cột '{CANDIDATE_GROUP_COLUMN}': {headers}"
            )
        name_key = (
            CANDIDATE_NAME_COLUMN
            if CANDIDATE_NAME_COLUMN in headers
            else headers[0]
        )
        for index, row in enumerate(reader):
            if candidate_limit is not None and index >= candidate_limit:
                break
            group = (row.get(CANDIDATE_GROUP_COLUMN) or "").strip()
            if not group:
                continue
            candidates.append((f"C{index}", group))

    jobs: list[tuple[Id, str]] = []
    with jobs_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if JOB_GROUP_COLUMN not in headers:
            raise ValueError(
                f"CSV mô tả công việc thiếu cột '{JOB_GROUP_COLUMN}': {headers}"
            )
        for index, row in enumerate(reader):
            if job_limit is not None and index >= job_limit:
                break
            group = (row.get(JOB_GROUP_COLUMN) or "").strip()
            if not group:
                continue
            jobs.append((f"J{index}", group))

    e_scores: ScoreMatrix = {}
    c_scores: ScoreMatrix = {}
    feasible_pairs: set[tuple[Id, Id]] = set()

    for candidate, candidate_group in candidates:
        e_row: dict[Id, float] = {}
        c_row: dict[Id, float] = {}
        for job, job_group in jobs:
            if candidate_group != job_group:
                continue
            e_row[job] = _mock_score("E", candidate, job, seed)
            c_row[job] = _mock_score("C", candidate, job, seed)
            feasible_pairs.add((candidate, job))
        e_scores[candidate] = e_row
        c_scores[candidate] = c_row

    quotas = {job: default_quota for job, _ in jobs}
    return e_scores, c_scores, quotas, feasible_pairs


def _mock_score(channel: str, candidate: Id, job: Id, seed: int) -> float:
    """Sinh điểm mô phỏng trong khoảng ``(0.05, 1.0]`` một cách tất định."""

    raw = stable_hash(channel, candidate, job, seed=seed)
    return 0.05 + (raw % 9500) / 10_000.0
