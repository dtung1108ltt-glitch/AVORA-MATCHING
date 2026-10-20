#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AVORA Matching Dashboard - giao diện Streamlit cho engine Gale-Shapley.

Chạy local:

    pip install streamlit
    streamlit run dashboard.py

Trình duyệt sẽ tự mở tại http://localhost:8501

Dashboard cho phép:

* Sinh thị trường mẫu, đọc CSV sẵn có của repo, upload file, hoặc dán tay
  ma trận điểm ``E`` (phía nhà tuyển dụng), ``C`` (phía ứng viên) và quota ``q_j``.
* Chạy Gale-Shapley Many-to-One Deferred Acceptance ở chế độ C-optimal
  (ứng viên chủ động) hoặc J-optimal (nhà tuyển dụng chủ động).
* Xem bảng phân bổ ứng viên - công việc, chỉ số Blocking Pairs, Fill Rate,
  mức hài lòng hai phía và danh sách Top-K gợi ý dự phòng.

File này chỉ cần ``streamlit`` + ``pandas`` (xem ``requirements-dashboard.txt``)
và package ``src`` của repo, không thêm dependency nào khác; engine không cần
PostgreSQL hay model embedding.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="AVORA Matching Dashboard",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:  # pragma: no cover - phụ thuộc môi trường chạy
    from src.core import (
        GaleShapleyMatching,
        MatchingResult,
        PreferenceProfile,
        PreprocessConfig,
        Preprocessor,
        Recommendation,
        ScoreMatrix,
        generate_top_k,
        load_market_from_csv,
    )
    from src.evaluation.benchmark import (
        BenchmarkResult,
        generate_synthetic_market,
        run_benchmark,
    )
    from src.evaluation.metrics import EvaluationReport, evaluate
except ImportError as exc:  # pragma: no cover
    st.error(
        "Không import được package `src`. Hãy chạy dashboard từ thư mục gốc của "
        "repo (nơi có thư mục `src/`) bằng lệnh `streamlit run dashboard.py`.\n\n"
        f"Chi tiết lỗi: {exc}"
    )
    st.stop()


CELL_PATTERN = re.compile(r"[,;\t]")
QUOTA_PATTERN = re.compile(r"[,;:=\t]")
CSV_CANDIDATES = ROOT / "scripts" / "data" / "candidates.csv"
CSV_JOBS = ROOT / "scripts" / "data" / "job_descriptions.csv"


@st.cache_resource(show_spinner=False)
def get_cached_preprocessor_resource(
    max_candidate_preferences: int = 40,
    max_employer_preferences: int | None = None,
    min_e_score: float = 0.0,
    min_c_score: float = 0.0,
    require_positive_scores: bool = True,
    tie_break_seed: int = 20_240_517,
    use_skill_tie_break: bool = True,
) -> Preprocessor:
    """Khởi tạo và lưu trữ đệm tài nguyên Preprocessor bằng @st.cache_resource."""
    return Preprocessor(
        PreprocessConfig(
            max_candidate_preferences=max_candidate_preferences,
            max_employer_preferences=max_employer_preferences,
            min_e_score=min_e_score,
            min_c_score=min_c_score,
            require_positive_scores=require_positive_scores,
            tie_break_seed=tie_break_seed,
            use_skill_tie_break=use_skill_tie_break,
        )
    )


@st.cache_resource(show_spinner=False)
def get_cached_embedding_model_resource() -> Any:
    """Nạp và lưu đệm mô hình embedding ONNX Runtime / FastEmbed bằng @st.cache_resource."""
    from app.embeddings import get_embedding_model

    return get_embedding_model()


@st.cache_data(show_spinner=False)
def run_cached_benchmark(
    n_candidates: int = 2_000,
    n_jobs: int = 400,
    avg_quota: int = 5,
    density: float = 0.4,
    max_preferences: int = 40,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Tính toán và lưu đệm kết quả benchmark từ `src/evaluation/benchmark.py` bằng @st.cache_data."""
    results: list[BenchmarkResult] = run_benchmark(
        n_candidates=n_candidates,
        n_jobs=n_jobs,
        avg_quota=avg_quota,
        density=density,
        max_preferences=max_preferences,
        seed=seed,
    )
    return [r.as_dict() for r in results]


@dataclass
class MarketInput:
    """Thị trường hai phía đã sẵn sàng đưa vào ``Preprocessor``.

    Attributes:
        name: Nhãn nguồn dữ liệu (hiển thị trên giao diện).
        e_scores: ``E[candidate][job]`` - điểm phía nhà tuyển dụng.
        c_scores: ``C[candidate][job]`` - điểm phía ứng viên.
        quotas: ``q_j`` - hạn mức từng vị trí.
        feasible: Tập cặp khả thi (Hard Constraints) hoặc ``None``.
        candidate_ids: Thứ tự chuẩn của ứng viên.
        job_ids: Thứ tự chuẩn của vị trí.
        notes: Ghi chú hiển thị cho người dùng.
    """

    name: str
    e_scores: ScoreMatrix
    c_scores: ScoreMatrix
    quotas: dict[str, int]
    candidate_ids: tuple[str, ...]
    job_ids: tuple[str, ...]
    feasible: set[tuple[str, str]] | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        """``(số ứng viên, số vị trí)``."""

        return len(self.candidate_ids), len(self.job_ids)

    @property
    def total_quota(self) -> int:
        """Tổng hạn mức tuyển dụng."""

        return sum(max(0, value) for value in self.quotas.values())

    def pair_count(self) -> int:
        """Số cặp ``(c, j)`` thực sự có điểm ở cả hai phía."""

        return sum(
            1
            for candidate, row in self.e_scores.items()
            for job in row
            if job in self.c_scores.get(candidate, {})
        )


# ---------------------------------------------------------------------------
# Bộ dựng dữ liệu thị trường (4 nguồn: mẫu / CSV repo / upload / dán tay)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def build_synthetic_market(
    n_candidates: int,
    n_jobs: int,
    avg_quota: int,
    density: float,
    seed: int,
) -> MarketInput:
    """Sinh thị trường mô phỏng tất định để thử thuật toán (có lưu đệm @st.cache_data).

    Điểm ``E`` và ``C`` được sinh từ cùng một "độ hợp" ẩn cộng nhiễu riêng, nên
    hai phía vừa có tương quan vừa có xung đột - đủ để kết quả matching có ý
    nghĩa. Cùng ``seed`` luôn cho cùng kết quả.

    Args:
        n_candidates: Số ứng viên ``N``.
        n_jobs: Số vị trí ``M``.
        avg_quota: Hạn mức trung bình mỗi vị trí.
        density: Tỷ lệ cặp ``(c, j)`` có điểm (1.0 = ma trận đặc).
        seed: Hạt giống ngẫu nhiên.

    Returns:
        ``MarketInput`` sẵn sàng cho engine.
    """

    rng = random.Random(seed)
    e_scores: ScoreMatrix = {}
    c_scores: ScoreMatrix = {}
    feasible: set[tuple[str, str]] = set()

    for candidate_index in range(n_candidates):
        candidate = f"C{candidate_index + 1}"
        ability = rng.random()
        e_row: dict[str, float] = {}
        c_row: dict[str, float] = {}
        for job_index in range(n_jobs):
            job = f"J{job_index + 1}"
            if rng.random() > density:
                continue
            fit = rng.random()
            e_value = 0.30 + 0.45 * fit + 0.25 * ability + 0.10 * rng.random()
            c_value = 0.30 + 0.45 * fit + 0.25 * rng.random()
            e_row[job] = round(min(0.999, e_value), 4)
            c_row[job] = round(min(0.999, c_value), 4)
            feasible.add((candidate, job))
        e_scores[candidate] = e_row
        c_scores[candidate] = c_row

    quotas: dict[str, int] = {}
    for job_index in range(n_jobs):
        job = f"J{job_index + 1}"
        jitter = rng.gauss(avg_quota, max(0.35 * avg_quota, 0.5))
        quotas[job] = max(1, min(2 * avg_quota, int(round(jitter))))

    return MarketInput(
        name="Thị trường mô phỏng",
        e_scores=e_scores,
        c_scores=c_scores,
        quotas=quotas,
        candidate_ids=tuple(e_scores),
        job_ids=tuple(quotas),
        feasible=feasible,
        notes=[
            f"N = {n_candidates} ứng viên, M = {n_jobs} vị trí, "
            f"density = {density:.2f}, quota TB = {avg_quota}.",
            "Nguồn: sinh mô phỏng tất định bằng `random.Random(seed)` "
            "(không dùng `hash()` nên tái lập được giữa các lần chạy).",
        ],
    )


def parse_score_grid(
    text: str, label: str
) -> tuple[ScoreMatrix, tuple[str, ...], tuple[str, ...]]:
    """Đọc ma trận điểm dạng văn bản (CSV/TSV) thành ``dict[candidate][job]``.

    Định dạng: dòng đầu là header chứa **mã vị trí**, ô đầu tiên để trống (hoặc
    ghi tiêu đề cho cột ứng viên). Mỗi dòng tiếp theo bắt đầu bằng **mã ứng
    viên**, sau đó là điểm tương ứng. Ô để trống = cặp không khả thi.

    Args:
        text: Nội dung ma trận.
        label: Tên ma trận (``"E"`` hoặc ``"C"``) dùng trong thông báo lỗi.

    Returns:
        ``(matrix, candidate_ids, job_ids)``.

    Raises:
        ValueError: Nếu thiếu dòng, lệch số ô, hoặc điểm không phải số.
    """

    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if len(rows) < 2:
        raise ValueError(
            f"Ma trận {label} cần ít nhất 1 dòng header và 1 dòng dữ liệu."
        )

    header = [cell.strip() for cell in CELL_PATTERN.split(rows[0])]
    if header and header[0] in {"", "-", "candidate", "ung vien", "ứng viên", "Ứng viên"}:
        header = header[1:]
    job_ids = tuple(cell for cell in header if cell)
    if not job_ids:
        raise ValueError(f"Ma trận {label} thiếu mã vị trí ở dòng header.")

    matrix: ScoreMatrix = {}
    candidate_ids: list[str] = []
    for line_number, row_text in enumerate(rows[1:], start=2):
        cells = [cell.strip() for cell in CELL_PATTERN.split(row_text)]
        candidate = cells[0]
        if not candidate:
            raise ValueError(f"Ma trận {label}: dòng {line_number} thiếu mã ứng viên.")
        values = cells[1:]
        if len(values) != len(job_ids):
            raise ValueError(
                f"Ma trận {label}: dòng {line_number} có {len(values)} điểm "
                f"nhưng header có {len(job_ids)} vị trí."
            )
        entry: dict[str, float] = {}
        for job, raw in zip(job_ids, values, strict=True):
            if raw == "":
                continue
            try:
                score = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"Ma trận {label}: dòng {line_number}, vị trí {job} có giá trị "
                    f"không phải số: {raw!r}"
                ) from exc
            entry[job] = score
        if candidate in matrix:
            raise ValueError(f"Ma trận {label}: trùng mã ứng viên {candidate!r}.")
        matrix[candidate] = entry
        candidate_ids.append(candidate)

    return matrix, tuple(candidate_ids), job_ids


def parse_quotas(text: str, job_ids: tuple[str, ...], default: int) -> dict[str, int]:
    """Đọc quota, mỗi dòng có dạng ``<mã vị trí><dấu phân cách><số nguyên>``.

    Dấu phân cách chấp nhận ``,``, ``;``, ``:``, ``=`` hoặc tab để gõ tay nhanh
    (ví dụ ``J1, 3`` hay ``J1=3`` đều hợp lệ).

    Args:
        text: Nội dung quota; rỗng nghĩa là dùng ``default`` cho mọi vị trí.
        job_ids: Danh sách vị trí cần quota.
        default: Quota mặc định khi không nhập.

    Returns:
        ``{job: quota}`` đủ cho mọi vị trí trong ``job_ids``.

    Raises:
        ValueError: Nếu quota âm hoặc dòng sai định dạng.
    """

    parsed: dict[str, int] = {}
    for row_text in text.splitlines():
        line = row_text.strip()
        if not line:
            continue
        cells = [cell.strip() for cell in QUOTA_PATTERN.split(line)]
        if len(cells) < 2:
            raise ValueError(
                f"Dòng quota không hợp lệ: {line!r} (cần dạng 'J1, 3' hoặc 'J1=3')."
            )
        try:
            value = int(float(cells[1]))
        except ValueError as exc:
            raise ValueError(f"Quota không phải số nguyên: {line!r}") from exc
        if value < 0:
            raise ValueError(f"Quota không được âm: {line!r}")
        parsed[cells[0]] = value

    return {job: parsed.get(job, default) for job in job_ids}


@st.cache_data(show_spinner=False)
def load_repo_csv_market(
    *,
    default_quota: int,
    seed: int,
    candidate_limit: int | None,
    job_limit: int | None,
) -> MarketInput:
    """Nạp thị trường từ ``scripts/data/*.csv`` sẵn có của repo (có lưu đệm @st.cache_data).

    Dùng đúng adapter :func:`src.core.load_market_from_csv`: Hard Constraint bắt
    buộc là **trùng nhóm khuyết tật**, điểm ``E``/``C`` sinh mô phỏng tất định.

    Args:
        default_quota: Hạn mức mặc định cho mỗi dòng mô tả công việc.
        seed: Hạt giống sinh điểm mô phỏng.
        candidate_limit: Giới hạn số ứng viên đọc vào.
        job_limit: Giới hạn số dòng mô tả công việc đọc vào.

    Returns:
        ``MarketInput`` dựng từ CSV.

    Raises:
        FileNotFoundError: Nếu thiếu file CSV trong ``scripts/data``.
    """

    e_scores, c_scores, quotas, feasible = load_market_from_csv(
        CSV_CANDIDATES,
        CSV_JOBS,
        default_quota=default_quota,
        seed=seed,
        candidate_limit=candidate_limit,
        job_limit=job_limit,
    )
    candidate_ids = tuple(e_scores)
    job_ids = tuple(quotas)

    notes = [
        f"Đọc từ `{CSV_CANDIDATES.relative_to(ROOT)}` và "
        f"`{CSV_JOBS.relative_to(ROOT)}`.",
        "Hard Constraint: ứng viên chỉ khả thi với mô tả **trùng nhóm khuyết tật**.",
        "Điểm E/C hiện là điểm mô phỏng tất định (dữ liệu CSV chưa có điểm thực).",
    ]
    if not feasible:
        notes.append("Không có cặp khả thi nào -> kết quả sẽ rỗng.")

    return MarketInput(
        name="CSV của repo",
        e_scores=e_scores,
        c_scores=c_scores,
        quotas=quotas,
        candidate_ids=candidate_ids,
        job_ids=job_ids,
        feasible=feasible,
        notes=notes,
    )


def build_grid_market(
    e_text: str,
    c_text: str,
    quota_text: str,
    *,
    default_quota: int,
    name: str,
) -> MarketInput:
    """Dựng thị trường từ hai ma trận dạng văn bản (dán tay hoặc upload).

    Args:
        e_text: Nội dung ma trận ``E``.
        c_text: Nội dung ma trận ``C``.
        quota_text: Nội dung quota (rỗng = dùng ``default_quota``).
        default_quota: Quota mặc định cho vị trí không khai báo riêng.
        name: Nhãn nguồn dữ liệu.

    Returns:
        ``MarketInput`` đã khớp cột giữa hai ma trận.

    Raises:
        ValueError: Nếu hai ma trận lệch nhau về ứng viên hoặc vị trí.
    """

    e_scores, e_candidates, e_jobs = parse_score_grid(e_text, "E")
    c_scores, c_candidates, c_jobs = parse_score_grid(c_text, "C")

    if e_candidates != c_candidates:
        raise ValueError(
            "Hai ma trận lệch danh sách ứng viên. E có "
            f"{len(e_candidates)} dòng, C có {len(c_candidates)} dòng."
        )
    if e_jobs != c_jobs:
        raise ValueError(
            "Hai ma trận lệch danh sách vị trí. E có "
            f"{len(e_jobs)} cột, C có {len(c_jobs)} cột."
        )

    quotas = parse_quotas(quota_text, e_jobs, default_quota)
    filled = sum(
        1
        for candidate in e_candidates
        for job in e_scores[candidate]
        if job in c_scores.get(candidate, {})
    )

    return MarketInput(
        name=name,
        e_scores=e_scores,
        c_scores=c_scores,
        quotas=quotas,
        candidate_ids=e_candidates,
        job_ids=e_jobs,
        feasible=None,
        notes=[
            f"N = {len(e_candidates)} ứng viên, M = {len(e_jobs)} vị trí, "
            f"{filled} cặp có điểm ở cả hai phía.",
            "Mọi cặp có mặt ở cả hai ma trận đều được coi là khả thi "
            "(không áp thêm Hard Constraint).",
        ],
    )


@dataclass
class RunOutput:
    """Toàn bộ kết quả của một lần chạy Gale-Shapley.

    Attributes:
        market: Dữ liệu đầu vào đã dùng.
        profile: ``PreferenceProfile`` sau tiền xử lý.
        engine: Đối tượng engine (giữ Rank Lookup Table để tra cứu ``O(1)``).
        result: Kết quả matching.
        report: Báo cáo chỉ số đánh giá.
        recommendations: Danh sách Top-K gợi ý dự phòng theo từng ứng viên.
        elapsed_ms: Tổng thời gian chạy (mili-giây).
    """

    market: MarketInput
    profile: PreferenceProfile
    engine: GaleShapleyMatching
    result: MatchingResult
    report: EvaluationReport
    recommendations: dict[str, list[Recommendation]]
    elapsed_ms: float


def run_matching(
    market: MarketInput,
    *,
    config: PreprocessConfig,
    optimal: str,
    top_k: int,
    weight_c: float,
    weight_e: float,
) -> RunOutput:
    """Chạy trọn pipeline: tiền xử lý -> Gale-Shapley -> đánh giá -> Top-K.

    Args:
        market: Thị trường hai phía.
        config: Tham số tiền xử lý (``L``, ``M``, ngưỡng sàn...).
        optimal: ``"candidate"`` hoặc ``"employer"``.
        top_k: Số gợi ý dự phòng mỗi ứng viên (0 = không sinh).
        weight_c: Trọng số điểm ``C`` khi tính điểm danh dư.
        weight_e: Trọng số điểm ``E`` khi tính điểm danh dư.

    Returns:
        ``RunOutput`` với đầy đủ kết quả và chỉ số.

    Raises:
        ValueError: Nếu ``optimal`` không hợp lệ hoặc quota âm.
    """

    started = time.perf_counter()

    preprocessor = get_cached_preprocessor_resource(
        max_candidate_preferences=config.max_candidate_preferences,
        max_employer_preferences=config.max_employer_preferences,
        min_e_score=config.min_e_score,
        min_c_score=config.min_c_score,
        require_positive_scores=config.require_positive_scores,
        tie_break_seed=config.tie_break_seed,
        use_skill_tie_break=config.use_skill_tie_break,
    )
    profile = preprocessor.build_profile(
        market.e_scores,
        market.c_scores,
        market.quotas,
        feasible_pairs=market.feasible,
        candidate_ids=market.candidate_ids,
        job_ids=market.job_ids,
    )
    engine = GaleShapleyMatching(profile, optimal=optimal)
    result = engine.solve()
    report = evaluate(profile, result)
    recommendations = (
        generate_top_k(
            profile, result, k=top_k, weight_c=weight_c, weight_e=weight_e
        )
        if top_k > 0
        else {}
    )

    return RunOutput(
        market=market,
        profile=profile,
        engine=engine,
        result=result,
        report=report,
        recommendations=recommendations,
        elapsed_ms=(time.perf_counter() - started) * 1000.0,
    )


DISPLAY_LIMIT = 2000


def matching_dataframe(output: RunOutput) -> pd.DataFrame:
    """Bảng phân bổ ứng viên - công việc kèm hạng nguyện vọng hai phía.

    Hạng hiển thị là **1-based** (hạng 1 = lựa chọn tốt nhất), lấy trực tiếp từ
    Rank Lookup Table của engine nên tra cứu là ``O(1)``.
    """

    profile = output.profile
    engine = output.engine
    rows: list[dict[str, object]] = []

    for candidate in profile.candidate_ids:
        job = output.result.matches.get(candidate)
        if job is None:
            rows.append(
                {
                    "Ứng viên": str(candidate),
                    "Vị trí": "—",
                    "Điểm E": None,
                    "Điểm C": None,
                    "Hạng nguyện vọng của ứng viên": None,
                    "Hạng của ứng viên trong mắt vị trí": None,
                }
            )
            continue

        candidate_rank = engine.candidate_rank(candidate, job)
        job_rank = engine.job_rank(job, candidate)
        rows.append(
            {
                "Ứng viên": str(candidate),
                "Vị trí": str(job),
                "Điểm E": round(profile.e_scores.get(candidate, {}).get(job, 0.0), 4),
                "Điểm C": round(profile.c_scores.get(candidate, {}).get(job, 0.0), 4),
                "Hạng nguyện vọng của ứng viên": (
                    candidate_rank + 1 if candidate_rank is not None else None
                ),
                "Hạng của ứng viên trong mắt vị trí": (
                    job_rank + 1 if job_rank is not None else None
                ),
            }
        )

    return pd.DataFrame(rows)


def job_dataframe(output: RunOutput) -> pd.DataFrame:
    """Bảng tình trạng tuyển dụng của từng vị trí (quota, đã tuyển, fill rate)."""

    profile = output.profile
    rows: list[dict[str, object]] = []

    for job in profile.job_ids:
        quota = profile.quotas.get(job, 0)
        members = output.result.job_assignments.get(job, [])
        rows.append(
            {
                "Vị trí": str(job),
                "Hạn mức": quota,
                "Đã tuyển": len(members),
                "Còn trống": max(0, quota - len(members)),
                "Tỷ lệ lấp đầy": (len(members) / quota) if quota > 0 else 0.0,
                "Ứng viên": ", ".join(str(member) for member in members) or "—",
            }
        )

    return pd.DataFrame(rows)


def recommendation_dataframe(output: RunOutput) -> pd.DataFrame:
    """Bảng Top-K gợi ý dự phòng cho từng ứng viên."""

    columns = ["Ứng viên", "Hạng gợi ý", "Vị trí", "Điểm danh dư", "Ghép chính thức"]
    rows: list[dict[str, object]] = []

    for candidate in output.profile.candidate_ids:
        for rec in output.recommendations.get(candidate, []):
            rows.append(
                {
                    "Ứng viên": str(candidate),
                    "Hạng gợi ý": rec.position,
                    "Vị trí": str(rec.job_id),
                    "Điểm danh dư": round(rec.score, 4),
                    "Ghép chính thức": "✔" if rec.is_primary else "",
                }
            )

    return pd.DataFrame(rows, columns=columns)


@st.cache_data(show_spinner=False)
def matrix_dataframe(
    scores: ScoreMatrix,
    candidate_ids: tuple[str, ...],
    job_ids: tuple[str, ...],
    *,
    row_limit: int,
    col_limit: int,
) -> pd.DataFrame:
    """Chuyển ma trận thưa thành DataFrame để xem trước (lưu đệm bằng @st.cache_data)."""

    rows = list(candidate_ids)[:row_limit]
    columns = list(job_ids)[:col_limit]
    data = {
        str(job): [scores.get(candidate, {}).get(job) for candidate in rows]
        for job in columns
    }
    return pd.DataFrame(data, index=[str(candidate) for candidate in rows])


def to_csv_bytes(frame: pd.DataFrame, *, index: bool = False) -> bytes:
    """Xuất DataFrame thành bytes CSV (UTF-8 BOM để Excel mở đúng tiếng Việt)."""

    return frame.to_csv(index=index).encode("utf-8-sig")


def report_json(output: RunOutput) -> str:
    """Xuất toàn bộ kết quả + chỉ số thành JSON để lưu trữ hoặc so sánh."""

    payload = {
        "nguon_du_lieu": output.market.name,
        "che_do": output.result.optimal,
        "thoi_gian_ms": round(output.elapsed_ms, 4),
        "so_phep_tinh": output.result.proposals,
        "matches": {str(k): str(v) for k, v in output.result.matches.items()},
        "chi_so": output.report.as_dict(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Giao diện Streamlit
# ---------------------------------------------------------------------------

SOURCE_SYNTHETIC = "🎲 Thị trường mô phỏng"
SOURCE_CSV = "📁 CSV của repo (scripts/data)"
SOURCE_PASTE = "✍️ Dán ma trận E/C"
SOURCE_UPLOAD = "⬆️ Upload file ma trận"
SOURCES = (SOURCE_SYNTHETIC, SOURCE_CSV, SOURCE_PASTE, SOURCE_UPLOAD)

DEFAULT_E_TEXT = """J1,J2,J3,J4
C1,0.90,0.40,0.20,0.10
C2,0.80,0.70,0.60,0.30
C3,0.30,0.50,0.90,0.70
C4,0.20,0.60,0.40,0.95
C5,0.70,0.90,0.50,0.40
C6,0.60,0.30,0.70,0.20"""

DEFAULT_C_TEXT = """J1,J2,J3,J4
C1,0.40,0.90,0.40,0.20
C2,0.70,0.60,0.30,0.30
C3,0.50,0.20,0.80,0.60
C4,0.30,0.50,0.60,0.90
C5,0.90,0.40,0.50,0.50
C6,0.50,0.70,0.40,0.30"""

DEFAULT_QUOTA_TEXT = """J1,2
J2,1
J3,2
J4,1"""


@dataclass
class DashboardSettings:
    """Toàn bộ lựa chọn của người dùng trên sidebar.

    Attributes:
        source: Nhãn nguồn dữ liệu đang chọn.
        market: Thị trường đã dựng xong.
        config: Tham số ``PreprocessConfig``.
        optimal: Chế độ DA (``"candidate"`` / ``"employer"``).
        top_k: Số gợi ý dự phòng mỗi ứng viên.
        weight_c: Trọng số điểm ``C`` khi trộn điểm danh dư.
        weight_e: Trọng số điểm ``E`` khi trộn điểm danh dư.
        row_limit: Số dòng ma trận xem trước.
        col_limit: Số cột ma trận xem trước.
    """

    source: str
    market: MarketInput
    config: PreprocessConfig
    optimal: str
    top_k: int
    weight_c: float
    weight_e: float
    row_limit: int
    col_limit: int


def render_sidebar() -> DashboardSettings:
    """Vẽ toàn bộ sidebar và dựng thị trường theo lựa chọn của người dùng.

    Returns:
        ``DashboardSettings`` đã sẵn sàng để chạy.

    Raises:
        ValueError: Nếu ma trận/quota nhập sai định dạng.
        FileNotFoundError: Nếu chọn nguồn CSV nhưng thiếu file trong repo.
    """

    sidebar = st.sidebar
    sidebar.header("⚙️ Cấu hình")
    sidebar.caption("Chọn nguồn dữ liệu, tham số tiền xử lý và chế độ chạy.")

    source = sidebar.radio("Nguồn dữ liệu", SOURCES, index=0)
    default_quota = int(
        sidebar.number_input("Quota mặc định mỗi vị trí", 1, 50, 2, 1)
    )
    market_seed = int(sidebar.number_input("Seed dữ liệu", 0, 1_000_000, 42, 1))

    if source == SOURCE_SYNTHETIC:
        n_candidates = int(sidebar.slider("Số ứng viên (N)", 2, 5000, 60, 1))
        n_jobs = int(sidebar.slider("Số vị trí (M)", 2, 2000, 20, 1))
        avg_quota = int(sidebar.slider("Quota trung bình", 1, 20, 3, 1))
        density = float(sidebar.slider("Mật độ cặp có điểm", 0.05, 1.0, 0.5, 0.05))
        market = build_synthetic_market(
            n_candidates, n_jobs, avg_quota, density, market_seed
        )

    elif source == SOURCE_CSV:
        limited = sidebar.checkbox("Giới hạn số dòng đọc vào", value=True)
        candidate_limit = (
            int(sidebar.slider("Ứng viên tối đa", 1, 5000, 300, 1)) if limited else None
        )
        job_limit = (
            int(sidebar.slider("Mô tả công việc tối đa", 1, 2000, 120, 1))
            if limited
            else None
        )
        market = load_repo_csv_market(
            default_quota=default_quota,
            seed=market_seed,
            candidate_limit=candidate_limit,
            job_limit=job_limit,
        )

    elif source == SOURCE_PASTE:
        e_text = sidebar.text_area("Ma trận E (nhà tuyển dụng)", DEFAULT_E_TEXT, height=190)
        c_text = sidebar.text_area("Ma trận C (ứng viên)", DEFAULT_C_TEXT, height=190)
        quota_text = sidebar.text_area("Quota q_j", DEFAULT_QUOTA_TEXT, height=110)
        market = build_grid_market(
            e_text,
            c_text,
            quota_text,
            default_quota=default_quota,
            name="Dán tay",
        )

    else:
        e_upload = sidebar.file_uploader("File ma trận E", type=["csv", "txt", "tsv"])
        c_upload = sidebar.file_uploader("File ma trận C", type=["csv", "txt", "tsv"])
        quota_upload = sidebar.file_uploader("File quota (tùy chọn)", type=["csv", "txt", "tsv"])
        if e_upload is None or c_upload is None:
            raise ValueError(
                "Hãy chọn đủ **file ma trận E** và **file ma trận C**. "
                "Định dạng: dòng đầu là mã vị trí, mỗi dòng sau là mã ứng viên "
                "kèm điểm."
            )
        market = build_grid_market(
            e_upload.getvalue().decode("utf-8-sig"),
            c_upload.getvalue().decode("utf-8-sig"),
            quota_upload.getvalue().decode("utf-8-sig") if quota_upload else "",
            default_quota=default_quota,
            name="Upload",
        )

    sidebar.divider()
    return _render_algorithm_settings(sidebar, source, market)


def _render_algorithm_settings(
    sidebar: "st.delta_generator.DeltaGenerator",
    source: str,
    market: MarketInput,
) -> DashboardSettings:
    """Vẽ nhóm widget tiền xử lý + thuật toán và gói thành ``DashboardSettings``."""

    sidebar.subheader("🧹 Tiền xử lý")
    max_candidate_preferences = int(
        sidebar.slider(
            "L - số nguyện vọng tối đa mỗi ứng viên",
            1,
            200,
            min(40, max(1, len(market.job_ids))),
            1,
            help="Danh sách ưu tiên bị cắt còn L phần tử -> độ phức tạp O(N x L).",
        )
    )
    limit_employer = sidebar.checkbox(
        "Giới hạn M ứng viên mỗi vị trí", value=False
    )
    max_employer_preferences = (
        int(
            sidebar.slider(
                "M - số ứng viên tối đa mỗi vị trí",
                1,
                500,
                min(50, max(1, len(market.candidate_ids))),
                1,
            )
        )
        if limit_employer
        else None
    )
    min_e_score = float(sidebar.number_input("Ngưỡng sàn điểm E", 0.0, 1.0, 0.0, 0.05))
    min_c_score = float(sidebar.number_input("Ngưỡng sàn điểm C", 0.0, 1.0, 0.0, 0.05))
    require_positive = sidebar.checkbox("Loại cặp có điểm <= 0", value=True)
    tie_break_seed = int(
        sidebar.number_input("Seed phá hòa", 0, 100_000_000, 20_240_517, 1)
    )

    sidebar.subheader("🤝 Thuật toán")
    optimal = sidebar.radio(
        "Chế độ Gale-Shapley",
        ("candidate", "employer"),
        format_func=lambda value: (
            "Ứng viên chủ động (C-optimal)" if value == "candidate"
            else "Nhà tuyển dụng chủ động (J-optimal)"
        ),
    )
    top_k = int(sidebar.slider("Top-K gợi ý dự phòng", 0, 10, 3, 1))
    weight_c = float(sidebar.slider("Trọng số điểm C", 0.0, 1.0, 0.5, 0.05))
    weight_e = round(1.0 - weight_c, 4)

    sidebar.subheader("🔍 Xem trước ma trận")
    row_limit = int(sidebar.slider("Số dòng hiển thị", 5, 200, 20, 5))
    col_limit = int(sidebar.slider("Số cột hiển thị", 5, 200, 20, 5))

    return DashboardSettings(
        source=source,
        market=market,
        config=PreprocessConfig(
            max_candidate_preferences=max_candidate_preferences,
            max_employer_preferences=max_employer_preferences,
            min_e_score=min_e_score,
            min_c_score=min_c_score,
            require_positive_scores=require_positive,
            tie_break_seed=tie_break_seed,
        ),
        optimal=optimal,
        top_k=top_k,
        weight_c=weight_c,
        weight_e=weight_e,
        row_limit=row_limit,
        col_limit=col_limit,
    )


def render_market_summary(settings: DashboardSettings) -> None:
    """Hiển thị thông tin thị trường và xem trước hai ma trận điểm."""

    market = settings.market
    st.subheader("1️⃣ Dữ liệu đầu vào")

    columns = st.columns(4)
    columns[0].metric("Ứng viên (N)", market.shape[0])
    columns[1].metric("Vị trí (M)", market.shape[1])
    columns[2].metric("Tổng hạn mức", market.total_quota)
    columns[3].metric("Cặp có điểm", market.pair_count())

    st.markdown(f"**Nguồn:** {settings.source} - `{market.name}`")
    for note in market.notes:
        st.caption(f"• {note}")

    with st.expander("Xem trước ma trận E và C (ô trống = cặp không khả thi)", expanded=False):
        left, right = st.columns(2)
        left.markdown("**E[c][j] - điểm phía nhà tuyển dụng**")
        left.dataframe(
            matrix_dataframe(
                market.e_scores,
                market.candidate_ids,
                market.job_ids,
                row_limit=settings.row_limit,
                col_limit=settings.col_limit,
            )
        )
        right.markdown("**C[c][j] - điểm phía ứng viên**")
        right.dataframe(
            matrix_dataframe(
                market.c_scores,
                market.candidate_ids,
                market.job_ids,
                row_limit=settings.row_limit,
                col_limit=settings.col_limit,
            )
        )
        st.caption(
            f"Chỉ hiển thị {min(settings.row_limit, market.shape[0])} dòng x {min(settings.col_limit, market.shape[1])} cột "
            "đầu tiên để giao diện mượt; thuật toán vẫn chạy trên toàn bộ dữ liệu."
        )


def render_metrics(output: RunOutput) -> None:
    """Hiển thị khối chỉ số: ổn định, blocking pairs, fill rate, hài lòng."""

    report = output.report
    st.subheader("3️⃣ Chỉ số đo lường")
    st.caption(
        f"Kết quả của nguồn `{output.market.name}` - chế độ "
        f"**{output.result.optimal}** - "
        f"N = {len(output.profile.candidate_ids)}, M = {len(output.profile.job_ids)}."
    )

    if report.blocking.is_stable:
        st.success(
            "✅ **Matching ỔN ĐỊNH** - không tồn tại blocking pair nào. "
            "Đây là bảo đảm lý thuyết của Gale-Shapley (1962) với danh sách ưu tiên "
            "nghiêm ngặt sau khi đã phá hòa tất định."
        )
    else:
        st.error(
            f"❌ Phát hiện **{report.blocking.count} blocking pair** "
            f"(tỷ lệ {report.blocking.rate:.4%}) - kết quả không ổn định."
        )

    top = st.columns(4)
    top[0].metric("Cặp ghép được", f"{report.satisfaction.matched_pairs}")
    top[1].metric(
        "Coverage ứng viên",
        f"{report.satisfaction.coverage:.2%}",
        help="Tỷ lệ ứng viên tìm được ít nhất một vị trí / tổng số ứng viên.",
    )
    top[2].metric(
        "Fill rate (slot)",
        f"{report.fill_rate.overall:.2%}",
        help=(
            f"{report.fill_rate.filled_slots}/{report.fill_rate.total_quota} slot "
            "hạn mức đã được lấp đầy."
        ),
    )
    top[3].metric("Số phép tính (proposals)", f"{output.result.proposals}")

    bottom = st.columns(4)
    bottom[0].metric(
        "Điểm E trung bình",
        f"{report.satisfaction.average_e:.4f}",
        help=f"Thấp nhất trong các cặp ghép: {report.satisfaction.min_e:.4f}",
    )
    bottom[1].metric(
        "Điểm C trung bình",
        f"{report.satisfaction.average_c:.4f}",
        help=f"Thấp nhất trong các cặp ghép: {report.satisfaction.min_c:.4f}",
    )
    bottom[2].metric(
        "Blocking pairs",
        f"{report.blocking.count}",
        delta="ỔN ĐỊNH" if report.blocking.is_stable else "KHÔNG ỔN ĐỊNH",
        delta_color="normal" if report.blocking.is_stable else "inverse",
    )
    bottom[3].metric("Thời gian chạy", f"{output.elapsed_ms:.1f} ms")

    if report.blocking.pairs:
        st.markdown("**Danh sách blocking pair đầu tiên:**")
        st.write(
            [
                f"{candidate} ⭢ {job}"
                for candidate, job in report.blocking.pairs[:20]
            ]
        )

    st.caption(
        "Công thức: `Blocking pairs = 0` là điều kiện ổn định; "
        "`Fill rate = filled_slots / total_quota`; "
        "`Coverage = matched_pairs / N`."
    )


def render_tables(output: RunOutput) -> None:
    """Hiển thị các bảng kết quả và nút tải file."""

    st.subheader("4️⃣ Bảng kết quả")

    match_frame = matching_dataframe(output)
    job_frame = job_dataframe(output)
    rec_frame = recommendation_dataframe(output)

    tab_match, tab_jobs, tab_topk, tab_unmatched, tab_export = st.tabs(
        [
            "🧑‍💼 Phân bổ ứng viên - công việc",
            "🏢 Tình trạng vị trí",
            "⭐ Top-K gợi ý",
            "🚫 Không ghép được",
            "⬇️ Tải kết quả",
        ]
    )

    with tab_match:
        st.caption(
            "Hai cột hạng lấy từ Rank Lookup Table của engine - tra cứu O(1). "
            "Hạng 1 = lựa chọn tốt nhất của phía đó."
        )
        st.dataframe(match_frame.head(DISPLAY_LIMIT))
        if len(match_frame) > DISPLAY_LIMIT:
            st.caption(f"Chỉ hiển thị {DISPLAY_LIMIT} dòng đầu; tải file để xem đủ.")

    with tab_jobs:
        st.caption(
            "`Tỷ lệ lấp đầy` = đã tuyển / hạn mức. Vị trí thiếu người là nguồn "
            "khan hiếm chính trên thị trường."
        )
        st.dataframe(job_frame)

    with tab_topk:
        if rec_frame.empty:
            st.info(
                "Chưa có gợi ý nào. Hãy đặt **Top-K gợi ý dự phòng** > 0 ở sidebar "
                "rồi chạy lại."
            )
        else:
            st.caption(
                "Điểm danh dư = trọng số x điểm C chuẩn hóa + trọng số x điểm E "
                "chuẩn hóa (min-max theo từng ứng viên / từng vị trí)."
            )
            st.dataframe(rec_frame)

    with tab_unmatched:
        left, right = st.columns(2)
        unmatched_candidates = output.result.unmatched_candidates
        unmatched_jobs = output.result.unmatched_jobs
        left.markdown(f"**Ứng viên không được nhận ({len(unmatched_candidates)})**")
        left.write(", ".join(str(c) for c in unmatched_candidates[:200]) or "—")
        right.markdown(f"**Vị trí còn trống ({len(unmatched_jobs)})**")
        right.write(", ".join(str(j) for j in unmatched_jobs[:200]) or "—")
        if len(unmatched_candidates) > 200 or len(unmatched_jobs) > 200:
            st.caption("Chỉ hiển thị 200 phần tử đầu tiên.")

    with tab_export:
        st.caption("File CSV xuất kèm BOM UTF-8 nên mở bằng Excel hiển thị đúng tiếng Việt.")
        download_left, download_right = st.columns(2)
        download_left.download_button(
            "⬇️ matching.csv",
            data=to_csv_bytes(match_frame),
            file_name="matching_assignments.csv",
            mime="text/csv",
        )
        download_left.download_button(
            "⬇️ job_fill_rate.csv",
            data=to_csv_bytes(job_frame),
            file_name="job_fill_rate.csv",
            mime="text/csv",
        )
        download_right.download_button(
            "⬇️ top_k_recommendations.csv",
            data=to_csv_bytes(rec_frame),
            file_name="top_k_recommendations.csv",
            mime="text/csv",
        )
        download_right.download_button(
            "⬇️ report.json",
            data=report_json(output).encode("utf-8"),
            file_name="matching_report.json",
            mime="application/json",
        )
        with st.expander("Xem JSON đầy đủ"):
            st.code(report_json(output), language="json")


def main() -> None:
    """Điểm khởi chạy của dashboard Streamlit."""

    st.title("🤝 AVORA Matching Dashboard")
    st.caption(
        "Hai phía (ứng viên - nhà tuyển dụng) được ghép bằng Gale-Shapley "
        "Many-to-One Deferred Acceptance có cắt ngắn danh sách ưu tiên, "
        "kèm đo lường Blocking Pairs và Fill Rate. Engine nằm ở package `src/`."
    )

    try:
        settings = render_sidebar()
    except (ValueError, FileNotFoundError, TypeError) as exc:
        st.error(f"⚠️ Chưa dựng được dữ liệu đầu vào: {exc}")
        st.info("Chỉnh lại tham số ở sidebar rồi thao tác tiếp.")
        return

    render_market_summary(settings)

    st.subheader("2️⃣ Chạy thuật toán")
    run_column, info_column = st.columns([1, 3])
    run_clicked = run_column.button("🚀 Chạy Gale-Shapley", type="primary")
    mode_label = (
        "C-optimal (ứng viên chủ động)"
        if settings.optimal == "candidate"
        else "J-optimal (nhà tuyển dụng chủ động)"
    )
    info_column.caption(
        f"Chế độ: **{mode_label}** · "
        f"L = {settings.config.max_candidate_preferences} · "
        f"M = {settings.config.max_employer_preferences or 'không cắt'} · "
        f"Top-K = {settings.top_k} · "
        f"trọng số C/E = {settings.weight_c}/{settings.weight_e}"
    )

    if run_clicked:
        try:
            with st.spinner("Đang chạy Deferred Acceptance..."):
                st.session_state["output"] = run_matching(
                    settings.market,
                    config=settings.config,
                    optimal=settings.optimal,
                    top_k=settings.top_k,
                    weight_c=settings.weight_c,
                    weight_e=settings.weight_e,
                )
            st.session_state["run_error"] = None
        except Exception as exc:  # noqa: BLE001 - hiển thị mọi lỗi ngay trên UI
            st.session_state["output"] = None
            st.session_state["run_error"] = f"{type(exc).__name__}: {exc}"

    run_error = st.session_state.get("run_error")
    if run_error:
        st.error(f"❌ Không chạy được thuật toán: {run_error}")
        return

    output = st.session_state.get("output")
    if output is None:
        st.info(
            "Nhấn **🚀 Chạy Gale-Shapley** để thực thi. Kết quả sẽ gồm bảng phân bổ "
            "ứng viên - công việc, Blocking Pairs, Fill Rate và Top-K gợi ý."
        )
        return

    render_metrics(output)
    render_tables(output)

    st.divider()
    with st.expander("💻 Lệnh terminal để chạy lại dashboard này"):
        st.code(
            "pip install streamlit\n"
            "streamlit run dashboard.py\n"
            "# hoặc cố định cổng / mở cho máy khác trong LAN:\n"
            "streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0",
            language="bash",
        )
    st.caption(
        "Engine không phụ thuộc Streamlit: cùng một package `src/` cũng được dùng "
        "cho `pytest tests -q`, `python -m src.evaluation.benchmark` và API FastAPI."
    )


if __name__ == "__main__":
    main()
