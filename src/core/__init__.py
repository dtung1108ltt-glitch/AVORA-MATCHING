"""Core của engine AVORA Matching.

Bao gồm 3 giai đoạn chính của pipeline:

1. :mod:`src.core.preprocessor`     - Giai đoạn 1 & 2: chuẩn hóa dữ liệu,
   lọc Hard Constraints, cắt ngắn danh sách ưu tiên và xây dựng
   ``PreferenceProfile``.
2. :mod:`src.core.matching_engine`  - Giai đoạn 3: Gale-Shapley Many-to-One
   Deferred Acceptance với Rank Lookup Table :math:`\\mathcal{O}(1)`.
3. :mod:`src.core.postprocessor`    - Giai đoạn 4: sinh danh sách Top-K gợi ý.

Các hằng số và kiểu dữ liệu dùng chung được tái xuất ở đây để import gọn hơn.
"""

from __future__ import annotations

from src.core.matching_engine import GaleShapleyMatching, MatchingResult
from src.core.postprocessor import (
    Recommendation,
    generate_top_k,
    iterative_clearance_rounds,
)
from src.core.preprocessor import (
    Id,
    PreferenceProfile,
    PreprocessConfig,
    Preprocessor,
    ScoreMatrix,
    build_profile_from_profiles,
    load_market_from_csv,
)
from src.core.profile_models import (
    CandidateProfile,
    ExecutionAudit,
    JobProfile,
)
from src.core.scoring import (
    EmployerWeightsConfig,
    calculate_composite_c_score,
    calculate_composite_e_score,
    check_hard_constraints,
    explain_match,
)

__all__ = [
    "CandidateProfile",
    "EmployerWeightsConfig",
    "ExecutionAudit",
    "GaleShapleyMatching",
    "Id",
    "JobProfile",
    "MatchingResult",
    "PreferenceProfile",
    "PreprocessConfig",
    "Preprocessor",
    "Recommendation",
    "ScoreMatrix",
    "build_profile_from_profiles",
    "calculate_composite_c_score",
    "calculate_composite_e_score",
    "check_hard_constraints",
    "explain_match",
    "generate_top_k",
    "iterative_clearance_rounds",
    "load_market_from_csv",
]

