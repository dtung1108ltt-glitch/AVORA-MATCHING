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
    load_market_from_csv,
)

__all__ = [
    "GaleShapleyMatching",
    "Id",
    "MatchingResult",
    "PreferenceProfile",
    "PreprocessConfig",
    "Preprocessor",
    "Recommendation",
    "ScoreMatrix",
    "generate_top_k",
    "iterative_clearance_rounds",
    "load_market_from_csv",
]
