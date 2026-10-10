"""Đo lường chất lượng kết quả matching.

* :mod:`src.evaluation.metrics`   - blocking pairs, satisfaction, fill rate.
* :mod:`src.evaluation.benchmark` - chạy thực nghiệm và so sánh với baseline.
"""

from __future__ import annotations

from src.evaluation.metrics import (
    BlockingPairReport,
    EvaluationReport,
    FillRateReport,
    SatisfactionReport,
    average_satisfaction,
    calculate_blocking_pairs,
    evaluate,
    fill_rate,
    find_blocking_pairs,
)

__all__ = [
    "BlockingPairReport",
    "EvaluationReport",
    "FillRateReport",
    "SatisfactionReport",
    "average_satisfaction",
    "calculate_blocking_pairs",
    "evaluate",
    "fill_rate",
    "find_blocking_pairs",
]
