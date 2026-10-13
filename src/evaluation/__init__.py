"""Đo lường chất lượng kết quả matching.

* :mod:`src.evaluation.metrics`   - blocking pairs, satisfaction, fill rate.
* :mod:`src.evaluation.benchmark` - chạy thực nghiệm và so sánh với baseline.
"""

from __future__ import annotations

from src.evaluation.benchmark import (
    BenchmarkResult,
    GreedyBaselineMatcher,
    run_benchmark,
    run_profile_benchmark,
    to_evaluation_comparison_table,
    to_markdown_table,
)
from src.evaluation.metrics import (
    BlockingPairReport,
    ConflictReport,
    CoverageReport,
    EvaluationReport,
    FairnessAuditReport,
    FillRateReport,
    RankReport,
    SatisfactionReport,
    audit_fairness,
    average_ranks,
    average_satisfaction,
    calculate_blocking_pairs,
    calculate_ndcg_at_k,
    calculate_precision_at_k,
    calculate_vector_recall,
    conflict_metrics,
    coverage_metrics,
    evaluate,
    fill_rate,
    find_blocking_pairs,
)

__all__ = [
    "BenchmarkResult",
    "BlockingPairReport",
    "ConflictReport",
    "CoverageReport",
    "EvaluationReport",
    "FairnessAuditReport",
    "FillRateReport",
    "GreedyBaselineMatcher",
    "RankReport",
    "SatisfactionReport",
    "audit_fairness",
    "average_ranks",
    "average_satisfaction",
    "calculate_blocking_pairs",
    "calculate_ndcg_at_k",
    "calculate_precision_at_k",
    "calculate_vector_recall",
    "conflict_metrics",
    "coverage_metrics",
    "evaluate",
    "fill_rate",
    "find_blocking_pairs",
    "run_benchmark",
    "run_profile_benchmark",
    "to_evaluation_comparison_table",
    "to_markdown_table",
]

