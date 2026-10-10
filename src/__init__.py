"""AVORA Matching core engine.

Package ``src`` chứa engine ghép cặp ổn định hai phía (Two-Sided Stable
Matching) cho bài toán thị trường việc làm giữa Ứng viên (Candidates) và
Nhà tuyển dụng / Vị trí tuyển dụng (Employers / Jobs).

Cấu trúc:

* :mod:`src.core`       - tiền xử lý, engine Gale-Shapley, hậu xử lý Top-K.
* :mod:`src.evaluation` - đo lường chất lượng (blocking pairs, satisfaction,
  fill rate) và benchmark so với baseline.
"""

from __future__ import annotations

__all__ = ["core", "evaluation"]
