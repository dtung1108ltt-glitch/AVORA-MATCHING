"""Cấu hình pytest: đảm bảo repo root nằm trên ``sys.path``.

Nhờ vậy các test có thể ``import src.core...`` dù chạy ``pytest`` từ bất kỳ đâu.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
