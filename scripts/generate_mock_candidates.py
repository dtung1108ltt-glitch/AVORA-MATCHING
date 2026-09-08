#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_mock_candidates.py

Tạo dữ liệu ứng viên giả phục vụ AVORA Matching.
Database đầu ra chỉ gồm đúng 2 cột:
- Họ và tên
- Nhóm khuyết tật

Workflow:
    POST /match
        -> tìm nhóm khuyết tật phù hợp từ job_descriptions
        -> lọc candidates.csv theo cột "Nhóm khuyết tật"
        -> trả danh sách ứng viên thuộc nhóm tương ứng

Dữ liệu được tạo hoàn toàn giả lập, không đại diện cho người thật.
Output mặc định: scripts/data/candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_COUNT = 3000
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "candidates.csv"

DISABILITY_GROUPS = (
    "Khiếm thính",
    "Khiếm thị",
    "Khuyết tật vận động",
    "Khuyết tật trí tuệ nhẹ",
    "Khuyết tật ngôn ngữ",
    "Khuyết tật thần kinh, tâm thần (ổn định)",
    "Đa khuyết tật nhẹ",
)

FAMILY_NAMES = (
    "Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ",
    "Võ", "Đặng", "Bùi", "Đỗ", "Ngô", "Dương", "Lý", "Đinh",
)

MIDDLE_NAMES = (
    "Văn", "Thị", "Minh", "Quang", "Ngọc", "Thu", "Hữu", "Kim",
    "Bích", "Anh", "Thanh", "Gia", "Hoài", "Khánh", "Đức", "Tường",
)

GIVEN_NAMES = (
    "An", "Anh", "Bảo", "Châu", "Dũng", "Đạt", "Đức", "Giang",
    "Hà", "Hải", "Hiếu", "Hoa", "Hùng", "Hương", "Khang", "Khoa",
    "Kiên", "Lan", "Linh", "Long", "Mai", "Minh", "Nam", "Nga",
    "Ngân", "Ngọc", "Nhân", "Phong", "Phúc", "Quân", "Quỳnh", "Sơn",
    "Tâm", "Thảo", "Thành", "Thiện", "Trang", "Trung", "Tuấn", "Uyên",
    "Việt", "Vinh", "Vy", "Xuân", "Yến",
)


def normalize_text(value: str) -> str:
    """Chuẩn hóa Unicode và khoảng trắng."""
    return " ".join(unicodedata.normalize("NFC", value).split())


def build_name_pool() -> list[str]:
    """Tạo tập tên giả đủ lớn và không trùng."""
    names = {
        normalize_text(f"{family} {middle} {given}")
        for family in FAMILY_NAMES
        for middle in MIDDLE_NAMES
        for given in GIVEN_NAMES
    }
    return sorted(names)


def allocate_group_counts(total: int) -> dict[str, int]:
    """
    Chia ứng viên gần đều cho tất cả nhóm.
    Phần dư được phân lần lượt từ nhóm đầu tiên.
    """
    base, remainder = divmod(total, len(DISABILITY_GROUPS))
    return {
        group: base + (1 if index < remainder else 0)
        for index, group in enumerate(DISABILITY_GROUPS)
    }


def generate_candidates(count: int, rng: random.Random) -> list[dict[str, str]]:
    """Sinh đúng count ứng viên, tên không trùng, các nhóm được phân bố gần đều."""
    name_pool = build_name_pool()
    if count > len(name_pool):
        raise ValueError(
            f"Số lượng yêu cầu {count} vượt quá {len(name_pool)} tên giả duy nhất có thể tạo."
        )

    selected_names = rng.sample(name_pool, count)
    group_counts = allocate_group_counts(count)

    assigned_groups: list[str] = []
    for group, group_count in group_counts.items():
        assigned_groups.extend([group] * group_count)
    rng.shuffle(assigned_groups)

    rows = [
        {
            "Họ và tên": name,
            "Nhóm khuyết tật": group,
        }
        for name, group in zip(selected_names, assigned_groups, strict=True)
    ]

    rng.shuffle(rows)
    return rows


def validate_candidates(rows: list[dict[str, str]], expected_count: int) -> None:
    """Kiểm tra số dòng, tên, nhãn và phân bố nhóm."""
    if len(rows) != expected_count:
        raise ValueError(f"Sai số dòng: cần {expected_count}, nhận {len(rows)}.")

    names = [row["Họ và tên"].strip() for row in rows]
    if any(not name for name in names):
        raise ValueError("Có ứng viên thiếu họ tên.")
    if len(names) != len(set(name.casefold() for name in names)):
        raise ValueError("Có họ tên bị trùng.")

    allowed = set(DISABILITY_GROUPS)
    invalid_groups = sorted({
        row["Nhóm khuyết tật"]
        for row in rows
        if row["Nhóm khuyết tật"] not in allowed
    })
    if invalid_groups:
        raise ValueError(f"Có nhóm khuyết tật không hợp lệ: {invalid_groups}")

    counts = Counter(row["Nhóm khuyết tật"] for row in rows)
    if expected_count >= len(DISABILITY_GROUPS):
        missing = [group for group in DISABILITY_GROUPS if counts[group] == 0]
        if missing:
            raise ValueError(f"Thiếu dữ liệu cho các nhóm: {missing}")

    if counts and max(counts.values()) - min(counts.values()) > 1:
        raise ValueError("Phân bố giữa các nhóm không cân bằng như dự kiến.")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    """Ghi CSV UTF-8 BOM để mở đúng tiếng Việt trong Excel."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Họ và tên", "Nhóm khuyết tật"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo candidates.csv giả gồm họ tên và nhóm khuyết tật."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Số ứng viên cần tạo, mặc định {DEFAULT_COUNT}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed, mặc định {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"File CSV đầu ra, mặc định {DEFAULT_OUTPUT}.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.count <= 0:
        print("Lỗi: --count phải lớn hơn 0.", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rows = generate_candidates(args.count, rng)
    validate_candidates(rows, args.count)
    write_csv(args.output, rows)

    counts = Counter(row["Nhóm khuyết tật"] for row in rows)
    print(f"Đã tạo {len(rows)} ứng viên giả.")
    print(f"File output: {args.output}")
    print("Các cột: Họ và tên, Nhóm khuyết tật")
    print("Phân bố:")
    for group in DISABILITY_GROUPS:
        print(f"  - {group}: {counts[group]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
