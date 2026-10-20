"""
Convert file Excel (.xlsx) sang CSV sử dụng Polars (engine="calamine" / fastexcel)
để tăng tốc độ đọc và xuất dữ liệu sang chuẩn Apache Arrow.
Dùng chung được cho cả file job listing và file candidate.

Cách dùng:
    python scripts/convert_xlsx_to_csv.py <đường_dẫn_file.xlsx> <đường_dẫn_file.csv> [header_row]
"""

from __future__ import annotations

import sys
import polars as pl


def convert(xlsx_path: str, csv_path: str, header_row: int = 0) -> None:
    """Đọc file Excel tại xlsx_path bằng engine calamine, ghi ra CSV tại csv_path.
    header_row: dòng chứa tên cột (0-indexed).
    """
    df = pl.read_excel(
        xlsx_path,
        engine="calamine",
        read_options={"header_row": header_row},
    )
    df.write_csv(csv_path, include_bom=True)
    print(f"Đã convert {xlsx_path} -> {csv_path} ({df.height} dòng).")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Cách dùng: python convert_xlsx_to_csv.py <input.xlsx> <output.csv> [header_row]")
        sys.exit(1)

    xlsx_arg = sys.argv[1]
    csv_arg = sys.argv[2]
    header_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    convert(xlsx_arg, csv_arg, header_arg)
