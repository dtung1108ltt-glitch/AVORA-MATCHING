"""
Convert file Excel (.xlsx) sang CSV để dễ đọc bằng pandas ở bước ingest.
Dùng chung được cho cả file job listing và file candidate.

Cách dùng:
    python scripts/convert_xlsx_to_csv.py <đường_dẫn_file.xlsx> <đường_dẫn_file.csv> [header_row]
"""

import sys                       # đọc tham số dòng lệnh
import pandas as pd              # đọc Excel, ghi CSV


def convert(xlsx_path: str, csv_path: str, header_row: int = 0):
    """Đọc file Excel tại xlsx_path, ghi ra CSV tại csv_path. header_row: dòng chứa tên cột (0-indexed)."""
    df = pd.read_excel(xlsx_path, header=header_row)         # đọc sheet đầu tiên, lấy header ở đúng dòng chỉ định
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")    # ghi CSV, utf-8-sig để Excel/pandas đọc lại đúng tiếng Việt
    print(f"Đã convert {xlsx_path} -> {csv_path} ({len(df)} dòng).")


if __name__ == "__main__":
    if len(sys.argv) < 3:                                      # cần ít nhất 2 tham số: input và output
        print("Cách dùng: python convert_xlsx_to_csv.py <input.xlsx> <output.csv> [header_row]")
        sys.exit(1)

    xlsx_arg = sys.argv[1]                                      # đường dẫn file Excel đầu vào
    csv_arg = sys.argv[2]                                        # đường dẫn file CSV đầu ra
    header_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 0     # dòng header, mặc định dòng đầu tiên

    convert(xlsx_arg, csv_arg, header_arg)
