"""
Đọc file CSV ứng viên và gửi từng dòng lên API POST /candidates.

PLACEHOLDER: COLUMN_MAP đang là tên cột TẠM. Khi có file Excel/CSV ứng viên
thật, chỉ cần sửa key bên trái cho khớp tên cột thật, không cần đổi gì khác.

Cách dùng:
    1. Chạy `docker compose up -d` để API sống ở http://localhost:8000
    2. Convert Excel -> CSV bằng convert_xlsx_to_csv.py (nếu cần)
    3. Đặt file CSV vào scripts/data/candidates.csv
    4. python scripts/ingest_candidates.py
"""

from pathlib import Path         # thao tác đường dẫn file
import pandas as pd              # đọc CSV
import requests                  # gửi HTTP request

API_URL = "http://localhost:8000/candidates"                     # endpoint tạo ứng viên
DATA_DIR = Path(__file__).parent / "data"                          # thư mục chứa dữ liệu
CSV_PATH = DATA_DIR / "candidates.csv"                               # đường dẫn file CSV ứng viên

# Ánh xạ tên cột trong file (PLACEHOLDER) -> tên field trong CandidateCreate
COLUMN_MAP = {
    "Họ tên": "ho_ten",
    "Nhóm khuyết tật": "nhom_khuyet_tat",   # PHẢI cùng hệ giá trị với cột nhóm khuyết tật của job_listings
    "Kỹ năng": "ky_nang",
    "Khu vực": "khu_vuc",
    "Liên hệ": "lien_he",
    "Ghi chú": "ghi_chu",
}


def row_to_payload(row: pd.Series) -> dict:
    """Chuyển 1 dòng CSV thành dict đúng field name của CandidateCreate."""
    payload = {}
    for csv_col, field in COLUMN_MAP.items():
        value = row.get(csv_col, "")                     # lấy giá trị theo tên cột gốc
        payload[field] = None if pd.isna(value) else str(value).strip()   # chuẩn hóa NaN -> None
    return payload


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")      # đọc CSV vào DataFrame
    df = df.dropna(how="all")                                # bỏ dòng trống hoàn toàn
    print(f"Đọc được {len(df)} dòng ứng viên, bắt đầu gửi lên API...")

    success, failed = 0, 0                                    # đếm kết quả
    for _, row in df.iterrows():                               # duyệt từng ứng viên
        payload = row_to_payload(row)                            # convert dòng -> payload JSON
        response = requests.post(API_URL, json=payload)           # gọi POST /candidates
        if response.status_code == 200:
            success += 1
        else:
            failed += 1
            print(f"Lỗi dòng {payload.get('ho_ten')}: {response.status_code} - {response.text}")

    print(f"Hoàn tất: {success} thành công, {failed} lỗi.")


if __name__ == "__main__":
    main()
