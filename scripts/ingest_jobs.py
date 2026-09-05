"""
Đọc file CSV công việc và gửi từng dòng lên API POST /documents.
Gửi qua API (thay vì ghi thẳng DB) để tận dụng luôn logic tính embedding
đã viết sẵn trong app/main.py, tránh lặp code.

PLACEHOLDER: COLUMN_MAP đang là tên cột TẠM. Khi có file Excel/CSV thật,
chỉ cần sửa key bên trái (tên cột trong file) cho khớp.

Cách dùng:
    1. Chạy `docker compose up -d` để API sống ở http://localhost:8000
    2. Convert Excel -> CSV bằng convert_xlsx_to_csv.py (nếu cần)
    3. Đặt file CSV vào scripts/data/job_listings.csv
    4. python scripts/ingest_jobs.py
"""

from pathlib import Path         # thao tác đường dẫn file gọn hơn os.path
import pandas as pd              # đọc CSV thành DataFrame
import requests                  # gửi HTTP request tới API

API_URL = "http://localhost:8000/documents"                      # endpoint tạo job listing
DATA_DIR = Path(__file__).parent / "data"                          # thư mục chứa file dữ liệu
CSV_PATH = DATA_DIR / "job_listings.csv"                            # đường dẫn file CSV công việc

# Ánh xạ tên cột trong file (PLACEHOLDER) -> tên field trong JobListingCreate
COLUMN_MAP = {
    "Nghề/Công việc": "nghe_cong_viec",
    "Ngành/Lĩnh vực": "nganh_linh_vuc",
    "Mô tả công việc": "mo_ta_cong_viec",
    "Mức lương": "muc_luong",
    "Khu vực phổ biến": "khu_vuc_pho_bien",
    "Nhóm khuyết tật": "nhom_khuyet_tat",
}


def row_to_payload(row: pd.Series) -> dict:
    """Chuyển 1 dòng CSV thành dict đúng field name của JobListingCreate."""
    payload = {}
    for csv_col, field in COLUMN_MAP.items():
        value = row.get(csv_col, "")                       # lấy giá trị theo tên cột gốc trong file
        payload[field] = None if pd.isna(value) else str(value).strip()   # chuẩn hóa NaN -> None
    return payload


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")        # đọc toàn bộ CSV vào DataFrame
    df = df.dropna(how="all")                                  # bỏ các dòng trống hoàn toàn
    print(f"Đọc được {len(df)} dòng công việc, bắt đầu gửi lên API...")

    success, failed = 0, 0                                      # đếm số dòng thành công / lỗi
    for _, row in df.iterrows():                                 # duyệt từng dòng dữ liệu
        payload = row_to_payload(row)                              # convert dòng -> payload JSON
        response = requests.post(API_URL, json=payload)             # gọi POST /documents
        if response.status_code == 200:                              # thành công
            success += 1
        else:                                                          # lỗi -> in ra để debug
            failed += 1
            print(f"Lỗi dòng {payload.get('nghe_cong_viec')}: {response.status_code} - {response.text}")

    print(f"Hoàn tất: {success} thành công, {failed} lỗi.")


if __name__ == "__main__":
    main()
