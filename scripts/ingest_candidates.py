"""
Đọc file CSV ứng viên và gửi từng dòng lên API POST /candidates.

File CSV thật hiện có các cột: Họ tên, Nhóm khuyết tật, Kỹ năng, Khu vực,
Liên hệ, Ghi chú -- nhưng bảng candidates mới chỉ có 1 cột mô tả duy nhất
(`mo_ta`), vì vậy script này TỰ GỘP "Nhóm khuyết tật" + "Kỹ năng" (+ "Ghi chú"
nếu có) thành 1 đoạn `mo_ta` trước khi gửi lên API. Nội dung `mo_ta` càng đủ
thông tin thì embedding càng chính xác, vì đây là trường duy nhất được dùng
để tính vector và match với query của nhà tuyển dụng.

Cách dùng:
    1. Chạy `docker compose up -d` để API sống ở http://localhost:8000
    2. Đảm bảo file CSV nằm ở scripts/data/candidates.csv
    3. python scripts/ingest_candidates.py
"""

from pathlib import Path         # thao tác đường dẫn file
import pandas as pd              # đọc CSV
import requests                  # gửi HTTP request

API_URL = "http://localhost:8000/candidates"                     # endpoint tạo ứng viên
DATA_DIR = Path(__file__).parent / "data"                          # thư mục chứa dữ liệu
CSV_PATH = DATA_DIR / "candidates.csv"                               # đường dẫn file CSV ứng viên

# Tên các cột gốc trong file CSV thật hiện có
COL_HO_TEN = "Họ tên"
COL_NHOM_KHUYET_TAT = "Nhóm khuyết tật"
COL_KY_NANG = "Kỹ năng"
COL_KHU_VUC = "Khu vực"
COL_LIEN_HE = "Liên hệ"
COL_GHI_CHU = "Ghi chú"


def build_mo_ta(row: pd.Series) -> str:
    """
    Gộp các cột mô tả rời rạc (nhóm khuyết tật, kỹ năng, ghi chú) thành 1 đoạn
    văn bản duy nhất -- chính là nội dung sẽ được embed để tìm kiếm.
    """
    parts = []
    nhom = row.get(COL_NHOM_KHUYET_TAT, "")
    if pd.notna(nhom) and str(nhom).strip():
        parts.append(f"Dạng khuyết tật: {str(nhom).strip()}")

    ky_nang = row.get(COL_KY_NANG, "")
    if pd.notna(ky_nang) and str(ky_nang).strip():
        parts.append(f"Kỹ năng/kinh nghiệm: {str(ky_nang).strip()}")

    ghi_chu = row.get(COL_GHI_CHU, "")
    if pd.notna(ghi_chu) and str(ghi_chu).strip():
        parts.append(f"Ghi chú: {str(ghi_chu).strip()}")

    return ". ".join(parts) if parts else "Chưa có mô tả"


def row_to_payload(row: pd.Series) -> dict:
    """Chuyển 1 dòng CSV thành dict đúng field name của CandidateCreate."""
    ho_ten = row.get(COL_HO_TEN, "")
    khu_vuc = row.get(COL_KHU_VUC, "")
    lien_he = row.get(COL_LIEN_HE, "")
    ghi_chu = row.get(COL_GHI_CHU, "")

    return {
        "ho_ten": None if pd.isna(ho_ten) else str(ho_ten).strip(),
        "mo_ta": build_mo_ta(row),
        "khu_vuc": None if pd.isna(khu_vuc) else str(khu_vuc).strip(),
        "lien_he": None if pd.isna(lien_he) else str(lien_he).strip(),
        "ghi_chu": None if pd.isna(ghi_chu) else str(ghi_chu).strip(),
    }


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")      # đọc CSV vào DataFrame
    df = df.dropna(how="all")                                # bỏ dòng trống hoàn toàn
    print(f"Đọc được {len(df)} dòng ứng viên, bắt đầu gửi lên API...")

    success, failed = 0, 0                                    # đếm kết quả
    for i, row in df.iterrows():                               # duyệt từng ứng viên
        payload = row_to_payload(row)                            # convert dòng -> payload JSON
        response = requests.post(API_URL, json=payload)           # gọi POST /candidates
        if response.status_code == 200:
            success += 1
        else:
            failed += 1
            print(f"Lỗi dòng {i} ({payload.get('ho_ten')}): {response.status_code} - {response.text}")

        if (i + 1) % 100 == 0:                                   # in tiến độ mỗi 100 dòng
            print(f"... đã xử lý {i + 1}/{len(df)}")

    print(f"Hoàn tất: {success} thành công, {failed} lỗi.")


if __name__ == "__main__":
    main()