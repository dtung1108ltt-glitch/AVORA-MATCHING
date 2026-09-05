"""
Sinh mock data THỰC TẾ (không random vô nghĩa) cho job_listings và candidates,
dùng để test luồng /documents, /candidates, /match.

Cách làm: định nghĩa sẵn các "khuôn mẫu" công việc gắn với nhóm khuyết tật
phù hợp thật sự (vd: khiếm thị hợp công việc không cần nhìn màn hình liên tục,
khiếm thính hợp công việc không cần nghe điện thoại...), rồi nhân bản theo
khu vực + mức lương + biến thể mô tả để ra đủ số lượng dòng mong muốn.

Cách dùng:
    python scripts/generate_mock_data.py
    # sinh ra scripts/data/job_listings.csv và scripts/data/candidates.csv
"""

import csv
import random
from pathlib import Path

random.seed(42)   # cố định seed để chạy lại vẫn ra cùng 1 bộ dữ liệu (tái lập được)

DATA_DIR = Path(__file__).parent / "data"   # nơi lưu file CSV output
DATA_DIR.mkdir(parents=True, exist_ok=True)   # tạo thư mục nếu chưa có

N_JOB_ROWS = 3000          # số dòng job_listings muốn sinh ra
N_CANDIDATE_ROWS = 3000    # số dòng candidates muốn sinh ra

# ---------------------------------------------------------------------------
# Danh mục nhóm khuyết tật -- PHẢI khớp tuyệt đối giữa job và candidate để
# bước /match gom nhóm chính xác
# ---------------------------------------------------------------------------
DISABILITY_GROUPS = [
    "Khiếm thính",
    "Khiếm thị",
    "Khuyết tật vận động",
    "Khuyết tật trí tuệ nhẹ",
    "Khuyết tật ngôn ngữ",
    "Khuyết tật thần kinh, tâm thần (ổn định)",
    "Đa khuyết tật nhẹ",
]

# Danh sách khu vực (tỉnh/thành) để nhân bản dữ liệu theo địa lý
KHU_VUC_LIST = [
    "Quận 1, TP.HCM", "Quận 3, TP.HCM", "Quận Bình Thạnh, TP.HCM",
    "Quận Tân Bình, TP.HCM", "TP. Thủ Đức, TP.HCM", "Quận Cầu Giấy, Hà Nội",
    "Quận Đống Đa, Hà Nội", "Quận Hai Bà Trưng, Hà Nội", "TP. Đà Nẵng",
    "TP. Cần Thơ", "TP. Biên Hòa, Đồng Nai", "TP. Thủ Dầu Một, Bình Dương",
    "TP. Nha Trang, Khánh Hòa", "TP. Huế", "TP. Vinh, Nghệ An",
    "Làm việc từ xa (Remote toàn quốc)",
]

# ---------------------------------------------------------------------------
# Khuôn mẫu công việc: mỗi archetype gắn với các nhóm khuyết tật PHÙ HỢP THẬT SỰ
# (vd: công việc cần nghe điện thoại thì không phù hợp nhóm khiếm thính)
# ---------------------------------------------------------------------------
JOB_ARCHETYPES = [
    {
        "nghe": "Nhân viên nhập liệu",
        "nganh": "Hành chính - Văn phòng",
        "mo_ta": "Nhập liệu, đối chiếu và cập nhật dữ liệu vào hệ thống Excel/phần mềm nội bộ. Công việc bàn giấy, không yêu cầu giao tiếp qua điện thoại.",
        "luong": "6-8 triệu/tháng",
        "ky_nang": "Tin học văn phòng, gõ phím nhanh, cẩn thận, tỉ mỉ",
        "phu_hop": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "nghe": "Nhân viên thiết kế đồ họa",
        "nganh": "Thiết kế - Sáng tạo",
        "mo_ta": "Thiết kế banner, ấn phẩm truyền thông bằng Photoshop/Illustrator/Canva theo brief từ khách hàng, làm việc online.",
        "luong": "8-15 triệu/tháng",
        "ky_nang": "Photoshop, Illustrator, Canva, tư duy thẩm mỹ",
        "phu_hop": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ", "Đa khuyết tật nhẹ"],
    },
    {
        "nghe": "Lập trình viên Backend",
        "nganh": "Công nghệ thông tin",
        "mo_ta": "Phát triển và bảo trì API, làm việc với database, có thể làm remote toàn thời gian hoặc bán thời gian.",
        "luong": "12-25 triệu/tháng",
        "ky_nang": "Python, SQL, Git, tư duy logic",
        "phu_hop": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "nghe": "Nhân viên trực chat/CSKH online",
        "nganh": "Chăm sóc khách hàng",
        "mo_ta": "Trả lời tin nhắn khách hàng qua Fanpage/Zalo/website, tư vấn sản phẩm, xử lý khiếu nại qua kênh chat (không gọi điện thoại).",
        "luong": "6-9 triệu/tháng",
        "ky_nang": "Viết tin nhắn rõ ràng, kiên nhẫn, hiểu sản phẩm",
        "phu_hop": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "nghe": "Nhân viên tổng đài chăm sóc khách hàng",
        "nganh": "Chăm sóc khách hàng",
        "mo_ta": "Nghe và tư vấn khách hàng qua điện thoại, ghi nhận yêu cầu, giọng nói rõ ràng, chủ động.",
        "luong": "7-10 triệu/tháng",
        "ky_nang": "Giao tiếp tốt, giọng nói rõ, xử lý tình huống",
        "phu_hop": ["Khuyết tật vận động", "Khuyết tật trí tuệ nhẹ"],
    },
    {
        "nghe": "Kỹ thuật viên xoa bóp bấm huyệt",
        "nganh": "Y tế - Chăm sóc sức khỏe",
        "mo_ta": "Thực hiện các liệu trình xoa bóp, bấm huyệt cho khách hàng tại cơ sở massage y học cổ truyền, đã qua đào tạo nghề.",
        "luong": "7-12 triệu/tháng (có hoa hồng)",
        "ky_nang": "Xoa bóp bấm huyệt, cảm nhận cơ thể tốt",
        "phu_hop": ["Khiếm thị"],
    },
    {
        "nghe": "Nhân viên tổng đài hỗ trợ bằng giọng nói (voice IVR content)",
        "nganh": "Công nghệ thông tin",
        "mo_ta": "Nghe và phân loại nội dung ghi âm, hỗ trợ xây dựng kịch bản thoại cho hệ thống tổng đài tự động, làm việc bằng tai nghe.",
        "luong": "7-10 triệu/tháng",
        "ky_nang": "Nghe hiểu tốt, gõ phím nhanh, tập trung",
        "phu_hop": ["Khiếm thị", "Khuyết tật vận động"],
    },
    {
        "nghe": "Nhân viên viết nội dung (Content Writer)",
        "nganh": "Truyền thông - Marketing",
        "mo_ta": "Viết bài blog, mô tả sản phẩm, kịch bản video ngắn theo yêu cầu, làm việc từ xa, deadline linh hoạt.",
        "luong": "6-10 triệu/tháng",
        "ky_nang": "Viết lách, nghiên cứu thông tin, SEO cơ bản",
        "phu_hop": ["Khiếm thính", "Khuyết tật vận động", "Khiếm thị", "Khuyết tật ngôn ngữ"],
    },
    {
        "nghe": "Thợ may công nghiệp",
        "nganh": "May mặc - Dệt may",
        "mo_ta": "May các công đoạn theo dây chuyền tại xưởng, đã qua đào tạo nghề may cơ bản, làm việc theo ca cố định.",
        "luong": "6-9 triệu/tháng",
        "ky_nang": "May công nghiệp, tỉ mỉ, chịu được áp lực dây chuyền",
        "phu_hop": ["Khiếm thính", "Khuyết tật trí tuệ nhẹ"],
    },
    {
        "nghe": "Nhân viên gia công thủ công mỹ nghệ",
        "nganh": "Thủ công mỹ nghệ",
        "mo_ta": "Gia công sản phẩm handmade (vòng tay, đồ trang trí, hộp quà) theo mẫu có sẵn tại xưởng hoặc nhận về nhà làm.",
        "luong": "4-7 triệu/tháng",
        "ky_nang": "Khéo tay, kiên nhẫn, tỉ mỉ",
        "phu_hop": ["Khiếm thính", "Khuyết tật trí tuệ nhẹ", "Khuyết tật vận động", "Đa khuyết tật nhẹ"],
    },
    {
        "nghe": "Nhân viên đóng gói sản phẩm",
        "nganh": "Sản xuất - Kho vận",
        "mo_ta": "Đóng gói, dán nhãn, kiểm đếm sản phẩm tại kho, công việc lặp lại theo quy trình đơn giản.",
        "luong": "5-7 triệu/tháng",
        "ky_nang": "Cẩn thận, làm theo quy trình, sức khỏe ổn định",
        "phu_hop": ["Khuyết tật trí tuệ nhẹ", "Khiếm thính", "Đa khuyết tật nhẹ"],
    },
    {
        "nghe": "Kế toán viên",
        "nganh": "Kế toán - Tài chính",
        "mo_ta": "Hạch toán chứng từ, lập báo cáo thuế cơ bản, sử dụng phần mềm kế toán MISA, làm việc tại văn phòng hoặc remote.",
        "luong": "8-12 triệu/tháng",
        "ky_nang": "MISA, Excel, nghiệp vụ kế toán",
        "phu_hop": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "nghe": "Giáo viên dạy tin học cơ bản",
        "nganh": "Giáo dục - Đào tạo",
        "mo_ta": "Giảng dạy kỹ năng tin học văn phòng cho học viên tại trung tâm dạy nghề cho người khuyết tật.",
        "luong": "7-11 triệu/tháng",
        "ky_nang": "Tin học văn phòng, kỹ năng sư phạm, kiên nhẫn",
        "phu_hop": ["Khuyết tật vận động", "Khiếm thị"],
    },
    {
        "nghe": "Nhân viên trực page bán hàng online",
        "nganh": "Thương mại điện tử",
        "mo_ta": "Quản lý fanpage/shop online, lên đơn, nhắn tin tư vấn khách hàng qua Messenger/Zalo (không gọi điện thoại).",
        "luong": "6-9 triệu/tháng",
        "ky_nang": "Chốt sale qua tin nhắn, quản lý đơn hàng, Excel cơ bản",
        "phu_hop": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "nghe": "Nhân viên biên phiên dịch ngôn ngữ ký hiệu",
        "nganh": "Truyền thông - Marketing",
        "mo_ta": "Hỗ trợ biên dịch nội dung sang ngôn ngữ ký hiệu cho video truyền thông, làm việc bán thời gian hoặc remote.",
        "luong": "6-10 triệu/tháng",
        "ky_nang": "Ngôn ngữ ký hiệu, biên tập video cơ bản",
        "phu_hop": ["Khiếm thính"],
    },
]

# Câu mở đầu/kết thúc để tạo biến thể mô tả tự nhiên, tránh trùng lặp 100%
INTRO_VARIANTS = [
    "Tuyển gấp: ", "Cần tuyển: ", "Cơ hội việc làm: ", "Thông báo tuyển dụng: ", "",
]
OUTRO_VARIANTS = [
    " Ưu tiên ứng viên có tinh thần trách nhiệm cao.",
    " Có hỗ trợ đào tạo lại từ đầu nếu chưa có kinh nghiệm.",
    " Môi trường làm việc thân thiện, hòa nhập.",
    " Xét duyệt hồ sơ và phỏng vấn trong tuần.",
    "",
]


def generate_jobs(n_rows: int) -> list[dict]:
    """Sinh n_rows dòng job_listings bằng cách nhân bản archetype theo khu vực/lương/biến thể mô tả."""
    rows = []
    for _ in range(n_rows):
        archetype = random.choice(JOB_ARCHETYPES)           # chọn 1 khuôn mẫu công việc
        khu_vuc = random.choice(KHU_VUC_LIST)                 # chọn khu vực ngẫu nhiên trong danh sách thật
        nhom = random.choice(archetype["phu_hop"])              # chỉ chọn nhóm khuyết tật THỰC SỰ phù hợp với job này
        mo_ta = (
            random.choice(INTRO_VARIANTS)
            + archetype["mo_ta"]
            + random.choice(OUTRO_VARIANTS)
        )                                                        # ghép mô tả với biến thể đầu/cuối cho tự nhiên
        rows.append({
            "Nghề/Công việc": archetype["nghe"],
            "Ngành/Lĩnh vực": archetype["nganh"],
            "Mô tả công việc": mo_ta.strip(),
            "Mức lương": archetype["luong"],
            "Khu vực phổ biến": khu_vuc,
            "Nhóm khuyết tật": nhom,
        })
    return rows


# Ngân hàng họ tên Việt Nam thật để ghép ngẫu nhiên -- tránh tên bịa vô nghĩa
HO_LIST = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Ngô", "Dương", "Lý"]
DEM_NAM = ["Văn", "Hữu", "Đình", "Quang", "Minh", "Anh", "Thành", "Xuân"]
DEM_NU = ["Thị", "Ngọc", "Thu", "Mỹ", "Kim", "Bích", "Diệu"]
TEN_NAM = ["Hùng", "Sơn", "Nam", "Hải", "Long", "Đức", "Phong", "Tuấn", "Khoa", "Bảo", "Kiên", "Vinh"]
TEN_NU = ["Hoa", "Lan", "Hương", "Linh", "Trang", "Nga", "Yến", "Thảo", "Mai", "Ngọc", "Vy", "Anh"]


def random_name() -> str:
    """Sinh 1 họ tên Việt Nam ngẫu nhiên nhưng hợp lệ, không dùng tên bịa."""
    ho = random.choice(HO_LIST)                       # họ
    if random.random() < 0.5:                            # 50% là tên "nam" theo cách đặt tên đệm/tên phổ biến
        return f"{ho} {random.choice(DEM_NAM)} {random.choice(TEN_NAM)}"
    return f"{ho} {random.choice(DEM_NU)} {random.choice(TEN_NU)}"   # 50% còn lại theo kiểu tên "nữ"


def generate_candidates(n_rows: int) -> list[dict]:
    """Sinh n_rows ứng viên, mỗi người gắn kỹ năng THỰC SỰ khớp với nhóm khuyết tật của họ."""
    rows = []
    for i in range(n_rows):
        nhom = random.choice(DISABILITY_GROUPS)                       # chọn nhóm khuyết tật của ứng viên này
        # Lấy các archetype công việc phù hợp với nhóm này để mượn danh sách kỹ năng cho hợp lý
        matching_archetypes = [a for a in JOB_ARCHETYPES if nhom in a["phu_hop"]]
        archetype = random.choice(matching_archetypes) if matching_archetypes else random.choice(JOB_ARCHETYPES)
        khu_vuc = random.choice(KHU_VUC_LIST)                            # khu vực sinh sống

        rows.append({
            "Họ tên": random_name(),
            "Nhóm khuyết tật": nhom,
            "Kỹ năng": archetype["ky_nang"],                              # kỹ năng mượn từ job phù hợp -> có nghĩa, không random
            "Khu vực": khu_vuc,
            "Liên hệ": f"09{random.randint(10000000, 99999999)}",           # số điện thoại giả dạng thật (đầu 09)
            "Ghi chú": "Sẵn sàng đi làm ngay" if random.random() < 0.7 else "Có thể bắt đầu sau 2 tuần",
        })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    """Ghi list[dict] ra file CSV với encoding utf-8-sig để giữ đúng dấu tiếng Việt."""
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)   # writer ghi theo đúng thứ tự cột đã khai báo
        writer.writeheader()                                    # ghi dòng header trước
        writer.writerows(rows)                                    # ghi toàn bộ dữ liệu


def main():
    jobs = generate_jobs(N_JOB_ROWS)                             # sinh dữ liệu job_listings
    candidates = generate_candidates(N_CANDIDATE_ROWS)             # sinh dữ liệu candidates

    job_path = DATA_DIR / "job_listings.csv"
    candidate_path = DATA_DIR / "candidates.csv"

    write_csv(job_path, jobs, list(jobs[0].keys()))                # ghi file CSV job
    write_csv(candidate_path, candidates, list(candidates[0].keys()))   # ghi file CSV candidate

    print(f"Đã sinh {len(jobs)} dòng job_listings -> {job_path}")
    print(f"Đã sinh {len(candidates)} dòng candidates -> {candidate_path}")


if __name__ == "__main__":
    main()
