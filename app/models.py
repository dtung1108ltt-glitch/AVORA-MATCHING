from datetime import datetime, timezone          # sinh timestamp mặc định khi tạo bản ghi
from sqlalchemy import Integer, Text, DateTime, ForeignKey   # kiểu cột SQL cơ bản
from sqlalchemy.orm import Mapped, mapped_column, relationship  # khai báo cột kiểu ORM 2.0
from pgvector.sqlalchemy import Vector             # kiểu cột vector do pgvector cung cấp

from app.database import Base
from app.config import VECTOR_DIM


class DisabilityGroup(Base):
    """
    Bảng "nhóm khuyết tật" -- đúng ô 'nhóm khuyết tật' trong sơ đồ.
    Đây là bảng danh mục (lookup table), cả job_listings và candidates
    đều tham chiếu (FK) tới bảng này, giống 2 mũi tên nét đứt trong hình.
    """
    __tablename__ = "disability_groups"

    # Khóa chính, tự tăng
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)

    # Tên nhóm khuyết tật, ví dụ: "Khiếm thính", "Khiếm thị", "Vận động"...
    ten_nhom: Mapped[str] = mapped_column(Text, unique=True, nullable=False, index=True)

    # Quan hệ ngược lại: từ 1 nhóm có thể lấy ra list job / candidate thuộc nhóm đó (tiện khi debug)
    job_listings: Mapped[list["JobListing"]] = relationship(back_populates="disability_group")
    candidates: Mapped[list["Candidate"]] = relationship(back_populates="disability_group")


class JobListing(Base):
    """
    Bảng "mô tả công việc" -- ứng với luồng trên của sơ đồ:
    database -> mô tả công việc -> lấy mô tả dạng text -> mô tả cv
    -> model embedding -> vector embedding -> lưu vào DB.
    """
    __tablename__ = "job_listings"

    # Khóa chính, tự tăng
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)

    # Tên nghề / vị trí công việc
    nghe_cong_viec: Mapped[str] = mapped_column(Text, nullable=False)

    # Ngành / lĩnh vực hoạt động
    nganh_linh_vuc: Mapped[str] = mapped_column(Text, nullable=True)

    # Mô tả chi tiết công việc (nội dung chính dùng để embedding)
    mo_ta_cong_viec: Mapped[str] = mapped_column(Text, nullable=False)

    # Mức lương đề xuất (dạng text vì dữ liệu gốc có thể ghi khoảng lương)
    muc_luong: Mapped[str] = mapped_column(Text, nullable=True)

    # Khu vực làm việc phổ biến
    khu_vuc_pho_bien: Mapped[str] = mapped_column(Text, nullable=True)

    # FK tới bảng disability_groups -- ứng với mũi tên nét đứt "mô tả cv -> nhóm khuyết tật"
    disability_group_id: Mapped[int] = mapped_column(
        ForeignKey("disability_groups.id"), nullable=False, index=True
    )

    # Đoạn text tổng hợp từ các trường trên, chính là ô "mô tả cv" trong sơ đồ,
    # được ghép lại trước khi đưa vào model embedding
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Vector embedding của "content", sinh ra bởi Qwen3-Embedding-0.6B (1024 chiều)
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIM), nullable=False)

    # Thời điểm bản ghi được tạo
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Quan hệ để truy ngược ra tên nhóm khuyết tật từ 1 job (job.disability_group.ten_nhom)
    disability_group: Mapped["DisabilityGroup"] = relationship(back_populates="job_listings")


class Candidate(Base):
    """
    Bảng "danh sách ứng viên" -- ứng với ô 'danh sách ứng viên' trong sơ đồ,
    cũng tham chiếu tới disability_groups (mũi tên nét đứt thứ 2), và là
    nguồn dữ liệu cho bước cuối "Lấy ra danh sách ứng viên của nhóm đối tượng đó".
    """
    __tablename__ = "candidates"

    # Khóa chính, tự tăng
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)

    # Họ tên ứng viên
    ho_ten: Mapped[str] = mapped_column(Text, nullable=False)

    # FK tới bảng disability_groups -- xác định ứng viên thuộc nhóm khuyết tật nào
    disability_group_id: Mapped[int] = mapped_column(
        ForeignKey("disability_groups.id"), nullable=False, index=True
    )

    # Kỹ năng / kinh nghiệm của ứng viên (text tự do)
    ky_nang: Mapped[str] = mapped_column(Text, nullable=True)

    # Khu vực sinh sống / mong muốn làm việc
    khu_vuc: Mapped[str] = mapped_column(Text, nullable=True)

    # Thông tin liên hệ (SĐT / email) để nhà tuyển dụng liên hệ trực tiếp
    lien_he: Mapped[str] = mapped_column(Text, nullable=True)

    # Ghi chú thêm (mức độ khuyết tật, tình trạng sẵn sàng đi làm...)
    ghi_chu: Mapped[str] = mapped_column(Text, nullable=True)

    # Thời điểm ứng viên được thêm vào hệ thống
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Quan hệ để truy ngược ra tên nhóm khuyết tật từ 1 candidate (candidate.disability_group.ten_nhom)
    disability_group: Mapped["DisabilityGroup"] = relationship(back_populates="candidates")
