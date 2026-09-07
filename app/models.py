from datetime import datetime, timezone          # sinh timestamp mặc định khi tạo bản ghi
from sqlalchemy import Integer, Text, DateTime    # kiểu cột SQL cơ bản
from sqlalchemy.orm import Mapped, mapped_column   # khai báo cột kiểu ORM 2.0
from pgvector.sqlalchemy import Vector             # kiểu cột vector do pgvector cung cấp

from app.database import Base
from app.config import VECTOR_DIM


class Candidate(Base):
    """
    Bảng DUY NHẤT của hệ thống: ứng viên + mô tả của họ.

    Luồng: text (mo_ta) -> embedding vector -> lưu vào cột `embedding`.
    Khi nhà tuyển dụng gửi 1 câu mô tả yêu cầu công việc, câu đó cũng được
    encode thành vector rồi so cosine similarity trực tiếp với `embedding`
    của từng ứng viên trong bảng này -- không qua bảng trung gian nào cả.
    """
    __tablename__ = "candidates"

    # Khóa chính, tự tăng
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)

    # Họ tên ứng viên
    ho_ten: Mapped[str] = mapped_column(Text, nullable=False)

    # Mô tả về ứng viên: kỹ năng, kinh nghiệm, loại công việc phù hợp, tình trạng
    # sức khỏe/khuyết tật ảnh hưởng đến công việc, mong muốn... (text tự do,
    # đây chính là nội dung được embed để tìm kiếm)
    mo_ta: Mapped[str] = mapped_column(Text, nullable=False)

    # Khu vực sinh sống / mong muốn làm việc (chỉ để hiển thị & lọc thô nếu cần,
    # không dùng để tính embedding)
    khu_vuc: Mapped[str] = mapped_column(Text, nullable=True)

    # Thông tin liên hệ (SĐT / email) để nhà tuyển dụng liên hệ trực tiếp
    lien_he: Mapped[str] = mapped_column(Text, nullable=True)

    # Ghi chú thêm (tùy chọn)
    ghi_chu: Mapped[str] = mapped_column(Text, nullable=True)

    # Vector embedding của `mo_ta`, sinh ra bởi Qwen3-Embedding-0.6B (VECTOR_DIM chiều)
    embedding: Mapped[list[float]] = mapped_column(Vector(VECTOR_DIM), nullable=False)

    # Thời điểm ứng viên được thêm vào hệ thống
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )