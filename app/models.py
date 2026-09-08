from __future__ import annotations

from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Index, Integer, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.config import VECTOR_DIM
from app.database import Base


class JobDescription(Base):
    """
    Mô tả công việc mẫu dùng làm kho vector để matching.

    Workflow:
        mo_ta_cong_viec
            -> Qwen embedding với is_query=False
            -> lưu vào embedding

    Khi tìm kiếm:
        query
            -> Qwen embedding với is_query=True
            -> cosine similarity với JobDescription.embedding
            -> lấy nhom_khuyet_tat của kết quả phù hợp
            -> lọc bảng candidates theo nhóm đó
    """

    __tablename__ = "job_descriptions"
    __table_args__ = (
        Index("ix_job_descriptions_nhom", "nhom_khuyet_tat"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    nhom_khuyet_tat: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    mo_ta_cong_viec: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(VECTOR_DIM),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class Candidate(Base):
    """
    Danh sách ứng viên để lọc theo nhóm khuyết tật.

    Ứng viên không được tạo embedding trong workflow hiện tại.
    Sau khi /match xác định nhóm phù hợp, hệ thống truy vấn:
        Candidate.nhom_khuyet_tat IN selected_groups
    """

    __tablename__ = "candidates"
    __table_args__ = (
        Index("ix_candidates_nhom", "nhom_khuyet_tat"),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    ho_ten: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    nhom_khuyet_tat: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
