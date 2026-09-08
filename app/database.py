from sqlalchemy import create_engine, text          # tạo engine kết nối DB + chạy raw SQL
from sqlalchemy.orm import sessionmaker, DeclarativeBase   # tạo session và lớp Base cho ORM

from app.config import DATABASE_URL   # lấy chuỗi kết nối DB từ config

# Engine quản lý pool kết nối tới PostgreSQL
engine = create_engine(DATABASE_URL, echo=False)

# Factory tạo session làm việc với DB cho mỗi request
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Lớp cha cho tất cả model ORM (JobListing, Candidate, DisabilityGroup...)."""
    pass


def get_db():
    """Dependency của FastAPI: mở session, trả cho endpoint dùng, rồi luôn đóng lại."""
    db = SessionLocal()          # mở 1 session mới cho request hiện tại
    try:
        yield db                  # trả session cho endpoint sử dụng
    finally:
        db.close()                # đảm bảo session luôn được đóng, kể cả khi có lỗi


def init_db():
    """Bật extension pgvector và tạo toàn bộ bảng nếu chưa tồn tại."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))  # bật pgvector trong DB
        conn.commit()                                                    # lưu thay đổi extension
    Base.metadata.create_all(bind=engine)   # tạo tất cả bảng khai báo trong models.py
