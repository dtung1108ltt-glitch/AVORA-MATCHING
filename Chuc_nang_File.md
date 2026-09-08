generate_mock_data.py : tạo mock data nhóm khuyết tật | mô tả công việc
generate_mock_candidates.py: tạo mock data danh sách ứng viên nhóm khuyết tật | Họ và tên

ingest_candidates.py: 
- CandidateCreate phải có 2 trường: ho_ten, nhom_khuyet_tat
- POST /candidates không tạo embedding cho ứng viên
- Bảng candidates phải có: id, ho_ten, nhom_khuyet_tat
ingest_job_descriptions.py
1. Nhận nhóm khuyết tật và mô tả công việc.
2. Chuyển riêng mo_ta_cong_viec thành embedding với is_query=False.
3. Lưu nhóm, mô tả và vector vào PostgreSQL/pgvector.

