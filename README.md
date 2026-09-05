# Avora Matching

Hệ thống gợi ý **danh sách ứng viên khuyết tật** phù hợp nhất với một yêu cầu
tuyển dụng, dựa trên vector search bằng pgvector + Qwen3-Embedding-0.6B.

## Kiến trúc DB (3 bảng, có FK)

```
disability_groups (nhóm khuyết tật)
       ▲                    ▲
       │ FK                 │ FK
  job_listings          candidates
  (mô tả công việc)     (danh sách ứng viên)
```

## Luồng 1 — nạp dữ liệu công việc

```
POST /documents
  -> ghép field thành "content" (mô tả cv)
  -> Qwen3-Embedding-0.6B encode -> vector 1024 chiều
  -> lưu vào job_listings (kèm FK disability_group_id)
```

## Luồng 2 — nhà tuyển dụng tìm ứng viên

```
POST /match {"query": "..."}
  -> dịch/encode query bằng Qwen3-Embedding-0.6B (prompt "query")
  -> cosine similarity, lấy K job_listings giống nhất
  -> gom theo disability_group_id, tính điểm trung bình similarity
  -> chọn nhóm có điểm cao nhất
  -> query candidates theo disability_group_id đã chọn
  -> trả về selected_group + group_scores + matched_jobs + candidates
```

## Chạy thử

```bash
docker compose up -d --build      # dựng db + web
# API chạy ở http://localhost:8000, docs tự sinh ở http://localhost:8000/docs
```

Nạp dữ liệu mẫu (sau khi convert Excel -> CSV và đặt đúng field trong
`COLUMN_MAP` của từng script):

```bash
python scripts/convert_xlsx_to_csv.py scripts/data/job_listings.xlsx scripts/data/job_listings.csv
python scripts/ingest_jobs.py

python scripts/convert_xlsx_to_csv.py scripts/data/candidates.xlsx scripts/data/candidates.csv
python scripts/ingest_candidates.py
```

Gọi thử matching:

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"query": "Cần tuyển nhân viên nhập liệu, ưu tiên làm việc tại nhà"}'
```
