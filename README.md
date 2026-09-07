# Avora Matching

Hệ thống gợi ý **danh sách ứng viên khuyết tật** phù hợp nhất với một yêu cầu
tuyển dụng, dựa trên vector search bằng pgvector + Qwen3-Embedding-0.6B.

## Kiến trúc DB (1 bảng duy nhất)

```
candidates (ho_ten, mo_ta, khu_vuc, lien_he, ghi_chu, embedding)
```

Không có bảng trung gian "nhóm khuyết tật" hay "mô tả công việc" — chỉ có
ứng viên và mô tả về họ. Mọi thứ nằm gọn trong 1 bảng.

## Luồng 1 — nạp dữ liệu ứng viên

```
POST /candidates
  -> Qwen3-Embedding-0.6B encode "mo_ta" -> vector 1024 chiều
  -> lưu vào candidates (kèm embedding)
```

## Luồng 2 — nhà tuyển dụng tìm ứng viên

```
POST /match {"query": "..."}
  -> encode query bằng Qwen3-Embedding-0.6B (prompt "query")
  -> cosine similarity trực tiếp với embedding của TỪNG ứng viên
  -> xếp hạng theo similarity, lấy top K
  -> trả về candidates kèm điểm similarity
```

Đây là luồng text -> embedding vector -> query đơn giản nhất: không qua
bước "chọn nhóm" trung gian, hệ thống so trực tiếp mô tả yêu cầu tuyển dụng
với mô tả từng ứng viên.

## Chạy thử

```bash
docker compose up -d --build      # dựng db + web
# API chạy ở http://localhost:8000, docs tự sinh ở http://localhost:8000/docs
```

Nạp dữ liệu mẫu (sau khi convert Excel -> CSV và đặt đúng field trong
`COLUMN_MAP` của `scripts/ingest_candidates.py`):

```bash
python scripts/convert_xlsx_to_csv.py scripts/data/candidates.xlsx scripts/data/candidates.csv
python scripts/ingest_candidates.py
```

Gọi thử matching:

```bash
curl -X POST http://localhost:8000/match \
  -H "Content-Type: application/json" \
  -d '{"query": "Cần tuyển nhân viên nhập liệu, ưu tiên làm việc tại nhà"}'
```