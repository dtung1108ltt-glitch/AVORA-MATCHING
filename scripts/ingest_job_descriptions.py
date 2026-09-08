#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_job_descriptions.py

Đọc file CSV có đúng 2 cột:
- Nhóm khuyết tật
- Mô tả công việc

Sau đó gửi từng dòng tới:
    POST http://localhost:8000/job-descriptions

Payload:
    {
        "nhom_khuyet_tat": "Khiếm thính",
        "mo_ta_cong_viec": "Nhập và đối chiếu dữ liệu trên Excel..."
    }

Backend chịu trách nhiệm:
1. Nhận nhóm khuyết tật và mô tả công việc.
2. Chuyển riêng mo_ta_cong_viec thành embedding với is_query=False.
3. Lưu nhóm, mô tả và vector vào PostgreSQL/pgvector.

Dữ liệu CSV không chứa vector.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "job_descriptions.csv"
DEFAULT_API_URL = "http://localhost:8000/job-descriptions"
DEFAULT_TIMEOUT = 120.0
DEFAULT_PROGRESS_EVERY = 25

CSV_GROUP_COLUMN = "Nhóm khuyết tật"
CSV_DESCRIPTION_COLUMN = "Mô tả công việc"

ALLOWED_GROUPS = {
    "Khiếm thính",
    "Khiếm thị",
    "Khuyết tật vận động",
    "Khuyết tật trí tuệ nhẹ",
    "Khuyết tật ngôn ngữ",
    "Khuyết tật thần kinh, tâm thần (ổn định)",
    "Đa khuyết tật nhẹ",
}


def normalize_text(value: Any) -> str:
    """Chuẩn hóa Unicode và khoảng trắng, giữ nguyên nội dung tiếng Việt."""
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def create_session() -> requests.Session:
    """Tạo HTTP session có retry cho lỗi kết nối và lỗi server tạm thời."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Content-Type": "application/json"})
    return session


def read_job_descriptions(
    csv_path: Path,
    min_description_length: int,
) -> tuple[list[dict[str, str]], list[str]]:
    """
    Đọc và kiểm tra file CSV.

    Trả về:
    - payload hợp lệ để gửi API
    - danh sách cảnh báo cho các dòng bị bỏ qua
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Đường dẫn không phải file: {csv_path}")

    payloads: list[dict[str, str]] = []
    warnings: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        original_headers = reader.fieldnames or []
        normalized_headers = [normalize_text(header) for header in original_headers]
        expected_headers = [CSV_GROUP_COLUMN, CSV_DESCRIPTION_COLUMN]

        missing = [x for x in expected_headers if x not in normalized_headers]
        extra = [x for x in normalized_headers if x not in expected_headers]

        if missing:
            raise ValueError(
                "CSV thiếu cột bắt buộc: "
                + ", ".join(missing)
                + f". Header hiện tại: {normalized_headers}"
            )
        if extra:
            raise ValueError(
                "CSV phải có đúng 2 cột. Cột thừa: " + ", ".join(extra)
            )

        header_map = {
            normalize_text(original): original
            for original in original_headers
        }

        for line_number, row in enumerate(reader, start=2):
            group = normalize_text(row.get(header_map[CSV_GROUP_COLUMN]))
            description = normalize_text(row.get(header_map[CSV_DESCRIPTION_COLUMN]))

            if not group and not description:
                warnings.append(f"Dòng {line_number}: dòng trống, đã bỏ qua.")
                continue
            if not group:
                warnings.append(
                    f"Dòng {line_number}: thiếu Nhóm khuyết tật, đã bỏ qua."
                )
                continue
            if group not in ALLOWED_GROUPS:
                warnings.append(
                    f"Dòng {line_number}: nhóm không hợp lệ '{group}', đã bỏ qua."
                )
                continue
            if not description:
                warnings.append(
                    f"Dòng {line_number} ({group}): thiếu Mô tả công việc, đã bỏ qua."
                )
                continue
            if len(description) < min_description_length:
                warnings.append(
                    f"Dòng {line_number} ({group}): mô tả chỉ có "
                    f"{len(description)} ký tự, ngắn hơn {min_description_length}, đã bỏ qua."
                )
                continue

            pair = (group.casefold(), description.casefold())
            if pair in seen_pairs:
                warnings.append(
                    f"Dòng {line_number} ({group}): trùng cặp nhóm và mô tả, đã bỏ qua."
                )
                continue
            seen_pairs.add(pair)

            payloads.append({
                "nhom_khuyet_tat": group,
                "mo_ta_cong_viec": description,
            })

    return payloads, warnings


def api_base_url(api_url: str) -> str:
    suffix = "/job-descriptions"
    normalized = api_url.rstrip("/")
    if normalized.endswith(suffix):
        return normalized[:-len(suffix)]
    return normalized


def check_api(session: requests.Session, api_url: str, timeout: float) -> None:
    """Kiểm tra FastAPI có phản hồi trước khi bắt đầu ingest."""
    base_url = api_base_url(api_url)
    check_urls = [f"{base_url}/openapi.json", f"{base_url}/docs"]
    last_error: Exception | None = None

    for url in check_urls:
        try:
            response = session.get(url, timeout=min(timeout, 30.0))
            if response.status_code < 500:
                return
        except requests.RequestException as exc:
            last_error = exc

    message = (
        f"Không kết nối được FastAPI tại {base_url}. "
        "Hãy chạy 'docker compose up --build -d' trước khi ingest."
    )
    if last_error is not None:
        message += f" Chi tiết: {last_error}"
    raise ConnectionError(message)


def extract_error(response: requests.Response) -> str:
    """Lấy thông báo lỗi ngắn gọn từ response API."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip() or response.reason

    if isinstance(body, dict) and "detail" in body:
        return str(body["detail"])
    return str(body)


def ingest(
    payloads: list[dict[str, str]],
    api_url: str,
    timeout: float,
    progress_every: int,
    delay: float,
    dry_run: bool,
    max_errors: int,
) -> dict[str, Any]:
    """Gửi từng payload tới POST /job-descriptions."""
    stats: dict[str, Any] = {
        "total_valid": len(payloads),
        "success": 0,
        "failed": 0,
        "group_success": Counter(),
        "errors": [],
    }

    if dry_run:
        print("DRY RUN: không gửi dữ liệu lên API.")
        for index, payload in enumerate(payloads[:5], start=1):
            preview = payload["mo_ta_cong_viec"]
            if len(preview) > 180:
                preview = preview[:177] + "..."
            print(
                f"Mẫu {index}: "
                f"nhom_khuyet_tat={payload['nhom_khuyet_tat']!r}, "
                f"mo_ta_cong_viec={preview!r}"
            )
        if len(payloads) > 5:
            print(f"... còn {len(payloads) - 5} payload khác.")
        return stats

    session = create_session()
    check_api(session, api_url, timeout)

    try:
        for index, payload in enumerate(payloads, start=1):
            try:
                response = session.post(
                    api_url,
                    json=payload,
                    timeout=timeout,
                )

                if response.status_code in (200, 201):
                    stats["success"] += 1
                    stats["group_success"][payload["nhom_khuyet_tat"]] += 1
                else:
                    stats["failed"] += 1
                    error = (
                        f"Dòng hợp lệ {index} ({payload['nhom_khuyet_tat']}): "
                        f"HTTP {response.status_code} - {extract_error(response)}"
                    )
                    stats["errors"].append(error)
                    print(f"Lỗi: {error}", file=sys.stderr)

                    if response.status_code == 404:
                        raise RuntimeError(
                            "Endpoint POST /job-descriptions chưa tồn tại trong app/main.py."
                        )
                    if response.status_code == 422:
                        raise RuntimeError(
                            "Schema backend không khớp. JobDescriptionCreate phải nhận "
                            "nhom_khuyet_tat và mo_ta_cong_viec."
                        )

            except requests.RequestException as exc:
                stats["failed"] += 1
                error = (
                    f"Dòng hợp lệ {index} ({payload['nhom_khuyet_tat']}): "
                    f"lỗi kết nối - {exc}"
                )
                stats["errors"].append(error)
                print(f"Lỗi: {error}", file=sys.stderr)

            if progress_every > 0 and (
                index % progress_every == 0 or index == len(payloads)
            ):
                print(
                    f"Tiến độ: {index}/{len(payloads)} | "
                    f"thành công: {stats['success']} | lỗi: {stats['failed']}"
                )

            if max_errors > 0 and stats["failed"] >= max_errors:
                raise RuntimeError(
                    f"Đã đạt giới hạn {max_errors} lỗi, dừng ingest để tránh gửi tiếp."
                )

            if delay > 0:
                time.sleep(delay)
    finally:
        session.close()

    return stats


def print_summary(
    stats: dict[str, Any],
    warning_count: int,
    unique_description_count: int,
    dry_run: bool,
) -> None:
    print("\n===== KẾT QUẢ INGEST JOB DESCRIPTIONS =====")
    print(f"Cặp nhóm và mô tả hợp lệ: {stats['total_valid']}")
    print(f"Số mô tả duy nhất: {unique_description_count}")
    print(f"Dòng bị bỏ qua khi kiểm tra: {warning_count}")

    if dry_run:
        print("Chế độ: DRY RUN")
        return

    print(f"Thêm thành công: {stats['success']}")
    print(f"Thêm thất bại: {stats['failed']}")

    if stats["group_success"]:
        print("Số bản ghi đã thêm theo nhóm:")
        for group in sorted(stats["group_success"]):
            print(f"  - {group}: {stats['group_success'][group]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Đọc job_descriptions.csv 2 cột và gửi từng dòng tới "
            "POST /job-descriptions."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"File CSV đầu vào, mặc định: {DEFAULT_CSV_PATH}",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Endpoint API, mặc định: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout mỗi request, mặc định {DEFAULT_TIMEOUT} giây.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=DEFAULT_PROGRESS_EVERY,
        help=(
            f"In tiến độ sau mỗi N dòng, mặc định {DEFAULT_PROGRESS_EVERY}; "
            "dùng 0 để tắt."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Thời gian nghỉ giữa hai request, mặc định 0 giây.",
    )
    parser.add_argument(
        "--min-description-length",
        type=int,
        default=50,
        help="Độ dài mô tả tối thiểu, mặc định 50 ký tự.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=10,
        help="Dừng khi đạt N lỗi; mặc định 10, dùng 0 để không giới hạn.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra CSV và in payload mẫu, không gọi API.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.timeout <= 0:
        print("Lỗi: --timeout phải lớn hơn 0.", file=sys.stderr)
        return 2
    if args.progress_every < 0:
        print("Lỗi: --progress-every không được âm.", file=sys.stderr)
        return 2
    if args.delay < 0:
        print("Lỗi: --delay không được âm.", file=sys.stderr)
        return 2
    if args.min_description_length < 1:
        print("Lỗi: --min-description-length phải lớn hơn 0.", file=sys.stderr)
        return 2
    if args.max_errors < 0:
        print("Lỗi: --max-errors không được âm.", file=sys.stderr)
        return 2

    try:
        payloads, warnings = read_job_descriptions(
            args.csv,
            args.min_description_length,
        )
    except (OSError, ValueError) as exc:
        print(f"Lỗi đọc CSV: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"Cảnh báo: {warning}", file=sys.stderr)

    if not payloads:
        print("Không có mô tả công việc hợp lệ để ingest.", file=sys.stderr)
        return 1

    unique_description_count = len({
        payload["mo_ta_cong_viec"].casefold()
        for payload in payloads
    })

    print(f"Đã đọc {len(payloads)} cặp nhóm và mô tả hợp lệ từ {args.csv}")
    print(f"Số mô tả công việc duy nhất: {unique_description_count}")
    print(
        "Lưu ý: nếu một mô tả thuộc nhiều nhóm, backend hiện tại sẽ "
        "tạo lại cùng vector cho từng nhóm."
    )

    try:
        stats = ingest(
            payloads=payloads,
            api_url=args.api_url,
            timeout=args.timeout,
            progress_every=args.progress_every,
            delay=args.delay,
            dry_run=args.dry_run,
            max_errors=args.max_errors,
        )
    except (ConnectionError, RuntimeError) as exc:
        print(f"Lỗi ingest: {exc}", file=sys.stderr)
        return 1

    print_summary(
        stats=stats,
        warning_count=len(warnings),
        unique_description_count=unique_description_count,
        dry_run=args.dry_run,
    )

    return 0 if args.dry_run or stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
