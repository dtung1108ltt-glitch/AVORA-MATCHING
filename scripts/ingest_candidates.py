#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ingest_candidates.py

Đọc file CSV ứng viên gồm đúng 2 cột:
- Họ và tên
- Nhóm khuyết tật

Sau đó gửi từng ứng viên tới:
    POST http://localhost:8000/candidates

Payload:
    {
        "ho_ten": "Nguyễn Văn A",
        "nhom_khuyet_tat": "Khiếm thính"
    }

Yêu cầu backend:
- CandidateCreate phải có 2 trường: ho_ten, nhom_khuyet_tat
- POST /candidates không tạo embedding cho ứng viên
- Bảng candidates phải có: id, ho_ten, nhom_khuyet_tat
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

DEFAULT_API_URL = "http://localhost:8000/candidates"
DEFAULT_CSV_PATH = Path(__file__).resolve().parent / "data" / "candidates.csv"
DEFAULT_TIMEOUT = 30.0

CSV_NAME_COLUMN = "Họ và tên"
CSV_GROUP_COLUMN = "Nhóm khuyết tật"

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
    """Chuẩn hóa Unicode và khoảng trắng."""
    if value is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).split())


def create_http_session() -> requests.Session:
    """Tạo HTTP session có retry cho lỗi mạng và lỗi server tạm thời."""
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"Content-Type": "application/json"})
    return session


def read_candidates(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """
    Đọc và kiểm tra CSV.

    Trả về:
    - danh sách ứng viên hợp lệ
    - danh sách cảnh báo cho dòng bị bỏ qua
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file CSV: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Đường dẫn không phải file: {csv_path}")

    candidates: list[dict[str, str]] = []
    warnings: list[str] = []
    seen_pairs: set[tuple[str, str]] = set()

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        headers = reader.fieldnames or []
        normalized_headers = [normalize_text(header) for header in headers]

        expected = [CSV_NAME_COLUMN, CSV_GROUP_COLUMN]
        missing = [column for column in expected if column not in normalized_headers]
        extra = [column for column in normalized_headers if column not in expected]

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

        # Ánh xạ lại tên cột sau khi chuẩn hóa Unicode/khoảng trắng.
        header_map = {
            normalize_text(original): original
            for original in headers
        }

        for line_number, row in enumerate(reader, start=2):
            name = normalize_text(row.get(header_map[CSV_NAME_COLUMN]))
            group = normalize_text(row.get(header_map[CSV_GROUP_COLUMN]))

            if not name and not group:
                warnings.append(f"Dòng {line_number}: dòng trống, đã bỏ qua.")
                continue
            if not name:
                warnings.append(f"Dòng {line_number}: thiếu Họ và tên, đã bỏ qua.")
                continue
            if not group:
                warnings.append(
                    f"Dòng {line_number} ({name}): thiếu Nhóm khuyết tật, đã bỏ qua."
                )
                continue
            if group not in ALLOWED_GROUPS:
                warnings.append(
                    f"Dòng {line_number} ({name}): nhóm không hợp lệ '{group}', đã bỏ qua."
                )
                continue

            pair = (name.casefold(), group.casefold())
            if pair in seen_pairs:
                warnings.append(
                    f"Dòng {line_number} ({name}): trùng ứng viên và nhóm trong CSV, đã bỏ qua."
                )
                continue
            seen_pairs.add(pair)

            candidates.append({
                "ho_ten": name,
                "nhom_khuyet_tat": group,
            })

    return candidates, warnings


def check_api(session: requests.Session, api_url: str, timeout: float) -> None:
    """Kiểm tra server có phản hồi trước khi ingest."""
    base_url = api_url.rsplit("/candidates", 1)[0].rstrip("/")
    check_urls = [f"{base_url}/docs", base_url or api_url]

    last_error: Exception | None = None
    for url in check_urls:
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code < 500:
                return
        except requests.RequestException as exc:
            last_error = exc

    message = (
        f"Không kết nối được API tại {base_url}. "
        "Hãy chạy docker compose up -d trước khi ingest."
    )
    if last_error:
        message += f" Chi tiết: {last_error}"
    raise ConnectionError(message)


def extract_error(response: requests.Response) -> str:
    """Lấy nội dung lỗi ngắn gọn từ response."""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = body.get("detail", body)
            return str(detail)
        return str(body)
    except ValueError:
        return response.text.strip() or response.reason


def ingest_candidates(
    candidates: list[dict[str, str]],
    api_url: str,
    timeout: float,
    progress_every: int,
    delay: float,
    dry_run: bool,
) -> dict[str, Any]:
    """Gửi từng ứng viên lên API và trả về thống kê."""
    stats: dict[str, Any] = {
        "total_valid": len(candidates),
        "success": 0,
        "failed": 0,
        "group_success": Counter(),
        "errors": [],
    }

    if dry_run:
        print("DRY RUN: không gửi dữ liệu lên API.")
        for index, payload in enumerate(candidates[:5], start=1):
            print(f"Mẫu {index}: {payload}")
        if len(candidates) > 5:
            print(f"... còn {len(candidates) - 5} payload khác.")
        return stats

    session = create_http_session()
    check_api(session, api_url, timeout)

    try:
        for index, payload in enumerate(candidates, start=1):
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
                        f"Ứng viên {index} ({payload['ho_ten']}): "
                        f"HTTP {response.status_code} - {extract_error(response)}"
                    )
                    stats["errors"].append(error)
                    print(f"Lỗi: {error}", file=sys.stderr)

            except requests.RequestException as exc:
                stats["failed"] += 1
                error = f"Ứng viên {index} ({payload['ho_ten']}): lỗi kết nối - {exc}"
                stats["errors"].append(error)
                print(f"Lỗi: {error}", file=sys.stderr)

            if progress_every > 0 and (
                index % progress_every == 0 or index == len(candidates)
            ):
                print(
                    f"Tiến độ: {index}/{len(candidates)} | "
                    f"thành công: {stats['success']} | lỗi: {stats['failed']}"
                )

            if delay > 0:
                time.sleep(delay)
    finally:
        session.close()

    return stats


def print_summary(
    stats: dict[str, Any],
    warning_count: int,
    dry_run: bool,
) -> None:
    print("\n===== KẾT QUẢ INGEST =====")
    print(f"Dòng hợp lệ trong CSV: {stats['total_valid']}")
    print(f"Dòng bị bỏ qua khi kiểm tra: {warning_count}")

    if dry_run:
        print("Chế độ: DRY RUN")
        return

    print(f"Thêm thành công: {stats['success']}")
    print(f"Thêm thất bại: {stats['failed']}")

    if stats["group_success"]:
        print("Số ứng viên đã thêm theo nhóm:")
        for group in sorted(stats["group_success"]):
            print(f"  - {group}: {stats['group_success'][group]}")

    if stats["errors"]:
        print(f"Có {len(stats['errors'])} lỗi. Đã hiển thị lỗi trong quá trình chạy.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đọc candidates.csv 2 cột và gửi ứng viên tới POST /candidates."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Đường dẫn CSV, mặc định: {DEFAULT_CSV_PATH}",
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
        help=f"Timeout mỗi request, mặc định: {DEFAULT_TIMEOUT} giây.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="In tiến độ sau mỗi N dòng, mặc định 100; dùng 0 để tắt.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Thời gian nghỉ giữa hai request, mặc định 0 giây.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra CSV và xem payload mẫu, không gọi API.",
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

    try:
        candidates, warnings = read_candidates(args.csv)
    except (OSError, ValueError) as exc:
        print(f"Lỗi đọc CSV: {exc}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"Cảnh báo: {warning}", file=sys.stderr)

    if not candidates:
        print("Không có ứng viên hợp lệ để ingest.", file=sys.stderr)
        return 1

    print(f"Đã đọc {len(candidates)} ứng viên hợp lệ từ {args.csv}")

    try:
        stats = ingest_candidates(
            candidates=candidates,
            api_url=args.api_url,
            timeout=args.timeout,
            progress_every=args.progress_every,
            delay=args.delay,
            dry_run=args.dry_run,
        )
    except ConnectionError as exc:
        print(f"Lỗi API: {exc}", file=sys.stderr)
        return 1

    print_summary(stats, len(warnings), args.dry_run)
    return 0 if args.dry_run or stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
