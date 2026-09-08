#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tạo database gồm đúng 2 cột:
- Nhóm khuyết tật
- Mô tả công việc

Mỗi dòng là một cặp:
    nhóm khuyết tật <-> mô tả công việc có thể phù hợp

Một mô tả có thể xuất hiện nhiều dòng nếu phù hợp với nhiều nhóm.
Output mặc định: scripts/data/job_descriptions.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import unicodedata
from pathlib import Path

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data" / "job_descriptions.csv"
DEFAULT_COUNT = 100
DEFAULT_SEED = 42

# Mỗi mẫu chỉ chứa nội dung cần cho embedding và các nhóm được gắn nhãn.
JOB_TEMPLATES = [
    {
        "tasks": "Nhập, đối chiếu và cập nhật dữ liệu trên Excel hoặc phần mềm nội bộ; kiểm tra lỗi trước khi lưu",
        "conditions": "Làm việc tại bàn, sử dụng máy tính, trao đổi chủ yếu bằng văn bản và không bắt buộc nghe điện thoại",
        "requirements": "Biết sử dụng Excel cơ bản, gõ phím và làm theo quy trình kiểm tra dữ liệu",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Phát triển và bảo trì API, làm việc với cơ sở dữ liệu, viết kiểm thử và quản lý mã nguồn bằng Git",
        "conditions": "Làm việc trên máy tính, có thể làm từ xa hoặc tại văn phòng, trao đổi qua chat và công cụ quản lý công việc",
        "requirements": "Có kiến thức Python, SQL, REST API, Git và khả năng đọc tài liệu kỹ thuật",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Thiết kế banner, hình ảnh sản phẩm và ấn phẩm truyền thông; chỉnh sửa sản phẩm theo phản hồi",
        "conditions": "Làm việc trên máy tính, nhận yêu cầu bằng văn bản và có thể làm từ xa",
        "requirements": "Sử dụng được Photoshop, Illustrator hoặc Canva và có sản phẩm mẫu phù hợp",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Tiếp nhận câu hỏi, tư vấn sản phẩm và xử lý yêu cầu của khách hàng qua kênh chat",
        "conditions": "Không gọi điện thoại, giao tiếp qua tin nhắn trên website, fanpage hoặc ứng dụng và có kịch bản trả lời",
        "requirements": "Viết rõ ràng, biết sử dụng máy tính và xử lý tình huống theo quy trình",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Nghiên cứu chủ đề, viết bài blog và mô tả sản phẩm; chỉnh sửa nội dung theo phản hồi biên tập",
        "conditions": "Làm việc trên máy tính, trao đổi bằng văn bản, có thể làm từ xa và quản lý theo thời hạn",
        "requirements": "Có khả năng viết, tìm kiếm thông tin và sử dụng công cụ soạn thảo văn bản",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ", "Khiếm thị"],
    },
    {
        "tasks": "Kiểm tra chứng từ, nhập số liệu, đối chiếu sổ sách và lập báo cáo kế toán định kỳ",
        "conditions": "Làm việc trên máy tính tại văn phòng hoặc theo hình thức kết hợp, trao đổi qua email và phần mềm nội bộ",
        "requirements": "Có nghiệp vụ kế toán, sử dụng Excel và phần mềm kế toán MISA",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Quản lý fanpage, tiếp nhận đơn hàng, trả lời tin nhắn và cập nhật trạng thái đơn trên hệ thống",
        "conditions": "Giao tiếp với khách hàng qua tin nhắn, làm việc bằng máy tính và không bắt buộc gọi điện thoại",
        "requirements": "Biết sử dụng mạng xã hội, Excel cơ bản và có khả năng viết tin nhắn rõ ràng",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Lắp ráp và hoàn thiện sản phẩm thủ công theo mẫu; kiểm tra số lượng và chất lượng trước khi bàn giao",
        "conditions": "Làm việc theo hướng dẫn trực quan và quy trình lặp lại; một số công đoạn có thể nhận về nhà",
        "requirements": "Khéo tay, kiên nhẫn và có khả năng thực hiện đúng mẫu",
        "groups": ["Khiếm thính", "Khuyết tật trí tuệ nhẹ", "Khuyết tật vận động", "Đa khuyết tật nhẹ"],
    },
    {
        "tasks": "Gấp hộp, dán nhãn, kiểm đếm và đóng gói sản phẩm nhẹ theo danh sách và quy trình có sẵn",
        "conditions": "Làm việc tại trạm cố định, không nâng vật nặng, có bảng hướng dẫn trực quan và thời gian nghỉ theo ca",
        "requirements": "Có khả năng làm theo từng bước, kiểm đếm cơ bản và duy trì sự tập trung",
        "groups": ["Khiếm thính", "Khuyết tật trí tuệ nhẹ", "Đa khuyết tật nhẹ"],
    },
    {
        "tasks": "Chuyển nội dung nói hoặc văn bản sang ngôn ngữ ký hiệu và phối hợp biên tập phụ đề cho video",
        "conditions": "Làm việc với video và tài liệu số, nhận kịch bản trước, có thể làm bán thời gian hoặc từ xa",
        "requirements": "Thành thạo ngôn ngữ ký hiệu và có kỹ năng biên tập video cơ bản",
        "groups": ["Khiếm thính"],
    },
    {
        "tasks": "Thực hiện các kỹ thuật xoa bóp và bấm huyệt theo liệu trình; ghi nhận thông tin dịch vụ sau khi hoàn thành",
        "conditions": "Làm việc trực tiếp tại cơ sở có không gian và quy trình an toàn, không làm từ xa",
        "requirements": "Đã qua đào tạo nghề, có chứng chỉ phù hợp và tuân thủ quy trình vệ sinh",
        "groups": ["Khiếm thị"],
    },
    {
        "tasks": "May các công đoạn đơn giản theo mẫu, kiểm tra đường may và sắp xếp sản phẩm sau khi hoàn thành",
        "conditions": "Làm việc tại vị trí cố định trong xưởng, có hướng dẫn trực quan và quy trình an toàn rõ ràng",
        "requirements": "Biết sử dụng máy may, có khả năng tập trung và thực hiện đúng mẫu",
        "groups": ["Khiếm thính", "Khuyết tật trí tuệ nhẹ"],
    },
    {
        "tasks": "Phân loại tài liệu điện tử, đặt tên tệp và lưu tài liệu vào đúng thư mục theo danh mục có sẵn",
        "conditions": "Làm việc trên máy tính, sử dụng hướng dẫn bằng văn bản và không yêu cầu giao tiếp qua điện thoại",
        "requirements": "Biết thao tác tệp, thư mục và có khả năng làm theo quy tắc đặt tên",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Kiểm tra lỗi chính tả, định dạng văn bản và đối chiếu nội dung với bản hướng dẫn trước khi xuất bản",
        "conditions": "Làm việc trên máy tính, trao đổi bằng nhận xét văn bản và có thể làm từ xa",
        "requirements": "Đọc hiểu tốt, cẩn thận và sử dụng được công cụ soạn thảo văn bản",
        "groups": ["Khiếm thính", "Khuyết tật vận động", "Khuyết tật ngôn ngữ"],
    },
    {
        "tasks": "Ghi âm nội dung theo kịch bản có sẵn, kiểm tra chất lượng tệp âm thanh và đặt tên tệp theo quy định",
        "conditions": "Làm việc trong không gian yên tĩnh, sử dụng micro và phần mềm ghi âm trên máy tính",
        "requirements": "Có giọng nói rõ ràng, đọc đúng kịch bản và thao tác được với tệp âm thanh",
        "groups": ["Khiếm thị", "Khuyết tật vận động"],
    },
    {
        "tasks": "Nghe tệp âm thanh, phân loại nội dung và nhập kết quả vào biểu mẫu theo các nhãn có sẵn",
        "conditions": "Làm việc bằng tai nghe và máy tính, không yêu cầu xử lý hình ảnh phức tạp",
        "requirements": "Nghe hiểu tốt, tập trung và gõ phím cơ bản",
        "groups": ["Khiếm thị", "Khuyết tật vận động"],
    },
]

INTRO_VARIANTS = [
    "Nhiệm vụ chính gồm: ",
    "Người lao động thực hiện: ",
    "Công việc bao gồm: ",
    "Các nhiệm vụ cần thực hiện gồm: ",
]

CONDITION_VARIANTS = [
    "Điều kiện thực hiện: ",
    "Môi trường và cách thức làm việc: ",
    "Công việc được thực hiện trong điều kiện: ",
]

REQUIREMENT_VARIANTS = [
    "Yêu cầu để thực hiện công việc: ",
    "Kỹ năng cần thiết: ",
    "Người thực hiện cần: ",
]

ENDING_VARIANTS = [
    "",
    " Công việc có quy trình hướng dẫn rõ ràng.",
    " Đơn vị cung cấp hướng dẫn ban đầu trước khi nhận việc.",
    " Kết quả được đánh giá dựa trên độ chính xác và mức độ hoàn thành.",
    " Công việc được chia thành các bước cụ thể để dễ theo dõi.",
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    return text


def create_description(template: dict, rng: random.Random) -> str:
    description = (
        f"{rng.choice(INTRO_VARIANTS)}{template['tasks']}. "
        f"{rng.choice(CONDITION_VARIANTS)}{template['conditions']}. "
        f"{rng.choice(REQUIREMENT_VARIANTS)}{template['requirements']}."
        f"{rng.choice(ENDING_VARIANTS)}"
    )
    return normalize_text(description)


def generate_rows(unique_descriptions: int, rng: random.Random) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    attempts = 0
    max_attempts = unique_descriptions * 200

    while len(seen) < unique_descriptions:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                "Không thể tạo đủ mô tả duy nhất. Hãy giảm --count hoặc bổ sung mẫu công việc."
            )

        template = rng.choice(JOB_TEMPLATES)
        description = create_description(template, rng)
        key = description.casefold()
        if key in seen:
            continue
        seen.add(key)

        # Một mô tả có thể tạo nhiều dòng, mỗi dòng ứng với một nhóm.
        for group in template["groups"]:
            rows.append({
                "Nhóm khuyết tật": group,
                "Mô tả công việc": description,
            })

    rng.shuffle(rows)
    return rows


def validate_rows(rows: list[dict[str, str]], expected_descriptions: int) -> None:
    if not rows:
        raise ValueError("Không có dữ liệu được tạo.")

    allowed_groups = {
        "Khiếm thính",
        "Khiếm thị",
        "Khuyết tật vận động",
        "Khuyết tật trí tuệ nhẹ",
        "Khuyết tật ngôn ngữ",
        "Khuyết tật thần kinh, tâm thần (ổn định)",
        "Đa khuyết tật nhẹ",
    }

    unique_descriptions = {row["Mô tả công việc"] for row in rows}
    if len(unique_descriptions) != expected_descriptions:
        raise ValueError(
            f"Sai số mô tả duy nhất: cần {expected_descriptions}, nhận {len(unique_descriptions)}."
        )

    seen_pairs: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=2):
        group = row["Nhóm khuyết tật"].strip()
        description = row["Mô tả công việc"].strip()
        if group not in allowed_groups:
            raise ValueError(f"Nhóm khuyết tật không hợp lệ tại dòng {index}: {group}")
        if len(description) < 100:
            raise ValueError(f"Mô tả quá ngắn tại dòng {index}.")
        pair = (group, description.casefold())
        if pair in seen_pairs:
            raise ValueError(f"Trùng cặp nhóm và mô tả tại dòng {index}.")
        seen_pairs.add(pair)


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["Nhóm khuyết tật", "Mô tả công việc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tạo database chỉ gồm nhóm khuyết tật và mô tả công việc."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Số mô tả công việc duy nhất, mặc định {DEFAULT_COUNT}.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("Lỗi: --count phải lớn hơn 0.", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    rows = generate_rows(args.count, rng)
    validate_rows(rows, args.count)
    write_csv(args.output, rows)

    print(f"Đã tạo {args.count} mô tả công việc duy nhất.")
    print(f"Đã ghi {len(rows)} dòng quan hệ nhóm khuyết tật - mô tả công việc.")
    print(f"File output: {args.output}")
    print("Database chỉ gồm 2 cột: Nhóm khuyết tật, Mô tả công việc.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
