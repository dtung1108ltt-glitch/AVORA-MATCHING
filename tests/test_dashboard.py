#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test cho giao diện Streamlit ``dashboard.py``.

Module này kiểm tra:

* Các hàm tiện ích thuần Python của dashboard (đọc ma trận, đọc quota, sinh
  thị trường mô phỏng) - không cần Streamlit runtime.
* Smoke test chạy thật toàn bộ dashboard bằng ``streamlit.testing.v1.AppTest``:
  mở trang, bấm nút chạy Gale-Shapley rồi kiểm tra Blocking Pairs = 0.

Nếu môi trường chưa cài ``streamlit`` (đây là thư viện tùy chọn cho UI), toàn bộ
file sẽ bị skip để không ảnh hưởng bộ test của engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="Cần streamlit để test dashboard.")

import dashboard  # noqa: E402  (import sau importorskip là chủ ý)

DASHBOARD_PATH = Path(dashboard.__file__).resolve()


def test_parse_score_grid_doc_vi_ma_tran_dang_csv() -> None:
    """Đọc ma trận dạng CSV, ô trống nghĩa là cặp không khả thi."""

    text = "\n".join(
        [
            ",J1,J2,J3",
            "C1,0.9,0.4,0.2",
            "C2,0.8,,0.3",
            "C3,0.5,0.6,0.7",
        ]
    )

    matrix, candidates, jobs = dashboard.parse_score_grid(text, "E")

    assert candidates == ("C1", "C2", "C3")
    assert jobs == ("J1", "J2", "J3")
    assert matrix["C1"] == {"J1": 0.9, "J2": 0.4, "J3": 0.2}
    assert matrix["C2"] == {"J1": 0.8, "J3": 0.3}, "Ô trống phải bị bỏ qua."


def test_parse_score_grid_bao_loi_khi_lech_so_o() -> None:
    """Số điểm trên dòng phải khớp số cột ở header."""

    with pytest.raises(ValueError, match="nhưng header có"):
        dashboard.parse_score_grid(",J1,J2\nC1,0.9\n", "E")


def test_parse_quotas_dung_mac_dinh_cho_vi_tri_thieu() -> None:
    """Vị trí không khai báo quota sẽ nhận giá trị mặc định.

    Dấu phân cách linh hoạt: cả ``J1, 3`` và ``J3=2`` đều được chấp nhận.
    """

    quotas = dashboard.parse_quotas("J1, 3\nJ3=2", ("J1", "J2", "J3"), default=5)

    assert quotas == {"J1": 3, "J2": 5, "J3": 2}


def test_parse_quotas_tu_choi_gia_tri_am() -> None:
    """Quota âm không được phép xuất hiện trong ``PreferenceProfile``."""

    with pytest.raises(ValueError, match="không được âm"):
        dashboard.parse_quotas("J1,-2", ("J1",), default=1)


def test_build_synthetic_market_tai_lap_voi_cung_seed() -> None:
    """Cùng seed phải cho cùng thị trường (deterministic)."""

    first = dashboard.build_synthetic_market(30, 10, 3, 0.6, 7)
    second = dashboard.build_synthetic_market(30, 10, 3, 0.6, 7)

    assert first.e_scores == second.e_scores
    assert first.c_scores == second.c_scores
    assert first.quotas == second.quotas
    assert first.shape == (30, 10)
    assert all(score > 0 for row in first.e_scores.values() for score in row.values())


def test_build_grid_market_tu_van_ban_mau() -> None:
    """Dựng thị trường từ hai ma trận mặc định của dashboard."""

    market = dashboard.build_grid_market(
        dashboard.DEFAULT_E_TEXT,
        dashboard.DEFAULT_C_TEXT,
        dashboard.DEFAULT_QUOTA_TEXT,
        default_quota=1,
        name="test",
    )

    assert market.shape == (6, 4)
    assert market.quotas == {"J1": 2, "J2": 1, "J3": 2, "J4": 1}
    assert market.pair_count() == 24

    output = dashboard.run_matching(
        market,
        config=dashboard.PreprocessConfig(),
        optimal="candidate",
        top_k=2,
        weight_c=0.5,
        weight_e=0.5,
    )

    assert output.report.blocking.count == 0
    assert output.report.fill_rate.overall > 0
    assert set(output.recommendations) == set(market.candidate_ids)


def test_dashboard_chay_that_qua_apptest() -> None:
    """Mở dashboard, bấm nút chạy và kiểm chứng kết quả hiển thị trên UI."""

    app_test = pytest.importorskip(
        "streamlit.testing.v1", reason="Cần Streamlit >= 1.28 để chạy AppTest."
    ).AppTest.from_file(str(DASHBOARD_PATH), default_timeout=120)

    app_test.run()
    assert not app_test.exception, "Trang phải render được mà không lỗi."

    run_buttons = [button for button in app_test.button if "Chạy" in button.label]
    assert run_buttons, "Phải có nút '🚀 Chạy Gale-Shapley'."
    run_buttons[0].click()
    app_test.run()
    assert not app_test.exception, "Bấm chạy không được phát sinh lỗi."

    metrics = {metric.label: metric.value for metric in app_test.metric}
    assert metrics.get("Blocking pairs") == "0", "Kết quả hiển thị phải ổn định."
    assert metrics.get("Cặp ghép được")
    assert float(metrics["Fill rate (slot)"].rstrip("%")) > 0

    assert app_test.dataframe, "Phải render ít nhất một bảng kết quả."
    assert app_test.download_button, "Phải có nút tải kết quả."

    assert app_test is not None
