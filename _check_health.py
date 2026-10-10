"""Tiện ích tạm thời: kiểm tra server Streamlit + proxy (sẽ xoá sau)."""

import os
import time
import urllib.request

out = []
out.append("PROXY_ENV=" + str({k: v for k, v in os.environ.items() if "proxy" in k.lower()}))

deadline = time.time() + 60
up = False
while time.time() < deadline:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8501/_stcore/health", timeout=5) as resp:
            up = True
            break
    except Exception:  # noqa: BLE001
        time.sleep(2)
out.append(f"server_reachable={up}")

for name, url in (
    ("127.0.0.1", "http://127.0.0.1:8501/_stcore/health"),
    ("localhost", "http://localhost:8501/_stcore/health"),
    ("127.0.0.1-root", "http://127.0.0.1:8501/"),
):
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            out.append(f"{name} -> HTTP {resp.status} {resp.read(40).decode('utf-8', 'replace')!r}")
    except Exception as exc:  # noqa: BLE001
        out.append(f"{name} -> ERROR {type(exc).__name__}: {exc}")

with open("_health.log", "w", encoding="utf-8") as handle:
    handle.write("\n".join(out))
