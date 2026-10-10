"""Tiện ích tạm thời: canh cổng 8501 rồi ghi báo cáo (sẽ xoá sau)."""

import pathlib
import socket
import subprocess
import time


def port_open() -> bool:
    with socket.socket() as sock:
        sock.settimeout(3)
        try:
            sock.connect(("127.0.0.1", 8501))
            return True
        except Exception:  # noqa: BLE001
            return False


step = time.time()
opened = None
while time.time() - step < 120:
    if port_open():
        opened = round(time.time() - step, 1)
        break
    time.sleep(2)

lines = [f"port_8501_open_after_watch_start={opened}s"]
for name in ("_run_marker.txt", "_streamlit2.log"):
    path = pathlib.Path(name)
    lines.append(f"--- {name} exists={path.exists()} ---")
    if path.exists():
        lines.append(path.read_text(encoding="utf-8", errors="replace"))

net = subprocess.run(
    'netstat -ano | findstr LISTENING | findstr ":8501"',
    shell=True,
    capture_output=True,
    text=True,
)
lines.append("--- netstat LISTENING :8501 ---")
lines.append(net.stdout or "(khong co gi lang nghe)")

pathlib.Path("_watch.log").write_text("\n".join(lines), encoding="utf-8")
