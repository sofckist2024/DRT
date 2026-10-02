"""
Desktop launcher for the DRT Streamlit app (used by the PyInstaller build).

Starts the Streamlit server in-process and opens the default web browser once
the server is reachable. When frozen by PyInstaller the app files (app.py,
drt.py, impedance_fit.py) are unpacked next to this launcher (sys._MEIPASS),
which Streamlit adds to sys.path so the app's imports resolve.
"""

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Import the heavy deps here so PyInstaller's analysis bundles them (the app
# itself is loaded by Streamlit at runtime via file path, so it is invisible to
# static analysis).
import numpy          # noqa: F401
import pandas         # noqa: F401
import scipy          # noqa: F401
import plotly         # noqa: F401
import streamlit      # noqa: F401


PORT = 8501


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)                # PyInstaller unpack dir
    return Path(__file__).resolve().parent


def _open_browser_when_ready(port: int, timeout: float = 90.0) -> None:
    url = f"http://localhost:{port}"
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    try:
        webbrowser.open(url)
    except Exception:  # noqa: BLE001
        pass


def main() -> int:
    base = _base_dir()

    # headless=true so Streamlit neither prompts for an email on first run nor
    # tries to open the browser itself (we do that reliably below).
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    threading.Thread(target=_open_browser_when_ready, args=(PORT,), daemon=True).start()

    import streamlit.web.cli as stcli
    sys.argv = [
        "streamlit", "run", str(base / "app.py"),
        f"--server.port={PORT}",
        "--server.headless=true",
        "--global.developmentMode=false",
    ]
    return stcli.main()


if __name__ == "__main__":
    raise SystemExit(main())
