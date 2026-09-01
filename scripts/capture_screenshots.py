"""Task 3: capture dashboard screenshots into docs/ for the README.

Launches `streamlit run app.py` as a subprocess, waits for the one-time
pipeline load to finish, then uses Playwright (headless Chromium) to
capture: the review queue, a cluster detail view (first queue row
selected), and the model performance tab. Requires `pip install
playwright` and `playwright install chromium` (not in requirements.txt --
this is a one-off authoring tool, not something the pipeline or dashboard
needs at runtime).

Usage:
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
PORT = 8765
URL = f"http://localhost:{PORT}"
LOAD_WAIT_SECONDS = 110  # one-time pipeline load + queue build


def main() -> None:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright isn't installed. Run:\n"
            "  pip install playwright\n"
            "  playwright install chromium\n"
            "then re-run this script. Alternatively, run `streamlit run "
            "app.py` yourself and take screenshots by hand into docs/."
        )
        sys.exit(1)

    DOCS_DIR.mkdir(exist_ok=True)

    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        print(f"Started streamlit (pid {server.pid}), waiting {LOAD_WAIT_SECONDS}s for the one-time pipeline load...")
        time.sleep(10)  # let the server bind before navigating
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1700, "height": 1200})
            page.goto(URL, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(LOAD_WAIT_SECONDS * 1000)

            page.screenshot(path=str(DOCS_DIR / "screenshot_queue.png"), full_page=False)
            print("Wrote docs/screenshot_queue.png")

            canvas = page.locator("canvas").first
            box = canvas.bounding_box()
            if box:
                page.mouse.click(box["x"] + 15, box["y"] + 45)
                page.wait_for_timeout(5000)
            page.screenshot(path=str(DOCS_DIR / "screenshot_detail.png"), full_page=True)
            print("Wrote docs/screenshot_detail.png")

            page.get_by_text("Model performance", exact=True).click()
            page.wait_for_timeout(3000)
            page.screenshot(path=str(DOCS_DIR / "screenshot_performance.png"), full_page=True)
            print("Wrote docs/screenshot_performance.png")

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        print("Stopped the temporary streamlit server.")


if __name__ == "__main__":
    main()
