"""Task 3 (dashboard) / Task 5 (this session): capture dashboard screenshots
into docs/ for the README.

Launches `streamlit run app.py` as a subprocess, waits for the one-time
pipeline load to finish, then uses Playwright (headless Chromium) to
capture: the review queue, a cluster detail view (first queue row
selected -- entity graph + MODEL score attribution, this session's new
Task 1/2 additions, plus the existing LLM narrative below it), and the
model performance tab. Requires `pip install playwright` and `playwright
install chromium` (not in requirements.txt -- this is a one-off authoring
tool, not something the pipeline or dashboard needs at runtime).

Clicking the queue's row-selection checkbox is the fragile part of this
script: it's a canvas-rendered grid (glide-data-grid, via st.dataframe),
and headless Chromium's hit-testing on it is unreliable with a single
click-and-fixed-wait -- confirmed by the fact that every screenshot this
script had EVER produced before this session still showed the unselected
"Select a cluster..." placeholder, not an actual detail view, despite the
script appearing to click and wait 5s. Two fixes, found by testing this
directly rather than assuming the old approach worked: (1) app.py's queue
dataframe now has an explicit key= (see app.py's build note), which was
letting Streamlit reset the selection on the very rerun the click itself
triggered; (2) this script now polls for the detail pane's own content
instead of a fixed sleep, since the detail pane now does real work (SHAP,
graph layout, an LLM call) that can legitimately take up to a minute.

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
LOAD_TIMEOUT_SECONDS = 240  # one-time pipeline load + queue build, polled for
DETAIL_TIMEOUT_SECONDS = 90  # SHAP + graph layout + a real LLM call


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
        print(f"Started streamlit (pid {server.pid}), waiting for the one-time pipeline load...")
        time.sleep(10)  # let the server bind before navigating
        with sync_playwright() as p:
            browser = p.chromium.launch()
            # Tall enough to fit the entity graph + MODEL attribution block
            # (this session's new Task 1/2 additions) in one screenshot
            # without relying on full-page scroll capture.
            page = browser.new_page(viewport={"width": 1700, "height": 2200})
            page.goto(URL, timeout=30000, wait_until="domcontentloaded")

            deadline = time.time() + LOAD_TIMEOUT_SECONDS
            loaded = False
            while time.time() < deadline:
                if page.get_by_text("Cluster queue, ranked by priority").count() > 0:
                    loaded = True
                    break
                page.wait_for_timeout(3000)
            if not loaded:
                print(f"WARNING: queue did not load within {LOAD_TIMEOUT_SECONDS}s -- capturing whatever is on screen.")
            else:
                # The "loaded" text can appear before the canvas grid has
                # actually painted its rows (a separate async draw step) --
                # give it a moment so the queue table isn't captured empty.
                page.wait_for_timeout(3000)

            page.screenshot(path=str(DOCS_DIR / "screenshot_queue.png"), full_page=False)
            print("Wrote docs/screenshot_queue.png")

            if loaded:
                canvas = page.locator("canvas").first
                box = canvas.bounding_box()
                if box:
                    # Row 1's selection checkbox -- see this file's module
                    # docstring for why this coordinate and the polling
                    # below (not a fixed sleep) both matter here.
                    page.mouse.click(box["x"] + 18, box["y"] + 55)
                    detail_deadline = time.time() + DETAIL_TIMEOUT_SECONDS
                    detail_loaded = False
                    while time.time() < detail_deadline:
                        if page.get_by_text("Score attribution").count() > 0:
                            detail_loaded = True
                            break
                        page.wait_for_timeout(3000)
                    if not detail_loaded:
                        print(f"WARNING: detail pane did not load within {DETAIL_TIMEOUT_SECONDS}s -- capturing whatever is on screen.")
                    page.wait_for_timeout(2000)  # let the last chart/narrative paint settle
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
