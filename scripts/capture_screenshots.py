"""Task 3 (dashboard) / Task 5 / this session: capture dashboard screenshots
into docs/ for the README.

Launches `streamlit run app.py` as a subprocess, waits for the one-time
pipeline load to finish, then uses Playwright (headless Chromium) to
capture five images: the review queue, a cluster detail view (entity
graph only, cluster 74986 -- a real 3-uid ring with a fraud-coloured
node, picked because it's small enough to read as a ring rather than a
dense blob), the SHAP score-attribution panel for that same cluster
(transaction-vs-cluster split included), the model performance tab, and
the Live replay tab mid-playback (after a REVIEW crossing has already
fired, so its narrative panel is visible). Requires `pip install
playwright` and `playwright install chromium` (not in requirements.txt
-- this is a one-off authoring tool, not something the pipeline or
dashboard needs at runtime).

Clicking the queue's row-selection checkbox is the fragile part of this
script: it's a canvas-rendered grid (glide-data-grid, via st.dataframe),
so cell text isn't in the DOM and rows can only be reached by raw pixel
coordinates on the canvas, not by locator/text lookup. Row 1's checkbox
is at canvas (+18, +55) -- found empirically in a prior session. Cluster
74986 is the 5th-ranked row in the current, deterministic priority
ordering; its checkbox was found empirically at canvas (+18, +196) (see
DEVLOG.md). If the trained model or queue ranking ever changes enough to
reorder the top rows, this offset will silently click the wrong cluster
-- the script prints the resulting "### Cluster N" heading so a wrong
selection is visible in the log, not just in a mis-labeled screenshot.

The detail view's content sits inside a fixed-height inner container
(page.screenshot(full_page=True) does not capture anything beyond the
2200px viewport here -- confirmed by diffing against a full_page=False
capture of the same state), so the graph and SHAP images are produced
by cropping one 1700x2200 viewport capture with PIL at empirically-found
y-boundaries, rather than by scrolling or by locator screenshots (the
panel's opening/closing "<div>" tags are written across separate
st.markdown calls, so they don't form one queryable DOM element).
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
LOAD_TIMEOUT_SECONDS = 360  # one-time pipeline load + queue build, polled for
DETAIL_TIMEOUT_SECONDS = 90  # SHAP + graph layout + a real LLM call
REPLAY_TIMEOUT_SECONDS = 360  # replay precompute (incl. 2 real LLM calls) + playback to the first crossing

# Cluster 74986's checkbox in the queue canvas -- see module docstring.
CLUSTER_74986_ROW_Y_OFFSET = 196

# Empirically-found crop boundaries (pixels, in the 1700x2200 viewport) for
# the two sub-images cut from one cluster-detail capture: the entity graph
# (full width, includes the queue table on the left for context) and the
# SHAP attribution panel (cropped to the right column only, since the left
# side is blank past the queue table's last row at this scroll position).
GRAPH_CROP = (0, 0, 1700, 1063)
SHAP_CROP = (555, 1063, 1700, 1600)


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

    from PIL import Image

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
            # in one screenshot without relying on full-page scroll capture
            # (which this app's fixed-height inner container defeats anyway).
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
                    page.mouse.click(box["x"] + 18, box["y"] + CLUSTER_74986_ROW_Y_OFFSET)
                    heading = page.locator("h3", has_text="Cluster").first
                    try:
                        heading_text = heading.inner_text(timeout=5000)
                    except Exception:
                        heading_text = "(no heading appeared)"
                    print(f"Selected row -> {heading_text}")
                    if "74986" not in heading_text:
                        print(
                            "WARNING: expected 'Cluster 74986' but got "
                            f"'{heading_text}' -- the queue ranking may have "
                            "changed; CLUSTER_74986_ROW_Y_OFFSET needs "
                            "re-finding (see this file's module docstring)."
                        )

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

            raw_detail_path = DOCS_DIR / "_raw_detail.png"
            page.screenshot(path=str(raw_detail_path), full_page=False)
            detail_im = Image.open(raw_detail_path)
            detail_im.crop(GRAPH_CROP).save(DOCS_DIR / "screenshot_detail.png")
            detail_im.crop(SHAP_CROP).save(DOCS_DIR / "screenshot_shap.png")
            raw_detail_path.unlink()
            print("Wrote docs/screenshot_detail.png (entity graph) and docs/screenshot_shap.png (score attribution)")

            page.get_by_text("Model performance", exact=True).click()
            page.wait_for_timeout(3000)
            page.screenshot(path=str(DOCS_DIR / "screenshot_performance.png"), full_page=True)
            print("Wrote docs/screenshot_performance.png")

            # Live replay tab, played to its first real REVIEW_THRESHOLD
            # crossing so the narrative payoff panel is visible. Widened
            # first: at 1700px the scored-feed table's rightmost (action)
            # column render just past the visible edge, clipping mid-word --
            # confirmed by inspecting the capture, not assumed.
            page.set_viewport_size({"width": 2000, "height": 2200})
            page.get_by_role("tab", name="Live replay").click()
            deadline2 = time.time() + REPLAY_TIMEOUT_SECONDS
            replay_ready = False
            while time.time() < deadline2:
                if page.get_by_text("Entity graph, building incrementally").count() > 0:
                    replay_ready = True
                    break
                page.wait_for_timeout(3000)
            if not replay_ready:
                print(f"WARNING: replay tab did not load within {REPLAY_TIMEOUT_SECONDS}s -- capturing whatever is on screen.")
            else:
                page.get_by_role("button", name="Play").click()
                narrative_deadline = time.time() + REPLAY_TIMEOUT_SECONDS
                narrative_seen = False
                while time.time() < narrative_deadline:
                    if page.get_by_text("crossed at replay transaction").count() > 0:
                        narrative_seen = True
                        break
                    page.wait_for_timeout(3000)
                if not narrative_seen:
                    print(f"WARNING: no REVIEW_THRESHOLD crossing appeared within {REPLAY_TIMEOUT_SECONDS}s of playback -- capturing whatever is on screen.")
                page.get_by_role("button", name="Pause").click()
                page.wait_for_timeout(1500)

            raw_replay_path = DOCS_DIR / "_raw_replay.png"
            page.screenshot(path=str(raw_replay_path), full_page=False)
            replay_im = Image.open(raw_replay_path)
            # Trim to the true content bounding box (not a hardcoded pixel
            # crop): both how tall the narrative panel is (varies with how
            # many clusters have crossed by the time Play is paused) and how
            # wide the feed table renders (varies with column content) are
            # data-dependent, so measure them from the actual capture.
            import numpy as np

            arr = np.array(replay_im.convert("RGB"))
            bg = arr[5, 5]
            nonbg = (np.abs(arr.astype(int) - bg.astype(int)) > 12).any(axis=2)
            content_rows = nonbg.any(axis=1).nonzero()[0]
            # Exclude the top nav bar (Deploy / menu icons, far right) from
            # the width scan -- it's always present regardless of how wide
            # the actual tab content below it is.
            content_cols = nonbg[230:, :].any(axis=0).nonzero()[0]
            bottom = int(content_rows.max()) + 24 if len(content_rows) else replay_im.height
            right = int(content_cols.max()) + 24 if len(content_cols) else replay_im.width
            replay_im.crop((0, 0, min(right, replay_im.width), min(bottom, replay_im.height))).save(DOCS_DIR / "screenshot_replay.png")
            raw_replay_path.unlink()
            print("Wrote docs/screenshot_replay.png")

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
