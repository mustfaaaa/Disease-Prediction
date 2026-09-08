"""Capture real screenshots of the running application for the README.

Starts the API on a spare port, drives a real Chromium through Playwright, and
writes PNGs to ``outputs/figures/ui/``.  Every image is a photograph of the
actual interface talking to the actual model - nothing is mocked up.

    python -m pip install playwright && python -m playwright install chromium
    python scripts/capture_ui.py
"""
from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "figures" / "ui"
PORT = 8011
BASE = f"http://127.0.0.1:{PORT}"

# Two records inside the documented schema, chosen to land on opposite sides of
# the decision threshold. The probabilities they produce come from the model.
HIGH = {"age": 62, "sex": 1, "cp": 4, "trestbps": 150, "chol": 268, "fbs": 0,
        "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 3.6, "slope": 2,
        "ca": 2, "thal": 7}
LOW = {"age": 41, "sex": 0, "cp": 2, "trestbps": 120, "chol": 198, "fbs": 0,
       "restecg": 0, "thalach": 178, "exang": 0, "oldpeak": 0.0, "slope": 1,
       "ca": 0, "thal": 3}

# Sticky elements bleed into Playwright's stitched element screenshots, so they
# are pinned to static for the capture only. Animations are frozen too.
SETTLE = """
() => {
  document.documentElement.style.scrollBehavior = 'auto';
  const css = document.createElement('style');
  css.textContent = `
    *, *::before, *::after { animation: none !important; transition: none !important; }
    .site-header, .result-panel, thead th { position: static !important; }
    .site-header { backdrop-filter: none !important; background: var(--surface) !important; }
  `;
  document.head.appendChild(css);
}
"""


def wait_for_server(timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/api/health", timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("API did not become healthy in time")


def only(page, section_id: str | None) -> None:
    page.evaluate(
        """(id) => document.querySelectorAll('main > section').forEach(
             s => s.style.display = (id === null || s.id === id) ? '' : 'none')""",
        section_id,
    )
    page.evaluate("() => window.scrollTo(0, 0)")


def fill(page, record: dict) -> None:
    for name, value in record.items():
        sel = f"#f-{name}"
        tag = page.eval_on_selector(sel, "e => e.tagName")
        if tag == "SELECT":
            page.select_option(sel, str(float(value)) if float(value) != int(float(value))
                               else str(int(value)))
        else:
            page.fill(sel, f"{value:g}")


def analyse(page) -> str:
    page.click("#submit-btn")
    page.wait_for_selector(".result-card .meter-value", timeout=15000)
    page.wait_for_timeout(700)
    return page.inner_text(".result-card .meter-value")


def shot(target, name: str, max_height: int | None = None) -> None:
    """Capture *target*; optionally keep only the top ``max_height`` pixels so a
    very tall element stays readable when embedded in the README."""
    path = OUT / name
    target.screenshot(path=str(path))
    if max_height:
        from PIL import Image
        with Image.open(path) as img:
            if img.height > max_height:
                img.crop((0, 0, img.width, max_height)).save(path)
    print(f"  wrote {path.relative_to(ROOT)}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT),
         "--log-level", "warning"],
        cwd=str(ROOT),
    )
    try:
        wait_for_server()
        print(f"API healthy on {BASE}")

        with sync_playwright() as pw:
            browser = pw.chromium.launch()

            # ---------------------------------------------------- desktop light
            page = browser.new_page(viewport={"width": 1440, "height": 900},
                                    device_scale_factor=2,
                                    color_scheme="light")
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_selector("#hero-stats .stat-value", timeout=15000)
            page.evaluate(SETTLE)
            page.wait_for_timeout(600)

            shot(page, "01_home.png")

            only(page, "prediction")
            page.wait_for_timeout(200)
            shot(page.locator(".form-card"), "02_prediction_form.png")

            # positive-side record
            fill(page, HIGH)
            prob_high = analyse(page)
            shot(page.locator(".result-card"), "03_result_positive.png")
            print(f"  positive record -> {prob_high}")

            # negative-side record
            page.click("#reset-btn")
            page.wait_for_timeout(300)
            fill(page, LOW)
            prob_low = analyse(page)
            shot(page.locator(".result-card"), "04_result_negative.png")
            print(f"  negative record -> {prob_low}")

            # validation state
            page.click("#reset-btn")
            page.wait_for_timeout(300)
            page.fill("#f-age", "250")
            page.click("#submit-btn")
            page.wait_for_selector("#error-summary:not([hidden])", timeout=8000)
            page.wait_for_timeout(400)
            shot(page.locator(".form-card"), "05_validation.png", max_height=1500)

            # how it works
            only(page, "how")
            page.wait_for_timeout(300)
            shot(page.locator("#how .shell"), "06_how_it_works.png", max_height=1250)

            # model + performance
            only(page, "model")
            page.wait_for_timeout(300)
            shot(page.locator("#model-content .grid").first, "07_model_selection.png")

            only(page, "performance")
            page.wait_for_timeout(400)
            content = page.locator("#performance-content")
            shot(content.locator("> .grid").first, "08_performance_tiles.png")
            shot(content.locator("> .table-wrap").first, "09_model_comparison.png")
            shot(content.locator("> .grid").nth(1), "10_roc_pr.png")
            shot(content.locator("> .grid").nth(2), "11_confusion_matrix.png")
            shot(content.locator("> .grid").nth(3), "12_threshold_calibration.png")
            shot(content.locator("> .grid").nth(5), "13_explainability.png")
            page.close()

            # ----------------------------------------------------- desktop dark
            page = browser.new_page(viewport={"width": 1440, "height": 900},
                                    device_scale_factor=2, color_scheme="dark")
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_selector("#hero-stats .stat-value", timeout=15000)
            page.evaluate(SETTLE)
            page.wait_for_timeout(600)
            shot(page, "14_home_dark.png")

            only(page, "prediction")
            fill(page, HIGH)
            analyse(page)
            shot(page.locator(".result-card"), "15_result_dark.png")

            only(page, "performance")
            page.wait_for_timeout(400)
            shot(page.locator("#performance-content > .grid").nth(1), "16_roc_pr_dark.png")
            page.close()

            # ------------------------------------------------------------ mobile
            page = browser.new_page(viewport={"width": 390, "height": 844},
                                    device_scale_factor=3, color_scheme="light",
                                    is_mobile=True, has_touch=True)
            page.goto(BASE, wait_until="networkidle")
            page.wait_for_selector("#hero-stats .stat-value", timeout=15000)
            page.evaluate(SETTLE)
            page.wait_for_timeout(600)
            shot(page, "17_mobile_home.png")

            page.click("#nav-toggle")
            page.wait_for_timeout(400)
            shot(page, "18_mobile_nav.png")
            page.click("#nav-toggle")

            only(page, "prediction")
            fill(page, HIGH)
            analyse(page)
            page.evaluate("() => window.scrollTo(0, 0)")
            page.wait_for_timeout(300)
            shot(page.locator(".result-card"), "19_mobile_result.png", max_height=3400)
            page.close()

            browser.close()
        print("\nAll screenshots captured from the live application.")
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except Exception:
            server.kill()


if __name__ == "__main__":
    main()
