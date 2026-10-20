"""Capture README screenshots from the real app without hosted API calls.

Requires trusted local trained artifacts, Playwright, and an installed Edge browser.
Install the optional capture dependency with: python -m pip install playwright
No training is started and no private test row is used as prediction input.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["GROQ_API_KEY"] = ""  # Use the local fallback; never call the hosted API.

from flask import jsonify
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

import ids_server


def public_values(value):
    """Remove the local project prefix from documentation-only JSON responses."""
    if isinstance(value, dict):
        return {key: public_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public_values(item) for item in value]
    if isinstance(value, str):
        value = value.replace(str(ROOT) + os.sep, "")
        if "\\" in value and ("Saved model" in value or "federated" in value):
            value = value.replace("\\", "/")
    return value


def main():
    output_dir = ROOT / "docs" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_server._sync_settings.update(auto_save_new=False, auto_federate=False)

    @ids_server.app.after_request
    def scrub_local_paths(response):
        if response.is_json:
            cleaned = jsonify(public_values(response.get_json()))
            cleaned.status_code = response.status_code
            return cleaned
        return response

    with ids_server.app.test_client() as client:
        for path in ("/", "/predict-ui", "/fl", "/rl", "/advisor", "/history", "/about", "/settings", "/info", "/live"):
            assert client.get(path).status_code == 200, path
        assert client.post("/predict", json={"features": [0.0]}).status_code == 400
        prediction = client.post("/predict", json={"features": [0.0] * ids_server.N_FEATURES})
        assert prediction.status_code == 200, "Synthetic-vector model inference failed"
        assert client.post("/predict", data=b" " * (1024 * 1024 + 1), content_type="application/json").status_code == 413
    print("PASS: page/API smoke checks, real synthetic-vector inference, invalid input, and 1 MiB request limit.", flush=True)

    print("Starting localhost screenshot server...", flush=True)
    server = make_server("127.0.0.1", 0, ids_server.app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    print(f"Screenshot server ready at {origin}", flush=True)
    try:
        with sync_playwright() as playwright:
            print("Starting headless Edge...", flush=True)
            browser = playwright.chromium.launch(channel="msedge", headless=True, timeout=20000)
            try:
                context = browser.new_context(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
                context.add_init_script("localStorage.setItem('prism_role', 'admin');")
                page = context.new_page()
                page.set_default_timeout(20000)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))

                def ready():
                    page.evaluate("document.fonts.ready")
                    page.locator("#roleModalOverlay").wait_for(state="hidden")

                page.goto(origin, wait_until="networkidle")
                ready()
                page.wait_for_function("document.querySelector('#statFeatures').textContent === '42'")
                page.screenshot(path=str(output_dir / "dashboard-home.png"), full_page=True)

                page.goto(origin + "/predict-ui", wait_until="networkidle")
                ready()
                page.wait_for_function("document.querySelector('#featTotal').textContent === '42'")
                page.locator("#feat").fill(", ".join(["0"] * ids_server.N_FEATURES))
                page.locator("#predictBtn").click()
                page.wait_for_function("document.querySelector('#pred').textContent.trim() !== '—'")
                page.wait_for_function("document.querySelector('#xaiBars').textContent.indexOf('Computing') === -1")
                page.wait_for_function("document.querySelector('#llmText').textContent.indexOf('Run Predict') === -1")
                for selector in ("#predTopoCard", "#learnCard", "#advisorCard"):
                    page.locator(selector).evaluate("element => element.style.display = 'none'")
                page.screenshot(path=str(output_dir / "dashboard-predict.png"), full_page=True)

                for path, name, selector in (
                    ("/fl", "dashboard-federated.png", "#flArch"),
                    ("/rl", "dashboard-rl.png", "#rlCount"),
                ):
                    page.goto(origin + path, wait_until="networkidle")
                    ready()
                    page.wait_for_function("selector => document.querySelector(selector).textContent.trim() !== '—'", arg=selector)
                    page.screenshot(path=str(output_dir / name), full_page=True)
                if errors:
                    raise RuntimeError("Browser errors during capture: " + "; ".join(errors))
                for path in sorted(output_dir.glob("dashboard-*.png")):
                    print(f"Saved {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
