"""
Local dashboard where the inputs can be edited and the model rerun.

    python dashboard_server.py            # then open http://127.0.0.1:8000
    python dashboard_server.py 8050       # another port

Each run from the page calls model.run() (equations from derivatives.py),
rewrites results.csv and saves the inputs used to results_config.json.
The server only listens on this computer (127.0.0.1).
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import model
from dashboard import read_csv, render_page
from plan import load_plan

HERE = Path(__file__).parent
CSV_PATH = HERE / "results.csv"
CONFIG_PATH = HERE / "results_config.json"
MAX_BODY = 1_000_000

run_lock = threading.Lock()
state = {"config": model.DEFAULT_CONFIG}


def run_config(config):
    """Validate, simulate, write results.csv + results_config.json."""
    with run_lock:
        clean = model.validate_config(config)
        start = time.perf_counter()
        model.run(clean, CSV_PATH)
        seconds = time.perf_counter() - start
        CONFIG_PATH.write_text(json.dumps(clean, indent=2), encoding="utf-8")
        state["config"] = clean
        return {"source": CSV_PATH.name, "columns": read_csv(CSV_PATH),
                "config": clean, "seconds": seconds}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, content_type):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            with run_lock:
                data = {"source": CSV_PATH.name, "columns": read_csv(CSV_PATH),
                        "config": state["config"], "defaults": model.DEFAULT_CONFIG,
                        "controllers": model.controller_info(), "editable": True}
            self._send(200, render_page(data), "text/html; charset=utf-8")
        elif path == "/results.csv":
            self._send(200, CSV_PATH.read_bytes(), "text/csv")
        elif path == "/api/controllers":
            # Controller files in controllers/ with their settings, for the dashboard.
            self._json(200, {"controllers": model.controller_info()})
        else:
            self._send(404, "Not found", "text/plain")

    def do_POST(self):
        if self.path not in ("/api/run", "/api/plan"):
            self._send(404, "Not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_BODY:
            self._json(413, {"error": "Request is too large."})
            return
        try:
            body = json.loads(self.rfile.read(length))
            if self.path == "/api/plan":
                # Plan CSV -> config for the form; nothing is run yet.
                config, summary = load_plan(str(body.get("text", "")), body.get("base") or state["config"],
                                            str(body.get("name") or "plan.csv")[:200])
                self._json(200, {"config": model.validate_config(config), "summary": summary})
            else:
                self._json(200, run_config(body))
        except (ValueError, KeyError, TypeError) as exc:
            self._json(400, {"error": str(exc) or "The inputs could not be read."})
        except RuntimeError as exc:           # solver failure
            self._json(500, {"error": f"The solver failed: {exc}"})

    def log_message(self, fmt, *args):
        if self.command == "POST":
            what = "plan loaded" if self.path == "/api/plan" else "run requested"
            sys.stderr.write(f"[{self.log_date_time_string()}] {what}\n")


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    saved = None
    if CONFIG_PATH.exists():
        try:
            saved = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except ValueError:
            saved = None
    print("Running the starting simulation...", flush=True)
    try:
        run_config(saved or model.DEFAULT_CONFIG)
    except (ValueError, RuntimeError) as exc:
        print(f"Saved inputs in {CONFIG_PATH.name} failed ({exc}); using the defaults.")
        run_config(model.DEFAULT_CONFIG)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dashboard at http://127.0.0.1:{port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
