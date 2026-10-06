"""
Build a self-contained HTML dashboard from the simulation CSV.

    python model.py          # writes results.csv
    python dashboard.py      # writes dashboard.html (open it in a browser)

    python dashboard.py other.csv out.html

The CSV data is embedded in the page, so the HTML file works on its own.
To change inputs from the page and rerun the model, use
dashboard_server.py instead.
"""

import csv
import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
HEAD = ('<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n')


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    columns = {name: [] for name in header}
    for row in body:
        for name, value in zip(header, row):
            columns[name].append(float(value) if value != "" else None)
    return columns


def render_page(data):
    """Full HTML page with `data` embedded as the dashboard's DATA object.

    data keys: source, columns, and for the editable version also
    editable=True, config (current run) and defaults (DEFAULT_CONFIG).
    """
    payload = json.dumps(data).replace("</", "<\\/")
    return HEAD + TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", payload)


def build(csv_path="results.csv", html_path="dashboard.html"):
    data = {"source": Path(csv_path).name, "columns": read_csv(csv_path)}
    # The run inputs (molecular weights, ML constants, volume) fill in the equations.
    config = Path(csv_path).with_name("results_config.json")
    if config.exists():
        data["config"] = json.loads(config.read_text(encoding="utf-8"))
    Path(html_path).write_text(render_page(data), encoding="utf-8")
    return html_path


if __name__ == "__main__":
    args = sys.argv[1:]
    out = build(*args) if args else build()
    print(f"Wrote {out}")
