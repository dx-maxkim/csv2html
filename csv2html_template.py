#!/usr/bin/env python3
"""
CSV -> HTML (method #2): Lightweight templating (no deps besides pandas)
- Groups by 'Task' and renders each section with a clean HTML template.
- Renders Source/Compiled/onnx/json as short anchor texts.
Usage:
    python csv2html_template.py <input.csv> <output.html>
"""
import sys
import pandas as pd
from html import escape
from pathlib import Path

TASK_COL = "Task"
LINK_KEYS = ["Source", "Compiled", "onnx", "json"]

def linkify(val, label):
    if pd.isna(val) or str(val).strip()=="":
        return ""
    return f'<a href="{escape(str(val))}" target="_blank" rel="noopener">{escape(label)}</a>'

def main(inp, outp):
    try:
        df = pd.read_csv(inp, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(inp, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    colmap = {c.lower(): c for c in df.columns}
    link_cols = [colmap[c.lower()] for c in LINK_KEYS if c.lower() in colmap]
    cols = [c for c in df.columns if c != TASK_COL]

    groups = []
    for task, g in df.groupby(TASK_COL):
        g2 = g[cols].copy()
        for lc in link_cols:
            g2[lc] = g2[lc].apply(lambda v: linkify(v, lc.lower()))
        groups.append({"task": task, "columns": list(g2.columns), "rows": g2.values.tolist()})

    template = """<!doctype html>
<html lang="en"><meta charset="utf-8">
<title>CSV -> HTML (templated)</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:28px;}
section{margin-bottom:32px;}
h2{margin:28px 0 10px;}
.table{border:1px solid #e5e7eb;border-radius:12px;overflow:auto;}
table{border-collapse:collapse;width:100%;}
th,td{padding:8px 10px;border-top:1px solid #f0f0f0;text-align:center;}
thead th{background:#f8fafc;border-bottom:1px solid #e5e7eb;}
tbody tr:nth-child(odd){background:#fcfcfc;}
</style>
<body>
<h1>Model Zoo</h1>
%%SECTIONS%%
</body></html>"""

    parts = []
    for g in groups:
        head = "".join(f"<th>{escape(str(c))}</th>" for c in g["columns"])
        rows_html = []
        for row in g["rows"]:
            cells = []
            for cell in row:
                if isinstance(cell, str) and cell.startswith("<a "):
                    cells.append(f"<td>{cell}</td>")
                else:
                    cells.append(f"<td>{escape(str(cell))}</td>")
            rows_html.append(f"<tr>{''.join(cells)}</tr>")
        parts.append(f"<section><h2>{escape(str(g['task']))}</h2><div class='table'><table>"
                     f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div></section>")
    Path(outp).write_text(template.replace("%%SECTIONS%%", "".join(parts)), encoding="utf-8")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python csv2html_template.py <input.csv> <output.html>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
