#!/usr/bin/env python3
"""
CSV -> HTML (method #1): pandas groupby + DataFrame.to_html
- Groups by 'Task' column, drops it from tables.
- Renders Source/Compiled/onnx/json as short anchor texts.
Usage:
    python csv2html_pandas.py <input.csv> <output.html>
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
    cols = [c for c in df.columns if c != TASK_COL]

    # Case-insensitive link column matching
    colmap = {c.lower(): c for c in df.columns}
    link_cols = [colmap[c.lower()] for c in LINK_KEYS if c.lower() in colmap]

    def linkify_columns(frame):
        out = frame.copy()
        for col in link_cols:
            out[col] = out[col].apply(lambda v: linkify(v, col.lower()))
        return out

    html = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>CSV -> HTML</title>',
            '<style>body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:28px;}',
            'h2{margin:32px 0 8px;} .table-wrap{overflow:auto;border:1px solid #ddd;border-radius:10px;}',
            'table{border-collapse:collapse;width:100%;} th,td{border-top:1px solid #eee;padding:8px 10px;text-align:center;font-size:14px;}',
            'thead th{position:sticky;top:0;background:#fafafa;border-bottom:1px solid #ddd;} tbody tr:nth-child(even){background:#fafafa;}</style>',
            '<body><h1>Model Zoo</h1>']
    for task, g in df.groupby(TASK_COL):
        g2 = linkify_columns(g[cols])
        html.append(f"<h2>{escape(str(task))}</h2><div class='table-wrap'>")
        html.append(g2.to_html(index=False, escape=False))
        html.append("</div>")
    html.append("</body></html>")
    Path(outp).write_text("\n".join(html), encoding="utf-8")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python csv2html_pandas.py <input.csv> <output.html>")
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
