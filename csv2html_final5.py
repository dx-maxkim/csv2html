#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


# ----------------------------
# Helpers & Constants
# ----------------------------
COLUMN_ALIAS_MAP = {
    # 표준화 대상(오탈자/대체 명)
    "Raw Accu": "Raw Accuracy",
    "Raw Accuracy(20250604)": "Raw Accuracy",
    "NPU Accracy": "NPU Accuracy",
    "3npus_fps": "FPS",
    "3npus_FPS/W": "FPS/W",
    "reference": "Source",
    "Reference": "Source",
    "dxnn": "Compiled",
    "DXNN": "Compiled",
    "onnx": "onnx",   # 타깃은 소문자 라벨
    "json": "json",   # 타깃은 소문자 라벨
    "Model ID": "Name",
    # 그대로 쓰는 것들(있을 때만 rename; 없으면 건너뜀)
    "Dataset": "Dataset",
    "Input Resolution": "Input Resolution",
    "Operations": "Operations",
    "Parameters": "Parameters",
    "License": "License",
    "Raw Accuracy": "Raw Accuracy",
    "NPU Accuracy": "NPU Accuracy",
    "FPS": "FPS",
    "FPS/W": "FPS/W",
    "Task": "Task",
}

FINAL_COLUMNS = [
    "Task", "Name", "Dataset", "Input Resolution", "Operations", "Parameters",
    "License", "Metric", "Raw Accuracy", "NPU Accuracy", "FPS", "FPS/W",
    "Source", "Compiled", "onnx", "json"
]

LINK_COLUMNS = ["Source", "Compiled", "onnx", "json"]
LINK_TEXT_MAP = {"Source": "Source", "Compiled": "Compiled", "onnx": "onnx", "json": "json"}


def create_html_link(url: str, text: str) -> str:
    """
    URL이 유효하면 HTML 앵커 태그를 생성, 아니면 빈 문자열.
    """
    if pd.notna(url) and isinstance(url, str):
        u = url.strip()
        if u.startswith("http://") or u.startswith("https://"):
            # 보안/성능: noopener/noreferrer 기본 적용
            return f'<a href="{u}" target="_blank" rel="noopener noreferrer">{text}</a>'
    return ""


def get_metric_type(value) -> str:
    """
    Raw Accuracy 문자열에서 metric 유형을 추출.
    지원 예: Top1, mAP/mAP50, AP, AP(Easy/Medium/Hard), PSNR/SSIM 등
    """
    if not isinstance(value, str):
        return ""
    s = value.strip()
    # 대표 케이스부터 빠르게 매칭
    if s.startswith("Top1"):
        return "Top1"
    if s.startswith("mAP"):
        return "mAP50" if "50" in s or "0.5" in s else "mAP"
    if s.startswith("Avg PSNR") or s.startswith("PSNR") or "SSIM" in s:
        return "PSNR/SSIM"
    if s.startswith("Val AP") or s.startswith("AP(") or s.startswith("AP " ) or s.startswith("AP"):
        return "AP"
    return ""


def normalize_and_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame 컬럼 표준화 + 파생 컬럼 생성 + 링크/숫자 포맷팅 적용
    """
    df = df.copy()

    # 1) Task 정리(예: "1. Image Classification" -> "Image Classification")
    if "Task" in df.columns:
        df["Task"] = df["Task"].astype(str).str.replace(r"^\d+\.\s*", "", regex=True).str.strip()
    else:
        df["Task"] = "Uncategorized Models"

    # 2) 컬럼 이름 표준화(존재하는 컬럼만 rename)
    rename_map = {c: COLUMN_ALIAS_MAP[c] for c in df.columns if c in COLUMN_ALIAS_MAP}
    df.rename(columns=rename_map, inplace=True)

    # 3) Name이 비어 있으면 filename/Model ID 등에서 채우기
    if "Name" not in df.columns or df["Name"].isna().all():
        # 위에서 rename으로 이미 'filename', 'Model ID'를 Name으로 보냈을 수 있음
        # 그래도 Name이 없거나 전부 NaN이면 대체
        if "filename" in df.columns:
            df["Name"] = df["filename"]
        elif "Model ID" in df.columns:
            df["Name"] = df["Model ID"]
        else:
            df["Name"] = "N/A"

    # 4) Metric 파생
    if "Raw Accuracy" in df.columns:
        df["Metric"] = df["Raw Accuracy"].apply(get_metric_type)
    else:
        df["Metric"] = ""

    # 5) 필수 컬럼 보강(없으면 빈값)
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # 6) 링크 컬럼 앵커 처리
    for col in LINK_COLUMNS:
        if col in df.columns:
            df.loc[:, col] = df[col].apply(lambda url: create_html_link(url, LINK_TEXT_MAP[col]))

    # 7) 숫자 포맷팅
    for col in ["Operations", "Parameters", "FPS", "FPS/W"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # 천단위 + 소수자릿수 통일
    df["Operations"] = df["Operations"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    df["Parameters"] = df["Parameters"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    df["FPS"] = df["FPS"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    df["FPS/W"] = df["FPS/W"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    # 8) 최종 컬럼 순서
    df = df[FINAL_COLUMNS]
    return df


def dataframe_grouped_html(df: pd.DataFrame) -> str:
    """
    Task 순서대로 섹션(<h2>) + 테이블(각각에 Task 컬럼 포함) HTML 생성
    """
    parts = []
    # 파일 등장 순서 유지
    task_order = df["Task"].astype(str).tolist() if "Task" in df.columns else []
    task_order = pd.Series(task_order).drop_duplicates().tolist() if task_order else ["Uncategorized Models"]

    for task in task_order:
        sub = df[df["Task"] == task]
        if sub.empty:
            continue
        # 섹션 제목
        parts.append(f'<h2 class="task-title">{task}</h2>')

        # 테이블로 변환(escape=False로 링크 허용)
        table_html = sub.to_html(
            escape=False,
            index=False,
            classes="model-zoo-table",
            na_rep=""
        )
        parts.append(f'<div class="table-container">{table_html}</div>')

    return "\n".join(parts)


def build_full_html(body: str) -> str:
    """
    전체 HTML 문서 스켈레톤 + 스타일
    """
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>DX Model Zoo</title>
  <style>
    :root {{
      --bg: #f8f9fa;
      --fg: #212529;
      --muted: #495057;
      --border: #dee2e6;
      --accent: #0d6efd;
      --card: #ffffff;
      --thead: #e9ecef;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--fg);
      line-height: 1.5;
    }}
    h1 {{
      text-align: center;
      margin: 0 0 28px 0;
      font-size: 2rem;
      color: #343a40;
    }}
    h2.task-title {{
      margin: 32px 0 12px 0;
      font-size: 1.5rem;
      color: #343a40;
      padding-bottom: 6px;
      border-bottom: 2px solid var(--border);
    }}
    .table-container {{
      overflow-x: auto;
      border-radius: 10px;
      box-shadow: 0 4px 8px rgba(0,0,0,.06);
      background: var(--card);
      margin-bottom: 22px;
    }}
    table.model-zoo-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.92rem;
      table-layout: auto;
    }}
    table.model-zoo-table th, table.model-zoo-table td {{
      border: 1px solid var(--border);
      padding: 10px 12px;
      text-align: center; /* 가운데 정렬 */
      vertical-align: middle;
      white-space: nowrap;
    }}
    table.model-zoo-table thead th {{
      background: var(--thead);
      color: var(--muted);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    table.model-zoo-table tbody tr:nth-of-type(even) {{
      background-color: #fdfdfd;
    }}
    table.model-zoo-table tbody tr:hover {{
      background-color: #f3f5f7;
    }}
    table.model-zoo-table a {{
      text-decoration: none;
      color: var(--accent);
      font-weight: 500;
    }}
    table.model-zoo-table a:hover {{ text-decoration: underline; }}
    @media (max-width: 768px) {{
      table.model-zoo-table th, table.model-zoo-table td {{
        padding: 8px 10px;
        font-size: 0.88rem;
      }}
    }}
  </style>
</head>
<body>
  <h1>DX Model Zoo</h1>
  {body}
</body>
</html>
"""


def generate_model_zoo_html(csv_path: Path, html_path: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    # CSV 로드
    df_raw = pd.read_csv(csv_path)

    # 정규화/가공
    df_processed = normalize_and_process(df_raw)

    # 섹션별 테이블 HTML 생성
    body_html = dataframe_grouped_html(df_processed)

    # 전체 HTML 문서 생성
    html = build_full_html(body_html)

    # 저장
    html_path.write_text(html, encoding="utf-8")
    print(f"[OK] Generated: {html_path}")


def main():
    p = argparse.ArgumentParser(description="Generate DX Model Zoo-style HTML from CSV.")
    p.add_argument("--csv", type=Path, default=Path("sample.csv"), help="Input CSV file path")
    p.add_argument("--out", type=Path, default=Path("model_zoo_grouped.html"), help="Output HTML file path")
    args = p.parse_args()

    generate_model_zoo_html(args.csv, args.out)


if __name__ == "__main__":
    main()

