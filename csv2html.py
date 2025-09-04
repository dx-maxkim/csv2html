#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import re
import pandas as pd


def _normalize_key(s: str) -> str:
    """Name 매칭을 위한 정규화: 소문자, 앞뒤공백 제거, 연속공백 단일화."""
    if not isinstance(s, str):
        return ""
    s = " ".join(s.strip().split())
    return s.lower()

def load_meta(path: Path) -> pd.DataFrame:
    """
    models_meta.(csv|parquet) 를 읽어 표준 스키마로 반환.
    필수 컬럼: Name, Input Resolution, Operations, Parameters
    """
    if not path.exists():
        raise FileNotFoundError(f"Enrichment file not found: {path}")

    if path.suffix.lower() == ".csv":
        meta = pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        meta = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported enrichment format. Use CSV or Parquet.")

    # 열 이름 방어적 표준화
    meta.columns = [str(c).strip() for c in meta.columns]
    rename_map = {
        "InputResolution": "Input Resolution",
        "Input_Resolution": "Input Resolution",
        "Ops": "Operations",
        "Params": "Parameters",
    }
    for k, v in rename_map.items():
        if k in meta.columns and v not in meta.columns:
            meta.rename(columns={k: v}, inplace=True)

    required = ["Name", "Input Resolution", "Operations", "Parameters"]
    missing = [c for c in required if c not in meta.columns]
    if missing:
        raise ValueError(f"Missing columns in enrichment: {missing}")

    # 키 정규화 컬럼 추가
    meta["_match_key_"] = meta["Name"].astype(str).map(_normalize_key)

    # 중복 Name 정책: 마지막(가장 아래) 레코드가 우선
    meta = meta.drop_duplicates(subset=["_match_key_"], keep="last")

    # 타입 정리
    meta["Operations"] = pd.to_numeric(meta["Operations"], errors="coerce")
    meta["Parameters"] = pd.to_numeric(meta["Parameters"], errors="coerce")

    return meta[["_match_key_", "Input Resolution", "Operations", "Parameters"]]


# ----------------------------
# Helpers & Constants
# ----------------------------
COLUMN_ALIAS_MAP = {
    # 표준화 대상(오탈자/대체 명)
    "Raw Accu": "Raw Accuracy",
    "Raw Accuracy(20250604)": "Raw Accuracy",
    "NPU Accracy": "NPU Accuracy",
    "3npus_fps": "FPS",
    "3npus_FPS/W": "FPS/Watt",
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
    "license": "License",
    "Raw Accuracy": "Raw Accuracy",
    "NPU Accuracy": "NPU Accuracy",
    "FPS": "FPS",
    "Task": "Task",
}

FINAL_COLUMNS = [
    "Task", "Name", "Dataset", "Input Resolution", "Operations", "Parameters",
    "License", "Metric", "Raw Accuracy", "NPU Accuracy", "FPS", "FPS/Watt",
    "Source", "Compiled", "onnx", "json"
]

LINK_COLUMNS = ["Source", "Compiled", "onnx", "json"]
LINK_TEXT_MAP = {"Source": "Source", "Compiled": "Compiled", "onnx": "onnx", "json": "json"}


def create_html_link(url: str, text: str) -> str:
    """
    URL과 링크 텍스트(종류)를 받아 적절한 아이콘이 포함된 HTML 앵커 태그를 생성한다.
    - text가 'Source'이면 바로가기 아이콘, 그 외에는 다운로드 아이콘을 사용한다.
    """
    if not isinstance(url, str) or not url.strip():
        return ""

    # 1. "바로가기" 아이콘 (External Link) SVG 코드 ↗️
    external_link_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
       <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
       <polyline points="15 3 21 3 21 9"></polyline>
       <line x1="10" y1="14" x2="21" y2="3"></line>
    </svg>
    """

    # 2. "다운로드" 아이콘 SVG 코드 📥
    download_svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="7 10 12 15 17 10"></polyline>
        <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
    """

    # 3. text 값에 따라 사용할 아이콘을 선택
    icon_svg = ""
    if text == "Source":
        icon_svg = external_link_svg
    else:
        icon_svg = download_svg
    
    # 4. 최종 HTML 태그를 생성하여 반환
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer" title="{text}">{icon_svg.strip()}</a>'


def get_metric_type(value) -> str:
    """
    Metric 대분류 (테이블의 'Metric' 컬럼용)
    """
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if s.startswith("Top1"):
        return "Top1"
    if s.startswith("mAP"):
        return "mAP50" if "50" in s or "0.5" in s else "mAP"
    if s.startswith("mIoU"):
        return "mIoU"
    if s.startswith("Avg PSNR") or s.startswith("PSNR") or "SSIM" in s:
        return "PSNR/SSIM"
    if s.startswith("Val AP") or s.startswith("AP(") or s.startswith("AP ") or s.startswith("AP"):
        return "AP(Easy/Med/Hard)"
    return ""


def get_metric_detail(value) -> str:
    """
    Metric 세부 라벨: AP(Easy)/AP(Medium)/AP(Hard) 등 구분
    """
    if not isinstance(value, str):
        return ""
    s = value.strip()
    if "AP(Easy)" in s:
        return "AP(Easy)"
    if "AP(Medium)" in s:
        return "AP(Medium)"
    if "AP(Hard)" in s:
        return "AP(Hard)"
    if s.startswith("Top1"):
        return "Top1"
    if s.startswith("mAP@0.5") or "mAP50" in s:
        return "mAP@0.5"
    if s.startswith("mAP"):
        return "mAP"
    if s.startswith("Avg PSNR") or s.startswith("PSNR") or "SSIM" in s:
        return "PSNR/SSIM"
    if s.startswith("Val AP") or s.startswith("AP(") or s.startswith("AP ") or s.startswith("AP"):
        return "AP"
    return ""


# ---- Accuracy 숫자 추출(요청 규칙) -------------------------------
_TOP1_RE   = re.compile(r'\bTop-?1\b[^0-9\-+]*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
_MAP_MAIN  = re.compile(r'\bmAP\b(?!\s*@?\s*0?\.?5|50)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
_MAP_050   = re.compile(r'\bmAP(?:@?\s*0?\.?5|50)\b\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
_FIRST_NUM = re.compile(r'([-+]?\d+(?:\.\d+)?)')

_AP_PREAMBLE_RE = re.compile(r'\b(?:Val\s*)?AP\b', re.IGNORECASE)
_AP_PAIR_RE     = re.compile(r'\b(Easy|Medium|Hard)\b\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
_AP_FUNC_RE     = re.compile(r'AP\((Easy|Medium|Hard)\)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
_NUM_RE         = re.compile(r'[-+]?\d+(?:\.\d+)?')

_PSNR_RE = re.compile(r'\b(?:Avg\s*)?PSNR\b[^0-9\-+]*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)
_SSIM_RE = re.compile(r'\b(?:Avg\s*)?SSIM\b[^0-9\-+]*([-+]?\d+(?:\.\d+)?)', re.IGNORECASE)


def extract_ap_triple_joined(raw: str) -> str:
    """
    'Val AP (Easy 95.44 / Medium 93.95 / Hard 85.664)' 같은 문자열에서
    Easy/Medium/Hard 값을 뽑아 '95.44 / 93.95 / 85.664'로 반환.
    실패 시 빈 문자열 반환.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()
    if not _AP_PREAMBLE_RE.search(s):
        return ""

    # 1) 라벨-값 쌍 우선
    pairs = {}
    for lbl, val in _AP_PAIR_RE.findall(s):
        pairs[lbl.capitalize()] = val
    for lbl, val in _AP_FUNC_RE.findall(s):
        pairs[lbl.capitalize()] = val

    ordered = [pairs.get("Easy"), pairs.get("Medium"), pairs.get("Hard")]
    if any(v is not None for v in ordered):
        return " / ".join(v or "" for v in ordered)

    # 2) 라벨이 없으면 괄호 안 등에서 숫자만 3개 순서대로 집계
    nums = _NUM_RE.findall(s)
    if len(nums) >= 3:
        return " / ".join(nums[:3])

    return ""


def extract_psnr_ssim_joined(raw: str) -> str:
    """
    'Avg PSNR: 31.709 / Avg SSIM: 0.8905' 같은 문자열에서
    '31.709 / 0.8905' 형태로 반환. (순서 PSNR / SSIM)
    다양한 표기(쉼표, 등호, 공백)도 허용.
    """
    if not isinstance(raw, str):
        return ""
    s = raw.strip()

    # 라벨 기반 우선
    psnr = None
    ssim = None
    m1 = _PSNR_RE.search(s)
    m2 = _SSIM_RE.search(s)
    if m1:
        psnr = m1.group(1)
    if m2:
        ssim = m2.group(1)
    if psnr or ssim:
        return " / ".join([psnr or "", ssim or ""]).strip()

    # 라벨이 없고 숫자만 나열된 경우: 앞의 두 숫자를 PSNR/SSIM으로 간주
    nums = _NUM_RE.findall(s)
    if len(nums) >= 2:
        return f"{nums[0]} / {nums[1]}"
    return ""


def extract_primary_accuracy(value) -> str:
    """
    Raw Accuracy 문자열에서 표시용 숫자 하나만 뽑아낸다.
    우선순위: Top1 > mAP(기본) > mAP@0.5/50 > 첫 숫자(Fallback)
    """
    if not isinstance(value, str):
        return ""
    s = value.strip()
    for pat in (_TOP1_RE, _MAP_MAIN, _MAP_050, _FIRST_NUM):
        m = pat.search(s)
        if m:
            return m.group(1)
    return ""
# ------------------------------------------------------------------

def _lower_alias_map(alias_map: dict) -> dict:
    """케이스 무관 매핑을 위해 소문자 키로 변환한 alias 맵을 만든다."""
    return {str(k).strip().lower(): v for k, v in alias_map.items()}


def dedupe_and_bfill_column(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """
    동일 라벨이 여러 번 생겼을 때 좌→우 우선으로 bfill하여 하나로 병합.
    """
    cols = df.filter(regex=fr"^{re.escape(colname)}$", axis=1)
    if cols.shape[1] > 1:
        merged = cols.bfill(axis=1).iloc[:, 0]
        extra = list(cols.columns[1:])
        df.drop(columns=extra, inplace=True)
        df[colname] = merged
    return df


def _metric_accuracy_parsor(df: pd.DataFrame) -> pd.DataFrame:
    """
    Metric, Accuracy 값을 원하는 형태로 변환
    """
    out_rows = []
    for _, row in df.iterrows():
        raw_str = str(row.get("Raw Accuracy", ""))
        npu_str = str(row.get("NPU Accuracy", ""))

        # 그대로 유지 (단, 숫자만 남기고 Metric/세부 라벨 정리)
        detail = get_metric_detail(raw_str)
        major = get_metric_type(raw_str)
        new_metric = detail if detail in ("AP(Easy)", "AP(Medium)", "AP(Hard)") else major
        row = row.copy()
        row["Metric"] = new_metric
        if new_metric == "AP(Easy/Med/Hard)":
            triple = extract_ap_triple_joined(raw_str)
            row["Raw Accuracy"] = triple if triple else extract_primary_accuracy(raw_str)
            triple = extract_ap_triple_joined(npu_str)
            row["NPU Accuracy"] = triple if triple else extract_primary_accuracy(npu_str)

        elif new_metric == "PSNR/SSIM":
            pair = extract_psnr_ssim_joined(raw_str)
            row["Raw Accuracy"] = pair if pair else extract_primary_accuracy(raw_str)
            row["NPU Accuracy"] = pair if pair else extract_primary_accuracy(npu_str)

        else:
            row["Raw Accuracy"] = extract_primary_accuracy(raw_str)
            row["NPU Accuracy"] = extract_primary_accuracy(npu_str)
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def normalize_and_process(df: pd.DataFrame, meta_path: Path | None = None) -> pd.DataFrame:
    """
    DataFrame 컬럼 표준화 + 파생 컬럼 생성 + 링크/숫자 포맷팅 적용
    """
    df = df.copy()

    # 0) 헤더 공백 제거
    df.columns = [str(c).strip() for c in df.columns]

    # 1) Task 정리
    if "Task" in df.columns:
        df["Task"] = df["Task"].astype(str).str.replace(r"^\d+\.\s*", "", regex=True).str.strip()
    else:
        df["Task"] = "Uncategorized Models"

    # 2) 컬럼 이름 표준화(케이스 무관 rename)
    lower_alias = _lower_alias_map(COLUMN_ALIAS_MAP)
    rename_map = {c: lower_alias[c.lower()] for c in df.columns if c.lower() in lower_alias}
    df.rename(columns=rename_map, inplace=True)

    # 2-1) Name/License 중복 라벨 병합
    df = dedupe_and_bfill_column(df, "Name")
    df = dedupe_and_bfill_column(df, "License")

    # 3) Name 보강(없거나 전부 NaN일 때)
    need_fill_name = False
    if "Name" not in df.columns:
        need_fill_name = True
    else:
        if df["Name"].isna().all():
            need_fill_name = True

    if need_fill_name:
        if "filename" in df.columns:
            df["Name"] = df["filename"]
        elif "Model ID" in df.columns:
            df["Name"] = df["Model ID"]
        else:
            df["Name"] = "N/A"

    # 4) Metric, Accuracy
    #if "Raw Accuracy" not in df.columns:
    #    df["Raw Accuracy"] = ""
    #if "NPU Accuracy" not in df.columns:
    #    df["NPU Accuracy"] = ""
    df = _metric_accuracy_parsor(df)

    # 5) 필수 컬럼 보강
    for col in FINAL_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    # 6) 링크 컬럼 앵커 처리
    for col in LINK_COLUMNS:
        df.loc[:, col] = df[col].apply(lambda url: create_html_link(url, LINK_TEXT_MAP[col]))

    # 7) 숫자 포맷팅
    for col in ["Operations", "Parameters", "FPS", "FPS/Watt"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Operations"] = df["Operations"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    df["Parameters"] = df["Parameters"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")
    df["FPS"] = df["FPS"].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")
    df["FPS/Watt"] = df["FPS/Watt"].map(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

    if "Name" not in df.columns:
        df["Name"] = "N/A"
    df["_match_key_"] = df["Name"].astype(str).map(_normalize_key)

    if "License" in df.columns:
        df["License"] = df["License"].fillna("Not Specified")
        df.loc[df["License"].astype(str).str.strip() == "", "License"] = "Not Specified"


    # --- 외부 보강 파일이 있으면 Left Join ---
    if meta_path is not None:
        meta = load_meta(meta_path)
        df = df.merge(meta, on="_match_key_", how="left", suffixes=("", "_meta"))

        # CSV에 없던 값만 메타로 채우기(기존값이 비었을 때만 덮어쓰기)
        def _fill_if_empty(base_col: str, meta_col: str):
            if base_col not in df.columns:
                df[base_col] = ""
            df[base_col] = df[base_col].where(df[base_col].astype(str).str.len() > 0, df[meta_col])

        _fill_if_empty("Input Resolution", "Input Resolution_meta")
        _fill_if_empty("Operations",        "Operations_meta")
        _fill_if_empty("Parameters",        "Parameters_meta")

        # 메타 컬럼은 정리
        df.drop(columns=[c for c in df.columns if c.endswith("_meta")], inplace=True)

    # 8) 최종 컬럼 순서
    df = df[FINAL_COLUMNS]
    return df


def dataframe_grouped_html(df: pd.DataFrame) -> str:
    """
    Task 순서대로 섹션(<h2>) + 테이블(각각에 Task 컬럼 포함) HTML 생성
    """
    parts = []
    task_order = df["Task"].astype(str).tolist() if "Task" in df.columns else []
    task_order = pd.Series(task_order).drop_duplicates().tolist() if task_order else ["Uncategorized Models"]

    for task in task_order:
        sub = df[df["Task"] == task]
        if sub.empty:
            continue
        parts.append(f'<h2 class="task-title">{task}</h2>')
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
      box-shadow: 0 4px 8px rgba(0,0,0,.3);
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
      text-align: center;
      vertical-align: middle;
      white-space: nowrap;
    }}
    table.model-zoo-table thead th {{
      background: #c0c0c0;
      color: var(--muted);
      font-weight: 600;
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    table.model-zoo-table tbody tr:nth-of-type(even) {{
      background-color: #e6e6e0;
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


def generate_model_zoo_html(csv_path: Path, html_path: Path, meta_path: Path | None = None) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    print(f"meta = {meta_path}")
    df_raw = pd.read_csv(csv_path)
    df_processed = normalize_and_process(df_raw, meta_path=meta_path)
    body_html = dataframe_grouped_html(df_processed)
    html = build_full_html(body_html)
    html_path.write_text(html, encoding="utf-8")
    print(f"[OK] Generated: {html_path}")


def main():
    p = argparse.ArgumentParser(description="Generate DX Model Zoo-style HTML from CSV.")
    p.add_argument("--csv", type=Path, default=Path("sample.csv"), help="Input CSV file path")
    p.add_argument("--out", type=Path, default=Path("output.html"), help="Output HTML file path")
    p.add_argument("--meta", type=Path, default=Path("meta.csv"), help="Metadata CSV file path")
    args = p.parse_args()
    generate_model_zoo_html(args.csv, args.out, args.meta)


if __name__ == "__main__":
    main()

