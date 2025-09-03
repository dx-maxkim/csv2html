import pandas as pd
from html import escape
from pathlib import Path

# 입력 CSV 경로와 출력 HTML 경로
csv_path = Path("sample.csv")
html_path = Path("output_model_zoo.html")

# CSV 읽기
df = pd.read_csv(csv_path)

# "Task" 열 제거
df = df.drop(columns=["Task"], errors="ignore")

# 열 이름 매핑
column_map = {
    "Model ID": "Name",
    "filename": "Name",
    "Dataset": "Dataset",
    "Raw Accu": "Raw Accuracy",
    "Raw Accuracy(20250604)": "Raw Accuracy",
    "NPU Accracy": "NPU Accuracy",
    "3npus_fps": "FPS",
    "3npus_FPS/W": "FPS/W",
    "reference": "Reference",
    "onnx": "ONNX",
    "dxnn": "DXNN",
    "json": "JSON",
}

# 매핑된 컬럼들만 추출
desired_columns = ["Name", "Dataset", "Input Resolution", "Operations", "Parameters",
                   "Raw Accuracy", "NPU Accuracy", "FPS", "FPS/W",
                   "Reference", "DXNN", "ONNX", "JSON"]

# CSV 내 컬럼명을 HTML용 컬럼명으로 매핑
df_renamed = pd.DataFrame()
for original_col, html_col in column_map.items():
    if original_col in df.columns:
        df_renamed[html_col] = df[original_col]

# 나머지 열(있을 경우) 직접 추가
for col in ["Input Resolution", "Operations", "Parameters"]:
    if col not in df_renamed.columns:
        df_renamed[col] = ""

# 열 순서 정렬
df_renamed = df_renamed[[col for col in desired_columns if col in df_renamed.columns]]

# HTML 생성
html_lines = [
    "<html>",
    "<head>",
    "<meta charset='utf-8'>",
    "<style>",
    "table { border-collapse: collapse; width: 100%; font-family: sans-serif; }",
    "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
    "th { background-color: #f2f2f2; }",
    "a { text-decoration: none; color: blue; }",
    "</style>",
    "</head>",
    "<body>",
    "<h2>DX Model Zoo</h2>",
    "<table>",
    "<thead><tr>" + "".join(f"<th>{escape(col)}</th>" for col in df_renamed.columns) + "</tr></thead>",
    "<tbody>"
]

# 테이블 본문
for idx, row in df_renamed.iterrows():
    html_lines.append("<tr>")
    for col in df_renamed.columns:
        val = row[col]
        if col in ["Reference", "ONNX", "DXNN", "JSON"]:
            if pd.notna(val) and val.startswith("http"):
                link_text = col  # e.g., "onnx"
                val_html = f'<a href="{escape(val)}" target="_blank">{link_text}</a>'
            else:
                val_html = ""
        else:
            val_html = escape(str(val)) if pd.notna(val) else ""
        html_lines.append(f"<td>{val_html}</td>")
    html_lines.append("</tr>")

# 종료 태그
html_lines += ["</tbody>", "</table>", "</body>", "</html>"]

# 저장
with open(html_path, "w", encoding="utf-8") as f:
    f.write("\n".join(html_lines))

