import pandas as pd
from pathlib import Path

def create_html_link(url, text):
    """
    URL이 유효하면 HTML 앵커 태그를 생성하고, 그렇지 않으면 빈 문자열을 반환합니다.
    """
    if pd.notna(url) and isinstance(url, str) and url.strip().startswith("http"):
        return f'<a href="{url}" target="_blank">{text}</a>'
    return ""

def process_dataframe_for_html(df: pd.DataFrame) -> pd.DataFrame:
    """
    HTML 변환 전 DataFrame에 필요한 모든 변환 작업을 적용합니다.
    (이름 정리, 열 이름 변경, 열 순서 지정, 링크 생성 등)
    """
    # 1. 여러 소스(filename, Model ID)에서 'Name' 열을 통합합니다.
    df_copy = df.copy()
    if 'filename' in df_copy.columns:
        df_copy['Name'] = df_copy['filename']
    elif 'Model ID' in df_copy.columns:
        df_copy['Name'] = df_copy['Model ID']
    else:
        df_copy['Name'] = 'N/A'

    # 2. 열 이름을 표시하려는 이름으로 변경합니다.
    column_map = {
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
        "Input Resolution": "Input Resolution",
        "Operations": "Operations",
        "Parameters": "Parameters",
    }
    df_copy.rename(columns={k: v for k, v in column_map.items() if k in df_copy.columns}, inplace=True)

    # 'Raw Accuracy' 열에서 'Metric' 열을 생성합니다.
    if 'Raw Accuracy' in df_copy.columns:
        def get_metric_type(value):
            if pd.isna(value):
                return ''
            val_str = str(value).strip()
            if val_str.startswith('Top1'):
                return 'Top1'
            elif val_str.startswith('mAP'):
                return 'mAP'
            elif val_str.startswith('Avg PSNR'):
                return 'PSNR/SSIM'
            elif val_str.startswith('Val AP'):
                return 'AP'
            else:
                return ''
        df_copy['Metric'] = df_copy['Raw Accuracy'].apply(get_metric_type)
    else:
        df_copy['Metric'] = ''


    # 'FPS/W' 열의 값을 소수점 2자리로 반올림합니다.
    if 'FPS/W' in df_copy.columns:
        # 먼저 숫자형으로 변환하고, 변환할 수 없는 값은 NaN으로 처리합니다.
        df_copy['FPS/W'] = pd.to_numeric(df_copy['FPS/W'], errors='coerce')
        # 소수점 둘째 자리에서 반올림합니다. NaN 값은 그대로 유지됩니다.
        df_copy['FPS/W'] = df_copy['FPS/W'].round(2)

    # 3. HTML 테이블에 표시할 최종 열 순서를 정의합니다.
    desired_columns = [
        "Name", "Dataset", "Input Resolution", "Operations", "Parameters",
        "Metric", "Raw Accuracy", "NPU Accuracy", "FPS", "FPS/W",
        "Reference", "DXNN", "ONNX", "JSON"
    ]

    # 원하는 열이 없는 경우, 빈 값으로 추가합니다.
    for col in desired_columns:
        if col not in df_copy.columns:
            df_copy[col] = ""

    # 정의된 순서대로 열을 선택합니다.
    df_processed = df_copy[desired_columns]

    # 4. 링크가 포함될 열의 형식을 지정합니다.
    link_columns = ["Reference", "ONNX", "DXNN", "JSON"]
    for col in link_columns:
        if col in df_processed.columns:
            # SettingWithCopyWarning을 피하기 위해 .loc 사용
            df_processed.loc[:, col] = df_processed[col].apply(lambda url: create_html_link(url, col))

    return df_processed

def generate_model_zoo_html(csv_path: Path, html_path: Path):
    """
    CSV 파일을 읽고 'Task'별로 그룹화하여
    각 Task에 대한 별도의 테이블이 있는 스타일 HTML 페이지를 생성합니다.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at {csv_path}")
        return

    # 1. 그룹화를 위해 'Task' 열을 정리합니다. (예: "1. Image Classification" -> "Image Classification")
    if 'Task' in df.columns:
        df['Task'] = df['Task'].str.replace(r'^\d+\.\s*', '', regex=True).str.strip()
    else:
        df['Task'] = 'Uncategorized Models' # Task 열이 없을 경우 기본값

    # 파일에 나타난 순서대로 Task 순서를 유지합니다.
    task_order = df['Task'].unique()
    all_html_parts = []

    # 2. 각 Task를 순회하며 별도의 부제와 테이블을 생성합니다.
    for task_name in task_order:
        # 현재 Task에 대한 부제를 추가합니다.
        all_html_parts.append(f'<h3 class="task-subtitle">{task_name}</h3>')

        # 현재 Task에 해당하는 데이터만 필터링합니다.
        task_df = df[df['Task'] == task_name]

        # 헬퍼 함수를 사용하여 데이터를 처리합니다.
        df_final = process_dataframe_for_html(task_df)

        # 현재 Task의 데이터를 HTML 테이블로 변환합니다.
        table_html = df_final.to_html(
            escape=False,
            index=False,
            classes="model-zoo-table",
            na_rep=""
        )

        # 생성된 테이블을 컨테이너로 감싸서 추가합니다.
        all_html_parts.append(f'<div class="table-container">{table_html}</div>')

    # 모든 HTML 조각(부제, 테이블)을 하나로 합칩니다.
    body_content = "\n".join(all_html_parts)

    # 3. 전체 HTML 문서를 스타일과 함께 정의합니다.
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DEEPX Model Zoo</title>
        <style>
            body {{
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                margin: 0;
                padding: 2rem;
                background-color: #f8f9fa;
                color: #212529;
            }}
            h2 {{
                text-align: center;
                color: #343a40;
                margin-bottom: 2rem;
            }}
            h3.task-subtitle {{
                color: #343a40;
                margin-top: 3rem;
                margin-bottom: 1.5rem;
                border-bottom: 2px solid #dee2e6;
                padding-bottom: 0.5rem;
                font-size: 1.75rem;
            }}
            body > h3.task-subtitle:first-of-type {{
                margin-top: 1rem;
            }}
            .table-container {{
                overflow-x: auto;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                background-color: #ffffff;
            }}
            table.model-zoo-table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 0.9rem;
            }}
            table.model-zoo-table th, table.model-zoo-table td {{
                border: 1px solid #dee2e6;
                padding: 12px 15px;
                text-align: left;
                vertical-align: middle;
            }}
            table.model-zoo-table thead th {{
                background-color: #e9ecef;
                color: #495057;
                font-weight: 600;
                position: sticky;
                top: 0;
            }}
            table.model-zoo-table tbody tr:nth-of-type(even) {{
                background-color: #f8f9fa;
            }}
            table.model-zoo-table tbody tr:hover {{
                background-color: #e9ecef;
            }}
            table.model-zoo-table a {{
                text-decoration: none;
                color: #007bff;
                font-weight: 500;
            }}
            table.model-zoo-table a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <h2>DEEPX Model Zoo</h2>
        {body_content}
    </body>
    </html>
    """

    # 4. 생성된 HTML을 출력 파일에 씁니다.
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Successfully generated grouped HTML file at {html_path}")

if __name__ == '__main__':
    # 입출력 경로를 정의합니다.
    input_csv = Path("sample.csv")
    output_html = Path("model_zoo_grouped.html")
    generate_model_zoo_html(input_csv, output_html)


