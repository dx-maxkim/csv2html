# CSV to HTML Converter

이 프로젝트는 **CSV 파일을 HTML 테이블 형태**로 변환하는 파이썬 스크립트를 제공합니다.  
특징은 다음과 같습니다:

- `Task` 컬럼으로 섹션을 나눔 → 각 Task별 소제목(`<h2>`) + 테이블 출력
- `Source`, `Compiled`, `onnx`, `json` 컬럼은 **짧은 링크 텍스트**(`source`, `compiled`, `onnx`, `json`)로 출력
- 모든 행 텍스트는 **가운데 정렬**
- HTML 출력 시 기본적인 CSS 스타일 적용

---

## 스크립트 종류

1. **csv2html_pandas.py**
   - `pandas` 사용
   - `DataFrame.to_html()` 기반
   - 코드가 간결하고 유지보수 쉬움

2. **csv2html_template.py**
   - pandas로 데이터 처리 후 직접 HTML 템플릿 주입
   - CSS/마크업 커스터마이징이 쉬움

---

## 사용법

```bash
python csv2html_pandas.py input.csv output.html
python csv2html_template.py input.csv output.html
```

---

## 환경 설정

```bash
python3 -m venv .venv-csv2html
source .venv-scv2html/bin/activate
pip install -r requirements.txt
```
