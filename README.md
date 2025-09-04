# CSV to HTML Converter

이 프로젝트는 **CSV 파일을 HTML 테이블 형태**로 변환하는 파이썬 스크립트를 제공합니다.  
특징은 다음과 같습니다:

- `Task` 컬럼으로 섹션을 나눔 → 각 Task별 소제목(`<h2>`) + 테이블 출력
- `Source`, `Compiled`, `onnx`, `json` 컬럼은 **짧은 링크 텍스트**(`source`, `compiled`, `onnx`, `json`)로 출력
- 모든 행 텍스트는 **가운데 정렬**
- HTML 출력 시 기본적인 CSS 스타일 적용

---

## 파일 설명

1. **csv2html.py**
   - `pandas` 사용
   - csv 인풋을 받아 html 로 변환해 주는 스크립트
  
2. **sample.csv**
   - 아래 링크와 같이 개발팀에서 보내주는 excel 값을 csv 로 저장하여 input 으로 넣어주는 test sample
   - https://docs.google.com/spreadsheets/d/1Z6bEzwrRK17XCYSvEc2veg_eAC-eTe4S/edit?gid=1374016918#gid=1374016918

3. **meta.csv**
   - 개발팀에서 공유해 주는 excel 에 각 모델 별 Input Res / Operations / Parameters 정보가 누락되어 임시로 참조하는 값
   
---

## 사용법 (Example)

```bash
# 아래와 같이 실행하면 default param 값이 적용됨 (--csv sample.csv --meta meta.csv --out output.html)
python3 csv2html.py

# 3가지 파라미터를 줄 수 있음
python3 csv2html.py --csv sample.csv --meta meta.csv --out output.html
```

---

## 환경 설정

```bash
python3 -m venv .venv-csv2html
source .venv-csv2html/bin/activate
pip install -r requirements.txt
```
