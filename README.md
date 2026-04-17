

---
title: EVA Nexio DAIA v5
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false 
license: mit
---
# EVA NEXIO.DAIA v5

데이터 자동분석 플랫폼 | 품질 검증 · 피처 엔지니어링 · 분석 제안 · CSV 내보내기

- **LLM**: Gemini 1.5 Flash API
- **일일 제한**: IP당 5회


## 🚀 빠른 시작

```bash
conda activate daia_v3   # 기존 환경 재사용 가능
python app.py            # Gradio UI → http://localhost:7860
python main.py data.csv  # CLI 실행
```

## 📋 분석 11단계 파이프라인

| # | 단계 | 내용 |
|---|------|------|
| 1 | **Data Inventory** | 파일 로드, 크기·컬럼 파악 |
| 2 | **Data Profiling** | 스키마 감지, 컬럼 메타데이터 |
| 3 | **Data Typology** | 시계열/패널/이벤트/정적 분류 |
| 4 | **Quality Validation** | 결측치·이상값·중복·타입오류 |
| 5 | **Data Cleaning** | 결측처리·이상값 clip·타입변환 |
| 6 | **Data Structuring** | 중복제거·분석 가능 형태 변환 |
| 7 | **Data Integration** | 내부 피처 통합 |
| 8 | **Feature Engineering** | 시간피처·통계피처·lag·비율 |
| 9 | **Exploratory Analysis** | 차트 13종 + 인사이트 |
| 10 | **Analytical Framing** | 분석 문제 정의 |
| 11 | **Analysis Recommendation** | 분석 방향 제안 → PDF 포함 |

## 📁 산출물 (`daia_output/`)

```
daia_output/
  charts/{파일명}/     ← PNG 차트 (kaleido 필요)
  daia_report_*.html  ← HTML 인터랙티브 리포트
  daia_report_*.pdf   ← PDF 리포트 (reportlab 필요)
  datasets/
    daia_feature_*.csv ← 최종 분석용 Feature Dataset
```

## 🔧 추가 패키지 (선택)

```bash
pip install kaleido             # PNG 차트 저장
pip install reportlab pillow    # PDF 내보내기
pip install beautifulsoup4      # PDF 파싱 개선
pip install llama-cpp-python    # LLM 분석
```

## 🤖 LLM (Bllossom)

`models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf` 배치 시 자동 활성화

---

v4 변경사항: 11단계 파이프라인 명시화 / 품질 검증 리포트 / 박스플롯·PCA·피처중요도 차트 추가 /
Feature Engineering 로그 / 분석 제안 섹션 (HTML·PDF) / 최종 CSV 자동 저장 / 프로그레스바 개선

## 📂 models/ 폴더 구조

```
models/
  __init__.py          ← 패키지 init
  llm_client.py        ← LLM 통합 클라이언트 (JSON 추출, 폴백 포함)
  model_loader.py      ← GGUF 모델 로더 (Singleton)
  prompt_templates.py  ← 5종 프롬프트 템플릿
  llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf  ← (별도 배치 필요)
```

> GGUF 모델 파일은 용량 문제로 압축 파일에서 제외됩니다.  
> 기존 `models/` 폴더의 `.gguf` 파일을 그대로 유지하면 됩니다.

