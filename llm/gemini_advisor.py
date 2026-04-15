"""
DAIA - Gemini LLM 어드바이저 (클라우드 배포용)
LLMAdvisor와 동일한 public API — google-generativeai 기반

환경변수: GEMINI_API_KEY
"""
from __future__ import annotations
import os, json
from typing import Optional
from utils.logger import get_logger

logger = get_logger()


class GeminiAdvisor:
    """LLMAdvisor와 동일한 public API — Gemini Flash 기반"""

    MODEL = "gemini-1.5-flash"

    def __init__(self):
        self._available: Optional[bool] = None
        self._client = None

    # ── 초기화 ────────────────────────────────────────────────────────────────
    def _get_client(self):
        if self._client is not None:
            return self._client
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.warning("⚠️ GEMINI_API_KEY 환경변수 없음 → Gemini 비활성")
            self._available = False
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai.GenerativeModel(self.MODEL)
            self._available = True
            logger.info(f"✅ Gemini API 초기화 성공 ({self.MODEL})")
        except ImportError:
            logger.warning("⚠️ google-generativeai 미설치: pip install google-generativeai")
            self._available = False
        except Exception as e:
            logger.warning(f"⚠️ Gemini 초기화 실패: {e}")
            self._available = False
        return self._client

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        self._get_client()
        return self._available or False

    # ── public API (LLMAdvisor 동일 인터페이스) ───────────────────────────────
    def analyze_schema(self, df_info: dict, schema) -> str:
        if not self._get_client():
            return ""
        return self._safe_generate(self._prompt_schema(df_info, schema))

    def suggest_preprocessing(self, profile_summary: dict, schema) -> str:
        if not self._get_client():
            return ""
        return self._safe_generate(self._prompt_preprocessing(profile_summary, schema))

    def interpret_insights(self, insights: list, schema, stats: dict) -> str:
        if not self._get_client():
            return ""
        return self._safe_generate(self._prompt_insights(insights, schema, stats))

    # ── prompt builders ───────────────────────────────────────────────────────
    def _prompt_schema(self, df_info: dict, schema) -> str:
        return f"""당신은 데이터 분석 전문가입니다. 다음 데이터셋을 분석하고 한국어로 설명해주세요.

## 데이터 기본 정보
- 행 수: {df_info.get('rows', 'N/A'):,}
- 컬럼 수: {df_info.get('cols', 'N/A')}
- 스키마 타입: {schema.schema_type}
- 주요 컬럼: {df_info.get('columns', [])[:10]}

## 샘플 데이터
{df_info.get('sample', '')}

이 데이터의 도메인, 목적, 주요 특징을 2~3문단으로 설명하세요.
서두/인사 없이 분석 내용만 작성하세요."""

    def _prompt_preprocessing(self, profile: dict, schema) -> str:
        return f"""데이터 전처리 전문가로서 다음 데이터 프로파일을 보고 전처리 권장사항을 제시하세요.

## 데이터 프로파일
- 스키마: {schema.schema_type}
- 결측치 많은 컬럼: {profile.get('high_missing', [])}
- 수치형 컬럼: {schema.numeric_cols[:8]}
- 범주형 컬럼: {schema.categorical_cols[:8]}
- 혼합 타입 컬럼: {getattr(schema, 'mixed_cols', [])}
- 데이터 품질 점수: {profile.get('quality_score', 'N/A')}/100

구체적인 전처리 단계를 번호 목록으로 작성하세요.
각 항목: "컬럼명: 처리방법 (이유)" 형식."""

    def _prompt_insights(self, insights: list, schema, stats: dict) -> str:
        return f"""데이터 분석 전문가로서 다음 자동 분석 결과를 해석하고 인사이트를 도출하세요.

## 자동 감지된 특이사항
{chr(10).join(insights[:15])}

## 주요 통계
{json.dumps(stats, ensure_ascii=False, indent=2)}

## 스키마 타입: {schema.schema_type}

다음을 포함해 한국어로 간결하게 작성하세요:
1. 데이터 전반적 상태 평가
2. 주요 발견사항 3가지 이상
3. 추가 분석 권장사항
4. 데이터 품질 이슈 주의사항"""

    # ── 생성 헬퍼 ─────────────────────────────────────────────────────────────
    def _safe_generate(self, prompt: str) -> str:
        try:
            response = self._client.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 600,
                    "temperature": 0.2,
                }
            )
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini 생성 오류: {e}")
            return f"[Gemini 오류: {e}]"
