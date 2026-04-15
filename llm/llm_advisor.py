"""
DAIA - LLM 어드바이저
우선순위: GEMINI_API_KEY 환경변수 존재 시 Gemini API 사용,
          없으면 로컬 llama-cpp-python 폴백.

로컬 실행: config.yaml llm.model_path 지정 + llama-cpp-python 설치
클라우드:  GEMINI_API_KEY 환경변수 설정 (HF Spaces Secrets)
"""
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Optional
from core.schema_detector import SchemaInfo
from utils.logger import get_logger

logger = get_logger()


class LLMAdvisor:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._client = None
        self._available: Optional[bool] = None
        # Gemini 우선 시도
        self._gemini = None
        self._try_init_gemini()

    # ── Gemini 초기화 (우선) ──────────────────────────────────────────────────
    def _try_init_gemini(self):
        if not os.environ.get("GEMINI_API_KEY", "").strip():
            return
        try:
            from llm.gemini_advisor import GeminiAdvisor
            g = GeminiAdvisor()
            if g.is_available():
                self._gemini = g
                self._available = True
                logger.info("✅ LLM: Gemini API 활성화")
        except Exception as e:
            logger.warning(f"⚠️ Gemini 초기화 실패: {e}")

    # ── 로컬 llama-cpp-python (폴백) ──────────────────────────────────────────
    def _get_client(self):
        if self._gemini:          # Gemini 우선
            return self._gemini
        if self._client is not None:
            return self._client
        try:
            from models.llm_client import get_llm_client
            self._client = get_llm_client(self.config_path)
            self._client.initialize()
            self._available = True
            logger.info("✅ LLM: llama-cpp-python 초기화 성공")
        except FileNotFoundError as e:
            logger.warning(f"⚠️  LLM 모델 파일 없음: {e}")
            self._available = False
        except ImportError as e:
            logger.warning(f"⚠️  llama-cpp-python 미설치: {e}")
            self._available = False
        except Exception as e:
            logger.warning(f"⚠️  LLM 초기화 실패: {e}")
            self._available = False
        return self._client

    def is_available(self) -> bool:
        if self._gemini:
            return self._gemini.is_available()
        if self._available is not None:
            return self._available
        self._get_client()
        return self._available or False

    # ── public API ────────────────────────────────────────────────────────────
    def analyze_schema(self, df_info: dict, schema: SchemaInfo) -> str:
        if self._gemini:
            return self._gemini.analyze_schema(df_info, schema)
        client = self._get_client()
        if not client:
            return ""
        return self._safe_generate(client, self._prompt_schema(df_info, schema))

    def suggest_preprocessing(self, profile_summary: dict, schema: SchemaInfo) -> str:
        if self._gemini:
            return self._gemini.suggest_preprocessing(profile_summary, schema)
        client = self._get_client()
        if not client:
            return ""
        return self._safe_generate(client, self._prompt_preprocessing(profile_summary, schema))

    def interpret_insights(self, insights: list, schema: SchemaInfo, stats: dict) -> str:
        if self._gemini:
            return self._gemini.interpret_insights(insights, schema, stats)
        client = self._get_client()
        if not client:
            return ""
        return self._safe_generate(client, self._prompt_insights(insights, schema, stats))

    # ── prompt builders ───────────────────────────────────────────────────────
    def _prompt_schema(self, df_info: dict, schema: SchemaInfo) -> str:
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

    def _prompt_preprocessing(self, profile: dict, schema: SchemaInfo) -> str:
        return f"""데이터 전처리 전문가로서 다음 데이터 프로파일을 보고 전처리 권장사항을 제시하세요.

## 데이터 프로파일
- 스키마: {schema.schema_type}
- 결측치 많은 컬럼: {profile.get('high_missing', [])}
- 수치형 컬럼: {schema.numeric_cols[:8]}
- 범주형 컬럼: {schema.categorical_cols[:8]}
- 혼합 타입 컬럼: {schema.mixed_cols}
- 데이터 품질 점수: {profile.get('quality_score', 'N/A')}/100

구체적인 전처리 단계를 번호 목록으로 작성하세요.
각 항목: "컬럼명: 처리방법 (이유)" 형식."""

    def _prompt_insights(self, insights: list, schema: SchemaInfo, stats: dict) -> str:
        return f"""데이터 분석 전문가로서 다음 자동 분석 결과를 해석하고 인사이트를 도출하세요.

## 자동 감지된 특이사항
{chr(10).join(insights)}

## 주요 통계
{json.dumps(stats, ensure_ascii=False, indent=2)}

## 스키마 타입: {schema.schema_type}

다음을 포함해 한국어로 작성하세요:
1. 데이터 전반적 상태 평가
2. 주요 발견사항 3가지 이상
3. 추가 분석 권장사항
4. 데이터 품질 이슈 주의사항

간결하고 실용적으로 작성하세요."""

    # ── llama-cpp 전용 헬퍼 ───────────────────────────────────────────────────
    def _safe_generate(self, client, prompt: str) -> str:
        try:
            raw = client.loader.generate(
                prompt,
                max_tokens=300,
                temperature=0.2,
            )
            return self._dedup(raw)
        except Exception as e:
            return f"[LLM 오류: {e}]"

    @staticmethod
    def _dedup(text: str, max_chars: int = 600) -> str:
        if not text:
            return ""
        seen, unique = set(), []
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                continue
            key = s[:25]
            if key not in seen:
                seen.add(key)
                unique.append(s)
        result = "\n".join(unique)
        if len(result) > max_chars:
            result = result[:max_chars].rsplit("\n", 1)[0] + "\n..."
        return result
