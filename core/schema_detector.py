"""
DAIA v2 - 스키마 감지기
데이터 구조를 자동으로 파악합니다.

지원 스키마:
  signal_pool  : signalname + value + timestamp (narrow/long format, 센서·신호 데이터)
  time_series  : timestamp + 복수 value 컬럼
  wide_table   : 일반 ML 테이블 (행=샘플, 열=피처)
  cross_tab    : 피벗/크로스탭 형태
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class SchemaInfo:
    schema_type: str                        # signal_pool | time_series | wide_table | cross_tab
    confidence: float                       # 0~1

    # signal_pool 전용
    signal_name_col: Optional[str] = None
    value_col: Optional[str] = None

    # time_series / signal_pool 공통
    timestamp_col: Optional[str] = None
    value_cols: list = field(default_factory=list)   # time_series일 때 값 컬럼 목록

    # 파생 컬럼 (mixed value 분리)
    value_num_col: Optional[str] = None    # 숫자 파생
    value_text_col: Optional[str] = None  # 텍스트 파생

    # 일반
    id_cols: list = field(default_factory=list)
    numeric_cols: list = field(default_factory=list)
    categorical_cols: list = field(default_factory=list)
    datetime_cols: list = field(default_factory=list)
    text_cols: list = field(default_factory=list)
    mixed_cols: list = field(default_factory=list)

    # 계층 구조 (signal_pool)
    hierarchy: dict = field(default_factory=dict)   # device → path → [signals]

    def summary(self) -> str:
        lines = [
            f"스키마 타입  : {self.schema_type} (신뢰도 {self.confidence:.0%})",
        ]
        if self.signal_name_col:
            lines.append(f"신호명 컬럼  : {self.signal_name_col}")
        if self.value_col:
            lines.append(f"값 컬럼      : {self.value_col}")
        if self.timestamp_col:
            lines.append(f"타임스탬프   : {self.timestamp_col}")
        if self.value_cols:
            lines.append(f"값 컬럼들    : {self.value_cols}")
        lines += [
            f"수치형       : {self.numeric_cols}",
            f"범주형       : {self.categorical_cols}",
            f"날짜/시간    : {self.datetime_cols}",
            f"혼합(숫+텍)  : {self.mixed_cols}",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
_SIG_HINTS  = ["signalname","signal_name","tag_name","tagname","sensor_name","parameter","item"]
_VAL_HINTS  = ["value","val","measurement","reading","data"]
_TS_HINTS   = ["updatedate","timestamp","datetime","date","time","created_at",
                "updated_at","event_time","log_time","recordtime"]


class SchemaDetector:
    def __init__(self, config: dict = None):
        self.cfg = (config or {}).get("schema", {})

    # ── public ───────────────────────────────────────────────────────────────
    def detect(self, df: pd.DataFrame) -> SchemaInfo:
        forced = self.cfg.get("force_schema")
        if forced:
            return self._build_forced(df, forced)

        scores = {
            "signal_pool" : self._score_signal_pool(df),
            "time_series" : self._score_time_series(df),
            "wide_table"  : self._score_wide_table(df),
            "cross_tab"   : self._score_cross_tab(df),
        }
        best = max(scores, key=scores.get)
        conf = scores[best]

        if best == "signal_pool":
            return self._build_signal_pool(df, conf)
        elif best == "time_series":
            return self._build_time_series(df, conf)
        elif best == "cross_tab":
            return self._build_cross_tab(df, conf)
        else:
            return self._build_wide_table(df, conf)

    # ── scoring ──────────────────────────────────────────────────────────────
    def _score_signal_pool(self, df: pd.DataFrame) -> float:
        cols_lower = {c.lower(): c for c in df.columns}
        sig_col  = self._find_col(cols_lower, _SIG_HINTS)
        val_col  = self._find_col(cols_lower, _VAL_HINTS)
        ts_col   = self._find_col(cols_lower, _TS_HINTS)

        score = 0.0
        if sig_col:  score += 0.4
        if val_col:  score += 0.3
        if ts_col:   score += 0.2

        # value 컬럼이 혼합 타입인지 확인
        if val_col and df[val_col].dtype == object:
            num_ratio = pd.to_numeric(df[val_col], errors="coerce").notna().mean()
            if 0.1 < num_ratio < 0.95:
                score += 0.1   # 진짜 혼합

        # 행 수 >> 컬럼 수 → narrow format 가능성
        if len(df) > len(df.columns) * 50:
            score += 0.05

        return min(score, 1.0)

    def _score_time_series(self, df: pd.DataFrame) -> float:
        cols_lower = {c.lower(): c for c in df.columns}
        ts_col = self._find_col(cols_lower, _TS_HINTS)
        if not ts_col:
            return 0.0

        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols) < 1:
            return 0.1

        score = 0.3
        if ts_col:
            score += 0.4
        if len(num_cols) >= 1:
            score += 0.2
        # 시그널 컬럼 없으면 signal_pool보다 time_series 가능성 높음
        sig_col = self._find_col(cols_lower, _SIG_HINTS)
        if not sig_col:
            score += 0.1
        return min(score, 1.0)

    def _score_wide_table(self, df: pd.DataFrame) -> float:
        # 일반 wide table : 컬럼 수 많고, 각 컬럼이 독립적 의미를 가짐
        num_cols = df.select_dtypes(include=[np.number]).columns
        score = 0.3
        if len(num_cols) > 3:
            score += 0.3
        if len(df.columns) >= 5:
            score += 0.2
        # 행 수 대비 컬럼 수 비율
        if len(df.columns) > 10:
            score += 0.1
        return min(score, 1.0)

    def _score_cross_tab(self, df: pd.DataFrame) -> float:
        # 행/열 인덱스가 모두 범주형인 경우
        obj_ratio = len(df.select_dtypes(include="object").columns) / max(len(df.columns), 1)
        if obj_ratio > 0.5 and len(df) < 200:
            return 0.5
        return 0.1

    # ── builders ─────────────────────────────────────────────────────────────
    def _build_signal_pool(self, df: pd.DataFrame, conf: float) -> SchemaInfo:
        cols_lower = {c.lower(): c for c in df.columns}
        sig_col = self._find_col(cols_lower, _SIG_HINTS)
        val_col = self._find_col(cols_lower, _VAL_HINTS)
        ts_col  = self._find_col(cols_lower, _TS_HINTS)

        # 나머지 컬럼 분류
        known = {c for c in [sig_col, val_col, ts_col] if c}
        rest = [c for c in df.columns if c not in known]

        id_cols, num_cols, cat_cols, dt_cols, txt_cols, mix_cols = \
            self._classify_columns(df, rest)

        info = SchemaInfo(
            schema_type="signal_pool",
            confidence=conf,
            signal_name_col=sig_col,
            value_col=val_col,
            timestamp_col=ts_col,
            id_cols=id_cols,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            datetime_cols=dt_cols,
            text_cols=txt_cols,
            mixed_cols=mix_cols,
        )
        return info

    def _build_time_series(self, df: pd.DataFrame, conf: float) -> SchemaInfo:
        cols_lower = {c.lower(): c for c in df.columns}
        ts_col   = self._find_col(cols_lower, _TS_HINTS)
        num_cols_all = list(df.select_dtypes(include=[np.number]).columns)
        val_cols = [c for c in num_cols_all if c != ts_col]

        rest = [c for c in df.columns if c not in [ts_col] + val_cols]
        id_cols, num_cols, cat_cols, dt_cols, txt_cols, mix_cols = \
            self._classify_columns(df, rest)

        return SchemaInfo(
            schema_type="time_series",
            confidence=conf,
            timestamp_col=ts_col,
            value_cols=val_cols,
            id_cols=id_cols,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            datetime_cols=dt_cols,
            text_cols=txt_cols,
            mixed_cols=mix_cols,
        )

    def _build_wide_table(self, df: pd.DataFrame, conf: float) -> SchemaInfo:
        id_cols, num_cols, cat_cols, dt_cols, txt_cols, mix_cols = \
            self._classify_columns(df, list(df.columns))
        return SchemaInfo(
            schema_type="wide_table",
            confidence=conf,
            id_cols=id_cols,
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            datetime_cols=dt_cols,
            text_cols=txt_cols,
            mixed_cols=mix_cols,
        )

    def _build_cross_tab(self, df: pd.DataFrame, conf: float) -> SchemaInfo:
        return self._build_wide_table(df, conf)

    def _build_forced(self, df: pd.DataFrame, schema_type: str) -> SchemaInfo:
        builders = {
            "signal_pool" : lambda: self._build_signal_pool(df, 1.0),
            "time_series" : lambda: self._build_time_series(df, 1.0),
            "wide_table"  : lambda: self._build_wide_table(df, 1.0),
        }
        return builders.get(schema_type, lambda: self._build_wide_table(df, 1.0))()

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _find_col(cols_lower: dict, hints: list) -> Optional[str]:
        for h in hints:
            if h in cols_lower:
                return cols_lower[h]
        # 부분 매칭
        for h in hints:
            for k, v in cols_lower.items():
                if h in k:
                    return v
        return None

    @staticmethod
    def _classify_columns(df: pd.DataFrame, cols: list):
        id_cols, num_cols, cat_cols, dt_cols, txt_cols, mix_cols = [], [], [], [], [], []

        for col in cols:
            s = df[col]
            # Datetime
            if pd.api.types.is_datetime64_any_dtype(s):
                dt_cols.append(col)
                continue

            # Numeric
            if pd.api.types.is_numeric_dtype(s):
                uniq_r = s.nunique() / max(len(s), 1)
                if ("id" in col.lower() or "_id" in col.lower()) and uniq_r > 0.8:
                    id_cols.append(col)
                elif s.nunique() <= 20 and uniq_r < 0.05:
                    cat_cols.append(col)
                else:
                    num_cols.append(col)
                continue

            # Object 계열
            if s.dtype == object:
                # 날짜 문자열 시도
                try:
                    parsed = pd.to_datetime(s.dropna().head(100), errors="coerce")
                    if parsed.notna().mean() > 0.8:
                        dt_cols.append(col)
                        continue
                except:
                    pass

                # 혼합(숫자+텍스트)
                num_ratio = pd.to_numeric(s, errors="coerce").notna().mean()
                non_null_ratio = s.notna().mean()
                if non_null_ratio > 0.05 and 0.05 < num_ratio < 0.95:
                    mix_cols.append(col)
                    continue

                # 고유값 비율로 범주/텍스트 구분
                uniq = s.nunique()
                if uniq <= 50 or uniq / max(len(s), 1) < 0.05:
                    cat_cols.append(col)
                else:
                    txt_cols.append(col)

        return id_cols, num_cols, cat_cols, dt_cols, txt_cols, mix_cols
