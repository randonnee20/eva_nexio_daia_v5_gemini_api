"""
DAIA v2 - 스마트 전처리기
스키마 정보를 기반으로 지능형 전처리를 수행합니다.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
from core.schema_detector import SchemaInfo


class SmartPreprocessor:
    def __init__(self, config: dict = None):
        self.cfg = (config or {}).get("preprocessing", {})
        self.report: list[str] = []   # 수행한 작업 기록

    def run(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        self.report = []
        df = df.copy()

        # 1. 컬럼명 정리
        df = self._clean_column_names(df)

        # 2. 날짜 파싱
        df = self._parse_datetimes(df, schema)

        # 3. 혼합 value 컬럼 분리 (signal_pool)
        if schema.schema_type == "signal_pool" and schema.value_col:
            df, schema = self._split_mixed_value(df, schema)

        # 4. 결측치 처리
        df = self._handle_missing(df, schema)

        # 5. 이상치 처리
        df = self._handle_outliers(df, schema)

        # 6. datetime 피처 추출
        if self.cfg.get("datetime", {}).get("extract_features", True):
            df = self._extract_datetime_features(df, schema)

        return df

    # ─────────────────────────────────────────────────────────────
    def _clean_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        return df

    def _parse_datetimes(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        if not self.cfg.get("datetime", {}).get("auto_parse", True):
            return df

        ts_col = schema.timestamp_col
        if ts_col and ts_col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
                df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                self.report.append(f"날짜 파싱: {ts_col}")

        for col in schema.datetime_cols:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors="coerce")
                self.report.append(f"날짜 파싱: {col}")

        return df

    def _split_mixed_value(self, df: pd.DataFrame, schema: SchemaInfo):
        """value 컬럼 → value_num (숫자) + value_text (텍스트) 분리"""
        col = schema.value_col
        if col not in df.columns:
            return df, schema

        num_parsed = pd.to_numeric(df[col], errors="coerce")
        is_num = num_parsed.notna()
        is_txt = df[col].notna() & ~is_num

        num_col  = col + "_num"
        text_col = col + "_text"

        df[num_col]  = num_parsed
        df[text_col] = df[col].where(is_txt)

        schema.value_num_col  = num_col
        schema.value_text_col = text_col

        n_num = int(is_num.sum())
        n_txt = int(is_txt.sum())
        self.report.append(
            f"혼합 컬럼 분리: '{col}' → '{num_col}'({n_num:,}행) + '{text_col}'({n_txt:,}행)"
        )
        return df, schema

    def _handle_missing(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        miss_cfg = self.cfg.get("missing", {})
        drop_thresh = miss_cfg.get("drop_col_threshold", 0.95)

        # 거의 전부 결측인 컬럼 삭제
        before = len(df.columns)
        miss_ratio = df.isnull().mean()
        drop_cols = miss_ratio[miss_ratio >= drop_thresh].index.tolist()

        # signal_pool 핵심 컬럼은 삭제 금지
        protect = {schema.signal_name_col, schema.value_col,
                   schema.value_num_col, schema.value_text_col,
                   schema.timestamp_col} - {None}
        drop_cols = [c for c in drop_cols if c not in protect]

        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)
            self.report.append(f"결측 컬럼 삭제({drop_thresh:.0%} 이상): {drop_cols}")

        # 수치형 채우기
        fill_num = miss_cfg.get("fill_numeric", "median")
        fill_cat = miss_cfg.get("fill_categorical", "mode")

        for col in schema.numeric_cols:
            if col not in df.columns:
                continue
            if df[col].isnull().any():
                if fill_num == "median":
                    df[col].fillna(df[col].median(), inplace=True)
                elif fill_num == "mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                elif fill_num == "zero":
                    df[col].fillna(0, inplace=True)

        for col in schema.categorical_cols:
            if col not in df.columns:
                continue
            if df[col].isnull().any() and fill_cat == "mode":
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)

        return df

    def _handle_outliers(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        out_cfg = self.cfg.get("outlier", {})
        method  = out_cfg.get("method", "iqr")
        action  = out_cfg.get("action", "clip")

        if method == "none":
            return df

        # signal_pool에서는 value_num에만 적용
        if schema.schema_type == "signal_pool" and schema.value_num_col:
            target_cols = [schema.value_num_col]
        else:
            target_cols = [c for c in schema.numeric_cols if c in df.columns]

        for col in target_cols:
            s = df[col].dropna()
            if len(s) < 10:
                continue

            if method == "iqr":
                factor = out_cfg.get("iqr_factor", 2.5)
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - factor * iqr, q3 + factor * iqr
            else:  # zscore
                thr = out_cfg.get("zscore_threshold", 3.5)
                lo = float(s.mean() - thr * s.std())
                hi = float(s.mean() + thr * s.std())

            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_out == 0:
                continue

            if action == "clip":
                df[col] = df[col].clip(lower=lo, upper=hi)
                self.report.append(f"이상치 clip: {col} ({n_out:,}개)")
            elif action == "remove":
                mask = (df[col] >= lo) & (df[col] <= hi) | df[col].isna()
                df = df[mask].reset_index(drop=True)
                self.report.append(f"이상치 제거: {col} ({n_out:,}행)")
            elif action == "flag":
                df[col + "_outlier"] = (df[col] < lo) | (df[col] > hi)
                self.report.append(f"이상치 플래그: {col} ({n_out:,}개)")

        return df

    def _extract_datetime_features(self, df: pd.DataFrame,
                                   schema: SchemaInfo) -> pd.DataFrame:
        ts_col = schema.timestamp_col
        if ts_col and ts_col in df.columns:
            col = df[ts_col]
            if pd.api.types.is_datetime64_any_dtype(col):
                df[ts_col + "_year"]    = col.dt.year
                df[ts_col + "_month"]   = col.dt.month
                df[ts_col + "_day"]     = col.dt.day
                df[ts_col + "_hour"]    = col.dt.hour
                df[ts_col + "_weekday"] = col.dt.weekday
                self.report.append(f"날짜 피처 추출: {ts_col} → year/month/day/hour/weekday")
        return df
