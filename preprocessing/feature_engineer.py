"""
DAIA v4 - Feature Engineering
절차 8단계: 통계 feature / 시간 feature / 파생 변수 / 집계 변수
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from core.schema_detector import SchemaInfo


@dataclass
class FeatureReport:
    added_features: list[str] = field(default_factory=list)
    dropped_features: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)


class FeatureEngineer:
    def __init__(self, config: dict = None):
        self.cfg = (config or {}).get("preprocessing", {}).get(
            "feature_engineering", {})
        self.report = FeatureReport()

    def run(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        self.report = FeatureReport()
        if not self.cfg.get("enable_auto_features", True):
            return df
        df = df.copy()

        # 1. 시간 피처 (timestamp 컬럼)
        if self.cfg.get("date_features", True):
            df = self._time_features(df, schema)

        # 2. 통계 파생 변수 (wide_table: 수치형 컬럼간)
        if schema.schema_type == "wide_table":
            df = self._stat_features(df, schema)

        # 3. 비율 파생 변수
        df = self._ratio_features(df, schema)

        # 4. 집계 변수 (signal_pool: 신호별 통계)
        if schema.schema_type == "signal_pool" and schema.signal_name_col:
            df = self._signal_agg_features(df, schema)

        # 5. 시계열 lag 피처 (time_series)
        if schema.schema_type in ("time_series", "signal_pool") and schema.timestamp_col:
            df = self._lag_features(df, schema)

        return df

    # ─── 1. 시간 피처 ────────────────────────────────────────────────────────
    def _time_features(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        ts_col = schema.timestamp_col
        if not ts_col or ts_col not in df.columns:
            return df
        col = df[ts_col]
        if not pd.api.types.is_datetime64_any_dtype(col):
            return df

        added = []
        for feat, fn in [
            ("_year",    lambda c: c.dt.year),
            ("_month",   lambda c: c.dt.month),
            ("_day",     lambda c: c.dt.day),
            ("_hour",    lambda c: c.dt.hour),
            ("_weekday", lambda c: c.dt.weekday),
            ("_quarter", lambda c: c.dt.quarter),
            ("_is_weekend", lambda c: (c.dt.weekday >= 5).astype(int)),
        ]:
            new_col = ts_col + feat
            if new_col not in df.columns:
                try:
                    df[new_col] = fn(col)
                    added.append(new_col)
                except Exception:
                    pass

        if added:
            self.report.added_features.extend(added)
            self.report.log.append(f"⏰ 시간 피처 추출 ({len(added)}개): {', '.join(added[:4])}")
        return df

    # ─── 2. 통계 파생 변수 ──────────────────────────────────────────────────
    def _stat_features(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        num_cols = [c for c in schema.numeric_cols if c in df.columns]
        if len(num_cols) < 3:
            return df

        # 행 단위 통계
        num_df = df[num_cols[:15]].copy()
        added = []
        try:
            df["_row_mean"]  = num_df.mean(axis=1)
            df["_row_std"]   = num_df.std(axis=1)
            df["_row_max"]   = num_df.max(axis=1)
            df["_row_min"]   = num_df.min(axis=1)
            df["_row_range"] = df["_row_max"] - df["_row_min"]
            added = ["_row_mean", "_row_std", "_row_max", "_row_min", "_row_range"]
            self.report.added_features.extend(added)
            self.report.log.append(f"📊 행 단위 통계 피처 {len(added)}개 추가 (mean/std/max/min/range)")
        except Exception:
            pass
        return df

    # ─── 3. 비율 피처 ────────────────────────────────────────────────────────
    def _ratio_features(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        num_cols = [c for c in schema.numeric_cols if c in df.columns]
        added = []
        # 의미있는 비율 쌍 탐지 (amount/price 관련)
        ratio_hints = [
            ("amount", "price"), ("revenue", "cost"), ("success", "total"),
            ("pass", "total"), ("count", "total"),
        ]
        for a_hint, b_hint in ratio_hints:
            a_cols = [c for c in num_cols if a_hint in c.lower()]
            b_cols = [c for c in num_cols if b_hint in c.lower()]
            for a in a_cols[:2]:
                for b in b_cols[:2]:
                    if a == b:
                        continue
                    new_col = f"_ratio_{a}_{b}"
                    if new_col not in df.columns:
                        denom = df[b].replace(0, np.nan)
                        df[new_col] = df[a] / denom
                        added.append(new_col)
        if added:
            self.report.added_features.extend(added)
            self.report.log.append(f"📐 비율 파생 피처 {len(added)}개 추가")
        return df

    # ─── 4. 신호별 집계 피처 (signal_pool) ──────────────────────────────────
    def _signal_agg_features(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        sig_col = schema.signal_name_col
        val_col = schema.value_num_col or schema.value_col
        if not val_col or val_col not in df.columns:
            return df
        try:
            agg = df.groupby(sig_col)[val_col].agg(
                sig_mean="mean", sig_std="std",
                sig_min="min",  sig_max="max",
                sig_count="count"
            ).round(4)
            df = df.merge(agg, on=sig_col, how="left")
            added = list(agg.columns)
            self.report.added_features.extend(added)
            self.report.log.append(f"📡 신호별 집계 피처 {len(added)}개 추가 (mean/std/min/max/count)")
        except Exception:
            pass
        return df

    # ─── 5. Lag 피처 (time_series) ───────────────────────────────────────────
    def _lag_features(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        ts_col = schema.timestamp_col
        val_col = schema.value_num_col or (schema.numeric_cols[0] if schema.numeric_cols else None)
        if not val_col or val_col not in df.columns:
            return df
        if not pd.api.types.is_datetime64_any_dtype(df.get(ts_col, pd.Series())):
            return df
        try:
            df_s = df.sort_values(ts_col)
            added = []
            for lag in [1, 2, 3, 7]:
                new_col = f"{val_col}_lag{lag}"
                df_s[new_col] = df_s[val_col].shift(lag)
                added.append(new_col)
            # rolling statistics
            df_s[f"{val_col}_roll_mean_7"] = df_s[val_col].rolling(7, min_periods=1).mean()
            df_s[f"{val_col}_roll_std_7"]  = df_s[val_col].rolling(7, min_periods=1).std()
            added += [f"{val_col}_roll_mean_7", f"{val_col}_roll_std_7"]
            self.report.added_features.extend(added)
            self.report.log.append(f"⏳ Lag/Rolling 피처 {len(added)}개 추가 (lag1~7, roll_mean/std_7)")
            return df_s
        except Exception:
            return df

    # ─── 분석 추천 생성 ──────────────────────────────────────────────────────
    def build_analysis_proposals(self, df: pd.DataFrame, schema: SchemaInfo,
                                  quality_report=None) -> list[dict]:
        """데이터 구조 기반 분석 방향 제안"""
        proposals = []
        num_cols = schema.numeric_cols
        cat_cols = schema.categorical_cols
        ts_col   = schema.timestamp_col
        has_target = self._detect_target(df, schema)

        st = schema.schema_type

        # 예측 분석
        if num_cols and len(df) > 50:
            proposals.append({
                "type": "예측 분석 (Regression)",
                "icon": "📈",
                "desc": f"수치형 컬럼 {len(num_cols)}개를 이용한 회귀 예측 모델",
                "algorithm": "Linear Regression, Random Forest Regressor, XGBoost",
                "target": f"추천 타겟: {num_cols[0]}",
                "features": f"입력 변수: {', '.join(num_cols[1:6])}",
            })

        # 분류
        if has_target:
            proposals.append({
                "type": "분류 분석 (Classification)",
                "icon": "🎯",
                "desc": f"타겟 변수 '{has_target}' 예측 분류 모델",
                "algorithm": "Logistic Regression, Random Forest, XGBoost, LightGBM",
                "target": f"타겟: {has_target}",
                "features": f"입력 변수: {', '.join(num_cols[:6])}",
            })

        # 군집
        if len(num_cols) >= 2 and len(df) >= 30:
            proposals.append({
                "type": "군집 분석 (Clustering)",
                "icon": "🔵",
                "desc": "비지도 학습으로 데이터 패턴 및 그룹 발견",
                "algorithm": "K-Means, DBSCAN, Hierarchical Clustering",
                "target": "비지도 (라벨 없음)",
                "features": f"사용 변수: {', '.join(num_cols[:8])}",
            })

        # 시계열 예측
        if ts_col and num_cols:
            proposals.append({
                "type": "시계열 예측 (Time Series Forecast)",
                "icon": "⏱",
                "desc": "시간 흐름에 따른 미래 값 예측",
                "algorithm": "ARIMA, Prophet, LSTM, Transformer",
                "target": f"예측 대상: {num_cols[0]}",
                "features": f"시간축: {ts_col} / lag 피처 활용",
            })

        # 이상 탐지
        if len(num_cols) >= 2:
            proposals.append({
                "type": "이상 탐지 (Anomaly Detection)",
                "icon": "🔴",
                "desc": "정상 패턴에서 벗어난 이상 데이터 탐지",
                "algorithm": "Isolation Forest, LOF, Autoencoder, One-Class SVM",
                "target": "비지도",
                "features": f"사용 변수: {', '.join(num_cols[:8])}",
            })

        # 연관 분석 (categorical 많을 때)
        if len(cat_cols) >= 3 and st == "wide_table":
            proposals.append({
                "type": "연관 분석 (Association Analysis)",
                "icon": "🔗",
                "desc": "범주형 변수 간 연관 규칙 발견",
                "algorithm": "Apriori, FP-Growth",
                "target": "비지도",
                "features": f"범주형 변수: {', '.join(cat_cols[:6])}",
            })

        # 인과 분석
        if len(num_cols) >= 3:
            proposals.append({
                "type": "인과 분석 (Causal Inference)",
                "icon": "⚗️",
                "desc": "변수 간 인과관계 탐색",
                "algorithm": "DoWhy, CausalML, Granger Causality",
                "target": f"타겟: {num_cols[0]}",
                "features": f"원인 후보: {', '.join(num_cols[1:5])}",
            })

        # 이탈 예측 (고객 데이터 패턴)
        if any(k in ' '.join(df.columns).lower() for k in
               ["customer", "user", "client", "고객", "churn", "active"]):
            proposals.append({
                "type": "이탈 예측 (Churn Prediction)",
                "icon": "🚨",
                "desc": "고객/사용자 이탈 가능성 예측",
                "algorithm": "Logistic Regression, Gradient Boosting, Survival Analysis",
                "target": "churn 여부 (이진 분류)",
                "features": f"활동 지표, 기간, 빈도 변수 활용",
            })

        return proposals

    def _detect_target(self, df, schema):
        for col in df.columns:
            clean = col.lower().replace("_", "").replace(" ", "")
            if clean in ["passfail","passorfail","label","target","result","outcome","churn","judge"]:
                return col
        for col in schema.categorical_cols:
            if col in df.columns and 2 <= df[col].nunique() <= 5:
                return col
        return None
