"""
DAIA v5 - 자동 분석 실행 엔진 (AutoRunner)
지원 분석: 회귀 / 분류 / 군집 / 이상탐지 / 시계열 예측

## 실행 흐름
  DAIAPipeline.run_analysis() → AutoRunner.run(analysis_type, df, schema)
      → 결과: AutoResult(metrics, charts, summary, recommendations)
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Optional
from core.schema_detector import SchemaInfo

_COLORS = px.colors.qualitative.Plotly


# ─── 결과 컨테이너 ─────────────────────────────────────────────────────────────
@dataclass
class AutoResult:
    analysis_type: str                          # regression / classification / clustering / anomaly / timeseries
    success: bool = True
    error_msg: str = ""

    # 성능 지표
    metrics: dict = field(default_factory=dict)

    # 시각화 (plotly Figure)
    figures: dict[str, go.Figure] = field(default_factory=dict)

    # 텍스트 요약
    summary: str = ""
    feature_importances: dict = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    model_name: str = ""
    target_col: str = ""
    feature_cols: list[str] = field(default_factory=list)


# ─── 메인 실행기 ───────────────────────────────────────────────────────────────
class AutoRunner:
    """분석 유형에 따라 적절한 서브러너를 호출"""

    def __init__(self, config: dict = None):
        self.cfg = config or {}

    def run(self, analysis_type: str, df: pd.DataFrame,
            schema: SchemaInfo, target_col: str = None) -> AutoResult:
        runners = {
            "regression":      self._run_regression,
            "classification":  self._run_classification,
            "clustering":      self._run_clustering,
            "anomaly":         self._run_anomaly,
            "timeseries":      self._run_timeseries,
            "association":     self._run_association,
        }
        fn = runners.get(analysis_type)
        if fn is None:
            return AutoResult(analysis_type, success=False,
                              error_msg=f"지원하지 않는 분석 유형: {analysis_type}")
        try:
            return fn(df, schema, target_col)
        except Exception as e:
            import traceback
            return AutoResult(analysis_type, success=False,
                              error_msg=f"{e}\n{traceback.format_exc()}")

    # ══════════════════════════════════════════════════════════════════════════
    # 1. 회귀 분석
    # ══════════════════════════════════════════════════════════════════════════
    def _run_regression(self, df: pd.DataFrame, schema: SchemaInfo,
                        target_col: str = None) -> AutoResult:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        num_cols = [c for c in schema.numeric_cols if c in df.columns]
        if not target_col:
            target_col = num_cols[0] if num_cols else None
        if not target_col or target_col not in df.columns:
            return AutoResult("regression", success=False, error_msg="타겟 컬럼 없음")

        feat_cols = [c for c in num_cols if c != target_col][:20]
        if len(feat_cols) < 1:
            return AutoResult("regression", success=False, error_msg="입력 변수 부족")

        data = df[feat_cols + [target_col]].dropna()
        if len(data) < 20:
            return AutoResult("regression", success=False, error_msg=f"데이터 부족 ({len(data)}행)")

        X = data[feat_cols].values
        y = data[target_col].values
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 비교
        models = {
            "Linear Regression":    LinearRegression(),
            "Ridge Regression":     Ridge(alpha=1.0),
            "Random Forest":        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting":    GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        results = {}
        best_model, best_r2, best_name = None, -np.inf, ""
        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                r2   = r2_score(y_te, pred)
                rmse = np.sqrt(mean_squared_error(y_te, pred))
                mae  = mean_absolute_error(y_te, pred)
                results[name] = {"R²": round(r2,4), "RMSE": round(rmse,4), "MAE": round(mae,4)}
                if r2 > best_r2:
                    best_r2, best_model, best_name = r2, model, name
            except Exception:
                pass

        pred_best = best_model.predict(X_te)

        # 피처 중요도
        fi = {}
        if hasattr(best_model, "feature_importances_"):
            fi = dict(zip(feat_cols, best_model.feature_importances_.round(4)))
        elif hasattr(best_model, "coef_"):
            fi = dict(zip(feat_cols, np.abs(best_model.coef_).round(4)))

        figs = {}
        figs["model_comparison"]  = _fig_model_comparison(results, "R²", "회귀 — 모델 성능 비교 (R²)")
        figs["actual_vs_pred"]    = _fig_actual_vs_pred(y_te, pred_best, best_name, target_col)
        figs["residuals"]         = _fig_residuals(y_te, pred_best, best_name)
        figs["feature_importance"]= _fig_feature_importance(fi, f"피처 중요도 ({best_name})")

        metrics = results.get(best_name, {})
        summary = (
            f"**최적 모델:** {best_name}  \n"
            f"**R² (결정계수):** {metrics.get('R²','N/A')} — "
            f"{'높음 (잘 맞음)' if metrics.get('R²',0)>0.8 else '보통' if metrics.get('R²',0)>0.5 else '낮음 (개선 필요)'}  \n"
            f"**RMSE:** {metrics.get('RMSE','N/A')}  \n"
            f"**MAE:** {metrics.get('MAE','N/A')}  \n"
            f"**학습 샘플:** {len(X_tr):,}개 / **테스트 샘플:** {len(X_te):,}개"
        )
        recs = _regression_recommendations(metrics, fi)

        return AutoResult(
            analysis_type="regression",
            metrics=metrics, figures=figs,
            summary=summary, feature_importances=fi,
            recommendations=recs, model_name=best_name,
            target_col=target_col, feature_cols=feat_cols,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. 분류 분석
    # ══════════════════════════════════════════════════════════════════════════
    def _run_classification(self, df: pd.DataFrame, schema: SchemaInfo,
                             target_col: str = None) -> AutoResult:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import (accuracy_score, f1_score,
                                     classification_report, confusion_matrix)

        # 타겟 자동 감지
        if not target_col:
            target_col = _detect_target(df, schema)
        if not target_col or target_col not in df.columns:
            return AutoResult("classification", success=False, error_msg="타겟 컬럼 없음")

        num_cols = [c for c in schema.numeric_cols if c in df.columns and c != target_col][:20]
        if len(num_cols) < 1:
            return AutoResult("classification", success=False, error_msg="입력 변수 부족")

        data = df[num_cols + [target_col]].dropna()
        if len(data) < 20:
            return AutoResult("classification", success=False, error_msg=f"데이터 부족 ({len(data)}행)")

        le = LabelEncoder()
        y  = le.fit_transform(data[target_col].astype(str))
        X  = data[num_cols].values
        n_cls = len(le.classes_)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)
        avg = "binary" if n_cls == 2 else "weighted"

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
        }
        results = {}
        best_model, best_f1, best_name = None, -np.inf, ""
        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                acc  = accuracy_score(y_te, pred)
                f1   = f1_score(y_te, pred, average=avg, zero_division=0)
                results[name] = {"Accuracy": round(acc,4), "F1-Score": round(f1,4)}
                if f1 > best_f1:
                    best_f1, best_model, best_name = f1, model, name
            except Exception:
                pass

        pred_best = best_model.predict(X_te)
        cm        = confusion_matrix(y_te, pred_best)
        fi = {}
        if hasattr(best_model, "feature_importances_"):
            fi = dict(zip(num_cols, best_model.feature_importances_.round(4)))
        elif hasattr(best_model, "coef_"):
            fi = dict(zip(num_cols, np.abs(best_model.coef_[0]).round(4)))

        figs = {}
        figs["model_comparison"]   = _fig_model_comparison(results, "F1-Score", "분류 — 모델 성능 비교 (F1-Score)")
        figs["confusion_matrix"]   = _fig_confusion_matrix(cm, le.classes_, best_name)
        figs["feature_importance"] = _fig_feature_importance(fi, f"피처 중요도 ({best_name})")
        figs["class_distribution"] = _fig_class_dist(y_te, pred_best, le.classes_)

        metrics = results.get(best_name, {})
        summary = (
            f"**최적 모델:** {best_name}  \n"
            f"**Accuracy:** {metrics.get('Accuracy','N/A')}  \n"
            f"**F1-Score:** {metrics.get('F1-Score','N/A')}  \n"
            f"**클래스 수:** {n_cls}개 ({', '.join(le.classes_[:5])})  \n"
            f"**학습 샘플:** {len(X_tr):,}개 / **테스트 샘플:** {len(X_te):,}개"
        )
        recs = _classification_recommendations(metrics, n_cls, y)

        return AutoResult(
            analysis_type="classification",
            metrics=metrics, figures=figs,
            summary=summary, feature_importances=fi,
            recommendations=recs, model_name=best_name,
            target_col=target_col, feature_cols=num_cols,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. 군집 분석
    # ══════════════════════════════════════════════════════════════════════════
    def _run_clustering(self, df: pd.DataFrame, schema: SchemaInfo,
                        target_col: str = None) -> AutoResult:
        from sklearn.cluster import KMeans, DBSCAN
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from sklearn.metrics import silhouette_score, davies_bouldin_score

        num_cols = [c for c in schema.numeric_cols if c in df.columns][:20]
        if len(num_cols) < 2:
            return AutoResult("clustering", success=False, error_msg="수치형 컬럼 부족")

        data = df[num_cols].dropna()
        if len(data) < 20:
            return AutoResult("clustering", success=False, error_msg=f"데이터 부족 ({len(data)}행)")

        X_s = StandardScaler().fit_transform(data.values)

        # Elbow + Silhouette 로 최적 K 탐색 (2~8)
        k_range  = range(2, min(9, len(data)//5 + 2))
        inertias, sils = [], []
        for k in k_range:
            km  = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_s)
            inertias.append(km.inertia_)
            sils.append(silhouette_score(X_s, lbl))

        best_k   = list(k_range)[int(np.argmax(sils))]
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels   = km_final.fit_predict(X_s)
        sil_best = silhouette_score(X_s, labels)
        db_best  = davies_bouldin_score(X_s, labels)

        # PCA 2D 시각화
        pca    = PCA(n_components=min(2, X_s.shape[1]), random_state=42)
        coords = pca.fit_transform(X_s)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])
        evr = pca.explained_variance_ratio_ * 100

        # 군집별 프로파일
        data_with_cluster = data.copy()
        data_with_cluster["_cluster"] = labels
        cluster_profile = data_with_cluster.groupby("_cluster")[num_cols[:8]].mean().round(3)

        figs = {}
        figs["elbow_silhouette"]  = _fig_elbow_sil(list(k_range), inertias, sils, best_k)
        figs["cluster_scatter"]   = _fig_cluster_scatter(coords, labels, best_k, evr)
        figs["cluster_profile"]   = _fig_cluster_profile(cluster_profile)
        figs["cluster_size"]      = _fig_cluster_size(labels, best_k)

        metrics = {
            "최적 K":         best_k,
            "Silhouette":    round(sil_best, 4),
            "Davies-Bouldin": round(db_best, 4),
            "총 샘플":        len(data),
        }
        summary = (
            f"**최적 군집 수:** {best_k}개 (Silhouette 최대화 기준)  \n"
            f"**Silhouette Score:** {sil_best:.4f} — "
            f"{'높음 (잘 분리됨)' if sil_best>0.5 else '보통' if sil_best>0.25 else '낮음 (겹침 많음)'}  \n"
            f"**Davies-Bouldin:** {db_best:.4f} — 낮을수록 좋음  \n"
            f"**사용 변수:** {', '.join(num_cols[:6])}{'...' if len(num_cols)>6 else ''}"
        )
        recs = _clustering_recommendations(sil_best, best_k, cluster_profile)

        return AutoResult(
            analysis_type="clustering",
            metrics=metrics, figures=figs,
            summary=summary,
            recommendations=recs, model_name=f"K-Means (K={best_k})",
            feature_cols=num_cols,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 4. 이상 탐지
    # ══════════════════════════════════════════════════════════════════════════
    def _run_anomaly(self, df: pd.DataFrame, schema: SchemaInfo,
                     target_col: str = None) -> AutoResult:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        num_cols = [c for c in schema.numeric_cols if c in df.columns][:20]
        if len(num_cols) < 1:
            return AutoResult("anomaly", success=False, error_msg="수치형 컬럼 없음")

        data = df[num_cols].dropna()
        if len(data) < 20:
            return AutoResult("anomaly", success=False, error_msg=f"데이터 부족 ({len(data)}행)")

        X_s  = StandardScaler().fit_transform(data.values)

        # contamination 자동 추정 (IQR 기반 이상치 비율)
        iqr_outlier_ratio = 0.0
        for col in num_cols[:5]:
            s = data[col]
            q1,q3 = s.quantile(0.25), s.quantile(0.75)
            iqr   = q3 - q1
            iqr_outlier_ratio += ((s < q1-1.5*iqr) | (s > q3+1.5*iqr)).mean()
        contamination = float(np.clip(iqr_outlier_ratio / len(num_cols[:5]), 0.01, 0.3))

        iso   = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        preds = iso.fit_predict(X_s)            # -1=이상, 1=정상
        scores= iso.score_samples(X_s)          # 낮을수록 이상

        is_anom    = preds == -1
        n_anom     = int(is_anom.sum())
        anom_ratio = n_anom / len(data) * 100

        # PCA 2D
        pca    = PCA(n_components=min(2, X_s.shape[1]), random_state=42)
        coords = pca.fit_transform(X_s)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])
        evr = pca.explained_variance_ratio_ * 100

        # 이상값 샘플의 컬럼별 특성
        anom_profile = data[is_anom][num_cols[:8]].describe().T[["mean","std","min","max"]].round(3)

        figs = {}
        figs["anomaly_scatter"]  = _fig_anomaly_scatter(coords, is_anom, evr)
        figs["anomaly_scores"]   = _fig_anomaly_scores(scores, is_anom)
        figs["anomaly_profile"]  = _fig_anomaly_profile(data, is_anom, num_cols[:6])

        metrics = {
            "이상 샘플 수":   n_anom,
            "이상 비율(%)":  round(anom_ratio, 2),
            "정상 샘플 수":  int((~is_anom).sum()),
            "Contamination": round(contamination, 3),
        }
        summary = (
            f"**이상 탐지 모델:** Isolation Forest  \n"
            f"**이상 샘플:** {n_anom:,}개 ({anom_ratio:.1f}%)  \n"
            f"**정상 샘플:** {int((~is_anom).sum()):,}개  \n"
            f"**자동 추정 오염률:** {contamination:.1%}  \n"
            f"**사용 변수:** {', '.join(num_cols[:6])}{'...' if len(num_cols)>6 else ''}"
        )
        recs = _anomaly_recommendations(anom_ratio, n_anom, num_cols)

        # 이상 인덱스를 메트릭에 포함
        metrics["anomaly_indices"] = data[is_anom].index.tolist()[:100]

        return AutoResult(
            analysis_type="anomaly",
            metrics=metrics, figures=figs,
            summary=summary,
            recommendations=recs, model_name="Isolation Forest",
            feature_cols=num_cols,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # 5. 시계열 예측
    # ══════════════════════════════════════════════════════════════════════════
    def _run_timeseries(self, df: pd.DataFrame, schema: SchemaInfo,
                        target_col: str = None) -> AutoResult:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        ts_col  = schema.timestamp_col
        if not ts_col or ts_col not in df.columns:
            return AutoResult("timeseries", success=False, error_msg="타임스탬프 컬럼 없음")

        num_cols = [c for c in schema.numeric_cols if c in df.columns]
        if not target_col:
            target_col = num_cols[0] if num_cols else None
        if not target_col:
            return AutoResult("timeseries", success=False, error_msg="타겟 컬럼 없음")

        data = df[[ts_col, target_col]].dropna().sort_values(ts_col).reset_index(drop=True)
        if len(data) < 30:
            return AutoResult("timeseries", success=False, error_msg=f"데이터 부족 ({len(data)}행)")

        # Lag 피처 생성
        LAG_STEPS = [1, 2, 3, 7, 14]
        ROLL_WINS = [7, 14]
        y_ser = data[target_col].copy()
        feat_df = pd.DataFrame(index=data.index)
        feat_names = []
        for lag in LAG_STEPS:
            col = f"lag_{lag}"; feat_df[col] = y_ser.shift(lag); feat_names.append(col)
        for win in ROLL_WINS:
            col = f"roll_mean_{win}"; feat_df[col] = y_ser.shift(1).rolling(win).mean(); feat_names.append(col)
            col = f"roll_std_{win}";  feat_df[col] = y_ser.shift(1).rolling(win).std();  feat_names.append(col)
        # 시간 피처
        if pd.api.types.is_datetime64_any_dtype(data[ts_col]):
            feat_df["dayofweek"] = data[ts_col].dt.dayofweek
            feat_df["month"]     = data[ts_col].dt.month
            feat_names += ["dayofweek", "month"]

        feat_df[target_col] = y_ser
        feat_df = feat_df.dropna()
        if len(feat_df) < 20:
            return AutoResult("timeseries", success=False, error_msg="lag 생성 후 데이터 부족")

        X = feat_df[feat_names].values
        y = feat_df[target_col].values
        dates = data[ts_col].iloc[feat_df.index].values

        # 시계열 분할 (마지막 20% = 테스트)
        split   = int(len(X) * 0.8)
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]
        d_tr, d_te = dates[:split], dates[split:]

        models = {
            "Linear Regression":  LinearRegression(),
            "Gradient Boosting":  GradientBoostingRegressor(n_estimators=100, random_state=42),
        }
        results = {}
        best_model, best_rmse, best_name = None, np.inf, ""
        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                pred = model.predict(X_te)
                rmse = np.sqrt(mean_squared_error(y_te, pred))
                mae  = mean_absolute_error(y_te, pred)
                r2   = r2_score(y_te, pred)
                results[name] = {"RMSE": round(rmse,4), "MAE": round(mae,4), "R²": round(r2,4)}
                if rmse < best_rmse:
                    best_rmse, best_model, best_name = rmse, model, name
            except Exception:
                pass

        pred_best = best_model.predict(X_te)
        fi = {}
        if hasattr(best_model, "feature_importances_"):
            fi = dict(zip(feat_names, best_model.feature_importances_.round(4)))
        elif hasattr(best_model, "coef_"):
            fi = dict(zip(feat_names, np.abs(best_model.coef_).round(4)))

        # 미래 7스텝 예측 (간단 extrapolation)
        future_pred = _forecast_future(best_model, X_te[-1], feat_names, steps=7)

        figs = {}
        figs["model_comparison"]  = _fig_model_comparison(results, "RMSE", "시계열 — 모델 성능 비교 (RMSE, 낮을수록 좋음)", lower_better=True)
        figs["ts_prediction"]     = _fig_ts_prediction(d_tr, y_tr, d_te, y_te, pred_best,
                                                        best_name, target_col, future_pred)
        figs["feature_importance"]= _fig_feature_importance(fi, f"피처 중요도 ({best_name})")
        figs["residuals"]         = _fig_residuals(y_te, pred_best, best_name)

        metrics = results.get(best_name, {})
        summary = (
            f"**최적 모델:** {best_name}  \n"
            f"**RMSE:** {metrics.get('RMSE','N/A')}  \n"
            f"**MAE:** {metrics.get('MAE','N/A')}  \n"
            f"**R²:** {metrics.get('R²','N/A')}  \n"
            f"**학습 기간:** {len(y_tr):,}개 포인트 / **테스트:** {len(y_te):,}개 포인트  \n"
            f"**미래 7스텝 예측:** 포함"
        )
        recs = _timeseries_recommendations(metrics, fi)

        return AutoResult(
            analysis_type="timeseries",
            metrics=metrics, figures=figs,
            summary=summary, feature_importances=fi,
            recommendations=recs, model_name=best_name,
            target_col=target_col, feature_cols=feat_names,
        )


    # ══════════════════════════════════════════════════════════════════════════
    # 6. 연관 분석 (Association Analysis)
    # ══════════════════════════════════════════════════════════════════════════
    def _run_association(self, df: pd.DataFrame, schema: SchemaInfo,
                         target_col: str = None) -> AutoResult:
        """
        Apriori 알고리즘 기반 연관 규칙 분석
        - 범주형 컬럼이 충분할 때 의미있는 결과 도출
        - 수치형 컬럼은 분위수(3구간) 기반으로 자동 변환
        """
        try:
            from mlxtend.frequent_patterns import apriori, association_rules
            from mlxtend.preprocessing import TransactionEncoder
        except ImportError:
            return AutoResult("association", success=False,
                              error_msg="mlxtend 미설치 — pip install mlxtend")

        # ── 분석 대상 컬럼 선정 ──────────────────────────────────────────────
        cat_cols = [c for c in schema.categorical_cols if c in df.columns]
        num_cols = [c for c in schema.numeric_cols    if c in df.columns]

        # 수치형 → 3구간 범주화 (low/mid/high)
        df_enc = df.copy()
        converted = []
        for col in num_cols[:8]:
            try:
                df_enc[col + "_bin"] = pd.qcut(
                    df_enc[col], q=3,
                    labels=[col+"_low", col+"_mid", col+"_high"],
                    duplicates="drop"
                )
                converted.append(col + "_bin")
            except Exception:
                pass

        use_cols = cat_cols[:8] + converted
        if len(use_cols) < 2:
            return AutoResult("association", success=False,
                              error_msg="분석 가능한 범주형/변환 컬럼 부족 (최소 2개 필요)")

        data = df_enc[use_cols].dropna()
        if len(data) < 20:
            return AutoResult("association", success=False,
                              error_msg=f"데이터 부족 ({len(data)}행)")

        # ── 원-핫 인코딩 → Boolean 행렬 ──────────────────────────────────────
        try:
            oht = pd.get_dummies(data.astype(str), prefix_sep="=").astype(bool)
        except Exception as e:
            return AutoResult("association", success=False,
                              error_msg=f"인코딩 오류: {e}")

        # ── min_support 자동 조정 ─────────────────────────────────────────────
        for min_sup in [0.3, 0.2, 0.1, 0.05]:
            try:
                freq_items = apriori(oht, min_support=min_sup,
                                     use_colnames=True, max_len=3)
                if len(freq_items) >= 5:
                    break
            except Exception:
                freq_items = pd.DataFrame()
        else:
            return AutoResult("association", success=False,
                              error_msg="지지도 0.05 이하에서도 빈발 항목집합 부족")

        if freq_items.empty:
            return AutoResult("association", success=False,
                              error_msg="빈발 항목집합 없음 — 데이터가 너무 희소합니다")

        # ── 연관 규칙 생성 ────────────────────────────────────────────────────
        try:
            rules = association_rules(freq_items, metric="lift", min_threshold=1.0)
            rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
        except Exception as e:
            return AutoResult("association", success=False,
                              error_msg=f"규칙 생성 오류: {e}")

        if rules.empty:
            return AutoResult("association", success=False,
                              error_msg="lift ≥ 1.0 기준 연관 규칙 없음")

        top_rules = rules.head(20)
        top_freq  = freq_items.sort_values("support", ascending=False).head(20)

        # ── 지표 ──────────────────────────────────────────────────────────────
        metrics = {
            "총 빈발 항목집합": len(freq_items),
            "총 연관 규칙 수": len(rules),
            "최고 Lift": round(float(rules["lift"].max()), 4),
            "최고 Confidence": round(float(rules["confidence"].max()), 4),
            "min_support 적용값": round(min_sup, 2),
            "분석 컬럼 수": len(use_cols),
        }

        # ── 시각화 ───────────────────────────────────────────────────────────
        figs = {}
        figs["assoc_top_rules"]   = _fig_assoc_rules(top_rules)
        figs["assoc_scatter"]     = _fig_assoc_scatter(rules)
        figs["assoc_freq_items"]  = _fig_assoc_freq_items(top_freq)
        figs["assoc_heatmap"]     = _fig_assoc_heatmap(top_rules)

        # ── 요약 ─────────────────────────────────────────────────────────────
        best = top_rules.iloc[0]
        best_ant = ", ".join(list(best["antecedents"]))
        best_con = ", ".join(list(best["consequents"]))
        summary = (
            f"**분석 방법:** Apriori (min_support={min_sup})  \n"
            f"**빈발 항목집합:** {len(freq_items)}개 / **연관 규칙:** {len(rules)}개  \n"
            f"**최고 Lift 규칙:** {best_ant} → {best_con}  \n"
            f"  Lift={best['lift']:.4f}, Confidence={best['confidence']:.4f}, "
            f"Support={best['support']:.4f}  \n"
            f"**분석 컬럼:** {', '.join(use_cols[:6])}{'...' if len(use_cols)>6 else ''}"
        )

        recs = _association_recommendations(rules, min_sup, len(use_cols))

        return AutoResult(
            analysis_type="association",
            metrics=metrics, figures=figs,
            summary=summary,
            recommendations=recs,
            model_name=f"Apriori (min_support={min_sup})",
            feature_cols=use_cols,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 시각화 헬퍼
# ══════════════════════════════════════════════════════════════════════════════

def _fig_model_comparison(results: dict, key: str, title: str,
                           lower_better: bool = False) -> go.Figure:
    names  = list(results.keys())
    values = [results[n].get(key, 0) for n in names]
    best_i = int(np.argmin(values) if lower_better else np.argmax(values))
    colors = [
        "#e74c3c" if i == best_i else "#3498db"
        for i in range(len(names))
    ]
    fig = go.Figure(go.Bar(
        x=names, y=values, marker_color=colors,
        text=[f"{v:.4f}" for v in values], textposition="outside",
    ))
    fig.add_annotation(
        text=f"★ 최적: {names[best_i]}",
        x=names[best_i], y=values[best_i],
        yshift=30, showarrow=False,
        font=dict(color="#e74c3c", size=12, family="Malgun Gothic"),
    )
    fig.update_layout(height=360, template="plotly_white",
                      title_text=title, yaxis_title=key)
    return fig


def _fig_actual_vs_pred(y_true, y_pred, model_name: str, target: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers",
                             marker=dict(color="#3498db", size=5, opacity=0.6),
                             name="예측 vs 실제"))
    mn, mx = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                             line=dict(color="red", dash="dash"), name="완벽 예측선"))
    fig.update_layout(height=400, template="plotly_white",
                      title_text=f"실제 vs 예측 ({model_name})",
                      xaxis_title=f"실제 {target}", yaxis_title=f"예측 {target}")
    return fig


def _fig_residuals(y_true, y_pred, model_name: str) -> go.Figure:
    residuals = y_true - y_pred
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["잔차 분포", "잔차 vs 예측값"])
    fig.add_trace(go.Histogram(x=residuals, nbinsx=30,
                               marker_color="#9b59b6", opacity=0.75,
                               showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                             marker=dict(color="#e67e22", size=4, opacity=0.6),
                             showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    fig.update_layout(height=380, template="plotly_white",
                      title_text=f"잔차 분석 ({model_name})")
    return fig


def _fig_feature_importance(fi: dict, title: str) -> go.Figure:
    if not fi:
        fig = go.Figure()
        fig.add_annotation(text="피처 중요도 없음", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(height=200)
        return fig
    sorted_fi = sorted(fi.items(), key=lambda x: x[1])[-15:]
    names, vals = zip(*sorted_fi)
    fig = go.Figure(go.Bar(x=list(vals), y=list(names), orientation="h",
                           marker_color="#2980b9",
                           text=[f"{v:.4f}" for v in vals], textposition="outside"))
    fig.update_layout(height=max(300, 25 * len(names)),
                      template="plotly_white", title_text=title,
                      xaxis_title="중요도")
    return fig


def _fig_confusion_matrix(cm, classes, model_name: str) -> go.Figure:
    classes_str = [str(c)[:15] for c in classes]
    fig = go.Figure(go.Heatmap(
        z=cm, x=classes_str, y=classes_str,
        colorscale="Blues",
        text=cm, texttemplate="%{text}",
        textfont_size=12,
    ))
    fig.update_layout(height=max(380, 60 * len(classes)),
                      template="plotly_white",
                      title_text=f"혼동 행렬 ({model_name})",
                      xaxis_title="예측", yaxis_title="실제")
    return fig


def _fig_class_dist(y_true, y_pred, classes) -> go.Figure:
    classes_str = [str(c)[:15] for c in classes]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["실제 클래스 분포", "예측 클래스 분포"])
    for data, col, label in [(y_true, 1, "실제"), (y_pred, 2, "예측")]:
        vc = pd.Series(data).value_counts().sort_index()
        fig.add_trace(go.Bar(
            x=[classes_str[i] if i < len(classes_str) else str(i) for i in vc.index],
            y=vc.values,
            marker_color=_COLORS[:len(vc)],
            showlegend=False,
        ), row=1, col=col)
    fig.update_layout(height=380, template="plotly_white",
                      title_text="클래스 분포 비교")
    return fig


def _fig_elbow_sil(k_range, inertias, sils, best_k) -> go.Figure:
    k_list = list(k_range)
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Elbow (관성)", "Silhouette Score"])
    fig.add_trace(go.Scatter(x=k_list, y=inertias, mode="lines+markers",
                             marker_color="#3498db", showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=k_list, y=sils, mode="lines+markers",
                             marker_color="#e74c3c", showlegend=False), row=1, col=2)
    fig.add_vline(x=best_k, line_dash="dash", line_color="#27ae60",
                  annotation_text=f"최적 K={best_k}", row=1, col=2)
    fig.update_layout(height=380, template="plotly_white",
                      title_text="최적 군집 수 탐색")
    return fig


def _fig_cluster_scatter(coords, labels, n_clusters, evr) -> go.Figure:
    fig = go.Figure()
    for c in range(n_clusters):
        mask = labels == c
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1], mode="markers",
            marker=dict(color=_COLORS[c % len(_COLORS)], size=6, opacity=0.65),
            name=f"군집 {c} ({mask.sum():,}개)"))
    fig.update_layout(height=480, template="plotly_white",
                      title_text=f"군집 시각화 (K={n_clusters}, PCA 2D)",
                      xaxis_title=f"PC1 ({evr[0]:.1f}%)",
                      yaxis_title=f"PC2 ({evr[1]:.1f}%)")
    return fig


def _fig_cluster_profile(profile: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col in profile.columns:
        fig.add_trace(go.Bar(name=col, x=[f"군집 {i}" for i in profile.index],
                             y=profile[col].values))
    fig.update_layout(barmode="group", height=420, template="plotly_white",
                      title_text="군집별 변수 평균 프로파일")
    return fig


def _fig_cluster_size(labels, n_clusters) -> go.Figure:
    vc = pd.Series(labels).value_counts().sort_index()
    fig = go.Figure(go.Bar(
        x=[f"군집 {i}" for i in vc.index], y=vc.values,
        marker_color=_COLORS[:n_clusters],
        text=vc.values, textposition="outside",
    ))
    fig.update_layout(height=360, template="plotly_white",
                      title_text="군집별 샘플 수")
    return fig


def _fig_anomaly_scatter(coords, is_anom, evr) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=coords[~is_anom, 0], y=coords[~is_anom, 1], mode="markers",
        marker=dict(color="#3498db", size=5, opacity=0.5), name="정상"))
    fig.add_trace(go.Scatter(
        x=coords[is_anom, 0], y=coords[is_anom, 1], mode="markers",
        marker=dict(color="#e74c3c", size=8, symbol="x"), name="이상"))
    fig.update_layout(height=460, template="plotly_white",
                      title_text="이상 탐지 결과 (PCA 2D)",
                      xaxis_title=f"PC1 ({evr[0]:.1f}%)",
                      yaxis_title=f"PC2 ({evr[1]:.1f}%)")
    return fig


def _fig_anomaly_scores(scores, is_anom) -> go.Figure:
    fig = go.Figure()
    idx = np.arange(len(scores))
    fig.add_trace(go.Scatter(x=idx[~is_anom], y=scores[~is_anom], mode="markers",
                             marker=dict(color="#3498db", size=3, opacity=0.5), name="정상"))
    fig.add_trace(go.Scatter(x=idx[is_anom], y=scores[is_anom], mode="markers",
                             marker=dict(color="#e74c3c", size=6, symbol="x"), name="이상"))
    fig.update_layout(height=360, template="plotly_white",
                      title_text="이상 점수 분포 (낮을수록 이상)",
                      xaxis_title="샘플 인덱스", yaxis_title="Anomaly Score")
    return fig


def _fig_anomaly_profile(data, is_anom, num_cols) -> go.Figure:
    normal_mean = data[~is_anom][num_cols].mean()
    anom_mean   = data[is_anom][num_cols].mean()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["정상 평균", "이상 평균"])
    fig.add_trace(go.Bar(x=num_cols, y=normal_mean.values,
                         marker_color="#3498db", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=num_cols, y=anom_mean.values,
                         marker_color="#e74c3c", showlegend=False), row=1, col=2)
    fig.update_layout(height=380, template="plotly_white",
                      title_text="정상 vs 이상 — 변수별 평균 비교")
    return fig


def _fig_ts_prediction(d_tr, y_tr, d_te, y_te, y_pred,
                        model_name, target_col, future_pred=None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d_tr, y=y_tr, mode="lines",
                             line=dict(color="#95a5a6", width=1), name="학습 데이터"))
    fig.add_trace(go.Scatter(x=d_te, y=y_te, mode="lines",
                             line=dict(color="#3498db", width=2), name="실제 (테스트)"))
    fig.add_trace(go.Scatter(x=d_te, y=y_pred, mode="lines",
                             line=dict(color="#e74c3c", width=2, dash="dot"), name="예측"))
    if future_pred is not None and len(future_pred):
        fig.add_trace(go.Scatter(
            x=list(range(len(d_te), len(d_te) + len(future_pred))),
            y=future_pred, mode="lines+markers",
            line=dict(color="#27ae60", width=2, dash="dash"), name="미래 예측 (7스텝)"))
    fig.update_layout(height=460, template="plotly_white",
                      title_text=f"시계열 예측 ({model_name}) — {target_col}")
    return fig


# ── 미래 예측 ──────────────────────────────────────────────────────────────────
def _forecast_future(model, last_X_row: np.ndarray, feat_names: list,
                     steps: int = 7) -> list:
    """마지막 row 기준으로 간단 재귀 예측 (lag 피처 기반)"""
    try:
        current = last_X_row.copy().tolist()
        preds   = []
        for _ in range(steps):
            pred_val = float(model.predict([current])[0])
            preds.append(pred_val)
            # lag 피처 당기기 (lag_1 ← 이번 예측값)
            lag_idxs = [i for i, n in enumerate(feat_names) if n.startswith("lag_")]
            lag_idxs_sorted = sorted(lag_idxs, key=lambda i: int(feat_names[i].split("_")[1]))
            for li in reversed(lag_idxs_sorted[1:]):
                current[li] = current[lag_idxs_sorted[lag_idxs_sorted.index(li) - 1]]
            if lag_idxs_sorted:
                current[lag_idxs_sorted[0]] = pred_val
        return preds
    except Exception:
        return []


# ── 추천 메시지 생성 ──────────────────────────────────────────────────────────
def _detect_target(df, schema):
    for col in df.columns:
        if col.lower().replace("_","") in ["passfail","label","target","result","churn","judge"]:
            return col
    for col in schema.categorical_cols:
        if col in df.columns and 2 <= df[col].nunique() <= 10:
            return col
    return None


def _regression_recommendations(metrics, fi):
    r2 = metrics.get("R²", 0)
    recs = []
    if r2 < 0.5:
        recs.append("R²가 낮습니다. 비선형 모델(XGBoost, LightGBM) 또는 추가 피처를 시도해보세요.")
    if r2 > 0.95:
        recs.append("R²가 매우 높습니다. 학습 데이터 과적합(overfitting) 여부를 교차검증으로 확인하세요.")
    if fi:
        top_feat = max(fi, key=fi.get)
        recs.append(f"가장 중요한 변수는 '{top_feat}'입니다. 이 변수의 도메인 의미를 확인하세요.")
    recs.append("예측 모델을 운영에 적용하기 전 최신 데이터로 주기적인 재학습을 권장합니다.")
    return recs


def _classification_recommendations(metrics, n_cls, y):
    acc = metrics.get("Accuracy", 0)
    f1  = metrics.get("F1-Score", 0)
    recs = []
    class_counts = pd.Series(y).value_counts()
    imbalance    = class_counts.max() / max(class_counts.min(), 1)
    if imbalance > 5:
        recs.append(f"클래스 불균형이 심합니다 (비율 약 {imbalance:.0f}:1). SMOTE 또는 class_weight 조정을 고려하세요.")
    if f1 < 0.7:
        recs.append("F1-Score가 낮습니다. 피처 엔지니어링 또는 앙상블 모델(XGBoost, LightGBM) 시도를 권장합니다.")
    if acc > 0.99:
        recs.append("정확도가 매우 높습니다. 데이터 누수(data leakage) 여부를 확인하세요.")
    recs.append("혼동 행렬에서 자주 오분류되는 클래스 쌍을 확인하고 추가 학습 데이터를 확보하세요.")
    return recs


def _clustering_recommendations(sil, k, profile):
    recs = []
    if sil < 0.25:
        recs.append("Silhouette Score가 낮습니다. 변수 스케일링 재확인 또는 DBSCAN 시도를 권장합니다.")
    if k >= 5:
        recs.append(f"군집 수가 {k}개로 많습니다. 군집별 비즈니스 의미를 부여하기 어려울 수 있습니다.")
    recs.append("군집 프로파일 차트에서 각 군집의 특성을 확인하고 비즈니스 관점에서 이름을 붙여보세요.")
    recs.append("군집 결과를 기반으로 군집별 맞춤 전략 (가격/마케팅/운영)을 수립할 수 있습니다.")
    return recs


def _anomaly_recommendations(ratio, n_anom, num_cols):
    recs = []
    if ratio > 15:
        recs.append(f"이상 비율이 {ratio:.1f}%로 높습니다. contamination 파라미터를 낮추거나 도메인 기준으로 임계값을 재설정하세요.")
    if ratio < 1:
        recs.append("이상 비율이 매우 낮습니다. 실제 이상이 없거나 탐지 민감도를 높여야 할 수 있습니다.")
    recs.append(f"감지된 이상 {n_anom}개의 발생 시점/조건을 도메인 전문가와 함께 검토하세요.")
    recs.append("이상 샘플의 변수별 평균 프로파일을 정상과 비교하여 이상 원인 변수를 파악하세요.")
    return recs


def _timeseries_recommendations(metrics, fi):
    rmse = metrics.get("RMSE", None)
    recs = []
    recs.append("시계열 예측 정확도를 높이려면 외부 변수(계절성, 이벤트 등)를 추가 피처로 활용하세요.")
    if fi:
        top_feat = max(fi, key=fi.get)
        lag_num  = top_feat.split("_")[-1] if "lag" in top_feat else "?"
        recs.append(f"가장 중요한 피처는 '{top_feat}'입니다. 해당 시차(lag)의 패턴을 분석하세요.")
    recs.append("LSTM, Transformer 등 딥러닝 기반 시계열 모델로 추가 성능 향상을 시도해볼 수 있습니다.")
    recs.append("예측 구간(Prediction Interval)을 추가하면 불확실성을 함께 표현할 수 있습니다.")
    return recs


def _fig_assoc_rules(top_rules: pd.DataFrame) -> go.Figure:
    """상위 연관 규칙 수평 바차트 (Lift 기준)"""
    labels = [
        f"{', '.join(list(r['antecedents']))} → {', '.join(list(r['consequents']))}"
        for _, r in top_rules.iterrows()
    ]
    labels = [l[:60] + "..." if len(l) > 60 else l for l in labels]
    lifts  = top_rules["lift"].round(4).tolist()

    fig = go.Figure(go.Bar(
        x=lifts, y=labels, orientation="h",
        marker=dict(
            color=lifts,
            colorscale="Oranges",
            showscale=True,
            colorbar=dict(title="Lift"),
        ),
        text=[f"{v:.3f}" for v in lifts],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(400, 28 * len(labels)),
        template="plotly_white",
        title_text="상위 연관 규칙 (Lift 기준)",
        xaxis_title="Lift",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=320),
    )
    return fig


def _fig_assoc_scatter(rules: pd.DataFrame) -> go.Figure:
    """Support × Confidence 산점도 (크기=Lift)"""
    fig = go.Figure(go.Scatter(
        x=rules["support"],
        y=rules["confidence"],
        mode="markers",
        marker=dict(
            size=rules["lift"].clip(upper=10) * 5,
            color=rules["lift"],
            colorscale="RdYlGn",
            showscale=True,
            colorbar=dict(title="Lift"),
            opacity=0.7,
        ),
        text=[
            f"Ant: {', '.join(list(r['antecedents']))}<br>"
            f"Con: {', '.join(list(r['consequents']))}<br>"
            f"Lift={r['lift']:.3f}"
            for _, r in rules.iterrows()
        ],
        hoverinfo="text",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="grey",
                  annotation_text="Confidence=0.5")
    fig.update_layout(
        height=450, template="plotly_white",
        title_text="Support × Confidence 분포 (원 크기=Lift)",
        xaxis_title="Support (지지도)",
        yaxis_title="Confidence (신뢰도)",
    )
    return fig


def _fig_assoc_freq_items(top_freq: pd.DataFrame) -> go.Figure:
    """빈발 항목집합 지지도 바차트"""
    labels  = [str(list(s))[:50] for s in top_freq["itemsets"]]
    support = top_freq["support"].round(4).tolist()
    fig = go.Figure(go.Bar(
        x=support, y=labels, orientation="h",
        marker_color="#3498db",
        text=[f"{v:.3f}" for v in support],
        textposition="outside",
    ))
    fig.update_layout(
        height=max(360, 25 * len(labels)),
        template="plotly_white",
        title_text="빈발 항목집합 지지도 Top-20",
        xaxis_title="Support (지지도)",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=280),
    )
    return fig


def _fig_assoc_heatmap(top_rules: pd.DataFrame) -> go.Figure:
    """상위 규칙의 Support / Confidence / Lift 히트맵"""
    labels = [
        f"{', '.join(list(r['antecedents'])[:2])} → {', '.join(list(r['consequents'])[:2])}"
        for _, r in top_rules.iterrows()
    ]
    labels = [l[:55] for l in labels]
    metrics_arr = np.array([
        top_rules["support"].values,
        top_rules["confidence"].values,
        top_rules["lift"].values / top_rules["lift"].max(),   # 0~1 정규화
    ])
    fig = go.Figure(go.Heatmap(
        z=metrics_arr,
        x=labels,
        y=["Support", "Confidence", "Lift (정규화)"],
        colorscale="YlOrRd",
        text=np.array([
            [f"{v:.3f}" for v in top_rules["support"]],
            [f"{v:.3f}" for v in top_rules["confidence"]],
            [f"{v:.3f}" for v in top_rules["lift"]],
        ]),
        texttemplate="%{text}",
        textfont_size=8,
    ))
    fig.update_layout(
        height=300, template="plotly_white",
        title_text="상위 연관 규칙 — 지표 히트맵",
        xaxis=dict(tickangle=-40),
    )
    return fig


def _association_recommendations(rules: pd.DataFrame,
                                  min_sup: float, n_cols: int) -> list[str]:
    recs = []
    high_lift = (rules["lift"] > 2).sum()
    if high_lift:
        recs.append(
            f"Lift > 2 인 규칙이 {high_lift}개 발견됐습니다. "
            "이 규칙들은 우연 이상으로 강하게 연관된 항목쌍입니다.")
    if min_sup <= 0.1:
        recs.append(
            f"지지도 기준이 {min_sup:.0%}로 낮습니다. "
            "데이터가 희소하거나 카테고리가 많을 때 나타나는 현상으로, "
            "고지지도 규칙만 선별해서 활용하세요.")
    recs.append(
        "Confidence가 높고 Lift도 높은 규칙이 가장 활용 가치가 높습니다. "
        "산점도에서 오른쪽 상단 규칙을 우선 검토하세요.")
    recs.append(
        "연관 규칙을 실무에 적용하기 전 도메인 전문가의 검토를 통해 "
        "허위 상관(spurious correlation)을 제거하세요.")
    return recs