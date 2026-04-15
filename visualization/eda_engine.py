"""
DAIA v4 - EDA 엔진 (강화 버전)
추가: 데이터 인벤토리 차트 / 품질 레이더 / 피처 중요도 / 분포+박스 / PCA
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Optional
from core.schema_detector import SchemaInfo

_COLORS = px.colors.qualitative.Plotly


class EDAEngine:
    def __init__(self, config: dict = None):
        self.cfg      = (config or {}).get("eda", {})
        self.ma_s     = self.cfg.get("time_series", {}).get("ma_short", 20)
        self.ma_l     = self.cfg.get("time_series", {}).get("ma_long", 100)
        self.anom_sig = self.cfg.get("time_series", {}).get("anomaly_sigma", 2.5)
        self.top_n    = self.cfg.get("top_n_categorical", 15)
        self.scat_n   = self.cfg.get("sample_for_scatter", 5000)
        self.corr_thr = self.cfg.get("correlation_threshold", 0.7)
        self.max_card = self.cfg.get("max_cardinality_barplot", 30)
        self._chart_figs: dict[str, go.Figure] = {}

    # ─── public API ──────────────────────────────────────────────────────────
    def run_all(self, df: pd.DataFrame, schema: SchemaInfo) -> dict[str, str]:
        charts = {}
        self._chart_figs = {}

        charts["data_overview"]    = self._safe("data_overview",    self._chart_overview,    df, schema)
        charts["missing_analysis"] = self._safe("missing_analysis", self._chart_missing,     df, schema)
        charts["distributions"]    = self._safe("distributions",    self._chart_distributions,df, schema)
        charts["box_plots"]        = self._safe("box_plots",        self._chart_boxplots,    df, schema)

        c = self._safe("correlations", self._chart_correlation, df, schema)
        if c: charts["correlations"] = c

        c = self._safe("pca_analysis", self._chart_pca, df, schema)
        if c: charts["pca_analysis"] = c

        c = self._safe("pca_loadings", self._chart_pca_loadings, df, schema)
        if c: charts["pca_loadings"] = c

        target_col = self._find_target(df, schema)
        if target_col:
            charts["target_analysis"] = self._safe("target_analysis", self._chart_target, df, schema, target_col)
            c = self._safe("feature_importance", self._chart_feature_importance, df, schema, target_col)
            if c: charts["feature_importance"] = c

        if schema.schema_type == "wide_table" and len(schema.numeric_cols) >= 2 and len(df) >= 20:
            charts["cluster_distribution"]  = self._safe("cluster_distribution",  self._chart_cluster_dist, df, schema)
            charts["cluster_visualization"] = self._safe("cluster_visualization", self._chart_cluster_viz,  df, schema)

        if schema.schema_type in ("signal_pool", "time_series"):
            charts["time_series"] = self._safe("time_series", self._chart_time_series, df, schema)
        if schema.schema_type == "signal_pool":
            charts["event_timeline"] = self._safe("event_timeline", self._chart_event_timeline, df, schema)

        # 범주형 분포
        if schema.categorical_cols:
            c = self._safe("categorical_dist", self._chart_categorical, df, schema)
            if c: charts["categorical_dist"] = c

        return {k: v for k, v in charts.items() if v}

    def compute_insights(self, df: pd.DataFrame, schema: SchemaInfo) -> list[str]:
        insights = []
        try:
            # 결측치
            miss = df.isnull().mean()
            high_miss = miss[miss > 0.5].index.tolist()
            if high_miss:
                insights.append(f"⚠️  결측치 50% 이상 컬럼: {high_miss}")
            mid_miss = miss[(miss > 0.2) & (miss <= 0.5)].index.tolist()
            if mid_miss:
                insights.append(f"🟡 결측치 20~50% 컬럼: {mid_miss[:5]}")

            # 중복
            dups = df.duplicated().sum()
            if dups:
                insights.append(f"🔁 중복 행: {dups:,}개 ({dups/len(df)*100:.1f}%)")
            else:
                insights.append("✅ 중복 행 없음")

            # signal_pool 전용
            if schema.schema_type == "signal_pool" and schema.signal_name_col:
                n_sig = df[schema.signal_name_col].nunique()
                top   = df[schema.signal_name_col].value_counts().index[0]
                cnt   = df[schema.signal_name_col].value_counts().iloc[0]
                insights.append(f"📡 신호 종류: {n_sig}개 (최다 신호: {top} - {cnt:,}건)")

            if schema.value_num_col and schema.value_num_col in df.columns:
                num = df[schema.value_num_col].dropna()
                insights.append(f"📊 value 숫자: {len(num):,}행")

            # wide_table 상관관계
            if schema.schema_type == "wide_table":
                num_cols = [c for c in schema.numeric_cols if c in df.columns]
                if len(num_cols) >= 3:
                    corr = df[num_cols].corr().abs()
                    arr  = corr.values.copy()
                    np.fill_diagonal(arr, 0)
                    high = (arr > self.corr_thr).sum() // 2
                    if high:
                        insights.append(f"🔗 강한 상관관계 쌍: {high}개 (|r|>{self.corr_thr})")

            # 왜도 높은 컬럼
            num_cols = [c for c in schema.numeric_cols if c in df.columns]
            skewed = []
            for col in num_cols[:20]:
                try:
                    sk = abs(df[col].skew())
                    if sk > 2:
                        skewed.append(f"{col}({sk:.1f})")
                except Exception:
                    pass
            if skewed:
                insights.append(f"📐 고왜도(|skew|>2) 컬럼: {', '.join(skewed[:5])}")

            # 상수 컬럼
            const_cols = [c for c in df.columns if df[c].nunique() <= 1]
            if const_cols:
                insights.append(f"⚪ 단일값(상수) 컬럼: {const_cols}")

        except Exception as e:
            insights.append(f"[인사이트 계산 오류: {e}]")
        return insights

    def get_descriptive_stats(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        if schema.schema_type == "signal_pool" and schema.value_num_col and schema.value_num_col in df.columns:
            grp = df.groupby(schema.signal_name_col)[schema.value_num_col].agg(
                ["count","mean","std","min",
                 lambda x: x.quantile(0.25),
                 "median",
                 lambda x: x.quantile(0.75),
                 "max", "skew"]
            ).round(4)
            grp.columns = ["Count","Mean","Std","Min","Q1","Median","Q3","Max","Skew"]
            return grp
        else:
            num_cols = [c for c in schema.numeric_cols if c in df.columns][:30]
            if not num_cols:
                return pd.DataFrame()
            desc = df[num_cols].describe().T.round(4)
            # 왜도, 첨도 추가
            try:
                desc["skew"] = df[num_cols].skew().round(4)
                desc["kurt"] = df[num_cols].kurt().round(4)
            except Exception:
                pass
            return desc

    def save_charts_png(self, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for name, fig in self._chart_figs.items():
            try:
                p = output_dir / f"{name}.png"
                fig.write_image(str(p), width=1200, height=700, scale=1.5)
                saved.append(p)
            except Exception:
                pass
        return saved

    # ─── 안전 래퍼 ────────────────────────────────────────────────────────────
    def _safe(self, name, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"[EDA] {name} 실패: {e}\n{traceback.format_exc()}")
            return None

    def _fig2html(self, fig: go.Figure, name: str) -> str:
        self._chart_figs[name] = fig
        return fig.to_html(full_html=False, include_plotlyjs="cdn",
                           config={"displayModeBar": True})

    # ─── 1. 데이터 개요 ──────────────────────────────────────────────────────
    def _chart_overview(self, df, schema):
        if schema.schema_type == "signal_pool":
            miss_df = self._signal_pool_effective_missing(df, schema)
        else:
            miss_df = (df.isnull().mean() * 100).sort_values(ascending=False).head(15)
        miss_df = miss_df[miss_df > 0]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=["데이터 타입 분포","결측치 비율 Top-15","변수 타입 수",
                            "메모리 사용량","왜도 분포","데이터 품질 점수"],
            specs=[[{"type":"pie"},{"type":"xy"},{"type":"xy"}],
                   [{"type":"xy"},{"type":"xy"},{"type":"indicator"}]],
            vertical_spacing=0.20, horizontal_spacing=0.10,
        )
        # 타입 파이
        dc = df.dtypes.astype(str).value_counts()
        fig.add_trace(go.Pie(labels=dc.index.tolist(), values=dc.values.tolist(),
                             textinfo="label+percent"), row=1, col=1)
        # 결측치
        if len(miss_df):
            fig.add_trace(go.Bar(y=miss_df.index.tolist(), x=miss_df.values.tolist(),
                                 orientation="h", marker_color="#e74c3c",
                                 showlegend=False), row=1, col=2)
            fig.update_xaxes(range=[0, 100], title_text="결측치(%)", row=1, col=2)
        # 변수 타입
        tc = {k:v for k,v in {"수치형":len(schema.numeric_cols),
              "범주형":len(schema.categorical_cols),"날짜형":len(schema.datetime_cols),
              "혼합":len(schema.mixed_cols)}.items() if v > 0}
        colors = ["#3498db","#e67e22","#9b59b6","#e74c3c"]
        fig.add_trace(go.Bar(x=list(tc.keys()), y=list(tc.values()),
                             marker_color=colors[:len(tc)], showlegend=False), row=1, col=3)
        # 메모리
        mem = (df.memory_usage(deep=True).drop("Index")/1024**2).sort_values(ascending=False).head(10)
        fig.add_trace(go.Bar(y=mem.index.tolist(), x=mem.values.tolist(),
                             orientation="h", marker_color="#2ecc71", showlegend=False), row=2, col=1)
        fig.update_xaxes(title_text="MB", row=2, col=1)
        # 왜도 분포
        num_cols = [c for c in schema.numeric_cols if c in df.columns][:20]
        if num_cols:
            skews = [df[c].skew() for c in num_cols]
            fig.add_trace(go.Histogram(x=skews, nbinsx=20, marker_color="#9b59b6",
                                       showlegend=False), row=2, col=2)
            fig.update_xaxes(title_text="Skewness", row=2, col=2)
        # 품질 점수 (게이지)
        miss_rate = df.isnull().mean().mean() * 100
        dup_rate  = df.duplicated().mean() * 100
        q_score   = max(0, 100 - miss_rate*0.5 - dup_rate*1.5)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=round(q_score, 1),
            title={"text": "품질 점수"},
            gauge={"axis":{"range":[0,100]},
                   "bar":{"color":"#27ae60"},
                   "steps":[{"range":[0,50],"color":"#fadbd8"},
                             {"range":[50,80],"color":"#fdebd0"},
                             {"range":[80,100],"color":"#d5f5e3"}]},
        ), row=2, col=3)

        fig.update_layout(height=700, title_text="데이터 개요 (Data Inventory)",
                          template="plotly_white", showlegend=False)
        return self._fig2html(fig, "data_overview")

    def _signal_pool_effective_missing(self, df, schema):
        skip = {schema.signal_name_col, schema.value_col, schema.value_num_col,
                schema.timestamp_col, "value_num", "value_text"}
        cols = [c for c in df.columns if c not in skip]
        if not cols:
            return pd.Series(dtype=float)
        return (df[cols].isnull().mean() * 100).sort_values(ascending=False).head(15)

    # ─── 2. 결측치 (heatmap + 패턴) ──────────────────────────────────────────
    def _chart_missing(self, df, schema):
        if schema.schema_type == "signal_pool":
            miss = self._signal_pool_effective_missing(df, schema)
        else:
            miss = (df.isnull().mean() * 100).sort_values(ascending=False)
            miss = miss[miss > 0].head(20)

        if miss.empty:
            fig = go.Figure()
            fig.add_annotation(text="✅ 결측치 없음", x=0.5, y=0.5,
                               showarrow=False, font_size=22)
            fig.update_layout(height=300, template="plotly_white")
            return self._fig2html(fig, "missing_analysis")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["결측치 패턴 (샘플 300행)", "결측치 비율 (%)"])
        sample = df.sample(min(300, len(df)), random_state=42)
        miss_mat = sample[miss.index.tolist()].isnull().astype(int)
        fig.add_trace(go.Heatmap(z=miss_mat.values.T, y=miss_mat.columns.tolist(),
                                 colorscale=[[0,"#2c3e50"],[1,"#e74c3c"]],
                                 showscale=False), row=1, col=1)
        colors = ["#e74c3c" if v > 50 else "#e67e22" if v > 20 else "#f1c40f"
                  for v in miss.values]
        fig.add_trace(go.Bar(y=miss.index.tolist(), x=miss.values.tolist(),
                             orientation="h", marker_color=colors,
                             text=[f"{v:.1f}%" for v in miss.values],
                             textposition="outside",
                             showlegend=False), row=1, col=2)
        fig.update_xaxes(range=[0, 105], title_text="결측치(%)", row=1, col=2)
        fig.update_layout(height=450, template="plotly_white", title_text="결측치 분석")
        return self._fig2html(fig, "missing_analysis")

    # ─── 3. 수치형 분포 (히스토그램 + KDE) ───────────────────────────────────
    def _chart_distributions(self, df, schema):
        if schema.schema_type == "signal_pool":
            col = schema.value_num_col
            if not col or col not in df.columns:
                return None
            sigs = df[schema.signal_name_col].value_counts().head(8).index.tolist()
            cols_to_plot = sigs; use_signal = True
        else:
            cols_to_plot = [c for c in schema.numeric_cols if c in df.columns][:12]
            use_signal = False
        if not cols_to_plot: return None

        n = len(cols_to_plot); ncols = 4; nrows = (n + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=[str(c) for c in cols_to_plot],
                            vertical_spacing=0.14, horizontal_spacing=0.08)
        for i, item in enumerate(cols_to_plot):
            r, c_i = divmod(i, ncols)
            if use_signal:
                s = df[df[schema.signal_name_col] == item][col].dropna()
            else:
                s = df[item].dropna()
            if len(s) < 5: continue
            fig.add_trace(go.Histogram(x=s, nbinsx=30, name=str(item),
                                       marker_color=_COLORS[i % len(_COLORS)],
                                       opacity=0.7, showlegend=False), row=r+1, col=c_i+1)
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(s)
                xr  = np.linspace(s.min(), s.max(), 200)
                yr  = kde(xr) * len(s) * (s.max() - s.min()) / 30
                fig.add_trace(go.Scatter(x=xr, y=yr, mode="lines",
                                         line=dict(color="red", width=1.5),
                                         showlegend=False), row=r+1, col=c_i+1)
            except ImportError:
                pass

        title = "신호별 값 분포" if use_signal else "수치형 변수 분포"
        fig.update_layout(height=max(280*nrows, 400), template="plotly_white",
                          title_text=title, showlegend=False)
        return self._fig2html(fig, "distributions")

    # ─── 4. Box Plot (이상값 시각화) ─────────────────────────────────────────
    def _chart_boxplots(self, df, schema):
        num_cols = [c for c in schema.numeric_cols if c in df.columns][:16]
        if not num_cols: return None
        n = len(num_cols); ncols = 4; nrows = (n + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=num_cols,
                            vertical_spacing=0.12, horizontal_spacing=0.08)
        for i, col in enumerate(num_cols):
            r, c_i = divmod(i, ncols)
            s = df[col].dropna()
            fig.add_trace(go.Box(y=s, name=col,
                                 marker_color=_COLORS[i % len(_COLORS)],
                                 boxpoints="outliers", jitter=0.3,
                                 showlegend=False), row=r+1, col=c_i+1)
        fig.update_layout(height=max(280*nrows, 400), template="plotly_white",
                          title_text="수치형 변수 박스플롯 (이상값 포함)", showlegend=False)
        return self._fig2html(fig, "box_plots")

    # ─── 5. 상관관계 히트맵 ──────────────────────────────────────────────────
    def _chart_correlation(self, df, schema):
        num_cols = [c for c in schema.numeric_cols if c in df.columns]
        if len(num_cols) < 3: return None
        cols = num_cols[:25]
        corr = df[cols].corr().round(2)
        fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=corr.values.round(2), texttemplate="%{text}", textfont_size=8,
            colorbar=dict(title="r"),
        ))
        fig.update_layout(height=max(500, 28*len(cols)),
                          title_text="변수 간 상관관계 히트맵",
                          template="plotly_white",
                          xaxis=dict(tickangle=-45))
        return self._fig2html(fig, "correlations")

    # ─── 6. PCA 분석 ─────────────────────────────────────────────────────────
    def _chart_pca(self, df, schema):
        num_cols = [c for c in schema.numeric_cols if c in df.columns]
        if len(num_cols) < 3: return None
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            X = df[num_cols[:20]].fillna(df[num_cols[:20]].median())
            X_s = StandardScaler().fit_transform(X)
            n_comp = min(min(len(num_cols), 10), X_s.shape[0])
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(X_s)
            evr = pca.explained_variance_ratio_ * 100
            cum = np.cumsum(evr)

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["주성분별 설명력(%)","누적 설명력(%)"])
            fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(n_comp)],
                                 y=evr.round(2), marker_color="#3498db",
                                 showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=[f"PC{i+1}" for i in range(n_comp)],
                                     y=cum.round(2), mode="lines+markers",
                                     line=dict(color="#e74c3c", width=2),
                                     showlegend=False), row=1, col=2)
            fig.add_hline(y=80, line_dash="dash", line_color="grey", row=1, col=2)
            fig.update_yaxes(title_text="설명력(%)", row=1, col=1)
            fig.update_yaxes(title_text="누적 설명력(%)", range=[0, 105], row=1, col=2)
            fig.update_layout(height=400, template="plotly_white",
                              title_text="PCA — 주성분 분석")
            return self._fig2html(fig, "pca_analysis")
        except ImportError:
            return None

    # ─── 6-2. PCA 변수 기여도 (Loadings) ────────────────────────────────────
    def _chart_pca_loadings(self, df, schema):
        """각 주성분(PC)에 어떤 변수가 얼마나 기여하는지 시각화"""
        num_cols = [c for c in schema.numeric_cols if c in df.columns]
        if len(num_cols) < 3:
            return None
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            X       = df[num_cols[:20]].fillna(df[num_cols[:20]].median())
            X_s     = StandardScaler().fit_transform(X)
            n_comp  = min(min(len(num_cols), 6), X_s.shape[0])
            pca     = PCA(n_components=n_comp, random_state=42)
            pca.fit(X_s)

            # loadings: shape = (n_features, n_components)
            loadings = pca.components_.T          # (변수 수, PC 수)
            evr      = pca.explained_variance_ratio_ * 100
            pc_labels = [f"PC{i+1}\n({evr[i]:.1f}%)" for i in range(n_comp)]
            var_names = [c[:18] for c in num_cols[:20]]  # 긴 이름 truncate

            # ── subplot 구성 ──────────────────────────────────────────────
            # 행1: 전체 로딩 히트맵
            # 행2: PC1·PC2 상위 기여 변수 바차트 (각 PC별)
            n_bar_cols = min(n_comp, 4)
            fig = make_subplots(
                rows=2, cols=n_bar_cols,
                row_heights=[0.5, 0.5],
                subplot_titles=(
                    ["변수별 주성분 기여도 (Loadings Heatmap)"]
                    + [""] * (n_bar_cols - 1)
                    + [f"PC{i+1} 상위 기여 변수 (설명력 {evr[i]:.1f}%)"
                       for i in range(n_bar_cols)]
                ),
                specs=(
                    [{"colspan": n_bar_cols}, *[None] * (n_bar_cols - 1)],
                    [{"type": "xy"}] * n_bar_cols,
                ),
                vertical_spacing=0.18,
                horizontal_spacing=0.06,
            )

            # 히트맵: 변수(행) × PC(열), 색상 = 로딩 값
            fig.add_trace(
                go.Heatmap(
                    z=loadings,
                    x=pc_labels,
                    y=var_names,
                    colorscale="RdBu_r",
                    zmid=0, zmin=-1, zmax=1,
                    text=np.round(loadings, 2),
                    texttemplate="%{text}",
                    textfont_size=9,
                    colorbar=dict(title="기여도", len=0.45, y=0.78),
                    showscale=True,
                ),
                row=1, col=1,
            )

            # PC별 상위 기여 변수 바차트
            for pc_idx in range(n_bar_cols):
                ld   = loadings[:, pc_idx]          # 해당 PC의 모든 변수 로딩
                # 절댓값 기준 상위 8개
                top_idx  = np.argsort(np.abs(ld))[::-1][:8]
                top_vars = [var_names[i] for i in top_idx]
                top_vals = [ld[i] for i in top_idx]
                colors   = ["#e74c3c" if v >= 0 else "#3498db" for v in top_vals]

                fig.add_trace(
                    go.Bar(
                        x=top_vals,
                        y=top_vars,
                        orientation="h",
                        marker_color=colors,
                        showlegend=False,
                        text=[f"{v:+.3f}" for v in top_vals],
                        textposition="outside",
                    ),
                    row=2, col=pc_idx + 1,
                )
                fig.update_xaxes(
                    range=[-1.05, 1.05],
                    zeroline=True, zerolinecolor="#aaa", zerolinewidth=1,
                    row=2, col=pc_idx + 1,
                )

            # 해석 안내 주석
            fig.add_annotation(
                text="<b>읽는 법:</b> 절댓값이 클수록 해당 PC에 강하게 기여 | "
                     "빨강(+) = 양의 방향, 파랑(-) = 음의 방향 | "
                     "같은 PC에서 반대 부호 변수는 서로 상충 관계",
                xref="paper", yref="paper",
                x=0, y=-0.04,
                showarrow=False,
                font=dict(size=11, color="#555"),
                align="left",
            )

            total_h = max(420 + 260 * 2, 700)
            fig.update_layout(
                height=total_h,
                template="plotly_white",
                title_text="PCA 변수 기여도 분석",
                showlegend=False,
            )
            return self._fig2html(fig, "pca_loadings")

        except ImportError:
            return None

    # ─── 7. 타겟 분석 ────────────────────────────────────────────────────────
    def _find_target(self, df, schema):
        candidates = ["passfail","passorfail","pass_or_fail","label","target",
                      "result","outcome","결과","양불","합불","판정","judge","churn"]
        for col in df.columns:
            if col.lower().replace("_","").replace(" ","") in candidates:
                return col
        for col in schema.categorical_cols:
            if col not in df.columns: continue
            if 2 <= df[col].nunique() <= 5:
                return col
        return None

    def _chart_target(self, df, schema, target_col):
        vc  = df[target_col].value_counts()
        pct = (vc / len(df) * 100).round(1)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["타겟 분포","클래스 비율"],
                            specs=[[{"type":"xy"},{"type":"pie"}]])
        fig.add_trace(go.Bar(x=vc.index.tolist(), y=vc.values.tolist(),
                             text=vc.values, textposition="outside",
                             marker_color=_COLORS[:len(vc)],
                             showlegend=False), row=1, col=1)
        fig.add_trace(go.Pie(labels=vc.index.tolist(), values=vc.values.tolist(),
                             text=[f"{p}%" for p in pct],
                             textinfo="label+text"), row=1, col=2)
        fig.update_layout(height=420, template="plotly_white",
                          title_text=f"타겟 변수 분석: {target_col}")
        return self._fig2html(fig, "target_analysis")

    # ─── 8. 피처 중요도 ──────────────────────────────────────────────────────
    def _chart_feature_importance(self, df, schema, target_col):
        try:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            from sklearn.preprocessing import LabelEncoder
            num_cols = [c for c in schema.numeric_cols if c in df.columns and c != target_col]
            if len(num_cols) < 2: return None
            X = df[num_cols].fillna(df[num_cols].median())
            y = df[target_col]
            is_clf = y.dtype == object or y.nunique() <= 10
            if is_clf:
                le = LabelEncoder()
                y  = le.fit_transform(y.astype(str))
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                y     = y.fillna(y.median())
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)
            imp = pd.Series(model.feature_importances_, index=num_cols).sort_values(ascending=True)
            imp = imp.tail(15)
            fig = go.Figure(go.Bar(x=imp.values, y=imp.index.tolist(),
                                   orientation="h", marker_color="#3498db",
                                   text=imp.values.round(4), textposition="outside"))
            fig.update_layout(height=max(350, 25*len(imp)),
                              title_text=f"피처 중요도 (Random Forest) — 타겟: {target_col}",
                              template="plotly_white",
                              xaxis_title="중요도(Gini Impurity)")
            return self._fig2html(fig, "feature_importance")
        except ImportError:
            return None

    # ─── 9. 클러스터링 ───────────────────────────────────────────────────────
    def _chart_cluster_dist(self, df, schema):
        labels, _ = self._run_clustering(df, schema)
        vc = pd.Series(labels).value_counts().sort_index()
        fig = go.Figure(go.Bar(
            x=[f"군집 {i}" for i in vc.index], y=vc.values,
            marker_color=_COLORS[:len(vc)],
            text=vc.values, textposition="outside"))
        fig.update_layout(height=400, template="plotly_white",
                          title_text="군집 크기 분포 (K-Means Elbow)")
        return self._fig2html(fig, "cluster_distribution")

    def _chart_cluster_viz(self, df, schema):
        labels, n_clusters = self._run_clustering(df, schema)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        num_cols = [c for c in schema.numeric_cols if c in df.columns][:20]
        X = df[num_cols].fillna(df[num_cols].median())
        X_s = StandardScaler().fit_transform(X)
        n_comp = min(2, X_s.shape[1])
        pca    = PCA(n_components=n_comp, random_state=42)
        coords = pca.fit_transform(X_s)
        evr    = pca.explained_variance_ratio_ * 100
        if n_comp == 1:
            coords = np.column_stack([coords, np.zeros(len(coords))])
            evr    = [evr[0], 0.0]
        fig = go.Figure()
        for c in range(n_clusters):
            mask = labels == c
            fig.add_trace(go.Scatter(
                x=coords[mask, 0], y=coords[mask, 1], mode="markers",
                marker=dict(color=_COLORS[c % len(_COLORS)], size=5, opacity=0.6),
                name=f"군집 {c}"))
        fig.update_layout(height=500, template="plotly_white",
                          title_text=f"군집 시각화 (K={n_clusters})",
                          xaxis_title=f"PC1 ({evr[0]:.1f}%)",
                          yaxis_title=f"PC2 ({evr[1]:.1f}%)")
        return self._fig2html(fig, "cluster_visualization")

    def _run_clustering(self, df, schema):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        num_cols = [c for c in schema.numeric_cols if c in df.columns][:20]
        X = df[num_cols].fillna(df[num_cols].median())
        X_s = StandardScaler().fit_transform(X)
        best_k = 2; best_ratio = float("inf"); inertias = []
        for k in range(2, min(6, len(df) // 10 + 2)):
            km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=100)
            km.fit(X_s); inertias.append(km.inertia_)
            if len(inertias) >= 2:
                ratio = inertias[-1] / inertias[-2]
                if ratio < best_ratio:
                    best_ratio = ratio; best_k = k
        km_f = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        return km_f.fit_predict(X_s), best_k

    # ─── 10. 범주형 분포 ─────────────────────────────────────────────────────
    def _chart_categorical(self, df, schema):
        cat_cols = [c for c in schema.categorical_cols if c in df.columns][:8]
        if not cat_cols: return None
        n = len(cat_cols); ncols = min(4, n); nrows = (n + ncols - 1) // ncols
        fig = make_subplots(rows=nrows, cols=ncols,
                            subplot_titles=cat_cols,
                            vertical_spacing=0.15, horizontal_spacing=0.10)
        for i, col in enumerate(cat_cols):
            r, c_i = divmod(i, ncols)
            vc = df[col].value_counts().head(self.top_n)
            fig.add_trace(go.Bar(x=vc.index.astype(str).tolist(), y=vc.values,
                                 marker_color=_COLORS[i % len(_COLORS)],
                                 showlegend=False), row=r+1, col=c_i+1)
        fig.update_layout(height=max(300*nrows, 400), template="plotly_white",
                          title_text="범주형 변수 분포", showlegend=False)
        return self._fig2html(fig, "categorical_dist")

    # ─── 11. 시계열 ──────────────────────────────────────────────────────────
    def _chart_time_series(self, df, schema):
        ts_col  = schema.timestamp_col
        val_col = schema.value_num_col or (schema.numeric_cols[0] if schema.numeric_cols else None)
        if not ts_col or not val_col or ts_col not in df.columns or val_col not in df.columns:
            return None

        if schema.schema_type == "signal_pool" and schema.signal_name_col:
            top_sigs = df[schema.signal_name_col].value_counts().head(6).index.tolist()
            fig = make_subplots(rows=len(top_sigs), cols=1,
                                subplot_titles=top_sigs,
                                vertical_spacing=0.06, shared_xaxes=True)
            for i, sig in enumerate(top_sigs):
                sub = df[df[schema.signal_name_col] == sig].sort_values(ts_col)
                s   = sub[val_col].dropna(); t = sub[ts_col][s.index]
                fig.add_trace(go.Scatter(x=t, y=s, mode="lines",
                                         line=dict(color=_COLORS[i%len(_COLORS)], width=1),
                                         name=sig, showlegend=False), row=i+1, col=1)
                if len(s) >= self.ma_s:
                    ma = s.rolling(self.ma_s).mean()
                    fig.add_trace(go.Scatter(x=t, y=ma, mode="lines",
                                             line=dict(color="red", width=1.5, dash="dot"),
                                             showlegend=False), row=i+1, col=1)
                mu, std = s.mean(), s.std()
                anom = s[np.abs(s - mu) > self.anom_sig * std]
                if len(anom):
                    fig.add_trace(go.Scatter(x=t[anom.index], y=anom, mode="markers",
                                             marker=dict(color="orange", size=7, symbol="x"),
                                             showlegend=False), row=i+1, col=1)
            h = max(250*len(top_sigs), 500)
        else:
            df_s = df.sort_values(ts_col)
            s    = df_s[val_col].dropna(); t = df_s[ts_col][s.index]
            fig  = go.Figure()
            fig.add_trace(go.Scatter(x=t, y=s, mode="lines",
                                     line=dict(color="#3498db", width=1), name=val_col))
            if len(s) >= self.ma_l:
                ma = s.rolling(self.ma_l).mean()
                fig.add_trace(go.Scatter(x=t, y=ma, mode="lines",
                                         line=dict(color="red", width=2, dash="dot"),
                                         name=f"MA{self.ma_l}"))
            h = 400

        fig.update_layout(height=h, template="plotly_white",
                          title_text="시계열 분석 (이상값 표시)", showlegend=False)
        return self._fig2html(fig, "time_series")

    # ─── 12. 이벤트 타임라인 ─────────────────────────────────────────────────
    def _chart_event_timeline(self, df, schema):
        ts_col  = schema.timestamp_col
        val_col = schema.value_col
        sig_col = schema.signal_name_col
        if not ts_col or not val_col or ts_col not in df.columns or val_col not in df.columns:
            return None
        events = df[df[val_col].apply(lambda x: isinstance(x, str) and not _is_numeric(x))].copy()
        if len(events) < 1: return None
        events = events.sort_values(ts_col).head(500)
        top_events = events[val_col].value_counts().head(10).index.tolist()
        events     = events[events[val_col].isin(top_events)]
        cmap       = {e: _COLORS[i % len(_COLORS)] for i, e in enumerate(top_events)}
        fig = go.Figure()
        for evt in top_events:
            sub = events[events[val_col] == evt]
            ys  = [evt]*len(sub) if sig_col not in events.columns else sub[sig_col].tolist()
            fig.add_trace(go.Scatter(x=sub[ts_col], y=ys, mode="markers",
                                     marker=dict(color=cmap[evt], size=8),
                                     name=evt))
        fig.update_layout(height=450, template="plotly_white",
                          title_text="이벤트 타임라인", showlegend=True)
        return self._fig2html(fig, "event_timeline")


def _is_numeric(v):
    try:
        float(v); return True
    except (ValueError, TypeError):
        return False
