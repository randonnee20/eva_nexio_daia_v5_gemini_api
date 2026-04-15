"""
DAIA v3 - 빠른 산점도/플롯 (Gradio UI용)
데이터를 그냥 그려보는 기능
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def quick_plot(df: pd.DataFrame, x_col: str, y_col: str,
               color_col: str = None, plot_type: str = "scatter") -> str:
    """
    단순 플롯 → HTML div 반환
    plot_type: scatter | line | box | histogram | heatmap
    """
    if x_col not in df.columns or y_col not in df.columns:
        return "<p style='color:red'>컬럼이 존재하지 않습니다.</p>"

    # 대용량 샘플링
    sample = df.sample(min(10000, len(df)), random_state=42) if len(df) > 10000 else df

    try:
        if plot_type == "scatter":
            fig = px.scatter(sample, x=x_col, y=y_col, color=color_col,
                             opacity=0.6, template="plotly_white",
                             title=f"{x_col} vs {y_col}")
        elif plot_type == "line":
            fig = px.line(sample.sort_values(x_col), x=x_col, y=y_col,
                          color=color_col, template="plotly_white",
                          title=f"{x_col} → {y_col}")
        elif plot_type == "box":
            fig = px.box(sample, x=color_col or x_col, y=y_col,
                         template="plotly_white",
                         title=f"{y_col} 박스플롯")
        elif plot_type == "histogram":
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=[x_col, y_col])
            fig.add_trace(go.Histogram(x=sample[x_col], nbinsx=40,
                                       name=x_col, showlegend=False), row=1, col=1)
            fig.add_trace(go.Histogram(x=sample[y_col], nbinsx=40,
                                       name=y_col, showlegend=False), row=1, col=2)
            fig.update_layout(template="plotly_white",
                              title_text="히스토그램")
        elif plot_type == "heatmap":
            num_cols = sample.select_dtypes(include=np.number).columns[:20]
            corr = sample[num_cols].corr().round(2)
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(),
                y=corr.index.tolist(), colorscale="RdBu_r",
                zmid=0, text=corr.values.round(2),
                texttemplate="%{text}", textfont_size=9,
            ))
            fig.update_layout(template="plotly_white",
                              title_text="상관관계 히트맵",
                              height=500)
        else:
            return "<p>지원하지 않는 플롯 타입</p>"

        return fig.to_html(full_html=False, include_plotlyjs=False,
                           config={"displayModeBar": True})
    except Exception as e:
        return f"<p style='color:red'>플롯 오류: {e}</p>"
