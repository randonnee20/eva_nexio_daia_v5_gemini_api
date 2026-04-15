"""
DAIA v5 - Gradio UI [HF Spaces Safe Version]
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import pandas as pd
import yaml
import tempfile
from pathlib import Path

# =========================
# SAFE UTIL
# =========================
def safe_df(df, msg="데이터 없음"):
    if isinstance(df, pd.DataFrame) and not df.empty:
        return df
    return pd.DataFrame({"메시지": [msg]})

def safe_md(x):
    try:
        return str(x) if x is not None else ""
    except:
        return ""

def safe_list_to_str(x):
    if isinstance(x, (list, tuple)):
        return ", ".join(map(str, x))
    return str(x)

def empty_fig(msg="차트 없음"):
    import plotly.graph_objects as go
    f = go.Figure()
    f.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False)
    f.update_layout(height=200, template="plotly_white")
    return f


# =========================
# IMPORT SAFE
# =========================
_IMPORT_ERROR = None
try:
    from core.pipeline import DAIAPipeline
except Exception as e:
    import traceback
    _IMPORT_ERROR = traceback.format_exc()

CONFIG_PATH = "config/config.yaml"
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DAIAPipeline(CONFIG_PATH)
    return _pipeline


# =========================
# ERROR RETURN (핵심)
# =========================
def error_return(msg):
    return (empty_fig(),) * 14 + (
        pd.DataFrame({"error": [msg]}),
        msg, "", "", "", msg
    )


# =========================
# MAIN
# =========================
def run_analysis(file_obj, schema_choice, no_llm, progress=gr.Progress()):

    if _IMPORT_ERROR:
        return error_return(_IMPORT_ERROR[:300])

    if file_obj is None:
        return error_return("파일 업로드 필요")

    try:
        pipeline = get_pipeline()

        # 실행
        out_path = pipeline.run(file_obj.name)

        figs = pipeline.eda_engine._chart_figs

        def get_fig(k):
            return figs[k] if k in figs else empty_fig()

        # ===== 안전 처리 =====
        stats_df = safe_df(pipeline.last_stats, "통계 없음")
        quality = pipeline.last_quality

        if isinstance(quality, pd.DataFrame) and not quality.empty:
            quality_md = quality.to_markdown()
        else:
            quality_md = "품질 정보 없음"

        # feature
        feat_report = pipeline.last_feature_report
        feat_md = ""
        if feat_report:
            feat_md = "\n".join(map(str, feat_report.log))

        # proposals
        proposal_md = ""
        proposals = pipeline.last_proposals
        if proposals:
            for p in proposals:
                proposal_md += f"""
### {safe_md(p.get('type'))}
- target: {safe_md(p.get('target'))}
- features: {safe_list_to_str(p.get('features'))}
"""

        # insights
        insights = pipeline.last_insights
        insight_md = "\n".join(map(str, insights)) if insights else "없음"

        # schema
        schema = pipeline.last_schema
        schema_md = safe_md(schema)

        return (
            get_fig("data_overview"),
            get_fig("missing_analysis"),
            get_fig("distributions"),
            get_fig("box_plots"),
            get_fig("correlations"),
            get_fig("pca_analysis"),
            get_fig("pca_loadings"),
            get_fig("target_analysis"),
            get_fig("feature_importance"),
            get_fig("cluster_distribution"),
            get_fig("cluster_visualization"),
            get_fig("categorical_dist"),
            get_fig("time_series"),
            get_fig("event_timeline"),

            stats_df,
            safe_md(quality_md),
            safe_md(feat_md),
            safe_md(proposal_md),
            safe_md(insight_md),
            safe_md(schema_md),
            "완료"
        )

    except Exception as e:
        import traceback
        return error_return(traceback.format_exc()[:500])


# =========================
# UI
# =========================
with gr.Blocks() as demo:

    file_in = gr.File()
    schema_dd = gr.Dropdown(["자동 감지"], value="자동 감지")
    no_llm = gr.Checkbox()

    run_btn = gr.Button("실행")

    plots = [gr.Plot() for _ in range(14)]

    stats_table = gr.Dataframe()
    quality_md = gr.Markdown()
    feat_md = gr.Markdown()
    proposal_md = gr.Markdown()
    insight_md = gr.Markdown()
    schema_md = gr.Markdown()
    log_md = gr.Markdown()

    run_btn.click(
        fn=run_analysis,
        inputs=[file_in, schema_dd, no_llm],
        outputs=plots + [
            stats_table,
            quality_md,
            feat_md,
            proposal_md,
            insight_md,
            schema_md,
            log_md
        ]
    )

if __name__ == "__main__":
    demo.launch()