"""
DAIA v5 - Gradio UI  [클라우드 배포판]
Hugging Face Spaces + Gemini API + Supabase 사용량 제한

로컬 실행:
    conda activate daia_v3
    python app.py

클라우드 필수 환경변수 (HF Spaces → Settings → Secrets):
    GEMINI_API_KEY   — Gemini API 키
    SUPABASE_URL     — Supabase 프로젝트 URL
    SUPABASE_KEY     — Supabase anon key
    DAILY_LIMIT      — 하루 최대 사용 횟수 (기본 5)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 로컬 개발: .env 자동 로드 ─────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import gradio as gr
import pandas as pd
import yaml
import tempfile
from pathlib import Path
from core.pipeline import DAIAPipeline
from visualization.quick_plot import quick_plot
from report.pdf_exporter import PDFExporter
from utils.rate_limiter import check_and_increment, get_remaining, DAILY_LIMIT


# ── 로고 ─────────────────────────────────────────────────────────────────────
def _load_logo_html():
    import base64
    png_path = Path(__file__).parent / "assets" / "logo-title.png"
    try:
        b64 = base64.b64encode(png_path.read_bytes()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="height:88px; display:block;">'
    except Exception:
        return '<span style="color:#fff; font-size:1.4rem; font-weight:700;">EVA NEXIO.DAIA</span>'

_LOGO_HTML = _load_logo_html()

_last_html_path: Path  = None
_last_charts_dir: Path = None
_last_pipeline_ref     = None
_last_auto_result_list = []

CONFIG_PATH = "config/config.yaml"
_pipeline: DAIAPipeline | None = None


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DAIAPipeline(CONFIG_PATH)
    return _pipeline


def _schema_choices():
    return ["자동 감지", "signal_pool", "time_series", "wide_table", "cross_tab"]


# ── 메인 분석 함수 (request: gr.Request 추가) ─────────────────────────────────
def run_analysis(file_obj, schema_choice, no_llm, request: gr.Request, progress=gr.Progress()):
    global _last_html_path, _last_charts_dir, _last_pipeline_ref, _last_auto_result_list
    _last_auto_result_list = []

    # ── 빈 결과 템플릿 ────────────────────────────────────────────────────────
    def _empty_return(msg: str):
        import plotly.graph_objects as go
        def _ef():
            f = go.Figure()
            f.update_layout(height=200, template="plotly_white")
            return f
        return (_ef(),) * 14 + (pd.DataFrame(), "", "", "", "", msg)

    if file_obj is None:
        return _empty_return("❌ 파일을 먼저 업로드하세요.")

    # ── 사용량 제한 체크 ──────────────────────────────────────────────────────
    allowed, used, limit = check_and_increment(request)
    if not allowed:
        return _empty_return(
            f"⛔ 오늘 사용 한도 초과 ({limit}회/일). 내일 다시 이용해 주세요."
        )

    pipeline = _get_pipeline()
    cfg = pipeline._load_config(CONFIG_PATH)
    cfg["report"]["open_browser"] = False   # 클라우드에서 브라우저 열기 비활성
    if no_llm:
        cfg["llm"] = cfg.get("llm", {})
    if schema_choice != "자동 감지":
        cfg.setdefault("schema", {})["force_schema"] = schema_choice

    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
    yaml.dump(cfg, tmp, allow_unicode=True)
    tmp.close()

    try:
        pipeline.config = cfg
        pipeline.config_path = tmp.name
        from core.schema_detector import SchemaDetector
        from preprocessing.preprocessor import SmartPreprocessor
        from preprocessing.data_quality import DataQualityValidator
        from preprocessing.feature_engineer import FeatureEngineer
        from visualization.eda_engine import EDAEngine
        from report.report_builder import ReportBuilder
        pipeline.detector          = SchemaDetector(cfg)
        pipeline.preprocessor      = SmartPreprocessor(cfg)
        pipeline.quality_validator = DataQualityValidator(cfg)
        pipeline.feature_engineer  = FeatureEngineer(cfg)
        pipeline.eda_engine        = EDAEngine(cfg)
        pipeline.reporter          = ReportBuilder(cfg)
        if no_llm:
            pipeline.llm._available = False

        steps = []
        # 남은 횟수 안내
        remaining = limit - used
        steps.append(f"ℹ️ 오늘 남은 분석 횟수: {remaining}회")

        def prog_cb(msg, pct):
            steps.append(msg)
            progress(pct, desc=msg)

        out_path = pipeline.run(file_obj.name, progress_cb=prog_cb)

        charts      = pipeline.last_charts
        schema      = pipeline.last_schema
        stats_df    = pipeline.last_stats
        quality     = pipeline.last_quality
        feat_report = pipeline.last_feature_report
        proposals   = pipeline.last_proposals
        insights    = pipeline.last_insights
        csv_path    = pipeline.last_csv_path

        figs = pipeline.eda_engine._chart_figs

        def get_fig(key, fallback_keys=None):
            import plotly.graph_objects as go
            keys_to_try = [key] + (fallback_keys or [])
            for k in keys_to_try:
                if k in figs:
                    return figs[k]
            empty = go.Figure()
            empty.add_annotation(text="차트 없음", x=0.5, y=0.5,
                                  showarrow=False, font_size=16)
            empty.update_layout(height=200, template="plotly_white")
            return empty

        c_overview  = get_fig("data_overview")
        c_missing   = get_fig("missing_analysis")
        c_dist      = get_fig("distributions")
        c_box       = get_fig("box_plots")
        c_corr      = get_fig("correlations")
        c_pca       = get_fig("pca_analysis")
        c_pca_load  = get_fig("pca_loadings")
        c_target    = get_fig("target_analysis")
        c_feat_imp  = get_fig("feature_importance")
        c_cluster   = get_fig("cluster_distribution", ["cluster_visualization"])
        c_cat       = get_fig("categorical_dist")
        c_ts        = get_fig("time_series")
        c_event     = get_fig("event_timeline")
        c_cluster_v = get_fig("cluster_visualization")

        stats_show = stats_df if stats_df is not None and len(stats_df) else pd.DataFrame({"메시지": ["수치형 컬럼 없음"]})
        quality_md  = quality.to_markdown() if quality else "품질 정보 없음"

        feat_md = ""
        if feat_report:
            feat_md = f"**추가 피처 수:** {len(feat_report.added_features)}\n\n"
            feat_md += "\n".join(f"- {l}" for l in feat_report.log) or "- 추가된 피처 없음"
            if feat_report.added_features:
                feat_md += f"\n\n**피처 목록:** {', '.join(feat_report.added_features[:10])}"

        proposal_md = ""
        if proposals:
            for p in proposals:
                proposal_md += f"### {p['icon']} {p['type']}\n"
                proposal_md += f"**설명:** {p['desc']}\n\n"
                proposal_md += f"**알고리즘:** {p['algorithm']}\n\n"
                proposal_md += f"**타겟:** {p['target']}\n\n"
                proposal_md += f"**입력 변수:** {p['features']}\n\n---\n"

        insight_md = "\n".join(f"- {i}" for i in insights) or "인사이트 없음"

        global _last_html_path, _last_charts_dir
        _last_html_path    = out_path
        _last_pipeline_ref = pipeline
        out_dir = Path(pipeline.config.get("report", {}).get("output_dir", "/tmp/daia_output"))
        _last_charts_dir = out_dir / "charts" / Path(file_obj.name).stem

        schema_md = f"""
**스키마 타입:** {schema.schema_type}
**신뢰도:** {schema.confidence:.0%}
**수치형 ({len(schema.numeric_cols)}):** {', '.join(schema.numeric_cols[:8]) or '-'}
**범주형 ({len(schema.categorical_cols)}):** {', '.join(schema.categorical_cols[:8]) or '-'}
{"**신호명:** " + str(schema.signal_name_col) if schema.signal_name_col else ""}
{"**타임스탬프:** " + str(schema.timestamp_col) if schema.timestamp_col else ""}
**HTML 리포트:** `{out_path}`
{"**분석 CSV:** `" + str(csv_path) + "`" if csv_path else ""}
""".strip()

        log_md = "\n".join(f"- {s}" for s in steps)

        return (c_overview, c_missing, c_dist, c_box,
                c_corr, c_pca, c_pca_load, c_target, c_feat_imp,
                c_cluster, c_cluster_v, c_cat, c_ts, c_event,
                stats_show, quality_md, feat_md, proposal_md,
                insight_md, schema_md, log_md)

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        import plotly.graph_objects as go
        def _ef(msg="오류"):
            f = go.Figure()
            f.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                             font=dict(color="red", size=13))
            f.update_layout(height=200, template="plotly_white")
            return f
        ef = _ef()
        return (ef,) * 14 + (pd.DataFrame(), f"오류: {e}", err, err, err, err, err)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ── 자동 분석 함수 ────────────────────────────────────────────────────────────
def run_auto_analysis(analysis_type_label, target_input, progress=gr.Progress()):
    import plotly.graph_objects as go

    label_map = {
        "📈 회귀 분석 (Regression)":          "regression",
        "🎯 분류 분석 (Classification)":       "classification",
        "🔵 군집 분석 (Clustering)":           "clustering",
        "🔴 이상 탐지 (Anomaly Detection)":    "anomaly",
        "⏱ 시계열 예측 (Time Series)":         "timeseries",
        "🔗 연관 분석 (Association Analysis)":  "association",
    }
    atype = label_map.get(analysis_type_label, "clustering")

    def _empty_fig(msg):
        f = go.Figure()
        f.add_annotation(text=msg, x=0.5, y=0.5, showarrow=False,
                         font=dict(size=14, color="#888"))
        f.update_layout(height=200, template="plotly_white")
        return f

    p = _get_pipeline()
    if p.last_df_feature is None:
        msg = "❌ 먼저 메인 분석(🚀 분석 시작)을 실행하세요."
        return (_empty_fig("분석 데이터 없음"),) * 4 + (msg,)

    progress(0.1, desc=f"🔬 {analysis_type_label} 실행 중...")
    target_col = target_input.strip() if target_input and target_input.strip() else None

    global _last_auto_result_list
    result = p.run_analysis(atype, target_col=target_col)
    progress(0.9, desc="📊 결과 정리 중...")

    if not result.success:
        msg = f"❌ 분석 실패: {result.error_msg}"
        return (_empty_fig("실패"),) * 4 + (msg,)

    _last_auto_result_list.append(result)

    def gf(key):
        return result.figures.get(key, _empty_fig(f"{key} 없음"))

    chart_keys = {
        "regression":     ["model_comparison", "actual_vs_pred", "residuals", "feature_importance"],
        "classification": ["model_comparison", "confusion_matrix", "class_distribution", "feature_importance"],
        "clustering":     ["elbow_silhouette", "cluster_scatter", "cluster_profile", "cluster_size"],
        "anomaly":        ["anomaly_scatter", "anomaly_scores", "anomaly_profile", "anomaly_scores"],
        "timeseries":     ["model_comparison", "ts_prediction", "residuals", "feature_importance"],
        "association":    ["assoc_top_rules", "assoc_scatter", "assoc_freq_items", "assoc_heatmap"],
    }
    keys = chart_keys.get(atype, ["model_comparison"] * 4)
    figs = [gf(k) for k in keys]

    metric_md = "### 📊 성능 지표\n"
    for k, v in result.metrics.items():
        if k == "anomaly_indices":
            continue
        metric_md += f"- **{k}:** {v}\n"
    metric_md += f"\n### 📝 요약\n{result.summary}\n"
    metric_md += "\n### 💬 개선 추천\n"
    for r in result.recommendations:
        metric_md += f"- {r}\n"

    progress(1.0, desc=f"✅ {analysis_type_label} 완료")
    return tuple(figs) + (metric_md,)


# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="DAIA",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    css="""
    .run-btn { background:#27ae60 !important; }
    #progress-area { padding:4px 0; min-height:0; }
    #tabs-col .progress-bar-wrap { display: none !important; }
    #tabs-col .generating         { display: none !important; }
    #tabs-col .eta-bar            { display: none !important; }
    #tabs-col .progress-bar       { display: none !important; }
    """
) as demo:

    gr.HTML(f"""
        <style>
        .header-box {{
            background: #000;
            border-radius: 12px;
            padding: 2px 24px 8px 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        .header-sub {{
            margin-top: -4px;
            color: #ffffff !important;
            font-size: 0.95rem;
            padding-left: 4px;
            text-align: center;
        }}
        </style>
        <div class="header-box">
        {_LOGO_HTML}
        <div class="header-sub">
            데이터 자동분석 | 품질 검증 · 피처 엔지니어링 · 분석 제안 · 연관 분석 · CSV 내보내기
        </div>
        </div>
        """)

    with gr.Row():
        # ── 왼쪽 설정 패널 ────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=260):
            gr.Markdown("### ⚙️ 분석 설정")
            file_in   = gr.File(label="📁 데이터 파일 (CSV / XLSX / JSON)",
                                file_types=[".csv", ".xlsx", ".xls", ".json"])
            schema_dd = gr.Dropdown(_schema_choices(), value="자동 감지", label="스키마 선택")
            no_llm_cb = gr.Checkbox(label="LLM 없이 실행 (빠른 분석)", value=False)
            run_btn   = gr.Button("🚀 분석 시작", variant="primary", elem_classes="run-btn")
            gr.Markdown(f"""
---
**일일 사용 제한:** {DAILY_LIMIT}회/일 (IP 기준)

**LLM:** Gemini API (클라우드)

**출력** (`/tmp/daia_output/`)
- `daia_report_*.html` — HTML 리포트
- `datasets/` — 분석용 CSV
""")

        # ── 오른쪽 탭 패널 ────────────────────────────────────────────────
        with gr.Column(scale=4, elem_id="tabs-col"):
            with gr.Tabs():

                with gr.Tab("📊 데이터 개요"):
                    overview_plot = gr.Plot()

                with gr.Tab("❓ 결측치"):
                    missing_plot = gr.Plot()

                with gr.Tab("📈 분포"):
                    dist_plot = gr.Plot()

                with gr.Tab("📦 박스플롯"):
                    box_plot = gr.Plot()

                with gr.Tab("🔗 상관관계"):
                    corr_plot = gr.Plot()

                with gr.Tab("🔬 PCA"):
                    pca_plot = gr.Plot()

                with gr.Tab("📐 PCA 변수 기여도"):
                    pca_loading_plot = gr.Plot()

                with gr.Tab("🎯 타겟 분석"):
                    target_plot = gr.Plot()

                with gr.Tab("⭐ 피처 중요도"):
                    feat_imp_plot = gr.Plot()

                with gr.Tab("🔵 군집 분포"):
                    cluster_plot = gr.Plot()

                with gr.Tab("🗺️ 군집 시각화"):
                    cluster_viz_plot = gr.Plot()

                with gr.Tab("🏷️ 범주형"):
                    cat_plot = gr.Plot()

                with gr.Tab("⏱ 시계열"):
                    ts_plot = gr.Plot()

                with gr.Tab("📅 이벤트"):
                    event_plot = gr.Plot()

                with gr.Tab("📋 기술통계"):
                    stats_table = gr.Dataframe(label="기술통계 (왜도·첨도 포함)", wrap=True)

                with gr.Tab("🔬 데이터 품질"):
                    quality_md_out = gr.Markdown()

                with gr.Tab("⚗️ Feature Engineering"):
                    feat_md_out = gr.Markdown()

                with gr.Tab("🎯 분석 제안"):
                    proposal_md_out = gr.Markdown()

                with gr.Tab("🤖 자동 분석 실행"):
                    gr.Markdown("### 추천 분석을 바로 실행합니다")
                    with gr.Row():
                        auto_type_dd = gr.Dropdown(
                            choices=[
                                "📈 회귀 분석 (Regression)",
                                "🎯 분류 분석 (Classification)",
                                "🔵 군집 분석 (Clustering)",
                                "🔴 이상 탐지 (Anomaly Detection)",
                                "⏱ 시계열 예측 (Time Series)",
                                "🔗 연관 분석 (Association Analysis)",
                            ],
                            value="🔵 군집 분석 (Clustering)",
                            label="분석 유형 선택",
                        )
                        auto_target_in = gr.Textbox(
                            label="타겟 컬럼 (선택 — 비워두면 자동 감지)",
                            placeholder="예: PassOrFail, price, label ...",
                        )
                    auto_run_btn = gr.Button("▶ 분석 실행", variant="primary")
                    with gr.Row():
                        auto_chart1 = gr.Plot(label="차트 1")
                        auto_chart2 = gr.Plot(label="차트 2")
                    with gr.Row():
                        auto_chart3 = gr.Plot(label="차트 3")
                        auto_chart4 = gr.Plot(label="차트 4")
                    auto_result_md = gr.Markdown()

                    auto_run_btn.click(
                        fn=run_auto_analysis,
                        inputs=[auto_type_dd, auto_target_in],
                        outputs=[auto_chart1, auto_chart2,
                                 auto_chart3, auto_chart4, auto_result_md],
                        show_progress="minimal",
                    )

                with gr.Tab("💡 인사이트"):
                    insight_md_out = gr.Markdown()

                with gr.Tab("ℹ️ 분석 정보"):
                    schema_out = gr.Markdown()
                    log_out    = gr.Markdown()

                with gr.Tab("📊 빠른 플롯"):
                    gr.Markdown("### 컬럼 선택 즉시 시각화")
                    with gr.Row():
                        qp_file  = gr.File(label="CSV", file_types=[".csv"])
                        qp_type  = gr.Dropdown(["scatter", "line", "box", "histogram", "heatmap"],
                                               value="scatter", label="플롯 타입")
                    with gr.Row():
                        qp_x     = gr.Textbox(label="X 컬럼")
                        qp_y     = gr.Textbox(label="Y 컬럼")
                        qp_color = gr.Textbox(label="색상 컬럼 (선택)")
                    qp_btn = gr.Button("▶ 플롯", variant="secondary")
                    qp_out = gr.HTML()

                    def do_quick_plot(f, x, y, color, ptype):
                        if f is None:
                            return "<p style='color:red'>파일 업로드 필요</p>"
                        try:
                            df = pd.read_csv(f.name, nrows=50000)
                            color = color.strip() if color and color.strip() in df.columns else None
                            return quick_plot(df, x.strip(), y.strip(), color, ptype)
                        except Exception as e:
                            return f"<p style='color:red'>오류: {e}</p>"

                    qp_btn.click(fn=do_quick_plot,
                                 inputs=[qp_file, qp_x, qp_y, qp_color, qp_type],
                                 outputs=qp_out)

                with gr.Tab("📄 PDF 내보내기"):
                    gr.Markdown("### 분석 완료 후 PDF 변환")
                    pdf_btn    = gr.Button("📄 PDF 생성", variant="secondary")
                    pdf_status = gr.Markdown()

                    def do_pdf_export():
                        global _last_html_path, _last_charts_dir, _last_auto_result_list
                        if _last_html_path is None:
                            return "❌ 먼저 분석을 실행하세요."
                        try:
                            from datetime import datetime
                            p        = _get_pipeline()
                            exporter = PDFExporter()
                            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
                            stem     = _last_html_path.stem.replace("daia_report_", "")
                            has_auto = bool(_last_auto_result_list)
                            suffix   = "_full_" + ts if has_auto else "_" + ts
                            pdf_path = (_last_html_path.parent
                                        / ("daia_report_" + stem + suffix + ".pdf"))
                            pdf_path = exporter.export(
                                _last_html_path, _last_charts_dir,
                                output_path=pdf_path,
                                eda_engine=p.eda_engine,
                                stats_df=p.last_stats,
                                quality=p.last_quality,
                                feat_report=p.last_feature_report,
                                proposals=p.last_proposals,
                                auto_result=_last_auto_result_list,
                            )
                            label = "EDA+자동분석 통합" if has_auto else "EDA 분석"
                            return f"✅ PDF 저장 ({label}): `{pdf_path}`"
                        except RuntimeError as e:
                            return f"❌ {e}"
                        except Exception as e:
                            return f"❌ 오류: {e}"

                    pdf_btn.click(fn=do_pdf_export, outputs=pdf_status)

    # ── 하단 진행 로그 ────────────────────────────────────────────────────────
    with gr.Row(visible=True):
        gr.HTML("<div style='height:4px'></div>")
    with gr.Row():
        progress_log = gr.Markdown(value="", elem_id="progress-area", label=None)

    # ── 버튼 연결 ─────────────────────────────────────────────────────────────
    _outputs = [
        overview_plot, missing_plot, dist_plot, box_plot,
        corr_plot, pca_plot, pca_loading_plot, target_plot, feat_imp_plot,
        cluster_plot, cluster_viz_plot, cat_plot, ts_plot, event_plot,
        stats_table,
        quality_md_out, feat_md_out, proposal_md_out,
        insight_md_out, schema_out, progress_log,
    ]

    run_btn.click(
        fn=run_analysis,
        inputs=[file_in, schema_dd, no_llm_cb],   # request는 Gradio가 자동 주입
        outputs=_outputs,
        show_progress="minimal",
    )


if __name__ == "__main__":
    print("=" * 55)
    print("  DAIA v5 - Gradio UI (클라우드 배포판)")
    print("=" * 55)
    demo.launch()
