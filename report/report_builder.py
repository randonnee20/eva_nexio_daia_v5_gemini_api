"""
DAIA v4 - 리포트 빌더
추가: 품질 점수 / 데이터 유형 / 피처 엔지니어링 로그 / 분석 제안 섹션
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from core.schema_detector import SchemaInfo
from utils.logger import get_logger

logger = get_logger()

_SCHEMA_NAMES = {
    "signal_pool" : "Signal Pool (Narrow Format)",
    "time_series" : "Time Series",
    "wide_table"  : "Wide Table",
    "cross_tab"   : "Cross Tab",
}


class ReportBuilder:
    def __init__(self, config: dict = None):
        cfg = (config or {}).get("report", {})
        self.out_dir = Path(cfg.get("output_dir", "./daia_output"))

    def build(self, df_raw, df_proc, schema, charts, insights,
              preproc_log, llm_text, source_name,
              stats_df=None, quality=None, feat_report=None,
              auto_result=None,
              proposals=None) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.out_dir / f"daia_report_{source_name}_{ts}.html"
        html = self._build_html(df_raw, df_proc, schema, charts, insights,
                                preproc_log, llm_text, source_name, stats_df,
                                quality, feat_report, proposals, auto_result)
        out.write_text(html, encoding="utf-8")
        logger.info(f"💾 리포트 저장: {out}")
        return out

    def _build_html(self, df_raw, df_proc, schema, charts, insights,
                    preproc_log, llm_text, name, stats_df,
                    quality, feat_report, proposals, auto_result=None):
        ts_str       = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        schema_display = _SCHEMA_NAMES.get(schema.schema_type, schema.schema_type)
        miss_rate    = df_raw.isnull().mean().mean() * 100
        dup_count    = df_raw.duplicated().sum()
        mem_mb       = df_raw.memory_usage(deep=True).sum() / 1024**2
        q_score      = quality.quality_score if quality else 0
        typology     = quality.data_typology if quality else schema_display

        # 기술통계
        stats_html = ""
        if stats_df is not None and len(stats_df):
            stats_html = stats_df.to_html(
                classes="stats-table", border=0,
                float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x))

        # 차트 섹션
        chart_order = [
            ("data_overview",        "📊 데이터 개요"),
            ("missing_analysis",     "❓ 결측치 분석"),
            ("distributions",        "📈 변수 분포"),
            ("box_plots",            "📦 박스플롯 (이상값)"),
            ("correlations",         "🔗 상관관계"),
            ("pca_analysis",         "🔬 PCA 주성분 분석"),
            ("target_analysis",      "🎯 타겟 변수"),
            ("feature_importance",   "⭐ 피처 중요도"),
            ("cluster_distribution", "🔵 군집 크기"),
            ("cluster_visualization","🗺️ 군집 시각화"),
            ("categorical_dist",     "🏷️ 범주형 분포"),
            ("time_series",          "⏱ 시계열"),
            ("event_timeline",       "📅 이벤트 타임라인"),
        ]
        chart_sections = ""
        for key, title in chart_order:
            if key in charts:
                chart_sections += f"""
                <section class="chart-section" id="{key}">
                  <h2>{title}</h2>
                  <div class="chart-wrapper">{charts[key]}</div>
                </section>"""

        # 전처리 로그
        preproc_items = "".join(f"<li>{s}</li>" for s in preproc_log)

        # 피처 엔지니어링 로그
        feat_items = ""
        if feat_report:
            for log in feat_report.log:
                feat_items += f"<li>{log}</li>"
            if feat_report.added_features:
                feat_items += f"<li>➕ 추가된 피처 ({len(feat_report.added_features)}개): " \
                              f"{', '.join(feat_report.added_features[:8])}{'...' if len(feat_report.added_features)>8 else ''}</li>"

        # 품질 이슈
        quality_items = ""
        if quality:
            sev_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            for issue in quality.issues[:20]:
                icon = sev_icon.get(issue.severity, "⚪")
                quality_items += f"<li>{icon} [{issue.category}] <code>{issue.column}</code>: {issue.detail}</li>"
            for t in quality.type_issues[:5]:
                quality_items += f"<li>🔶 [type_error] {t}</li>"

        # 인사이트
        insight_items = "".join(f"<li>{s}</li>" for s in insights)

        # 분석 제안 카드
        proposal_cards = ""
        if proposals:
            for p in proposals:
                proposal_cards += f"""
                <div class="proposal-card">
                  <div class="proposal-icon">{p['icon']}</div>
                  <div class="proposal-content">
                    <div class="proposal-type">{p['type']}</div>
                    <div class="proposal-desc">{p['desc']}</div>
                    <div class="proposal-detail">
                      <span class="pd-label">알고리즘</span> {p['algorithm']}
                    </div>
                    <div class="proposal-detail">
                      <span class="pd-label">타겟</span> {p['target']}
                    </div>
                    <div class="proposal-detail">
                      <span class="pd-label">변수</span> {p['features']}
                    </div>
                  </div>
                </div>"""

        # 컬럼 프로파일
        profile_rows = ""
        for col in df_raw.columns[:50]:
            dtype   = str(df_raw[col].dtype)
            miss    = df_raw[col].isnull().mean() * 100
            nuniq   = df_raw[col].nunique()
            sample  = str(df_raw[col].dropna().iloc[:3].tolist())[:60] if df_raw[col].dropna().shape[0] else "-"
            miss_cls = "high-miss" if miss > 50 else ("mid-miss" if miss > 20 else "")
            profile_rows += f"""
            <tr>
              <td>{col}</td>
              <td><span class="dtype-tag">{dtype}</span></td>
              <td class="{miss_cls}">{miss:.1f}%</td>
              <td>{nuniq:,}</td>
              <td class="sample-val">{sample}</td>
            </tr>"""

        # ── 자동 분석 결과 섹션 ──────────────────────────────────────────────
        auto_html = ""
        if auto_result and auto_result.success:
            ar = auto_result
            type_labels = {
                "regression":     "📈 회귀 분석 결과",
                "classification": "🎯 분류 분석 결과",
                "clustering":     "🔵 군집 분석 결과",
                "anomaly":        "🔴 이상 탐지 결과",
                "timeseries":     "⏱ 시계열 예측 결과",
            }
            ar_title = type_labels.get(ar.analysis_type, "자동 분석 결과")
            # 지표 카드
            metric_cards = ""
            for k, v in ar.metrics.items():
                if k == "anomaly_indices": continue
                metric_cards += f"""
                <div class="metric-card">
                  <div class="val">{v}</div>
                  <div class="lbl">{k}</div>
                </div>"""
            # 추천 사항
            rec_items = "".join(f"<li>{r}</li>" for r in ar.recommendations)
            # 차트 HTML (plotly div)
            chart_divs = ""
            chart_titles = {
                "model_comparison":   "모델 성능 비교",
                "actual_vs_pred":     "실제 vs 예측",
                "residuals":          "잔차 분석",
                "feature_importance": "피처 중요도",
                "confusion_matrix":   "혼동 행렬",
                "class_distribution": "클래스 분포",
                "elbow_silhouette":   "최적 군집 수 탐색",
                "cluster_scatter":    "군집 시각화",
                "cluster_profile":    "군집 프로파일",
                "cluster_size":       "군집별 샘플 수",
                "anomaly_scatter":    "이상 탐지 산점도",
                "anomaly_scores":     "이상 점수",
                "anomaly_profile":    "정상 vs 이상 비교",
                "ts_prediction":      "시계열 예측",
            }
            for key, title in chart_titles.items():
                if key in ar.figures:
                    fig_html = ar.figures[key].to_html(
                        full_html=False, include_plotlyjs=False,
                        config={"displayModeBar": True})
                    chart_divs += f"""
                    <div class="auto-chart">
                      <h4>{title}</h4>
                      <div class="chart-wrapper">{fig_html}</div>
                    </div>"""
            auto_html = f"""
            <section id="auto_analysis" style="border-left:4px solid #27ae60;">
              <h2 style="border-color:#27ae60;">{ar_title}
                <span style="background:#27ae60;color:#fff;font-size:.72rem;
                             padding:2px 10px;border-radius:10px;vertical-align:middle;
                             margin-left:8px;">{ar.model_name}</span>
              </h2>
              <p style="color:#555;font-size:.9rem;margin-bottom:14px;">
                <b>타겟:</b> {ar.target_col or "비지도"} &nbsp;|&nbsp;
                <b>입력 변수:</b> {len(ar.feature_cols)}개
              </p>
              <!-- 요약 -->
              <div style="background:#f0fff4;padding:14px 18px;border-radius:8px;
                          margin-bottom:16px;font-size:.92rem;line-height:1.9;">
                {ar.summary.replace(chr(10),"<br>")}
              </div>
              <!-- 지표 카드 -->
              <div class="metric-grid" style="margin-bottom:18px;">{metric_cards}</div>
              <!-- 차트 -->
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px;">
                {chart_divs}
              </div>
              <!-- 추천 -->
              {"<h3 style='margin-top:18px;'>💬 개선 추천사항</h3><ul class='insights-list'>" + rec_items + "</ul>" if rec_items else ""}
            </section>"""

        # LLM
        def llm_section(key, title, icon):
            txt = llm_text.get(key, "").strip()
            if not txt or txt.startswith("[LLM"): return ""
            txt_html = txt.replace("\n", "<br>")
            return f"""<section class="llm-section">
              <h2>{icon} {title} <span class="llm-badge">AI</span></h2>
              <div class="llm-content">{txt_html}</div></section>"""
        llm_html = (llm_section("schema","데이터 해석","🤖") +
                    llm_section("preprocessing","전처리 권장사항","🔧") +
                    llm_section("insights","AI 인사이트","💡"))
        if not llm_html:
            llm_html = """<section class="llm-section no-llm">
              <p>⚠️ LLM 비활성. <code>models/</code>에 GGUF 모델 배치 후 활성화됩니다.</p></section>"""

        # nav items
        nav_items = ["요약","스키마","품질","기술통계","컬럼프로파일","피처","분석제안","차트","자동분석","전처리","인사이트","AI분석"]
        nav_ids   = ["summary","schema","quality","stats","profile","features","proposals","data_overview","auto_analysis","preproc","insights","llm"]
        nav_html  = "".join(f'<li><a href="#{i}">{n}</a></li>' for n, i in zip(nav_items, nav_ids))

        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DAIA v4 - {name}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  :root {{
    --primary:#2c3e50;--accent:#3498db;--success:#27ae60;
    --warn:#e67e22;--danger:#e74c3c;--bg:#f0f2f5;--card:#fff;
    --text:#2c3e50;--border:#dee2e6;--radius:12px;
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:"Segoe UI","Malgun Gothic",sans-serif;background:var(--bg);color:var(--text);line-height:1.6}}
  .header{{background:linear-gradient(135deg,#1a2533,#2980b9);color:#fff;padding:28px 40px}}
  .header h1{{font-size:1.9rem;font-weight:700}}
  .header .meta{{opacity:.8;margin-top:4px;font-size:.88rem}}
  .schema-badge{{display:inline-block;background:rgba(255,255,255,.18);border-radius:20px;padding:4px 14px;margin-top:8px;font-size:.82rem;font-weight:600}}
  .nav{{background:var(--card);border-bottom:1px solid var(--border);padding:0 40px;position:sticky;top:0;z-index:100;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
  .nav ul{{list-style:none;display:flex;gap:0;overflow-x:auto}}
  .nav a{{display:block;padding:12px 16px;color:var(--text);text-decoration:none;font-size:.85rem;white-space:nowrap;border-bottom:3px solid transparent;transition:all .2s}}
  .nav a:hover{{color:var(--accent);border-bottom-color:var(--accent)}}
  .container{{max-width:1440px;margin:0 auto;padding:24px 36px}}
  .metric-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin-bottom:28px}}
  .metric-card{{background:var(--card);border-radius:var(--radius);padding:18px;text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:4px solid var(--accent)}}
  .metric-card .val{{font-size:1.7rem;font-weight:700;color:var(--accent)}}
  .metric-card .lbl{{font-size:.78rem;color:#666;margin-top:4px}}
  section{{background:var(--card);border-radius:var(--radius);padding:24px 28px;margin-bottom:20px;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
  section h2{{font-size:1.15rem;font-weight:700;margin-bottom:16px;color:var(--primary);border-left:4px solid var(--accent);padding-left:12px}}
  .chart-wrapper{{overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.86rem}}
  th{{background:var(--primary);color:#fff;padding:9px 11px;text-align:left;white-space:nowrap}}
  td{{padding:8px 11px;border-bottom:1px solid var(--border)}}
  tr:hover td{{background:#f0f7ff}}
  .dtype-tag{{background:#eaf4ff;color:var(--accent);padding:2px 8px;border-radius:10px;font-size:.78rem}}
  .high-miss{{color:var(--danger);font-weight:700}}
  .mid-miss{{color:var(--warn)}}
  .sample-val{{color:#666;font-size:.8rem;font-family:monospace}}
  .stats-table{{font-size:.82rem}}
  .preproc-log,.insights-list,.quality-list,.feat-list{{list-style:none;padding:0}}
  .preproc-log li,.feat-list li{{padding:7px 12px;border-left:3px solid var(--success);background:#f0fff4;margin-bottom:5px;border-radius:0 8px 8px 0;font-size:.88rem}}
  .insights-list li{{padding:7px 12px;border-left:3px solid var(--accent);background:#eaf4ff;margin-bottom:5px;border-radius:0 8px 8px 0;font-size:.88rem}}
  .quality-list li{{padding:7px 12px;border-left:3px solid var(--warn);background:#fffbf0;margin-bottom:5px;border-radius:0 8px 8px 0;font-size:.88rem}}
  .llm-section{{border-left:4px solid #9b59b6}}
  .llm-section h2{{border-color:#9b59b6}}
  .llm-badge{{background:#9b59b6;color:#fff;font-size:.68rem;padding:2px 8px;border-radius:10px;vertical-align:middle}}
  .llm-content{{white-space:pre-wrap;line-height:1.8;background:#faf8ff;padding:14px;border-radius:8px;font-size:.9rem}}
  .no-llm{{background:#fffbf0;border-left-color:var(--warn)}}
  .schema-info{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}}
  .schema-item{{background:#f8f9fa;padding:11px;border-radius:8px}}
  .schema-item .sk{{font-size:.75rem;color:#666;text-transform:uppercase;font-weight:600}}
  .schema-item .sv{{font-size:.92rem;font-weight:600;color:var(--text);margin-top:3px;word-break:break-all}}
  /* 분석 제안 */
  .proposals-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px;margin-top:4px}}
  .proposal-card{{background:linear-gradient(135deg,#f8f9fa,#eaf4ff);border-radius:10px;padding:16px;border-left:4px solid var(--accent);display:flex;gap:12px}}
  .proposal-icon{{font-size:2rem;flex-shrink:0;line-height:1}}
  .proposal-type{{font-weight:700;color:var(--primary);font-size:.95rem;margin-bottom:4px}}
  .proposal-desc{{font-size:.85rem;color:#555;margin-bottom:6px}}
  .proposal-detail{{font-size:.8rem;color:#666;margin-top:3px}}
  .pd-label{{background:var(--accent);color:#fff;border-radius:4px;padding:1px 6px;font-size:.72rem;margin-right:5px}}
  @media(max-width:768px){{.container{{padding:14px}}.header{{padding:18px}}.metric-card .val{{font-size:1.3rem}}}}
</style>
</head>
<body>
<div class="header">
  <h1>DAIA v4 분석 리포트</h1>
  <div class="meta">📁 {name} &nbsp;|&nbsp; ⏱ {ts_str}</div>
  <div class="schema-badge">🔍 {schema_display} &nbsp;|&nbsp; 🗂 {typology}</div>
</div>
<nav class="nav"><ul>{nav_html}</ul></nav>
<div class="container">

<!-- 요약 메트릭 -->
<div class="metric-grid" id="summary">
  <div class="metric-card"><div class="val">{len(df_raw):,}</div><div class="lbl">총 행 수</div></div>
  <div class="metric-card"><div class="val">{len(df_raw.columns)}</div><div class="lbl">원본 컬럼</div></div>
  <div class="metric-card"><div class="val">{len(df_proc.columns)}</div><div class="lbl">피처 컬럼</div></div>
  <div class="metric-card"><div class="val" style="color:{'var(--danger)' if miss_rate>30 else 'var(--warn)' if miss_rate>10 else 'var(--success)'}">{miss_rate:.1f}%</div><div class="lbl">결측치 비율</div></div>
  <div class="metric-card"><div class="val" style="color:{'var(--warn)' if dup_count>0 else 'var(--success)'}">{dup_count:,}</div><div class="lbl">중복 행</div></div>
  <div class="metric-card"><div class="val" style="color:{'var(--danger)' if q_score<50 else 'var(--warn)' if q_score<80 else 'var(--success)'}">{q_score:.0f}</div><div class="lbl">품질 점수/100</div></div>
  <div class="metric-card"><div class="val">{mem_mb:.1f} MB</div><div class="lbl">메모리</div></div>
</div>

<!-- 스키마 -->
<section id="schema">
  <h2>🔍 스키마 분석</h2>
  <div class="schema-info">
    <div class="schema-item"><div class="sk">스키마 타입</div><div class="sv">{schema_display}</div></div>
    <div class="schema-item"><div class="sk">신뢰도</div><div class="sv">{schema.confidence:.0%}</div></div>
    <div class="schema-item"><div class="sk">데이터 유형</div><div class="sv">{typology}</div></div>
    <div class="schema-item"><div class="sk">수치형 컬럼</div><div class="sv">{len(schema.numeric_cols)}개: {', '.join(schema.numeric_cols[:5])}{'...' if len(schema.numeric_cols)>5 else ''}</div></div>
    <div class="schema-item"><div class="sk">범주형 컬럼</div><div class="sv">{len(schema.categorical_cols)}개: {', '.join(schema.categorical_cols[:5])}{'...' if len(schema.categorical_cols)>5 else ''}</div></div>
    {"<div class='schema-item'><div class='sk'>신호명 컬럼</div><div class='sv'>" + str(schema.signal_name_col) + "</div></div>" if schema.signal_name_col else ""}
    {"<div class='schema-item'><div class='sk'>타임스탬프</div><div class='sv'>" + str(schema.timestamp_col) + "</div></div>" if schema.timestamp_col else ""}
  </div>
</section>

<!-- 품질 검증 -->
<section id="quality">
  <h2>🔬 데이터 품질 검증 (Data Quality Validation)</h2>
  <div class="schema-info" style="margin-bottom:14px">
    <div class="schema-item"><div class="sk">품질 점수</div><div class="sv" style="color:{'#e74c3c' if q_score<50 else '#e67e22' if q_score<80 else '#27ae60'}">{q_score:.1f} / 100</div></div>
    <div class="schema-item"><div class="sk">데이터 유형</div><div class="sv">{typology}</div></div>
    <div class="schema-item"><div class="sk">결측치 컬럼</div><div class="sv">{len(quality.missing_summary) if quality else 0}개</div></div>
    <div class="schema-item"><div class="sk">이상값 컬럼</div><div class="sv">{len(quality.outlier_summary) if quality else 0}개</div></div>
    <div class="schema-item"><div class="sk">중복 행</div><div class="sv">{quality.duplicate_count if quality else 0:,}개</div></div>
    <div class="schema-item"><div class="sk">타입 이슈</div><div class="sv">{len(quality.type_issues) if quality else 0}건</div></div>
  </div>
  {"<ul class='quality-list'>" + quality_items + "</ul>" if quality_items else "<p style='color:#27ae60;font-weight:600'>✅ 주요 품질 이슈 없음</p>"}
</section>

<!-- 기술통계 -->
<section id="stats">
  <h2>📋 기술통계 (Descriptive Statistics)</h2>
  {"<div class='chart-wrapper'>" + stats_html + "</div>" if stats_html else "<p style='color:#999'>수치형 컬럼 없음</p>"}
</section>

<!-- 컬럼 프로파일 -->
<section id="profile">
  <h2>🗂 컬럼 프로파일 (Data Profiling)</h2>
  <div class="chart-wrapper">
    <table>
      <thead><tr><th>컬럼명</th><th>타입</th><th>결측치</th><th>고유값</th><th>샘플 값</th></tr></thead>
      <tbody>{profile_rows}</tbody>
    </table>
  </div>
</section>

<!-- 피처 엔지니어링 -->
<section id="features">
  <h2>⚗️ Feature Engineering</h2>
  {"<ul class='feat-list'>" + feat_items + "</ul>" if feat_items else "<p style='color:#999'>추가된 피처 없음</p>"}
</section>

<!-- 분석 제안 -->
<section id="proposals">
  <h2>🎯 분석 방향 제안 (Analysis Recommendation)</h2>
  <p style="color:#555;margin-bottom:14px;font-size:.9rem">
    데이터 구조와 변수 특성을 바탕으로 적용 가능한 분석 방법을 제안합니다.
  </p>
  {"<div class='proposals-grid'>" + proposal_cards + "</div>" if proposal_cards else "<p style='color:#999'>분석 제안 없음</p>"}
</section>

<!-- 차트 -->
{chart_sections}

<!-- 전처리 로그 -->
<section id="preproc">
  <h2>⚙️ 전처리 로그 (Data Cleaning)</h2>
  {"<ul class='preproc-log'>" + preproc_items + "</ul>" if preproc_items else "<p style='color:#999'>전처리 없음</p>"}
</section>

<!-- 인사이트 -->
<section id="insights">
  <h2>💡 자동 인사이트</h2>
  {"<ul class='insights-list'>" + insight_items + "</ul>" if insight_items else "<p style='color:#999'>인사이트 없음</p>"}
</section>

<!-- AI 분석 -->
<div id="llm">{llm_html}</div>

</div>
<footer style="text-align:center;padding:20px;color:#999;font-size:.8rem">
  DAIA v4 &nbsp;|&nbsp; Generated {ts_str}
</footer>
</body></html>"""
        return html
