"""
DAIA v5 - PDF 내보내기
추가: 분석 제안(Analysis Proposal) 섹션 / 품질 검증 섹션 / Feature 로그
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime


class PDFExporter:

    def export(self, html_path: Path, charts_dir: Path,
               output_path: Path = None,
               eda_engine=None,
               stats_df=None,
               quality=None,
               feat_report=None,
               proposals=None,
               auto_result=None) -> Path:  # auto_result: 단일 또는 리스트
        if output_path is None:
            output_path = html_path.with_suffix(".pdf")

        png_files = []
        if charts_dir and charts_dir.exists():
            png_files = sorted(charts_dir.glob("*.png"))
        if eda_engine and not png_files:
            png_files = self._render_figures(eda_engine, charts_dir)

        return self._build(html_path, png_files, output_path,
                           stats_df, quality, feat_report, proposals, auto_result)

    def _render_figures(self, eda_engine, charts_dir: Path) -> list[Path]:
        charts_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        try:
            import kaleido  # noqa
            for name, fig in eda_engine._chart_figs.items():
                try:
                    p = charts_dir / f"{name}.png"
                    fig.write_image(str(p), width=1400, height=800, scale=1.5)
                    saved.append(p)
                except Exception as e:
                    print(f"[PDF] PNG 실패 {name}: {e}")
        except ImportError:
            print("[PDF] kaleido 미설치 → pip install kaleido")
        return saved

    def _build(self, html_path: Path, png_files: list[Path],
               output_path: Path, stats_df, quality, feat_report, proposals,
               auto_result=None) -> Path:  # auto_result: 단일 또는 리스트
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.lib import colors
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                            Image as RLImage, PageBreak,
                                            HRFlowable, Table, TableStyle,
                                            KeepTogether)
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from PIL import Image as PILImage
        except ImportError:
            raise RuntimeError("설치 필요: pip install reportlab pillow")

        # 한글 폰트
        KR, KRB = "KR", "KRB"
        _candidates = [
            (r"C:\Windows\Fonts\malgun.ttf",    r"C:\Windows\Fonts\malgunbd.ttf"),
            (r"C:\Windows\Fonts\NanumGothic.ttf", r"C:\Windows\Fonts\NanumGothicBold.ttf"),
            ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
        ]
        ok = False
        for reg, bold in _candidates:
            try:
                pdfmetrics.registerFont(TTFont(KR, reg))
                try:
                    pdfmetrics.registerFont(TTFont(KRB, bold))
                except Exception:
                    pdfmetrics.registerFont(TTFont(KRB, reg))
                ok = True; break
            except Exception:
                continue
        if not ok:
            KR = KRB = "Helvetica"

        W, H = A4
        margin   = 1.5 * cm
        usable_w = W - 2 * margin

        S = {
            "title": ParagraphStyle("T",  fontName=KRB, fontSize=18, spaceAfter=4,
                                    textColor=colors.HexColor("#1a2533")),
            "sub":   ParagraphStyle("S",  fontName=KR,  fontSize=9,
                                    textColor=colors.grey, spaceAfter=8),
            "h2":    ParagraphStyle("H2", fontName=KRB, fontSize=12,
                                    spaceBefore=14, spaceAfter=6,
                                    textColor=colors.HexColor("#2980b9")),
            "h3":    ParagraphStyle("H3", fontName=KRB, fontSize=10,
                                    spaceBefore=8,  spaceAfter=4,
                                    textColor=colors.HexColor("#2c3e50")),
            "body":  ParagraphStyle("B",  fontName=KR, fontSize=9, leading=15, spaceAfter=3),
            "small": ParagraphStyle("SM", fontName=KR, fontSize=8, textColor=colors.grey),
            "badge": ParagraphStyle("BD", fontName=KRB, fontSize=9,
                                    textColor=colors.HexColor("#2980b9")),
        }

        def HR():
            return HRFlowable(width=usable_w, thickness=0.5,
                               color=colors.HexColor("#bdc3c7"), spaceAfter=6)
        def H2(txt): return Paragraph(txt, S["h2"])
        def H3(txt): return Paragraph(txt, S["h3"])

        source = html_path.stem.replace("daia_report_","").rsplit("_20",1)[0]
        ts_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                leftMargin=margin, rightMargin=margin,
                                topMargin=margin, bottomMargin=margin)
        story = []

        # ── 표지 ──────────────────────────────────────────────────────────
        story += [
            Paragraph("DAIA v4 분석 리포트", S["title"]),
            Paragraph(f"{source}  |  {ts_str}", S["sub"]),
            HRFlowable(width=usable_w, thickness=2,
                       color=colors.HexColor("#2980b9"), spaceAfter=14),
        ]

        # ── HTML 구조화 파싱 ───────────────────────────────────────────────
        sections = self._parse_html(html_path)
        for sec in sections:
            kind = sec.get("type")
            if kind == "metrics":
                story.append(H2("📋 요약"))
                story.append(HR())
                items = sec["items"]
                rows  = []
                for i in range(0, len(items), 2):
                    pair = items[i:i+2]
                    row  = []
                    for val, lbl in pair:
                        cell = [Paragraph(f"<b>{val}</b>", ParagraphStyle(
                                    "mv", fontName=KRB, fontSize=13,
                                    textColor=colors.HexColor("#2980b9"))),
                                Paragraph(lbl, S["small"])]
                        row.append(cell)
                    if len(row) == 1: row.append("")
                    rows.append(row)
                if rows:
                    col_w = usable_w / 2
                    tbl   = Table(rows, colWidths=[col_w, col_w])
                    tbl.setStyle(TableStyle([
                        ("BOX",        (0,0),(-1,-1), 0.5, colors.HexColor("#dce3ea")),
                        ("INNERGRID",  (0,0),(-1,-1), 0.3, colors.HexColor("#dce3ea")),
                        ("VALIGN",     (0,0),(-1,-1), "MIDDLE"),
                        ("TOPPADDING",(0,0),(-1,-1), 7),
                        ("BOTTOMPADDING",(0,0),(-1,-1),7),
                        ("LEFTPADDING",(0,0),(-1,-1), 10),
                    ]))
                    story.append(tbl); story.append(Spacer(1, 0.3*cm))

            elif kind == "schema":
                story.append(H2("🔍 스키마 분석"))
                story.append(HR())
                rows = [[
                    [Paragraph("스키마 타입", S["small"]),
                     Paragraph(f"<b>{sec.get('schema_type','')}</b>", S["body"])],
                    [Paragraph("신뢰도", S["small"]),
                     Paragraph(f"<b>{sec.get('confidence','')}</b>", S["body"])],
                ],[
                    [Paragraph("수치형 컬럼", S["small"]),
                     Paragraph(sec.get("numeric",""), S["body"])],
                    [Paragraph("범주형 컬럼", S["small"]),
                     Paragraph(sec.get("categorical",""), S["body"])],
                ]]
                for row in rows:
                    tbl = Table([row], colWidths=[usable_w/2, usable_w/2])
                    tbl.setStyle(TableStyle([
                        ("BOX",       (0,0),(-1,-1), 0.5, colors.HexColor("#dce3ea")),
                        ("INNERGRID", (0,0),(-1,-1), 0.3, colors.HexColor("#dce3ea")),
                        ("VALIGN",    (0,0),(-1,-1), "TOP"),
                        ("TOPPADDING",(0,0),(-1,-1), 5),
                        ("BOTTOMPADDING",(0,0),(-1,-1),5),
                        ("LEFTPADDING",(0,0),(-1,-1), 8),
                    ]))
                    story.append(tbl)
                story.append(Spacer(1, 0.3*cm))

            elif kind == "stats" and stats_df is not None and len(stats_df):
                story.append(H2("📊 기술통계"))
                story.append(HR())
                story.append(self._stats_table(stats_df, usable_w, KR, KRB, colors))
                story.append(Spacer(1, 0.3*cm))

            elif kind == "profile":
                story.append(H2("🗂 컬럼 프로파일"))
                story.append(HR())
                rows_data = [["컬럼명","타입","결측치","고유값","샘플"]]
                for r in sec.get("rows",[]):
                    rows_data.append([r.get("name",""),r.get("dtype",""),
                                      r.get("missing",""),r.get("nuniq",""),r.get("sample","")[:40]])
                if len(rows_data) > 1:
                    cw = [usable_w*0.22, usable_w*0.12, usable_w*0.10,
                          usable_w*0.10, usable_w*0.46]
                    tbl = Table(rows_data, colWidths=cw, repeatRows=1)
                    tbl.setStyle(TableStyle([
                        ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#2c3e50")),
                        ("TEXTCOLOR", (0,0),(-1,0), colors.white),
                        ("FONTNAME",  (0,0),(-1,0), KRB),
                        ("FONTNAME",  (0,1),(-1,-1),KR),
                        ("FONTSIZE",  (0,0),(-1,-1),8),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f7fa")]),
                        ("GRID",      (0,0),(-1,-1),0.3,colors.lightgrey),
                        ("VALIGN",    (0,0),(-1,-1),"TOP"),
                        ("TOPPADDING",(0,0),(-1,-1),3),
                        ("BOTTOMPADDING",(0,0),(-1,-1),3),
                        ("LEFTPADDING",(0,0),(-1,-1),3),
                    ]))
                    story.append(tbl); story.append(Spacer(1, 0.3*cm))

            elif kind == "text":
                title = sec.get("title","")
                lines = sec.get("lines",[])
                if not lines: continue
                story.append(H2(title)); story.append(HR())
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    try: story.append(Paragraph(f"• {line}", S["body"]))
                    except Exception: pass
                story.append(Spacer(1, 0.2*cm))

        # ── 품질 검증 섹션 ─────────────────────────────────────────────────
        if quality:
            story.append(PageBreak())
            story.append(H2("🔬 데이터 품질 검증 보고서"))
            story.append(HR())
            q_rows = [["항목","값"]]
            q_rows += [
                ["품질 점수", f"{quality.quality_score:.1f}/100"],
                ["데이터 유형", quality.data_typology],
                ["중복 행", f"{quality.duplicate_count:,}개"],
                ["결측치 컬럼", f"{len(quality.missing_summary)}개"],
                ["이상값 컬럼", f"{len(quality.outlier_summary)}개"],
                ["타입 이슈", f"{len(quality.type_issues)}건"],
            ]
            tbl = Table(q_rows, colWidths=[usable_w*0.4, usable_w*0.6])
            tbl.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#2c3e50")),
                ("TEXTCOLOR", (0,0),(-1,0), colors.white),
                ("FONTNAME",  (0,0),(-1,0), KRB),
                ("FONTNAME",  (0,1),(-1,-1),KR),
                ("FONTSIZE",  (0,0),(-1,-1),9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f5f7fa")]),
                ("GRID",      (0,0),(-1,-1),0.3,colors.lightgrey),
                ("TOPPADDING",(0,0),(-1,-1),5),
                ("LEFTPADDING",(0,0),(-1,-1),8),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 0.4*cm))

            # 결측치 상세
            if quality.missing_summary:
                story.append(H3("결측치 상세"))
                m_rows = [["컬럼명","결측률(%)","심각도"]]
                for col, pct in list(quality.missing_summary.items())[:20]:
                    sev = "높음" if pct>50 else "중간" if pct>20 else "낮음"
                    m_rows.append([col, f"{pct:.1f}%", sev])
                tbl2 = Table(m_rows, colWidths=[usable_w*0.5, usable_w*0.25, usable_w*0.25])
                tbl2.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e74c3c")),
                    ("TEXTCOLOR", (0,0),(-1,0),colors.white),
                    ("FONTNAME",  (0,0),(-1,0),KRB),
                    ("FONTNAME",  (0,1),(-1,-1),KR),
                    ("FONTSIZE",  (0,0),(-1,-1),8),
                    ("GRID",      (0,0),(-1,-1),0.3,colors.lightgrey),
                    ("TOPPADDING",(0,0),(-1,-1),4),
                    ("LEFTPADDING",(0,0),(-1,-1),5),
                ]))
                story.append(tbl2); story.append(Spacer(1, 0.3*cm))

        # ── Feature Engineering 섹션 ───────────────────────────────────────
        if feat_report and feat_report.log:
            story.append(H2("⚗️ Feature Engineering 결과"))
            story.append(HR())
            for log in feat_report.log:
                try: story.append(Paragraph(f"• {log}", S["body"]))
                except Exception: pass
            if feat_report.added_features:
                story.append(Paragraph(
                    f"추가 피처 ({len(feat_report.added_features)}개): " +
                    ", ".join(feat_report.added_features[:12]),
                    S["body"]
                ))
            story.append(Spacer(1, 0.3*cm))

        # ── 분석 방향 제안 (Analysis Proposal) ────────────────────────────
        if proposals:
            story.append(PageBreak())
            story.append(H2("🎯 분석 방향 제안 (Analysis Recommendation)"))
            story.append(HR())
            story.append(Paragraph(
                "데이터 구조와 변수 특성을 바탕으로 적용 가능한 분석 방법을 제안합니다.",
                S["body"]
            ))
            story.append(Spacer(1, 0.3*cm))

            for i, p in enumerate(proposals, 1):
                rows = [
                    [Paragraph(f"{i}. {p['icon']} {p['type']}", ParagraphStyle(
                        "pt", fontName=KRB, fontSize=11,
                        textColor=colors.HexColor("#2980b9")))],
                    [Paragraph(p['desc'], S["body"])],
                    [Table([
                        [Paragraph("<b>알고리즘</b>", S["small"]),
                         Paragraph(p['algorithm'][:80], S["body"])],
                        [Paragraph("<b>타겟 변수</b>", S["small"]),
                         Paragraph(p['target'][:80], S["body"])],
                        [Paragraph("<b>입력 변수</b>", S["small"]),
                         Paragraph(p['features'][:80], S["body"])],
                    ], colWidths=[usable_w*0.18, usable_w*0.72],
                    style=TableStyle([
                        ("FONTNAME",(0,0),(-1,-1),KR),
                        ("FONTSIZE",(0,0),(-1,-1),8),
                        ("TOPPADDING",(0,0),(-1,-1),2),
                        ("BOTTOMPADDING",(0,0),(-1,-1),2),
                        ("LEFTPADDING",(0,0),(-1,-1),4),
                        ("GRID",(0,0),(-1,-1),0.2,colors.lightgrey),
                    ]))],
                ]
                card = Table(rows, colWidths=[usable_w])
                card.setStyle(TableStyle([
                    ("BOX",      (0,0),(-1,-1), 0.8, colors.HexColor("#3498db")),
                    ("BACKGROUND",(0,0),(-1,0), colors.HexColor("#eaf4ff")),
                    ("TOPPADDING",(0,0),(-1,-1), 5),
                    ("BOTTOMPADDING",(0,0),(-1,-1),5),
                    ("LEFTPADDING",(0,0),(-1,-1), 10),
                    ("RIGHTPADDING",(0,0),(-1,-1),10),
                ]))
                story.append(KeepTogether([card, Spacer(1, 0.3*cm)]))


        # ── PNG 차트 ───────────────────────────────────────────────────────
        if png_files:
            story.append(PageBreak())
            story.append(H2("📈 분석 차트"))
            story.append(HR())
            for png in png_files:
                try:
                    img = PILImage.open(str(png))
                    iw, ih = img.size
                    ratio = min(usable_w / iw, (H * 0.5) / ih)
                    dw, dh = iw * ratio, ih * ratio
                    label  = png.stem.replace("_"," ").title()
                    block  = KeepTogether([
                        Paragraph(label, S["h3"]),
                        RLImage(str(png), width=dw, height=dh),
                        Spacer(1, 0.4*cm),
                    ])
                    story.append(block)
                except Exception as e:
                    story.append(Paragraph(f"[{png.stem} 오류: {e}]", S["small"]))
        else:
            story.append(Paragraph("PNG 차트 없음 — pip install kaleido 후 재실행", S["small"]))

        # ── 자동 분석 결과 섹션 (모든 실행 결과 누적 출력) ─────────────────────
        import unicodedata
        def _safe(txt):
            """이모지 및 reportlab 불가 문자 제거"""
            out = ""
            for ch in str(txt):
                cat = unicodedata.category(ch)
                if cat in ("Cc", "Cs", "So", "Cn"):
                    continue
                out += ch
            return out.strip()

        # auto_result = 단일 AutoResult 또는 리스트 모두 허용
        if auto_result:
            _ar_list = auto_result if isinstance(auto_result, list) else [auto_result]
            _ar_list = [r for r in _ar_list if r and getattr(r, "success", False)]
        else:
            _ar_list = []

        for _ar_idx, ar in enumerate(_ar_list):
            try:
                type_labels = {
                    "regression":     "회귀 분석 (Regression)",
                    "classification": "분류 분석 (Classification)",
                    "clustering":     "군집 분석 (Clustering)",
                    "anomaly":        "이상 탐지 (Anomaly Detection)",
                    "timeseries":     "시계열 예측 (Time Series)",
                    "association":    "연관 분석 (Association Analysis)",
                }
                ar_title = type_labels.get(ar.analysis_type, "자동 분석")

                story.append(PageBreak())
                story.append(H2(f"[자동 분석 {_ar_idx+1}] {ar_title}"))
                story.append(HR())

                # ── 모델 정보 + 성능 지표 테이블 ──────────────────────────
                info_rows = [["항목", "내용"]]
                info_rows.append(["분석 유형", ar_title])
                info_rows.append(["모델",      _safe(ar.model_name or "-")])
                info_rows.append(["타겟 변수", _safe(ar.target_col or "비지도")])
                info_rows.append(["입력 변수 수", str(len(ar.feature_cols))])

                story.append(H3("모델 정보"))
                info_tbl = Table(info_rows, colWidths=[usable_w*0.35, usable_w*0.65])
                info_tbl.setStyle(TableStyle([
                    ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#27ae60")),
                    ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
                    ("FONTNAME",       (0,0),(-1,0), KRB),
                    ("FONTNAME",       (0,1),(-1,-1),KR),
                    ("FONTSIZE",       (0,0),(-1,-1),9),
                    ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white, colors.HexColor("#f0fff4")]),
                    ("GRID",           (0,0),(-1,-1),0.3,colors.lightgrey),
                    ("TOPPADDING",     (0,0),(-1,-1),5),
                    ("BOTTOMPADDING",  (0,0),(-1,-1),5),
                    ("LEFTPADDING",    (0,0),(-1,-1),8),
                ]))
                story.append(info_tbl)
                story.append(Spacer(1, 0.3*cm))

                # ── 성능 지표 ─────────────────────────────────────────────
                metric_rows = [["지표", "값"]]
                for k, v in ar.metrics.items():
                    if k == "anomaly_indices":
                        continue
                    metric_rows.append([_safe(str(k)), _safe(str(v))])
                if len(metric_rows) > 1:
                    story.append(H3("성능 지표"))
                    m_tbl = Table(metric_rows, colWidths=[usable_w*0.5, usable_w*0.5])
                    m_tbl.setStyle(TableStyle([
                        ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#2980b9")),
                        ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
                        ("FONTNAME",       (0,0),(-1,0), KRB),
                        ("FONTNAME",       (0,1),(-1,-1),KR),
                        ("FONTSIZE",       (0,0),(-1,-1),9),
                        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white, colors.HexColor("#eaf4ff")]),
                        ("GRID",           (0,0),(-1,-1),0.3,colors.lightgrey),
                        ("TOPPADDING",     (0,0),(-1,-1),5),
                        ("BOTTOMPADDING",  (0,0),(-1,-1),5),
                        ("LEFTPADDING",    (0,0),(-1,-1),8),
                    ]))
                    story.append(m_tbl)
                    story.append(Spacer(1, 0.3*cm))

                # ── 분석 요약 ─────────────────────────────────────────────
                story.append(H3("분석 요약"))
                for line in ar.summary.split("\n"):
                    cleaned = _safe(line.strip().lstrip("*").strip())
                    if cleaned:
                        try:
                            story.append(Paragraph("- " + cleaned, S["body"]))
                        except Exception as _le:
                            print(f"[PDF] 요약 라인 오류: {_le}")
                story.append(Spacer(1, 0.3*cm))

                # ── 개선 추천사항 ─────────────────────────────────────────
                if ar.recommendations:
                    story.append(H3("개선 추천사항"))
                    for rec in ar.recommendations:
                        try:
                            story.append(Paragraph("- " + _safe(rec), S["body"]))
                        except Exception as _le:
                            print(f"[PDF] 추천 오류: {_le}")
                    story.append(Spacer(1, 0.3*cm))

                # ── 피처 중요도 테이블 ────────────────────────────────────
                if ar.feature_importances:
                    story.append(H3("피처 중요도 (상위 15개)"))
                    sorted_fi = sorted(ar.feature_importances.items(),
                                       key=lambda x: x[1], reverse=True)[:15]
                    fi_rows = [["변수명", "중요도"]]
                    for fname, fval in sorted_fi:
                        fi_rows.append([_safe(str(fname)), str(round(float(fval), 4))])
                    fi_tbl = Table(fi_rows, colWidths=[usable_w*0.65, usable_w*0.35])
                    fi_tbl.setStyle(TableStyle([
                        ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#2c3e50")),
                        ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
                        ("FONTNAME",       (0,0),(-1,0), KRB),
                        ("FONTNAME",       (0,1),(-1,-1),KR),
                        ("FONTSIZE",       (0,0),(-1,-1),8.5),
                        ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white, colors.HexColor("#f5f7fa")]),
                        ("GRID",           (0,0),(-1,-1),0.3,colors.lightgrey),
                        ("ALIGN",          (1,0),(-1,-1),"RIGHT"),
                        ("TOPPADDING",     (0,0),(-1,-1),4),
                        ("BOTTOMPADDING",  (0,0),(-1,-1),4),
                        ("LEFTPADDING",    (0,0),(-1,-1),6),
                    ]))
                    story.append(fi_tbl)
                    story.append(Spacer(1, 0.3*cm))

                # ── 분석 차트 PNG ─────────────────────────────────────────
                if ar.figures:
                    story.append(H3("분석 차트"))
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
                        "assoc_top_rules":    "상위 연관 규칙 (Lift 기준)",
                        "assoc_scatter":      "Support x Confidence 분포",
                        "assoc_freq_items":   "빈발 항목집합 지지도",
                        "assoc_heatmap":      "연관 규칙 지표 히트맵",
                    }
                    try:
                        import kaleido  # noqa
                        import tempfile as _tf
                        tmp_dir = Path(_tf.mkdtemp())
                        for key, title in chart_titles.items():
                            if key not in ar.figures:
                                continue
                            try:
                                png_p = tmp_dir / f"auto_{_ar_idx}_{key}.png"
                                ar.figures[key].write_image(
                                    str(png_p), width=1200, height=600, scale=1.5)
                                img    = PILImage.open(str(png_p))
                                iw, ih = img.size
                                ratio  = min(usable_w / iw, (H * 0.42) / ih)
                                dw, dh = iw * ratio, ih * ratio
                                story.append(KeepTogether([
                                    Paragraph(title, S["h3"]),
                                    RLImage(str(png_p), width=dw, height=dh),
                                    Spacer(1, 0.3*cm),
                                ]))
                            except Exception as _ce:
                                story.append(Paragraph(
                                    f"[{title} 차트 오류: {_safe(str(_ce))}]", S["small"]))
                    except ImportError:
                        story.append(Paragraph(
                            "차트 PNG 변환 불가 — pip install kaleido", S["small"]))

                story.append(Spacer(1, 0.5*cm))

            except Exception as _ae:
                import traceback as _tb
                print(f"[PDF] 자동분석[{_ar_idx}] 섹션 오류: {_ae}\n{_tb.format_exc()}")
                try:
                    story.append(Paragraph(
                        f"[자동 분석 {_ar_idx+1} 오류: {str(_ae)[:80]}]", S["small"]))
                except Exception:
                    pass


        # ── 푸터 ──────────────────────────────────────────────────────────
        story += [
            Spacer(1, 0.5*cm),
            HRFlowable(width=usable_w, thickness=0.5,
                       color=colors.lightgrey, spaceAfter=4),
            Paragraph(f"DAIA v5  |  Generated {ts_str}", S["small"]),
        ]
        doc.build(story)
        return output_path

    # ── 기술통계 테이블 ───────────────────────────────────────────────────
    def _stats_table(self, df, usable_w, KR, KRB, colors):
        from reportlab.platypus import Table, TableStyle
        data_df = df.reset_index()
        n_cols  = len(data_df.columns)
        col_w   = usable_w / n_cols
        header  = [str(c) for c in data_df.columns]
        data    = [header]
        for _, row in data_df.head(60).iterrows():
            data.append([str(round(v,4)) if isinstance(v,float) else str(v) for v in row])
        tbl = Table(data, colWidths=[col_w]*n_cols, repeatRows=1)
        tbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR",      (0,0),(-1,0), colors.white),
            ("FONTNAME",       (0,0),(-1,0), KRB),
            ("FONTSIZE",       (0,0),(-1,-1),7.5),
            ("FONTNAME",       (0,1),(-1,-1),KR),
            ("ROWBACKGROUNDS", (0,1),(-1,-1),[colors.white,colors.HexColor("#f5f7fa")]),
            ("GRID",           (0,0),(-1,-1),0.3,colors.lightgrey),
            ("ALIGN",          (1,0),(-1,-1),"RIGHT"),
            ("TOPPADDING",     (0,0),(-1,-1),2),
            ("BOTTOMPADDING",  (0,0),(-1,-1),2),
            ("LEFTPADDING",    (0,0),(-1,-1),3),
        ]))
        return tbl

    # ── HTML 파싱 ─────────────────────────────────────────────────────────
    def _parse_html(self, html_path: Path) -> list[dict]:
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            return self._parse_html_fallback(html_path)
        try:
            html = html_path.read_text(encoding="utf-8")
        except Exception:
            return []
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script","style","nav"]):
            tag.decompose()
        sections = []

        # 메트릭 카드
        vals = soup.select(".val"); lbls = soup.select(".lbl")
        if vals and lbls:
            items = [(v.get_text(strip=True), l.get_text(strip=True))
                     for v,l in zip(vals,lbls)]
            if items: sections.append({"type":"metrics","items":items})

        # 스키마
        schema_sec = soup.find(id="schema")
        if schema_sec:
            items = schema_sec.select(".schema-item"); info = {}
            for item in items:
                sk = item.select_one(".sk"); sv = item.select_one(".sv")
                if sk and sv:
                    key = sk.get_text(strip=True); val = sv.get_text(strip=True)
                    if "타입" in key:   info["schema_type"] = val
                    elif "신뢰도" in key: info["confidence"]  = val
                    elif "수치형" in key: info["numeric"]     = val
                    elif "범주형" in key: info["categorical"]  = val
            if info: info["type"] = "schema"; sections.append(info)

        # 기술통계
        if soup.find(id="stats"):
            sections.append({"type":"stats"})

        # 컬럼 프로파일
        profile_sec = soup.find(id="profile")
        if profile_sec:
            rows = []
            for tr in profile_sec.select("tbody tr"):
                tds = tr.select("td")
                if len(tds) >= 5:
                    rows.append({"name":tds[0].get_text(strip=True),
                                 "dtype":tds[1].get_text(strip=True),
                                 "missing":tds[2].get_text(strip=True),
                                 "nuniq":tds[3].get_text(strip=True),
                                 "sample":tds[4].get_text(strip=True)[:50]})
            if rows: sections.append({"type":"profile","rows":rows})

        # 텍스트 섹션
        text_map = [
            ("preproc",  "⚙️ 전처리 로그",  "preproc"),
            ("insights", "💡 자동 인사이트","insights"),
            ("llm",      "🤖 AI 분석",      "llm"),
        ]
        for key, title, elem_id in text_map:
            sec_el = soup.find(id=elem_id)
            if not sec_el: continue
            if key == "llm":
                content = sec_el.get_text(" ", strip=True)
                lines   = list(dict.fromkeys([s.strip() for s in content.split(".")
                                              if len(s.strip()) > 5]))[:12]
            else:
                lines = [li.get_text(strip=True)
                         for li in sec_el.select("li")
                         if li.get_text(strip=True)]
                if not lines:
                    lines = [sec_el.get_text(" ", strip=True)]
            if lines and any(lines):
                sections.append({"type":"text","title":title,"lines":lines})

        return sections

    def _parse_html_fallback(self, html_path: Path) -> list[dict]:
        import re
        try: html = html_path.read_text(encoding="utf-8")
        except Exception: return []
        html = re.sub(r'<script.*?</script>','',html,flags=re.DOTALL)
        html = re.sub(r'<style.*?</style>','',html,flags=re.DOTALL)
        text = re.sub(r'<[^>]+>',' ',html)
        text = re.sub(r'\s+',' ',text).strip()
        return [{"type":"text","title":"분석 요약",
                 "lines":[s.strip() for s in text.split('.') if len(s.strip())>5][:30]}]