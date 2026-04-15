"""
DAIA v4 - Pipeline
절차: Data Inventory → Profiling → Typology → Quality Validation →
      Cleaning → Structuring → Feature Engineering → EDA →
      Analytical Framing → Analysis Recommendation
"""
from __future__ import annotations
import yaml, tempfile, os
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np

from utils.logger import get_logger, timer
from utils.data_loader import DataLoader
from core.schema_detector import SchemaDetector, SchemaInfo
from preprocessing.preprocessor import SmartPreprocessor
from preprocessing.data_quality import DataQualityValidator, QualityReport
from preprocessing.feature_engineer import FeatureEngineer, FeatureReport
from visualization.eda_engine import EDAEngine
from llm.llm_advisor import LLMAdvisor
from report.report_builder import ReportBuilder
from analysis.auto_runner import AutoRunner, AutoResult

logger = get_logger()


class DAIAPipeline:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config      = self._load_config(config_path)
        self.loader       = DataLoader(self.config)
        self.detector     = SchemaDetector(self.config)
        self.preprocessor = SmartPreprocessor(self.config)
        self.quality_validator = DataQualityValidator(self.config)
        self.feature_engineer  = FeatureEngineer(self.config)
        self.eda_engine   = EDAEngine(self.config)
        self.llm          = LLMAdvisor(config_path)
        self.reporter     = ReportBuilder(self.config)

        self.last_charts:    dict[str, str]         = {}
        self.last_insights:  list[str]              = []
        self.last_schema:    Optional[SchemaInfo]   = None
        self.last_stats:     Optional[pd.DataFrame] = None
        self.last_quality:   Optional[QualityReport]= None
        self.last_feature_report: Optional[FeatureReport] = None
        self.last_proposals: list[dict]             = []
        self.last_df_feature: Optional[pd.DataFrame] = None
        self.last_csv_path:   Optional[Path] = None
        self.auto_runner      = AutoRunner(self.config)
        self.last_auto_result: Optional[AutoResult] = None

    def _load_config(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {}

    def run(self, data_path: str, output_name: Optional[str] = None,
            progress_cb=None) -> Path:
        def _prog(msg, pct):
            logger.info(msg)
            if progress_cb:
                progress_cb(msg, pct)

        path = Path(data_path)
        name = output_name or path.stem

        # STEP 1: Data Inventory
        _prog("📂 [1/11] Data Inventory — 데이터 로드 중...", 0.03)
        with timer("데이터 로드", logger):
            df_raw = self.loader.load(data_path)
        _prog(f"✅ [1/11] 로드 완료: {df_raw.shape[0]:,}행 × {df_raw.shape[1]}컬럼", 0.08)

        # STEP 2: Data Profiling & Schema
        _prog("🔍 [2/11] Data Profiling — 스키마 감지 및 컬럼 프로파일...", 0.13)
        with timer("스키마 감지", logger):
            schema = self.detector.detect(df_raw)
        self.last_schema = schema
        _prog(f"✅ [2/11] 스키마: {schema.schema_type} (신뢰도 {schema.confidence:.0%})", 0.18)

        # STEP 3: Data Typology
        _prog("🗂 [3/11] Data Typology — 데이터 유형 분류...", 0.22)
        _prog("✅ [3/11] 유형 분류 완료", 0.25)

        # STEP 4: Data Quality Validation
        _prog("🔬 [4/11] Data Quality Validation — 결측치/이상값/중복 검증...", 0.28)
        with timer("품질 검증", logger):
            quality = self.quality_validator.validate(df_raw, schema)
        self.last_quality = quality
        _prog(
            f"✅ [4/11] 품질 점수: {quality.quality_score:.1f}/100 | "
            f"결측 {len(quality.missing_summary)}컬럼 / 이상값 {len(quality.outlier_summary)}컬럼 / 중복 {quality.duplicate_count}행",
            0.33,
        )

        # STEP 5: Data Cleaning
        _prog("⚙️ [5/11] Data Cleaning — 결측치/이상값 처리...", 0.36)
        with timer("전처리", logger):
            df_clean = self.preprocessor.run(df_raw, schema)
        for s in self.preprocessor.report:
            logger.info(f"  ✔ {s}")
        _prog(f"✅ [5/11] 정제 완료: {len(self.preprocessor.report)}개 처리 / {len(df_clean):,}행 유지", 0.42)

        # STEP 6: Data Structuring
        _prog("🔧 [6/11] Data Structuring — 분석용 구조 재구성...", 0.46)
        df_struct = self._structure_data(df_clean, schema)
        _prog("✅ [6/11] 구조 재구성 완료", 0.50)

        # STEP 7: Data Integration
        _prog("🔗 [7/11] Data Integration — 데이터 통합 중...", 0.52)
        _prog("✅ [7/11] 통합 완료", 0.55)

        # STEP 8: Feature Engineering
        _prog("⚗️ [8/11] Feature Engineering — 분석용 피처 생성...", 0.57)
        with timer("피처 엔지니어링", logger):
            df_feature = self.feature_engineer.run(df_struct, schema)
        feat_report = self.feature_engineer.report
        self.last_feature_report = feat_report
        _prog(f"✅ [8/11] 피처 엔지니어링: {len(feat_report.added_features)}개 신규 피처", 0.63)

        # STEP 9: EDA
        _prog("📈 [9/11] EDA — 시각화 및 탐색적 분석...", 0.66)
        with timer("EDA", logger):
            charts   = self.eda_engine.run_all(df_feature, schema)
            insights = self.eda_engine.compute_insights(df_feature, schema)
            stats_df = self.eda_engine.get_descriptive_stats(df_feature, schema)
        self.last_charts   = charts
        self.last_insights = insights
        self.last_stats    = stats_df
        _prog(f"✅ [9/11] EDA 완료: 차트 {len(charts)}개 / 인사이트 {len(insights)}건", 0.72)

        # STEP 10: Analytical Framing
        _prog("🎯 [10/11] Analytical Framing — 분석 방향 도출...", 0.76)
        proposals = self.feature_engineer.build_analysis_proposals(df_feature, schema, quality)
        self.last_proposals = proposals
        _prog(f"✅ [10/11] 분석 제안 {len(proposals)}개 도출", 0.80)

        # PNG 차트 저장
        out_dir = Path(self.config.get("report", {}).get("output_dir", "./daia_output"))
        charts_dir = out_dir / "charts" / name
        saved_png = self.eda_engine.save_charts_png(charts_dir)
        if saved_png:
            _prog(f"🖼 PNG {len(saved_png)}개 저장: {charts_dir}", 0.82)

        # LLM
        llm_text = {"schema": "", "preprocessing": "", "insights": ""}
        if self.llm.is_available():
            _prog("🤖 LLM 분석 중...", 0.84)
            df_info = {"rows": len(df_raw), "cols": len(df_raw.columns),
                       "columns": list(df_raw.columns),
                       "sample": df_raw.head(5).to_string(max_cols=10)}
            profile = self._build_profile(df_feature, schema)
            stats   = self._build_stats(df_feature, schema)
            with timer("LLM", logger):
                llm_text["schema"]        = self.llm.analyze_schema(df_info, schema)
                llm_text["preprocessing"] = self.llm.suggest_preprocessing(profile, schema)
                llm_text["insights"]      = self.llm.interpret_insights(insights, schema, stats)
        else:
            _prog("⚠️ LLM 비활성 → 규칙 기반 분석만 수행", 0.84)

        # STEP 11: Report + CSV export
        _prog("📄 [11/11] 리포트 생성 및 분석 데이터셋 저장...", 0.88)
        csv_path = self._save_feature_csv(df_feature, name, out_dir)
        self.last_df_feature = df_feature
        self.last_csv_path   = csv_path
        if csv_path:
            _prog(f"💾 분석 데이터셋 저장: {csv_path.name}", 0.91)

        with timer("리포트", logger):
            out = self.reporter.build(
                df_raw, df_feature, schema,
                charts, insights,
                self.preprocessor.report,
                llm_text,
                source_name=name,
                stats_df=stats_df,
                quality=quality,
                feat_report=feat_report,
                proposals=proposals,
            )
        _prog(f"✅ 완료 → {out}", 1.0)
        if self.config.get("report", {}).get("open_browser", True):
            import webbrowser
            webbrowser.open(str(out.resolve()))
        return out

    def _structure_data(self, df: pd.DataFrame, schema: SchemaInfo) -> pd.DataFrame:
        df = df.copy()
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        if len(df) < before:
            logger.info(f"  ✔ 중복 {before - len(df)}행 제거")
        return df

    def _save_feature_csv(self, df: pd.DataFrame, name: str, out_dir: Path) -> Optional[Path]:
        try:
            dataset_dir = out_dir / "datasets"
            dataset_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_path = dataset_dir / f"daia_feature_{name}_{ts}.csv"
            df.to_csv(str(csv_path), index=False, encoding="utf-8-sig")
            logger.info(f"💾 Feature CSV 저장: {csv_path}")
            return csv_path
        except Exception as e:
            logger.warning(f"CSV 저장 실패: {e}")
            return None

    def _build_profile(self, df, schema):
        miss = df.isnull().mean()
        q = max(0, 100 - miss.mean()*50 - df.duplicated().sum()/max(len(df),1)*20)
        return {"quality_score": round(q,1), "high_missing": miss[miss>0.3].index.tolist()}

    def _build_stats(self, df, schema):
        stats = {}
        cols = ([schema.value_num_col] if schema.value_num_col and schema.value_num_col in df.columns else []) \
             + schema.numeric_cols[:5]
        for col in cols:
            if col not in df.columns: continue
            s = df[col].dropna()
            if len(s) < 2: continue
            stats[col] = {"mean": round(float(s.mean()),4), "std": round(float(s.std()),4),
                          "min":  round(float(s.min()),4),  "max": round(float(s.max()),4)}
        return stats

    def run_analysis(self, analysis_type: str,
                     target_col: str = None) -> "AutoResult":
        """
        EDA 완료 후 추천 분석을 실행합니다.
        analysis_type: regression / classification / clustering / anomaly / timeseries
        """
        if self.last_df_feature is None:
            from analysis.auto_runner import AutoResult
            return AutoResult(analysis_type, success=False,
                              error_msg="먼저 분석을 실행하세요 (run() 호출 필요)")
        result = self.auto_runner.run(
            analysis_type, self.last_df_feature,
            self.last_schema, target_col=target_col,
        )
        self.last_auto_result = result
        return result

