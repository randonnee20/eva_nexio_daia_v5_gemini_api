"""
DAIA v4 - 데이터 품질 검증 (Data Quality Validation)
절차: 결측치 / 이상값 / 중복 / 범위오류 / 타입오류 검증
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualityIssue:
    category: str       # missing | outlier | duplicate | range_error | type_error
    column: str
    severity: str       # high | medium | low
    count: int
    detail: str


@dataclass
class QualityReport:
    total_rows: int
    total_cols: int
    quality_score: float            # 0~100
    issues: list[QualityIssue] = field(default_factory=list)
    missing_summary: dict = field(default_factory=dict)
    outlier_summary: dict = field(default_factory=dict)
    duplicate_count: int = 0
    type_issues: list[str] = field(default_factory=list)
    data_typology: str = ""         # 데이터 유형 분류
    typology_reason: str = ""
    profile_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_markdown(self) -> str:
        lines = [
            f"**품질 점수:** {self.quality_score:.1f}/100",
            f"**총 행:** {self.total_rows:,} | **총 컬럼:** {self.total_cols}",
            f"**중복 행:** {self.duplicate_count:,}",
            f"**데이터 유형:** {self.data_typology}",
            "",
            "### 결측치",
        ]
        for col, pct in list(self.missing_summary.items())[:15]:
            bar = "🔴" if pct > 50 else "🟡" if pct > 20 else "🟢"
            lines.append(f"- {bar} `{col}`: {pct:.1f}%")
        if not self.missing_summary:
            lines.append("- ✅ 결측치 없음")
        lines += ["", "### 이상값"]
        for col, info in list(self.outlier_summary.items())[:10]:
            lines.append(f"- `{col}`: {info['count']:,}개 ({info['pct']:.1f}%)")
        if not self.outlier_summary:
            lines.append("- ✅ 이상값 없음")
        return "\n".join(lines)


class DataQualityValidator:
    def __init__(self, config: dict = None):
        self.cfg = (config or {}).get("preprocessing", {})

    def validate(self, df: pd.DataFrame, schema=None) -> QualityReport:
        report = QualityReport(
            total_rows=len(df),
            total_cols=len(df.columns),
            quality_score=100.0,
        )

        # 1. 결측치
        miss = df.isnull().mean() * 100
        miss_nonzero = miss[miss > 0].sort_values(ascending=False)
        report.missing_summary = miss_nonzero.round(2).to_dict()
        for col, pct in miss_nonzero.items():
            sev = "high" if pct > 50 else "medium" if pct > 20 else "low"
            report.issues.append(QualityIssue(
                "missing", col, sev, int(df[col].isnull().sum()),
                f"결측치 {pct:.1f}%"
            ))
            report.quality_score -= pct * 0.2

        # 2. 중복
        dup = df.duplicated().sum()
        report.duplicate_count = int(dup)
        if dup > 0:
            pct = dup / len(df) * 100
            report.issues.append(QualityIssue(
                "duplicate", "전체", "medium" if pct < 5 else "high",
                int(dup), f"중복 행 {pct:.1f}%"
            ))
            report.quality_score -= pct * 0.5

        # 3. 이상값 (수치형)
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - 2.5 * iqr
            hi = q3 + 2.5 * iqr
            n_out = int(((s < lo) | (s > hi)).sum())
            if n_out > 0:
                pct = n_out / len(s) * 100
                report.outlier_summary[col] = {
                    "count": n_out, "pct": round(pct, 2),
                    "lower": round(lo, 4), "upper": round(hi, 4),
                }
                sev = "high" if pct > 10 else "medium" if pct > 3 else "low"
                report.issues.append(QualityIssue(
                    "outlier", col, sev, n_out, f"IQR 기준 이상값 {pct:.1f}%"
                ))
                report.quality_score -= pct * 0.1

        # 4. 타입 오류 (object 컬럼 중 숫자인 것)
        for col in df.select_dtypes(include="object").columns:
            num_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
            if 0.8 < num_ratio < 1.0:
                report.type_issues.append(
                    f"`{col}` — 대부분 숫자이지만 문자형 (숫자 변환 권장)"
                )
                report.quality_score -= 2.0

        # 5. 범위 오류 (음수여서는 안 될 컬럼)
        for col in num_cols:
            if any(k in col.lower() for k in ["count", "qty", "quantity", "age", "price", "amount"]):
                neg = (df[col] < 0).sum()
                if neg > 0:
                    report.issues.append(QualityIssue(
                        "range_error", col, "medium", int(neg),
                        f"음수값 {neg}개 (범위 오류 의심)"
                    ))
                    report.quality_score -= 1.0

        # 6. 컬럼 프로파일 테이블 생성
        rows = []
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            miss_pct = s.isnull().mean() * 100
            nuniq = s.nunique()
            # 기술통계 (수치형)
            if pd.api.types.is_numeric_dtype(s) and not s.dropna().empty:
                mean_v = f"{s.mean():.3f}"
                std_v  = f"{s.std():.3f}"
                min_v  = f"{s.min():.3f}"
                max_v  = f"{s.max():.3f}"
            else:
                mean_v = std_v = min_v = max_v = "-"
            sample = str(s.dropna().iloc[:2].tolist())[:40] if s.dropna().shape[0] else "-"
            rows.append({
                "컬럼": col, "타입": dtype,
                "결측(%)": round(miss_pct, 1),
                "고유값": nuniq,
                "평균": mean_v, "표준편차": std_v,
                "최솟값": min_v, "최댓값": max_v,
                "샘플": sample,
            })
        report.profile_table = pd.DataFrame(rows)

        # 7. 데이터 유형 분류 (Data Typology)
        report.data_typology, report.typology_reason = _classify_typology(df, schema)

        report.quality_score = max(0.0, min(100.0, report.quality_score))
        return report


def _classify_typology(df: pd.DataFrame, schema=None) -> tuple[str, str]:
    """데이터를 유형별로 분류"""
    if schema is None:
        # 단순 판단
        has_ts = any(pd.api.types.is_datetime64_any_dtype(df[c]) for c in df.columns)
        has_id = any("id" in c.lower() for c in df.columns)
        if has_ts and has_id:
            return "패널 데이터 (Panel Data)", "entity ID + 타임스탬프 컬럼 존재"
        elif has_ts:
            return "시계열 데이터 (Time Series)", "타임스탬프 컬럼 존재"
        return "정적 테이블 (Wide Table)", "시계열/ID 없는 일반 테이블"

    st = schema.schema_type
    ts = schema.timestamp_col
    if st == "signal_pool":
        return "이벤트/신호 데이터 (Event/Signal Data)", "signalname + value + timestamp 구조"
    elif st == "time_series":
        if schema.id_cols:
            return "패널 데이터 (Panel Data)", f"entity({schema.id_cols}) + time({ts})"
        return "시계열 데이터 (Time Series)", f"타임스탬프 컬럼: {ts}"
    elif st == "wide_table":
        id_cols = schema.id_cols
        if id_cols and ts:
            return "트랜잭션 데이터 (Transaction Data)", f"ID({id_cols}) + 시간({ts}) 존재"
        elif id_cols:
            return "엔티티 데이터 (Entity Data)", f"ID 컬럼 존재: {id_cols}"
        return "정적 테이블 (Static Wide Table)", "행=샘플, 열=피처 구조"
    return "정적 테이블 (Static Wide Table)", "기타"
