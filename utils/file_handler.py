"""
EVA NEXIO DAIA - 파일 핸들러
CSV, Excel, JSON 등 다양한 형식 지원
빅데이터 대응: pandas / Dask / DuckDB 자동 전환 (RAM 32GB 기준)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import yaml
from typing import Union, Dict, Any, List
from datetime import datetime

from utils.logger import get_logger

logger = get_logger()

# ── 파일 크기 기준 (RAM 32GB 기준 최적화) ───────────────────────────────────
# pandas  : ~500MB  (빠르고 단순)
# Dask    : 500MB~6GB (멀티코어 병렬, compute() 시 pandas 변환)
# DuckDB  : 6GB~   (메모리 최소화, SQL 기반 스트리밍)
THRESHOLD_DASK_MB   = 500    # 이 이상이면 Dask 사용
THRESHOLD_DUCKDB_MB = 6_000  # 이 이상이면 DuckDB 사용
MAX_FILE_SIZE_MB    = 10_240 # 절대 상한 10GB (config보다 이 값이 우선)
CHUNK_SIZE          = 300_000  # chunked 읽기 단위 (행 수)


class FileHandler:
    """파일 읽기/쓰기 통합 핸들러 (빅데이터 대응)"""

    SUPPORTED_EXTENSIONS = {
        'csv':    ['.csv'],
        'excel':  ['.xlsx', '.xls'],
        'json':   ['.json'],
        'pickle': ['.pkl', '.pickle'],
        'yaml':   ['.yaml', '.yml']
    }

    def __init__(self, config_path="config/config.yaml"):
        self.config = self._load_config(config_path)

        # config의 max_file_size는 무시하고 RAM 기반 상한 사용
        self.max_file_size = MAX_FILE_SIZE_MB * 1024 ** 2
        self.allowed_extensions = self.config.get('security', {}).get(
            'allowed_extensions', ['.csv', '.xlsx']
        )

    # ────────────────────────────────────────────────────────────────────────
    # 설정 / 유효성
    # ────────────────────────────────────────────────────────────────────────
    def _load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"설정 파일 로드 실패: {e}")
            return {}

    def validate_file(self, file_path: Union[str, Path]) -> bool:
        """파일 유효성 검증 (크기 상한 10GB)"""
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"파일이 존재하지 않습니다: {file_path}")
            return False

        if file_path.suffix not in self.allowed_extensions:
            logger.error(f"지원하지 않는 파일 형식: {file_path.suffix}")
            return False

        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            logger.error(
                f"파일 크기 초과: {file_size / 1024**2:.2f}MB "
                f"(최대: {MAX_FILE_SIZE_MB:,}MB)"
            )
            return False

        return True

    # ────────────────────────────────────────────────────────────────────────
    # 메인 읽기 — 파일 크기에 따라 엔진 자동 선택
    # ────────────────────────────────────────────────────────────────────────
    def read_data(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        데이터 파일 읽기 (자동 형식 + 엔진 선택)

        크기별 처리 전략 (RAM 32GB 기준):
          < 500MB  → pandas (즉시 로드)
          500MB~6GB → Dask  (멀티코어 병렬 → pandas 변환)
          > 6GB    → DuckDB (SQL 스트리밍 → pandas 변환)
        """
        file_path = Path(file_path)

        if not self.validate_file(file_path):
            raise ValueError(f"유효하지 않은 파일: {file_path}")

        size_mb = file_path.stat().st_size / 1024 ** 2
        logger.info(f"📂 파일 읽기: {file_path.name} ({size_mb:.1f}MB)")

        try:
            if file_path.suffix == '.csv':
                df = self._read_csv_auto(file_path, size_mb, **kwargs)
            elif file_path.suffix in ['.xlsx', '.xls']:
                df = self._read_excel(file_path, **kwargs)
            elif file_path.suffix == '.json':
                df = self._read_json(file_path, **kwargs)
            else:
                raise ValueError(f"지원하지 않는 형식: {file_path.suffix}")

            # 컬럼명 정제
            if self.config.get('security', {}).get('sanitize_column_names', True):
                df = self._sanitize_columns(df)

            mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
            logger.info(f"✅ 읽기 완료: Shape {df.shape}, Memory {mem_mb:.2f}MB")
            return df

        except Exception:
            logger.exception(f"파일 읽기 실패: {file_path}")
            raise

    # ────────────────────────────────────────────────────────────────────────
    # CSV 엔진 분기
    # ────────────────────────────────────────────────────────────────────────
    def _read_csv_auto(self, file_path: Path, size_mb: float, **kwargs) -> pd.DataFrame:
        if size_mb >= THRESHOLD_DUCKDB_MB:
            return self._read_csv_duckdb(file_path, **kwargs)
        elif size_mb >= THRESHOLD_DASK_MB:
            return self._read_csv_dask(file_path, **kwargs)
        else:
            return self._read_csv_pandas(file_path, **kwargs)

    def _read_csv_pandas(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """pandas chunked 읽기 (< 500MB)"""
        encoding = self._detect_encoding(file_path)
        logger.info(f"  엔진: pandas | 인코딩: {encoding}")

        chunks = []
        for i, chunk in enumerate(
            pd.read_csv(file_path, encoding=encoding,
                        chunksize=CHUNK_SIZE, low_memory=False, **kwargs)
        ):
            chunks.append(chunk)
            logger.debug(f"  chunk {i+1}: {len(chunk):,} rows")

        return pd.concat(chunks, ignore_index=True)

    def _read_csv_dask(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Dask 병렬 읽기 (500MB ~ 6GB) → pandas 변환"""
        try:
            import dask.dataframe as dd
        except ImportError:
            logger.warning("Dask 미설치 → pandas chunked로 대체 (pip install dask)")
            return self._read_csv_pandas(file_path, **kwargs)

        encoding = self._detect_encoding(file_path)
        logger.info(f"  엔진: Dask | 인코딩: {encoding}")

        ddf = dd.read_csv(file_path, encoding=encoding,
                          blocksize="128MB", low_memory=False, **kwargs)
        logger.info(f"  Dask partitions: {ddf.npartitions}")

        df = ddf.compute()
        return df

    def _read_csv_duckdb(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """DuckDB SQL 스트리밍 (> 6GB) → pandas 변환"""
        try:
            import duckdb
        except ImportError:
            logger.warning("DuckDB 미설치 → Dask로 대체 (pip install duckdb)")
            return self._read_csv_dask(file_path, **kwargs)

        logger.info(f"  엔진: DuckDB")

        con = duckdb.connect()
        # DuckDB는 CSV를 직접 SQL로 쿼리 (메모리 최소화)
        query = f"SELECT * FROM read_csv_auto('{file_path.as_posix()}', header=true)"
        df = con.execute(query).df()
        con.close()
        return df

    # ────────────────────────────────────────────────────────────────────────
    # Excel / JSON
    # ────────────────────────────────────────────────────────────────────────
    def _read_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Excel 읽기 (대용량이면 경고)"""
        size_mb = file_path.stat().st_size / 1024 ** 2
        if size_mb > 200:
            logger.warning(
                f"Excel {size_mb:.0f}MB — 대용량 Excel은 CSV 변환 권장"
            )
        logger.info(f"  엔진: pandas (Excel)")
        return pd.read_excel(file_path, **kwargs)

    def _read_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        logger.info(f"  엔진: pandas (JSON)")
        return pd.read_json(file_path, **kwargs)

    # ────────────────────────────────────────────────────────────────────────
    # 인코딩 감지
    # ────────────────────────────────────────────────────────────────────────
    def _detect_encoding(self, file_path: Path) -> str:
        """샘플 읽기로 인코딩 자동 감지"""
        for enc in ['utf-8', 'cp949', 'euc-kr', 'latin1']:
            try:
                with open(file_path, encoding=enc) as f:
                    f.read(65536)  # 64KB 샘플
                return enc
            except UnicodeDecodeError:
                continue
        return 'utf-8'  # fallback

    # ────────────────────────────────────────────────────────────────────────
    # 컬럼명 정제
    # ────────────────────────────────────────────────────────────────────────
    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(r'[\s\.\-]+', '_', regex=True)
        df.columns = df.columns.str.replace(r'[^\w가-힣]', '', regex=True)

        if df.columns.duplicated().any():
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique():
                cols[cols == dup] = [f"{dup}_{i}" for i in range(sum(cols == dup))]
            df.columns = cols

        return df

    # ────────────────────────────────────────────────────────────────────────
    # 저장
    # ────────────────────────────────────────────────────────────────────────
    def save_dataframe(self, df: pd.DataFrame, file_path: Union[str, Path],
                       format: str = 'csv', **kwargs):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"💾 저장: {file_path.name}")

        try:
            if format == 'csv':
                df.to_csv(file_path, index=False, encoding='utf-8-sig', **kwargs)
            elif format == 'excel':
                df.to_excel(file_path, index=False, **kwargs)
            elif format == 'json':
                df.to_json(file_path, orient='records', force_ascii=False, **kwargs)
            elif format == 'pickle':
                df.to_pickle(file_path)
            else:
                raise ValueError(f"지원하지 않는 저장 형식: {format}")

            logger.info(f"✅ 저장 완료: {file_path}")
        except Exception:
            logger.exception(f"저장 실패: {file_path}")
            raise

    # ────────────────────────────────────────────────────────────────────────
    # 모델 / 파이프라인
    # ────────────────────────────────────────────────────────────────────────
    def save_model(self, model: Any, model_path: Union[str, Path],
                   metadata: Dict = None):
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"💾 모델 저장: {model_path.name}")

        if metadata:
            meta_path = model_path.with_suffix('.meta.json')
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['model_path'] = str(model_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"📋 메타데이터 저장: {meta_path.name}")

    def load_model(self, model_path: Union[str, Path]) -> tuple:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"📂 모델 로드: {model_path.name}")

        meta_path = model_path.with_suffix('.meta.json')
        metadata = None
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            logger.info(f"📋 메타데이터 로드: {meta_path.name}")

        return model, metadata

    def save_pipeline(self, pipeline: Dict, pipeline_path: Union[str, Path]):
        pipeline_path = Path(pipeline_path)
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        logger.info(f"💾 파이프라인 저장: {pipeline_path.name}")

    def load_pipeline(self, pipeline_path: Union[str, Path]) -> Dict:
        pipeline_path = Path(pipeline_path)
        if not pipeline_path.exists():
            raise FileNotFoundError(f"파이프라인 파일이 없습니다: {pipeline_path}")
        with open(pipeline_path, 'rb') as f:
            pipeline = pickle.load(f)
        logger.info(f"📂 파이프라인 로드: {pipeline_path.name}")
        return pipeline

    # ────────────────────────────────────────────────────────────────────────
    # 파일 정보
    # ────────────────────────────────────────────────────────────────────────
    def get_file_info(self, file_path: Union[str, Path]) -> Dict:
        file_path = Path(file_path)
        if not file_path.exists():
            return {"error": "파일이 존재하지 않습니다"}

        stat = file_path.stat()
        size_mb = stat.st_size / 1024 ** 2

        # 예상 엔진 표시
        if file_path.suffix == '.csv':
            if size_mb >= THRESHOLD_DUCKDB_MB:
                engine = "DuckDB"
            elif size_mb >= THRESHOLD_DASK_MB:
                engine = "Dask"
            else:
                engine = "pandas"
        else:
            engine = "pandas"

        return {
            "name":        file_path.name,
            "extension":   file_path.suffix,
            "size_mb":     round(size_mb, 2),
            "engine":      engine,
            "created":     datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified":    datetime.fromtimestamp(stat.st_mtime).isoformat(),
        }


# ────────────────────────────────────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    handler = FileHandler()

    df_test = pd.DataFrame({
        '고객ID': range(1, 101),
        '나이':   np.random.randint(20, 70, 100),
        '구매액': np.random.randint(1000, 100000, 100)
    })

    handler.save_dataframe(df_test, 'data/test.csv')
    df_loaded = handler.read_data('data/test.csv')
    print(df_loaded.head())

    info = handler.get_file_info('data/test.csv')
    print(f"\n파일 정보: {info}")
    print("파일 핸들러 테스트 완료")