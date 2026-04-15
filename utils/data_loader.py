"""
DAIA v2 - DataLoader wrapper
원본 utils/file_handler.py의 FileHandler를 래핑합니다.
pandas / Dask / DuckDB 자동 선택 (파일 크기 기준).
"""
from __future__ import annotations
from pathlib import Path
from utils.file_handler import FileHandler
from utils.logger import get_logger

logger = get_logger()


class DataLoader:
    def __init__(self, config: dict = None):
        # FileHandler는 config_path 문자열을 받음 → 임시 yaml 생성 없이
        # config dict를 직접 건네는 방식 대신, config_path를 통해 동작
        self._config = config or {}

    def load(self, path: str | Path):
        import tempfile, yaml, os
        # FileHandler가 config_path를 필요로 하므로 임시 yaml로 전달
        tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml',
                                          delete=False, encoding='utf-8')
        yaml.dump(self._config, tmp, allow_unicode=True)
        tmp.close()
        try:
            fh = FileHandler(config_path=tmp.name)
            return fh.read_data(path)
        finally:
            os.unlink(tmp.name)
