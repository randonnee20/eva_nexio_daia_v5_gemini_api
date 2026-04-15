"""
EVA NEXIO DAIA - 로깅 유틸리티 (v2 통합본)
"""

import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
import yaml


class DAIALogger:
    _instances = {}

    def __new__(cls, name="daia", config_path="config/config.yaml"):
        if name not in cls._instances:
            cls._instances[name] = super().__new__(cls)
        return cls._instances[name]

    def __init__(self, name="daia", config_path="config/config.yaml"):
        if hasattr(self, '_initialized'):
            return
        self.name = name
        self.logger = logging.getLogger(name)
        self._setup_logger(config_path)
        self._initialized = True

    def _setup_logger(self, config_path):
        log_config = {}
        for p in [config_path, str(Path(__file__).parent.parent / "config" / "config.yaml")]:
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    log_config = (yaml.safe_load(f) or {}).get('logging', {})
                break
            except Exception:
                pass

        level      = log_config.get('level', 'INFO')
        log_file   = log_config.get('file_path', './logs/app.log')
        max_bytes  = log_config.get('max_size', 10485760)
        backup_cnt = log_config.get('backup_count', 5)
        fmt        = log_config.get('format', '%(asctime)s [%(levelname)s] %(message)s')

        self.logger.setLevel(getattr(logging, level, logging.INFO))
        self.logger.handlers.clear()
        formatter = logging.Formatter(fmt, datefmt="%H:%M:%S")

        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        try:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(log_file, maxBytes=max_bytes,
                                     backupCount=backup_cnt, encoding='utf-8')
            fh.setLevel(getattr(logging, level, logging.INFO))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        except Exception:
            pass

    def debug(self, msg, *a, **kw):     self.logger.debug(msg, *a, **kw)
    def info(self, msg, *a, **kw):      self.logger.info(msg, *a, **kw)
    def warning(self, msg, *a, **kw):   self.logger.warning(msg, *a, **kw)
    def error(self, msg, *a, **kw):     self.logger.error(msg, *a, **kw)
    def critical(self, msg, *a, **kw):  self.logger.critical(msg, *a, **kw)
    def exception(self, msg, *a, **kw): self.logger.exception(msg, *a, **kw)

    def log_step(self, step_name, status="START"):
        sep = "=" * 60
        self.info(f"\n{sep}\n[{status}] {step_name}\n{sep}")

    def log_data_info(self, df, prefix=""):
        self.info(f"{prefix}Shape: {df.shape}")
        self.info(f"{prefix}Memory: {df.memory_usage(deep=True).sum()/1024**2:.2f} MB")

    def log_metric(self, metric_name, value):
        self.info(f"📊 {metric_name}: {value}")

    def log_timer(self, operation):
        return _LogTimer(self, operation)


class _LogTimer:
    def __init__(self, logger, operation):
        self.logger    = logger
        self.operation = operation
        self.t0        = None

    def __enter__(self):
        self.t0 = time.time()
        self.logger.info(f"⏱  시작: {self.operation}")
        return self

    def __exit__(self, exc_type, *_):
        elapsed = time.time() - self.t0
        if exc_type is None:
            self.logger.info(f"✅ 완료: {self.operation} ({elapsed:.2f}초)")
        else:
            self.logger.error(f"❌ 실패: {self.operation} ({elapsed:.2f}초)")
        return False


@contextmanager
def timer(label: str, logger=None):
    t0 = time.time()
    yield
    elapsed = time.time() - t0
    msg = f"⏱  {label}: {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


_global_logger = None

def get_logger(name="daia", config_path="config/config.yaml") -> DAIALogger:
    global _global_logger
    if _global_logger is None:
        _global_logger = DAIALogger(name, config_path)
    return _global_logger
