"""DAIA v4 - CLI 실행"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path
from core.pipeline import DAIAPipeline


def main():
    parser = argparse.ArgumentParser(description="DAIA 분석 파이프라인")
    parser.add_argument("data", help="데이터 파일 경로 (CSV/XLSX/JSON)")
    parser.add_argument("--config", default="config/config.yaml", help="설정 파일")
    parser.add_argument("--no-llm", action="store_true", help="LLM 비활성화")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"데이터 파일이 없습니다: {data_path}")

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일이 없습니다: {config_path}")

    pipeline = DAIAPipeline(str(config_path))
    if args.no_llm and hasattr(pipeline, "llm") and hasattr(pipeline.llm, "_available"):
        pipeline.llm._available = False

    def progress(msg, pct):
        bar_len = 30
        filled = int(pct * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {pct*100:.0f}% {msg}", end="", flush=True)
        if pct >= 1.0:
            print()

    try:
        out = pipeline.run(str(data_path), progress_cb=progress)
        print(f"\n✅ 완료: {out}")
        if getattr(pipeline, "last_csv_path", None):
            print(f"💾 분석 CSV: {pipeline.last_csv_path}")
    except Exception as e:
        import traceback
        print("\n❌ 실행 중 오류 발생")
        print(f"오류: {e}")
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()