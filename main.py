"""DAIA v4 - CLI 실행"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
from core.pipeline import DAIAPipeline

def main():
    parser = argparse.ArgumentParser(description="DAIA v4 분석 파이프라인")
    parser.add_argument("data", help="데이터 파일 경로 (CSV/XLSX/JSON)")
    parser.add_argument("--config", default="config/config.yaml", help="설정 파일")
    parser.add_argument("--no-llm", action="store_true", help="LLM 비활성화")
    args = parser.parse_args()

    pipeline = DAIAPipeline(args.config)
    if args.no_llm:
        pipeline.llm._available = False

    def progress(msg, pct):
        bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
        print(f"\r[{bar}] {pct*100:.0f}% {msg}", end="", flush=True)
        if pct >= 1.0:
            print()

    out = pipeline.run(args.data, progress_cb=progress)
    print(f"\n✅ 완료: {out}")
    if pipeline.last_csv_path:
        print(f"💾 분석 CSV: {pipeline.last_csv_path}")

if __name__ == "__main__":
    main()
