"""Run the whole EDA pipeline: python run_eda.py [--limit N] [--workers 2] [--force]

    01_extract_features.py  scan audio, write features.csv / mel_stats.npz
    02_analyze.py           statistics, leakage probes, shift and novelty analysis
    03_plots.py             figures
    04_report.py            outputs/EDA_REPORT.md

Set DSC_ROOT to the dataset root if this folder is not inside it (default: three levels up (repo/DSC_Pipeline/stages/01_eda -> DSC_Moyim)).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(script, *extra):
    t0 = time.time()
    print(f"\n=== {script} {' '.join(extra)}", flush=True)
    r = subprocess.run([sys.executable, str(HERE / script), *extra], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f"{script} failed with exit code {r.returncode}")
    print(f"=== {script} done in {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke test on N random files per split")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true", help="recompute features even if cached")
    ap.add_argument("--skip-extract", action="store_true", help="reuse existing features.csv")
    a = ap.parse_args()

    if not a.skip_extract:
        extra = ["--workers", str(a.workers)]
        if a.limit:
            extra += ["--limit", str(a.limit)]
        if a.force or a.limit:
            extra.append("--force")
        run("01_extract_features.py", *extra)
    run("02_analyze.py")
    run("03_plots.py")
    run("04_report.py")


if __name__ == "__main__":
    main()
