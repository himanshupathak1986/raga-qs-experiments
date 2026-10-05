from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / name), *args]
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--quality-limit", type=int, default=None)
    args = parser.parse_args()

    run_script("collect_swebench.py")
    run_script("collect_securityeval.py")
    run_script("extract_patch_metrics.py")
    run_script("analyze_securityeval.py")

    if not args.skip_quality:
        qargs = ["--only-resolved"]
        if args.quality_limit is not None:
            qargs += ["--limit", str(args.quality_limit)]
        run_script("extract_quality_metrics.py", *qargs)

    run_script("statistical_analysis.py")
    run_script("generate_figures.py")


if __name__ == "__main__":
    main()
