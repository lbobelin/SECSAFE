#!/usr/bin/env python3
"""Merge IAGR-vs-SEP stress-test array outputs and generate figures."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from run_iagr_sep_stress_parallel import merge_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/iagrc_campaign_stress")
    parser.add_argument("--figures-dir", default="figures/iagrc_campaign_stress")
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_path, summary_path, ratios_path = merge_results(out_dir)
    print("Merged outputs:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {ratios_path}")

    if not args.skip_figures:
        Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "make_iagr_sep_stress_figures.py",
            "--summary", str(summary_path),
            "--ratios", str(ratios_path),
            "--out-dir", args.figures_dir,
        ]
        print("Generating figures:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
