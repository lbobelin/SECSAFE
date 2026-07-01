#!/usr/bin/env python3
"""Run one configuration of the IAGR-vs-SEP stress-test grid.

This script is intended for Slurm array execution. The grid is generated
identically on every task, and SLURM_ARRAY_TASK_ID selects one configuration.
"""
from __future__ import annotations

import argparse
import itertools
import os
import traceback
from pathlib import Path

from run_iagr_sep_stress_parallel import (
    _parse_csv_floats,
    _parse_csv_ints,
    _parse_csv_strings,
    run_one,
)


def build_configs(args: argparse.Namespace) -> list[dict]:
    fleet_sizes = _parse_csv_ints(args.fleet_sizes)
    profiles = _parse_csv_strings(args.profiles)
    strengths = _parse_csv_floats(args.strengths)
    alphas = _parse_csv_ints(args.alphas)
    policies = _parse_csv_strings(args.policies)

    configs: list[dict] = []
    for n, profile, strength, alpha in itertools.product(fleet_sizes, profiles, strengths, alphas):
        configs.append({
            "n_agvs": n,
            "profile": profile,
            "strength": strength,
            "alpha": alpha,
            "beta": args.beta,
            "policies": policies,
            "runs": args.runs,
            "horizon": args.horizon,
            "seed": args.seed + n + int(100 * strength) + 1000 * alpha,
            "out_dir": str(Path(args.out_dir)),
            "resume": not args.no_resume,
            "campaign_cost_model": args.campaign_cost_model,
            "campaign_group_size": args.campaign_group_size,
            "campaign_marginal_cost_factor": args.campaign_marginal_cost_factor,
        })
    return configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleet-sizes", default="50,100,200,300")
    parser.add_argument("--profiles", default="convoy,sequential,shared_resource,coordination")
    parser.add_argument("--strengths", default="6,10,15")
    parser.add_argument("--alphas", default="1,2")
    parser.add_argument("--beta", type=int, default=2)
    parser.add_argument("--policies", default="IAGR_C,IAGR,SEP,II")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out-dir", default="results/iagrc_campaign_stress")
    parser.add_argument("--campaign-cost-model", action="store_true")
    parser.add_argument("--campaign-group-size", type=int, default=5)
    parser.add_argument("--campaign-marginal-cost-factor", type=float, default=0.45)
    parser.add_argument("--task-id", type=int, default=None,
                        help="Task index. Defaults to SLURM_ARRAY_TASK_ID.")
    parser.add_argument("--print-grid-size", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    configs = build_configs(args)
    if args.print_grid_size:
        print(len(configs))
        return

    task_id = args.task_id
    if task_id is None:
        env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_task_id is None:
            raise SystemExit("No --task-id provided and SLURM_ARRAY_TASK_ID is not set.")
        task_id = int(env_task_id)

    if task_id < 0 or task_id >= len(configs):
        raise SystemExit(f"Task id {task_id} outside grid [0, {len(configs)-1}].")

    cfg = configs[task_id]
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    try:
        res = run_one(cfg)
        print(f"Task {task_id}/{len(configs)-1}: {res['status']} {res['label']}")
        print(f"  raw: {res['raw']}")
        print(f"  summary: {res['summary']}")
    except Exception as exc:
        failure_dir = Path(args.out_dir) / "failures"
        failure_dir.mkdir(parents=True, exist_ok=True)
        failure_file = failure_dir / f"task_{task_id}.txt"
        failure_file.write_text(
            f"Task {task_id} failed\nConfig: {cfg}\nException: {exc}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
