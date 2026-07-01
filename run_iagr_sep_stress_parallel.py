#!/usr/bin/env python3
"""Focused IAGR-vs-SEP stress test for large fleets.

This runner executes online policies only. It is designed to test whether the
interaction-aware part of IAGR separates from the simpler SEP policy when fleet
coupling is stronger and maintenance resources are scarce.
"""
from __future__ import annotations

import argparse
import itertools
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import pandas as pd

from run_agv_momdp_experiments import Params, run_condition, summarize


def _parse_csv_ints(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def _parse_csv_strings(s: str) -> list[str]:
    return [x.strip() for x in s.split(',') if x.strip()]


def run_one(cfg: dict) -> dict:
    out_dir = Path(cfg["out_dir"])
    raw_dir = out_dir / "raw_parts"
    summary_dir = out_dir / "summary_parts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    label = (
        f"stress_n{cfg['n_agvs']}_profile-{cfg['profile']}"
        f"_strength-{cfg['strength']}_alpha-{cfg['alpha']}_beta-{cfg['beta']}"
    )
    raw_path = raw_dir / f"{label}.csv"
    summary_path = summary_dir / f"{label}.csv"

    if cfg.get("resume", True) and raw_path.exists() and summary_path.exists():
        return {"label": label, "status": "skipped", "raw": str(raw_path), "summary": str(summary_path)}

    p = Params(
        n_agvs=cfg["n_agvs"],
        n_runs=cfg["runs"],
        horizon=cfg["horizon"],
        seed=cfg["seed"],
        interaction_profile=cfg["profile"],
        interaction_strength=cfg["strength"],
        alpha_technicians=cfg["alpha"],
        beta_monitored_agvs=cfg["beta"],
        campaign_cost_model=cfg.get("campaign_cost_model", False),
        campaign_group_size=cfg.get("campaign_group_size", 5),
        campaign_marginal_cost_factor=cfg.get("campaign_marginal_cost_factor", 0.45),
    )
    policies = cfg["policies"]
    raw = run_condition(label, p, policies, offline_policy=None)
    summary = summarize(raw)
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {"label": label, "status": "done", "raw": str(raw_path), "summary": str(summary_path)}


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        cols = []
        for c in df.columns:
            parts = [str(x) for x in c if str(x) not in ("", "nan") and not str(x).startswith("Unnamed:")]
            cols.append("_".join(parts))
        df.columns = cols
    return df


def merge_results(out_dir: Path) -> tuple[Path, Path, Path]:
    raw_files = sorted((out_dir / "raw_parts").glob("*.csv"))
    summary_files = sorted((out_dir / "summary_parts").glob("*.csv"))
    if not raw_files or not summary_files:
        raise RuntimeError("No part CSV files found to merge.")

    raw = pd.concat((pd.read_csv(f) for f in raw_files), ignore_index=True)
    summary = pd.concat((pd.read_csv(f, header=[0, 1]) for f in summary_files), ignore_index=True)
    summary = flatten_columns(summary)

    raw_path = out_dir / "iagr_sep_stress_raw.csv"
    summary_path = out_dir / "iagr_sep_stress_summary.csv"
    ratios_path = out_dir / "iagr_sep_stress_ratios.csv"

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)

    # Robustly locate cost columns after flattened pandas aggregation output.
    cost_mean_col = "discounted_cost_mean"
    if cost_mean_col not in summary.columns:
        candidates = [c for c in summary.columns if c.startswith("discounted_cost") and c.endswith("mean")]
        if not candidates:
            raise RuntimeError(f"Could not find discounted cost mean column in {summary.columns.tolist()}")
        cost_mean_col = candidates[0]

    id_cols = [
        "condition_", "policy_", "n_agvs_", "interaction_profile_",
        "interaction_strength_", "alpha_technicians_", "beta_monitored_agvs_",
    ]
    # Depending on pandas version, non-aggregated columns may have no trailing underscore.
    remap = {}
    for c in summary.columns:
        if c.endswith("_"):
            remap[c] = c[:-1]
    s2 = summary.rename(columns=remap)

    key_cols = ["condition", "n_agvs", "interaction_profile", "interaction_strength", "alpha_technicians", "beta_monitored_agvs"]
    pivot = s2.pivot_table(index=key_cols, columns="policy", values=cost_mean_col, aggfunc="mean").reset_index()
    for method in ["IAGR", "IAGR_C"]:
        for baseline in ["SEP", "II", "IAGR"]:
            if method == baseline:
                continue
            if method in pivot.columns and baseline in pivot.columns:
                pivot[f"{method}_over_{baseline}"] = pivot[method] / pivot[baseline]
                pivot[f"{method}_gain_vs_{baseline}_pct"] = 100.0 * (1.0 - pivot[f"{method}_over_{baseline}"])
    pivot.to_csv(ratios_path, index=False)
    return raw_path, summary_path, ratios_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleet-sizes", default="50,100,200,300", help="Comma-separated fleet sizes")
    parser.add_argument("--profiles", default="convoy,sequential,shared_resource,coordination")
    parser.add_argument("--strengths", default="6,10,15")
    parser.add_argument("--alphas", default="1,2")
    parser.add_argument("--beta", type=int, default=2)
    parser.add_argument("--policies", default="IAGR_C,IAGR,SEP,II", help="Online policies to compare")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--out-dir", default="results/iagrc_campaign_stress")
    parser.add_argument("--campaign-cost-model", action="store_true", help="Enable shared setup/marginal campaign cost accounting")
    parser.add_argument("--campaign-group-size", type=int, default=5)
    parser.add_argument("--campaign-marginal-cost-factor", type=float, default=0.45)
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    fleet_sizes = _parse_csv_ints(args.fleet_sizes)
    profiles = _parse_csv_strings(args.profiles)
    strengths = _parse_csv_floats(args.strengths)
    alphas = _parse_csv_ints(args.alphas)
    policies = _parse_csv_strings(args.policies)

    configs = []
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
            "out_dir": str(out_dir),
            "resume": not args.no_resume,
            "campaign_cost_model": args.campaign_cost_model,
            "campaign_group_size": args.campaign_group_size,
            "campaign_marginal_cost_factor": args.campaign_marginal_cost_factor,
        })

    print(f"Stress-test configurations: {len(configs)}")
    print(f"Policies: {policies}")
    print(f"Workers: {args.workers}")
    print(f"Output directory: {out_dir}")

    failures = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(run_one, cfg): cfg for cfg in configs}
        for i, fut in enumerate(as_completed(futs), start=1):
            cfg = futs[fut]
            try:
                res = fut.result()
                print(f"[{i}/{len(configs)}] {res['status']}: {res['label']}", flush=True)
            except Exception as e:
                msg = f"FAILED cfg={cfg}: {e}\n{traceback.format_exc()}"
                print(msg, flush=True)
                failures.append(msg)

    if failures:
        fail_path = out_dir / "failures.txt"
        fail_path.write_text("\n\n".join(failures), encoding="utf-8")
        raise SystemExit(f"{len(failures)} configurations failed. See {fail_path}")

    raw_path, summary_path, ratios_path = merge_results(out_dir)
    print("Merged outputs:")
    print(f"  {raw_path}")
    print(f"  {summary_path}")
    print(f"  {ratios_path}")


if __name__ == "__main__":
    main()
