#!/usr/bin/env python3
"""
Reproducible single-node/Slurm AGV experiment campaign runner (v3).

Design goal
-----------
Run one independent parameter configuration per Python process.

Version 3 supports two execution modes:
  1. run-one: execute a single manifest row;
  2. schedule-local: reserve one Slurm allocation and keep many independent
     Python interpreters running concurrently on that node.

Every task writes one raw CSV and one summary CSV. Completed tasks are skipped
automatically unless --force is used, making the campaign restartable after
wall-time kills or node failures.

Typical workflow
----------------
1. Create a manifest:
   python agv_campaign.py prepare --out-root paper_experiments_v3

2. Run locally inside one Slurm allocation:
   python agv_campaign.py schedule-local --out-root paper_experiments_v3 --workers 63

3. Merge and plot:
   python agv_campaign.py merge --out-root paper_experiments_v3
   python make_paper_figures.py --results-dir paper_experiments_v3/outputs --fig-dir paper_experiments_v3/figures
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # needed for dataclasses on Python 3.12
    spec.loader.exec_module(mod)
    return mod


def build_base_params(mod: Any, args: argparse.Namespace):
    return mod.Params(
        n_runs=args.runs,
        horizon=args.horizon,
        seed=args.seed,
        pbvi_iterations=args.pbvi_iterations,
        pbvi_belief_points=args.belief_points,
        pbvi_max_obs_states=args.pbvi_max_obs_states,
        pbvi_rollouts=args.pbvi_rollouts,
        intervention_cost_multiplier=args.base_intervention_multiplier,
        intervention_downtime_penalty=args.base_downtime_penalty,
        downtime_patch_epochs=args.downtime_patch_epochs,
        downtime_rejuvenate_epochs=args.downtime_rejuvenate_epochs,
        downtime_maintain_epochs=args.downtime_maintain_epochs,
        downtime_isolate_epochs=args.downtime_isolate_epochs,
    )


def job_dict(name: str, sweep: str, group: str, updates: Dict[str, Any], policies: List[str]) -> Dict[str, Any]:
    return {
        "condition_name": name,
        "sweep": sweep,
        "condition_group": group,
        "updates": updates,
        "policies": policies,
    }


def build_jobs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    policies = ["RM", "RC", "SEP", "II", "IAGR"]
    jobs: List[Dict[str, Any]] = []

    base_cal = {
        "intervention_cost_multiplier": args.base_intervention_multiplier,
        "intervention_downtime_penalty": args.base_downtime_penalty,
        "downtime_patch_epochs": args.downtime_patch_epochs,
        "downtime_rejuvenate_epochs": args.downtime_rejuvenate_epochs,
        "downtime_maintain_epochs": args.downtime_maintain_epochs,
        "downtime_isolate_epochs": args.downtime_isolate_epochs,
        "iagr_w_interaction": args.base_iagr_interaction_weight,
    }

    # RQ1: main policy comparison on PBVI-compatible small fleet.
    jobs.append(job_dict(
        "rq1_baseline_n2_calibrated", "core", "rq1_baseline",
        {**base_cal, "n_agvs": 2, "interaction_profile": "convoy", "interaction_strength": 3.0},
        policies,
    ))

    # RQ2: observation quality.
    for ps, pc in [(0.01, 0.90), (0.02, 0.85), (0.05, 0.75), (0.10, 0.65)]:
        jobs.append(job_dict(
            f"rq2_obs_ps{ps}_pc{pc}", "core", "rq2_observation_quality",
            {**base_cal, "n_agvs": 2, "p_alert_secure": ps, "p_alert_compromised": pc},
            policies,
        ))

    # RQ3: interaction profile and strength.
    for profile in ["none", "convoy", "sequential", "shared_resource", "coordination"]:
        for strength in [0.0, 1.0, 3.0, 6.0]:
            jobs.append(job_dict(
                f"rq3_{profile}_{strength}", "core", "rq3_interactions",
                {**base_cal, "n_agvs": 2, "interaction_profile": profile, "interaction_strength": strength},
                policies,
            ))

    # RQ4a: PBVI scalability on small fleets.
    for n in [1, 2, 3]:
        jobs.append(job_dict(
            f"rq4_pbvi_scale_n{n}", "pbvi_scale", "rq4_pbvi_scale",
            {**base_cal, "n_agvs": n, "n_runs": min(args.runs, 80),
             "pbvi_max_obs_states": 64 if n <= 2 else 96,
             "pbvi_belief_points": 16 if n <= 2 else 12},
            policies,
        ))

    # RQ4b: realistic fleet scalability with online heuristics/baselines.
    for n in [1, 2, 3, 5, 10, 20, 50]:
        jobs.append(job_dict(
            f"rq4_fleet_scale_n{n}", "fleet_scale", "rq4_fleet_scale",
            {**base_cal, "n_agvs": n, "n_runs": min(args.runs, 80)},
            ["RM", "RC", "SEP", "II", "IAGR"],
        ))

    # RQ5a: intervention cost multiplier. This directly tests whether II wins
    # because one-shot interventions are too cheap.
    for mult in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]:
        jobs.append(job_dict(
            f"rq5_intervention_cost_x{mult}", "cost_sensitivity", "rq5_intervention_cost",
            {"n_agvs": 2, "interaction_profile": "convoy", "interaction_strength": 3.0,
             "intervention_cost_multiplier": mult,
             "intervention_downtime_penalty": args.base_downtime_penalty},
            policies,
        ))

    # RQ5b: intervention downtime penalty. This tests the second hypothesis:
    # interventions are currently too painless because they do not disrupt service.
    for penalty in [0.0, 0.5, 1.0, 2.0, 5.0]:
        jobs.append(job_dict(
            f"rq5_intervention_downtime_x{penalty}", "cost_sensitivity", "rq5_intervention_downtime",
            {"n_agvs": 2, "interaction_profile": "convoy", "interaction_strength": 3.0,
             "intervention_cost_multiplier": args.base_intervention_multiplier,
             "intervention_downtime_penalty": penalty},
            policies,
        ))

    # RQ5c: explicit intervention duration sweep. This keeps the one-shot
    # downtime penalty fixed and changes how many future epochs an intervention
    # makes the AGV unavailable in Monte Carlo evaluation.
    for maint_dur in [1, 2, 3, 5]:
        jobs.append(job_dict(
            f"rq5_maintenance_duration_{maint_dur}", "cost_sensitivity", "rq5_intervention_duration",
            {**base_cal, "n_agvs": 2, "interaction_profile": "convoy", "interaction_strength": 3.0,
             "downtime_maintain_epochs": maint_dur},
            policies,
        ))

    # RQ6: IAGR interaction-weight sensitivity. This checks when explicitly
    # optimizing interaction costs helps or hurts the scalable heuristic.
    for w in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
        jobs.append(job_dict(
            f"rq6_iagr_interaction_w{w}", "iagr_sensitivity", "rq6_iagr_interaction_weight",
            {**base_cal, "n_agvs": 10, "n_runs": min(args.runs, 80),
             "interaction_profile": "convoy", "interaction_strength": 3.0,
             "iagr_w_interaction": w},
            ["II", "IAGR", "SEP"],
        ))

    return jobs


def manifest_path(out_root: Path) -> Path:
    return out_root / "manifest" / "campaign_jobs.json"


def prepare(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    (out_root / "manifest").mkdir(parents=True, exist_ok=True)
    (out_root / "outputs").mkdir(parents=True, exist_ok=True)
    (out_root / "policy_cache").mkdir(parents=True, exist_ok=True)
    (out_root / "figures").mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    payload = {
        "campaign": "agv_resilience_v3",
        "description": "One independent Python process per parameter configuration; restartable single-node scheduler.",
        "base": {
            "runs": args.runs,
            "horizon": args.horizon,
            "seed": args.seed,
            "base_intervention_multiplier": args.base_intervention_multiplier,
            "base_downtime_penalty": args.base_downtime_penalty,
            "downtime_patch_epochs": args.downtime_patch_epochs,
            "downtime_rejuvenate_epochs": args.downtime_rejuvenate_epochs,
            "downtime_maintain_epochs": args.downtime_maintain_epochs,
            "downtime_isolate_epochs": args.downtime_isolate_epochs,
            "base_iagr_interaction_weight": args.base_iagr_interaction_weight,
            "pbvi_iterations": args.pbvi_iterations,
            "belief_points": args.belief_points,
            "pbvi_max_obs_states": args.pbvi_max_obs_states,
            "pbvi_rollouts": args.pbvi_rollouts,
        },
        "jobs": jobs,
    }
    manifest_path(out_root).write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Prepared {len(jobs)} jobs")
    print(f"Manifest: {manifest_path(out_root)}")
    print(f"Slurm array range: 0-{len(jobs)-1}")


def load_manifest(out_root: Path) -> Dict[str, Any]:
    path = manifest_path(out_root)
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}. Run prepare first.")
    return json.loads(path.read_text())


def output_prefix(job: Dict[str, Any]) -> str:
    return f"{job['sweep']}_{job['condition_name']}"


def already_done(out_dir: Path, job: Dict[str, Any]) -> bool:
    prefix = output_prefix(job)
    return any(out_dir.glob(f"summary_{prefix}_*.csv")) and any(out_dir.glob(f"raw_{prefix}_*.csv"))


def run_one(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    data = load_manifest(out_root)
    jobs = data["jobs"]
    if args.index < 0 or args.index >= len(jobs):
        raise IndexError(f"Job index {args.index} outside 0..{len(jobs)-1}")

    job = jobs[args.index]
    out_dir = out_root / "outputs"
    policy_dir = out_root / "policy_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    if already_done(out_dir, job) and not args.force:
        print(f"[SKIP] {args.index}: {job['condition_name']} already has raw+summary CSV")
        return

    mod = load_module("agv_sim_v2", Path(args.script))
    mgr = load_module("agv_manager_v2", Path(args.manager))

    # Build base Params from manifest values to make the campaign reproducible.
    base_cfg = data["base"]
    base = mod.Params(
        n_runs=int(base_cfg["runs"]),
        horizon=int(base_cfg["horizon"]),
        seed=int(base_cfg["seed"]),
        pbvi_iterations=int(base_cfg["pbvi_iterations"]),
        pbvi_belief_points=int(base_cfg["belief_points"]),
        pbvi_max_obs_states=int(base_cfg["pbvi_max_obs_states"]),
        pbvi_rollouts=int(base_cfg["pbvi_rollouts"]),
        intervention_cost_multiplier=float(base_cfg["base_intervention_multiplier"]),
        intervention_downtime_penalty=float(base_cfg["base_downtime_penalty"]),
        downtime_patch_epochs=int(base_cfg.get("downtime_patch_epochs", 1)),
        downtime_rejuvenate_epochs=int(base_cfg.get("downtime_rejuvenate_epochs", 2)),
        downtime_maintain_epochs=int(base_cfg.get("downtime_maintain_epochs", 3)),
        downtime_isolate_epochs=int(base_cfg.get("downtime_isolate_epochs", 1)),
        iagr_w_interaction=float(base_cfg.get("base_iagr_interaction_weight", 1.0)),
    )
    p = dataclasses.replace(base, **job["updates"])

    print(f"[RUN] index={args.index} condition={job['condition_name']} sweep={job['sweep']}")
    print(f"      n_agvs={p.n_agvs}, runs={p.n_runs}, horizon={p.horizon}, "
          f"cost_mult={p.intervention_cost_multiplier}, downtime_penalty={p.intervention_downtime_penalty}, "
          f"maint_dur={p.downtime_maintain_epochs}, patch_dur={p.downtime_patch_epochs}, "
          f"iagr_w_int={p.iagr_w_interaction}")

    raw, summary = mgr.run_one_condition(
        mod=mod,
        p=p,
        condition_name=job["condition_name"],
        policies=list(job["policies"]),
        sweep_name=job["sweep"],
        condition_group=job["condition_group"],
        out_dir=out_dir,
        policy_dir=policy_dir,
        overwrite_policy=args.overwrite_policy,
    )
    print(f"[DONE] raw rows={len(raw)} summary rows={len(summary)}")


def task_status(out_dir: Path, jobs: List[Dict[str, Any]]) -> Tuple[List[int], List[int]]:
    """Return (done_indices, pending_indices) according to raw+summary CSVs."""
    done, pending = [], []
    for idx, job in enumerate(jobs):
        if already_done(out_dir, job):
            done.append(idx)
        else:
            pending.append(idx)
    return done, pending


def schedule_local(args: argparse.Namespace) -> None:
    """Run pending manifest tasks as independent Python processes on one allocation.

    This intentionally avoids Python threading/multiprocessing. Each task is a
    separate interpreter launched via subprocess.Popen. The scheduler process
    keeps at most --workers child processes alive; when one finishes, the next
    pending task is launched.
    """
    out_root = Path(args.out_root)
    data = load_manifest(out_root)
    jobs = data["jobs"]
    out_dir = out_root / "outputs"
    log_dir = out_root / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    done, pending = task_status(out_dir, jobs)
    if args.only_missing:
        queue = list(pending)
    else:
        queue = list(range(len(jobs)))

    if args.limit is not None:
        queue = queue[: args.limit]

    workers = args.workers
    if workers <= 0:
        # Leave one CPU for the scheduler when SLURM_CPUS_PER_TASK is set.
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "2"))
        workers = max(1, cpus - 1)

    print(f"Campaign: {data.get('campaign', 'unknown')}", flush=True)
    print(f"Total jobs: {len(jobs)} | done: {len(done)} | pending: {len(pending)}", flush=True)
    print(f"Scheduler workers: {workers}", flush=True)
    print(f"Queue length: {len(queue)}", flush=True)

    running: Dict[int, Tuple[subprocess.Popen, Any, Any, Path, Path]] = {}
    completed = 0
    failed: List[int] = []
    skipped = 0

    def launch(idx: int) -> None:
        nonlocal skipped
        job = jobs[idx]
        if already_done(out_dir, job) and not args.force:
            print(f"[SKIP] {idx}: {job['condition_name']} already complete", flush=True)
            skipped += 1
            return

        stdout_path = log_dir / f"task_{idx:04d}_{job['condition_name']}.out"
        stderr_path = log_dir / f"task_{idx:04d}_{job['condition_name']}.err"
        fout = stdout_path.open("w")
        ferr = stderr_path.open("w")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-one",
            "--out-root", str(out_root),
            "--index", str(idx),
            "--script", args.script,
            "--manager", args.manager,
        ]
        if args.force:
            cmd.append("--force")
        if args.overwrite_policy:
            cmd.append("--overwrite-policy")
        print(f"[LAUNCH] {idx}: {job['condition_name']}", flush=True)
        proc = subprocess.Popen(cmd, stdout=fout, stderr=ferr)
        running[idx] = (proc, fout, ferr, stdout_path, stderr_path)

    queue_iter = iter(queue)
    no_more = False

    try:
        while running or not no_more:
            while len(running) < workers and not no_more:
                try:
                    idx = next(queue_iter)
                except StopIteration:
                    no_more = True
                    break
                launch(idx)

            if not running:
                break

            time.sleep(args.poll_seconds)
            finished = []
            for idx, (proc, fout, ferr, stdout_path, stderr_path) in list(running.items()):
                rc = proc.poll()
                if rc is not None:
                    fout.close()
                    ferr.close()
                    finished.append((idx, rc, stdout_path, stderr_path))
                    del running[idx]

            for idx, rc, stdout_path, stderr_path in finished:
                completed += 1
                if rc == 0:
                    print(f"[DONE] {idx} ({completed} finished, {len(running)} running)", flush=True)
                else:
                    failed.append(idx)
                    print(f"[FAIL] {idx} rc={rc}; see {stderr_path}", flush=True)

    except KeyboardInterrupt:
        print("Interrupted: terminating child processes...", flush=True)
        for idx, (proc, fout, ferr, _, _) in running.items():
            proc.terminate()
            fout.close()
            ferr.close()
        raise

    # Write status and rebuild merged outputs at the end.
    status = {
        "total_jobs": len(jobs),
        "initial_done": len(done),
        "initial_pending": len(pending),
        "queued": len(queue),
        "skipped": skipped,
        "completed_child_processes": completed,
        "failed_indices": failed,
    }
    (out_root / "manifest" / "local_scheduler_status.json").write_text(json.dumps(status, indent=2, sort_keys=True))

    if failed:
        print(f"Finished with failures: {failed}", flush=True)
        raise SystemExit(2)

    print("All scheduled tasks finished. Rebuilding merged CSVs...", flush=True)
    merge_args = argparse.Namespace(out_root=str(out_root))
    merge(merge_args)
    print("Done.", flush=True)


def merge(args: argparse.Namespace) -> None:
    out_root = Path(args.out_root)
    out_dir = out_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(p for p in out_dir.glob("raw_*.csv") if not p.name.endswith("_ALL.csv"))
    summary_files = sorted(p for p in out_dir.glob("summary_*.csv") if not p.name.endswith("_ALL.csv"))

    if not raw_files:
        raise FileNotFoundError(f"No raw_*.csv files in {out_dir}")
    if not summary_files:
        raise FileNotFoundError(f"No summary_*.csv files in {out_dir}")

    raw_all = pd.concat([pd.read_csv(p) for p in raw_files], ignore_index=True)
    summary_all = pd.concat([pd.read_csv(p) for p in summary_files], ignore_index=True)

    raw_all.to_csv(out_dir / "raw_ALL.csv", index=False)
    summary_all.to_csv(out_dir / "summary_ALL.csv", index=False)

    # Also write per-sweep merged files for backward compatibility with older plotting scripts.
    for sweep in sorted(raw_all["sweep"].dropna().unique()):
        raw_all[raw_all["sweep"] == sweep].to_csv(out_dir / f"raw_{sweep}_ALL.csv", index=False)
        summary_all[summary_all["sweep"] == sweep].to_csv(out_dir / f"summary_{sweep}_ALL.csv", index=False)

    # Report missing jobs.
    missing = []
    try:
        data = load_manifest(out_root)
        for idx, job in enumerate(data["jobs"]):
            if not already_done(out_dir, job):
                missing.append((idx, job["condition_name"]))
    except Exception:
        pass

    report = out_root / "manifest" / "campaign_status.txt"
    lines = [
        f"Raw files: {len(raw_files)}",
        f"Summary files: {len(summary_files)}",
        f"Raw rows: {len(raw_all)}",
        f"Summary rows: {len(summary_all)}",
        "",
        "Missing jobs:",
    ]
    if missing:
        lines.extend([f"- {i}: {name}" for i, name in missing])
    else:
        lines.append("- none")
    report.write_text("\n".join(lines))

    print(f"Wrote {out_dir / 'raw_ALL.csv'}")
    print(f"Wrote {out_dir / 'summary_ALL.csv'}")
    print(f"Status: {report}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    common_prepare = argparse.ArgumentParser(add_help=False)
    common_prepare.add_argument("--out-root", default="paper_experiments_v3")
    common_prepare.add_argument("--runs", type=int, default=150)
    common_prepare.add_argument("--horizon", type=int, default=120)
    common_prepare.add_argument("--seed", type=int, default=42)
    common_prepare.add_argument("--base-intervention-multiplier", type=float, default=3.0)
    common_prepare.add_argument("--base-downtime-penalty", type=float, default=1.0)
    common_prepare.add_argument("--downtime-patch-epochs", type=int, default=1)
    common_prepare.add_argument("--downtime-rejuvenate-epochs", type=int, default=2)
    common_prepare.add_argument("--downtime-maintain-epochs", type=int, default=3)
    common_prepare.add_argument("--downtime-isolate-epochs", type=int, default=1)
    common_prepare.add_argument("--base-iagr-interaction-weight", type=float, default=1.0)
    common_prepare.add_argument("--pbvi-iterations", type=int, default=10)
    common_prepare.add_argument("--belief-points", type=int, default=16)
    common_prepare.add_argument("--pbvi-max-obs-states", type=int, default=64)
    common_prepare.add_argument("--pbvi-rollouts", type=int, default=200)

    p_prepare = sub.add_parser("prepare", parents=[common_prepare])
    p_prepare.set_defaults(func=prepare)

    p_run = sub.add_parser("run-one")
    p_run.add_argument("--out-root", default="paper_experiments_v3")
    p_run.add_argument("--index", type=int, required=True)
    p_run.add_argument("--script", default="run_agv_momdp_experiments.py")
    p_run.add_argument("--manager", default="agv_experiment_manager_v2.py")
    p_run.add_argument("--force", action="store_true")
    p_run.add_argument("--overwrite-policy", action="store_true")
    p_run.set_defaults(func=run_one)

    p_sched = sub.add_parser("schedule-local")
    p_sched.add_argument("--out-root", default="paper_experiments_v3")
    p_sched.add_argument("--workers", type=int, default=0,
                         help="Number of concurrent Python child processes. Default: SLURM_CPUS_PER_TASK-1.")
    p_sched.add_argument("--script", default="run_agv_momdp_experiments.py")
    p_sched.add_argument("--manager", default="agv_experiment_manager_v2.py")
    p_sched.add_argument("--poll-seconds", type=float, default=2.0)
    p_sched.add_argument("--only-missing", action="store_true", default=True,
                         help="Run only tasks without raw+summary CSVs (default).")
    p_sched.add_argument("--force", action="store_true", help="Rerun tasks even if CSVs exist.")
    p_sched.add_argument("--overwrite-policy", action="store_true")
    p_sched.add_argument("--limit", type=int, default=None, help="Debug: run at most N queued tasks.")
    p_sched.set_defaults(func=schedule_local)

    p_merge = sub.add_parser("merge")
    p_merge.add_argument("--out-root", default="paper_experiments_v3")
    p_merge.set_defaults(func=merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
