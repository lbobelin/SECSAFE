#!/usr/bin/env python3
"""Generate paper figures from the v2 AGV campaign outputs."""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = ["discounted_cost", "raw_cost", "failsoft_epochs", "safestop_epochs", "interventions", "unavailable_epochs", "avg_automation"]
POLICY_ORDER = ["PBVI", "IAGR", "SEP", "II", "RM", "RC"]


def read_raw(results_dir: Path) -> pd.DataFrame:
    candidates = [results_dir / "raw_ALL.csv"]
    for c in candidates:
        if c.exists():
            return pd.read_csv(c)
    files = sorted(p for p in results_dir.glob("raw_*.csv") if not p.name.endswith("_ALL.csv"))
    if not files:
        raise FileNotFoundError(f"No raw CSV files found in {results_dir}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "policy" in df.columns:
        df["policy"] = df["policy"].replace({"JOINT": "IAGR"})
    if "param_n_agvs" not in df.columns and "n_agvs" in df.columns:
        df["param_n_agvs"] = df["n_agvs"]
    # Pandas 3 removed/changed some behavior around errors="ignore".
    # Also, not every param_* column is numeric: e.g. param_interaction_profile
    # contains strings such as convoy/sequential. Convert only columns that are
    # actually numeric, and leave categorical parameters untouched.
    for c in df.columns:
        if c.startswith("param_") or c in METRICS + ["n_agvs"]:
            original = df[c]
            converted = pd.to_numeric(original, errors="coerce")
            # Keep the converted values only if the column has at least one
            # numeric value, or if it was already entirely empty. Otherwise we
            # would accidentally turn categorical columns into all-NaN columns.
            if converted.notna().any() or original.isna().all():
                df[c] = converted
    return df


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in [
        "condition", "policy", "sweep", "condition_group",
        "param_n_agvs", "param_interaction_profile", "param_interaction_strength",
        "param_p_alert_secure", "param_p_alert_compromised",
        "param_intervention_cost_multiplier", "param_intervention_downtime_penalty",
        "param_downtime_maintain_epochs", "param_downtime_patch_epochs",
        "param_downtime_rejuvenate_epochs", "param_iagr_w_interaction",
    ] if c in raw.columns]
    s = raw.groupby(group_cols, dropna=False)[METRICS].agg(["mean", "std", "count"]).reset_index()
    s.columns = ["_".join([str(x) for x in col if str(x)]).strip("_") if isinstance(col, tuple) else str(col) for col in s.columns]
    return s


def order_policies(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["_policy_order"] = d["policy"].map(lambda p: POLICY_ORDER.index(p) if p in POLICY_ORDER else 999)
    return d.sort_values("_policy_order")


def save(fig_dir: Path, name: str) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}.pdf")
    plt.savefig(fig_dir / f"{name}.png", dpi=300)
    plt.close()


def bar_policy(summary: pd.DataFrame, condition_group: str, metric: str, title: str, ylabel: str, fig_dir: Path, name: str) -> bool:
    d = summary[summary["condition_group"] == condition_group].copy()
    col = f"{metric}_mean"
    err = f"{metric}_std"
    if d.empty or col not in d.columns:
        return False
    # Prefer calibrated n=2 baseline if duplicate rows exist.
    if condition_group == "rq1_baseline" and "param_n_agvs" in d.columns:
        d = d[d["param_n_agvs"] == 2]
    # Average over conditions only if accidentally duplicated.
    d = d.groupby("policy", as_index=False)[[col, err]].mean(numeric_only=True)
    d = order_policies(d)
    x = np.arange(len(d))
    plt.figure(figsize=(4.2, 2.8))
    plt.bar(x, d[col].astype(float))
    if err in d.columns:
        plt.errorbar(x, d[col].astype(float), yerr=d[err].fillna(0).astype(float), fmt="none", capsize=3)
    plt.xticks(x, d["policy"], rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    save(fig_dir, name)
    return True


def line_policy(summary: pd.DataFrame, d: pd.DataFrame, xcol: str, metric: str, title: str, xlabel: str, ylabel: str, fig_dir: Path, name: str) -> bool:
    col = f"{metric}_mean"
    if d.empty or xcol not in d.columns or col not in d.columns:
        return False
    plt.figure(figsize=(4.5, 2.9))
    for pol in POLICY_ORDER:
        dd = d[d["policy"] == pol].dropna(subset=[xcol, col]).sort_values(xcol)
        if not dd.empty:
            plt.plot(dd[xcol].astype(float), dd[col].astype(float), marker="o", label=pol)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=8)
    save(fig_dir, name)
    return True


def make_all(summary: pd.DataFrame, fig_dir: Path) -> Dict[str, bool]:
    made: Dict[str, bool] = {}
    made["fig_rq1_policy_cost"] = bar_policy(summary, "rq1_baseline", "discounted_cost", "Policy comparison", "Mean discounted cost", fig_dir, "fig_rq1_policy_cost")
    made["fig_rq1_interventions"] = bar_policy(summary, "rq1_baseline", "interventions", "Intervention frequency", "Mean interventions", fig_dir, "fig_rq1_interventions")
    made["fig_rq1_failsoft"] = bar_policy(summary, "rq1_baseline", "failsoft_epochs", "Degraded operation", "Mean fail-soft epochs", fig_dir, "fig_rq1_failsoft")
    made["fig_rq1_unavailable"] = bar_policy(summary, "rq1_baseline", "unavailable_epochs", "Intervention-induced unavailability", "Mean unavailable epochs", fig_dir, "fig_rq1_unavailable")

    d = summary[summary["condition_group"] == "rq2_observation_quality"].copy()
    made["fig_rq2_observation_quality"] = line_policy(summary, d, "param_p_alert_compromised", "discounted_cost", "Cyber observation quality", "P(alert | compromised)", "Mean discounted cost", fig_dir, "fig_rq2_observation_quality")

    d = summary[summary["condition_group"] == "rq3_interactions"].copy()
    if not d.empty and "param_interaction_profile" in d.columns:
        for profile in sorted(d["param_interaction_profile"].dropna().astype(str).unique()):
            dd = d[d["param_interaction_profile"].astype(str) == profile]
            made[f"fig_rq3_interaction_{profile}"] = line_policy(summary, dd, "param_interaction_strength", "discounted_cost", f"Interaction profile: {profile}", "Interaction strength", "Mean discounted cost", fig_dir, f"fig_rq3_interaction_{profile}")

    d = summary[summary["condition_group"] == "rq4_pbvi_scale"].copy()
    made["fig_rq4_pbvi_scale"] = line_policy(summary, d, "param_n_agvs", "discounted_cost", "PBVI small-fleet scaling", "Number of AGVs", "Mean discounted cost", fig_dir, "fig_rq4_pbvi_scale")

    d = summary[summary["condition_group"] == "rq4_fleet_scale"].copy()
    made["fig_rq4_fleet_scale_cost"] = line_policy(summary, d, "param_n_agvs", "discounted_cost", "Fleet scalability", "Number of AGVs", "Mean discounted cost", fig_dir, "fig_rq4_fleet_scale_cost")
    made["fig_rq4_fleet_scale_interventions"] = line_policy(summary, d, "param_n_agvs", "interventions", "Fleet scalability: interventions", "Number of AGVs", "Mean interventions", fig_dir, "fig_rq4_fleet_scale_interventions")

    d = summary[summary["condition_group"] == "rq5_intervention_cost"].copy()
    made["fig_rq5_intervention_cost"] = line_policy(summary, d, "param_intervention_cost_multiplier", "discounted_cost", "Intervention cost sensitivity", "Intervention cost multiplier", "Mean discounted cost", fig_dir, "fig_rq5_intervention_cost")

    d = summary[summary["condition_group"] == "rq5_intervention_downtime"].copy()
    made["fig_rq5_intervention_downtime"] = line_policy(summary, d, "param_intervention_downtime_penalty", "discounted_cost", "Intervention downtime sensitivity", "Downtime penalty per intervention", "Mean discounted cost", fig_dir, "fig_rq5_intervention_downtime")

    d = summary[summary["condition_group"] == "rq5_intervention_duration"].copy()
    made["fig_rq5_intervention_duration"] = line_policy(summary, d, "param_downtime_maintain_epochs", "discounted_cost", "Maintenance duration sensitivity", "Maintenance downtime epochs", "Mean discounted cost", fig_dir, "fig_rq5_intervention_duration")

    d = summary[summary["condition_group"] == "rq6_iagr_interaction_weight"].copy()
    made["fig_rq6_iagr_weight"] = line_policy(summary, d, "param_iagr_w_interaction", "discounted_cost", "IAGR interaction-weight sensitivity", "IAGR interaction weight", "Mean discounted cost", fig_dir, "fig_rq6_iagr_weight")

    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="paper_experiments_v3/outputs")
    ap.add_argument("--fig-dir", default="paper_experiments_v3/figures")
    args = ap.parse_args()
    results_dir = Path(args.results_dir)
    fig_dir = Path(args.fig_dir)
    raw = normalize(read_raw(results_dir))
    summary = summarize(raw)
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(fig_dir / "all_raw_for_figures.csv", index=False)
    summary.to_csv(fig_dir / "all_summary_for_figures.csv", index=False)
    made = make_all(summary, fig_dir)
    report = ["Generated figures:"] + [f"- {k}: {'OK' if v else 'missing input'}" for k, v in sorted(made.items())]
    (fig_dir / "figure_report.txt").write_text("\n".join(report))
    print(f"Wrote figures to {fig_dir}")
    print(fig_dir / "figure_report.txt")


if __name__ == "__main__":
    main()
