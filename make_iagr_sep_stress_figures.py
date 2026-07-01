#!/usr/bin/env python3
"""Generate focused IAGR-vs-SEP stress-test figures."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", default="results/iagrc_campaign_stress/iagr_sep_stress_summary.csv")
    p.add_argument("--ratios", default="results/iagrc_campaign_stress/iagr_sep_stress_ratios.csv")
    p.add_argument("--out-dir", default="figures/iagrc_campaign_stress")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary)
    ratios = pd.read_csv(args.ratios)

    # Normalize possible flattened names.
    summary = summary.rename(columns={c: c[:-1] for c in summary.columns if c.endswith("_")})
    cost_col = "discounted_cost_mean"
    auto_col = "avg_automation_mean"

    # Aggregate across stress dimensions to show overall trend.
    agg_cost = summary.groupby(["n_agvs", "policy"], as_index=False)[cost_col].mean()
    plt.figure(figsize=(7, 4.5))
    for pol, g in agg_cost.groupby("policy"):
        g = g.sort_values("n_agvs")
        plt.plot(g["n_agvs"], g[cost_col], marker="o", label=pol)
    plt.xlabel("Fleet size (AGVs)")
    plt.ylabel("Mean discounted cost")
    plt.title("IAGR-vs-SEP stress test: discounted cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "stress_discounted_cost.pdf")
    plt.savefig(out / "stress_discounted_cost.png", dpi=200)
    plt.close()

    if auto_col in summary.columns:
        agg_auto = summary.groupby(["n_agvs", "policy"], as_index=False)[auto_col].mean()
        plt.figure(figsize=(7, 4.5))
        for pol, g in agg_auto.groupby("policy"):
            g = g.sort_values("n_agvs")
            plt.plot(g["n_agvs"], g[auto_col], marker="o", label=pol)
        plt.xlabel("Fleet size (AGVs)")
        plt.ylabel("Mean automation level")
        plt.title("IAGR-vs-SEP stress test: automation preservation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "stress_avg_automation.pdf")
        plt.savefig(out / "stress_avg_automation.png", dpi=200)
        plt.close()

    # Gains for IAGR and IAGR-C against SEP/II/IAGR.
    rows = []
    for method in ["IAGR", "IAGR_C"]:
        for baseline in ["SEP", "II", "IAGR"]:
            if method == baseline:
                continue
            gain_col = f"{method}_gain_vs_{baseline}_pct"
            if gain_col not in ratios.columns:
                continue
            agg = ratios.groupby("n_agvs", as_index=False)[gain_col].mean().sort_values("n_agvs")
            plt.figure(figsize=(7, 4.5))
            plt.axhline(0.0, linewidth=1)
            plt.plot(agg["n_agvs"], agg[gain_col], marker="o")
            plt.xlabel("Fleet size (AGVs)")
            plt.ylabel(f"{method} gain vs {baseline} (%)")
            plt.title(f"Relative benefit of {method} over {baseline}")
            plt.tight_layout()
            safe_m = method.lower()
            plt.savefig(out / f"stress_{safe_m}_gain_vs_{baseline.lower()}.pdf")
            plt.savefig(out / f"stress_{safe_m}_gain_vs_{baseline.lower()}.png", dpi=200)
            plt.close()

            plt.figure(figsize=(7, 4.5))
            plt.axhline(0.0, linewidth=1)
            for profile, g in ratios.groupby("interaction_profile"):
                g2 = g.groupby("n_agvs", as_index=False)[gain_col].mean().sort_values("n_agvs")
                plt.plot(g2["n_agvs"], g2[gain_col], marker="o", label=profile)
            plt.xlabel("Fleet size (AGVs)")
            plt.ylabel(f"{method} gain vs {baseline} (%)")
            plt.title(f"{method} benefit over {baseline} by interaction profile")
            plt.legend()
            plt.tight_layout()
            plt.savefig(out / f"stress_{safe_m}_gain_vs_{baseline.lower()}_by_profile.pdf")
            plt.savefig(out / f"stress_{safe_m}_gain_vs_{baseline.lower()}_by_profile.png", dpi=200)
            plt.close()

            g = ratios.groupby("n_agvs")[gain_col].agg(["mean", "std", "min", "max"]).reset_index()
            g.insert(1, "method", method)
            g.insert(2, "baseline", baseline)
            rows.append(g)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out / "stress_gain_summary_by_fleet.csv", index=False)

    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()
