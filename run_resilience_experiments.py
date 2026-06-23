#!/usr/bin/env python3
"""
Simulation scaffold for resilience-aware cyber/maintenance orchestration.

This script is intentionally lightweight and self-contained. It is not a PBVI
implementation. It provides a Monte Carlo environment and baseline policies for
the experimental section. Replace `joint_policy` by your PBVI/MOMDP policy when
available.

Experiments included:
  1. baseline comparison
  2. IDS quality sensitivity
  3. interaction-cost sensitivity/profile comparison
  4. fleet-size scaling

Outputs:
  - results_summary.csv
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Callable, Dict, List, Tuple
import argparse
import itertools
import math
import random
import time

import numpy as np
import pandas as pd


class Health(IntEnum):
    HEALTHY = 0
    DEGRADED = 1
    FAILED = 2


class Cyber(IntEnum):
    SECURE = 0
    SUSPECTED = 1
    COMPROMISED = 2
    RECOVERING = 3


class Action(IntEnum):
    NOOP = 0
    DOWNGRADE = 1
    MAINTAIN = 2
    PATCH = 3
    REJUVENATE = 4
    ISOLATE = 5
    STOP = 6


@dataclass(frozen=True)
class Params:
    n_tractors: int = 3
    horizon: int = 365
    gamma: float = 0.99
    n_runs: int = 500
    seed: int = 42

    # Physical transitions
    p_degrade: float = 0.03
    p_fail_from_degraded: float = 0.05
    p_repair_success: float = 0.85

    # Cyber transitions
    p_suspect: float = 0.02
    p_compromise_from_suspected: float = 0.15
    p_patch_success: float = 0.70
    p_rejuvenate_success: float = 0.90

    # Observation model
    false_positive: float = 0.05
    false_negative: float = 0.15

    # Costs per epoch/action
    cost_automation_step: float = 100.0
    cost_failed: float = 800.0
    cost_degraded_phys: float = 150.0
    cost_suspected: float = 100.0
    cost_compromised: float = 500.0
    cost_maintain: float = 400.0
    cost_patch: float = 150.0
    cost_rejuvenate: float = 300.0
    cost_isolate: float = 120.0
    cost_stop: float = 500.0
    cost_downgrade: float = 50.0

    # Interaction costs
    interaction_profile: str = "convoy"  # none, convoy, sequential, shared_resource, coordination
    interaction_strength: float = 100.0

    # Policy thresholds
    belief_patch_threshold: float = 0.45
    belief_rejuvenate_threshold: float = 0.70


@dataclass
class State:
    # Arrays of length n_tractors
    automation: np.ndarray  # 0..5
    health: np.ndarray      # Health
    cyber: np.ndarray       # Cyber
    belief_compromised: np.ndarray  # belief P(c = compromised)


def initial_state(p: Params) -> State:
    return State(
        automation=np.full(p.n_tractors, 5, dtype=int),
        health=np.full(p.n_tractors, Health.HEALTHY, dtype=int),
        cyber=np.full(p.n_tractors, Cyber.SECURE, dtype=int),
        belief_compromised=np.full(p.n_tractors, 0.02, dtype=float),
    )


def observe_alerts(state: State, p: Params, rng: np.random.Generator) -> np.ndarray:
    alerts = np.zeros(p.n_tractors, dtype=int)
    for i, c in enumerate(state.cyber):
        compromised_like = c in (Cyber.COMPROMISED, Cyber.SUSPECTED)
        if compromised_like:
            alerts[i] = rng.random() > p.false_negative
        else:
            alerts[i] = rng.random() < p.false_positive
    return alerts


def update_belief_simple(state: State, actions: np.ndarray, alerts: np.ndarray, p: Params) -> None:
    """Simple approximate Bayesian filter over P(compromised)."""
    b = state.belief_compromised.copy()

    # prediction
    for i, a in enumerate(actions):
        # prior drift toward compromise
        b[i] = b[i] + (1.0 - b[i]) * p.p_suspect * p.p_compromise_from_suspected
        if a == Action.PATCH:
            b[i] *= (1.0 - p.p_patch_success)
        elif a == Action.REJUVENATE:
            b[i] *= (1.0 - p.p_rejuvenate_success)
        elif a == Action.ISOLATE:
            b[i] *= 0.8

    # correction
    for i, alert in enumerate(alerts):
        # likelihoods P(alert | compromised) and P(alert | not compromised)
        if alert:
            like_c = 1.0 - p.false_negative
            like_not = p.false_positive
        else:
            like_c = p.false_negative
            like_not = 1.0 - p.false_positive
        denom = like_c * b[i] + like_not * (1.0 - b[i])
        if denom > 0:
            b[i] = (like_c * b[i]) / denom

    state.belief_compromised[:] = np.clip(b, 0.0, 1.0)


def transition(state: State, actions: np.ndarray, p: Params, rng: np.random.Generator) -> None:
    n = p.n_tractors

    for i in range(n):
        a = actions[i]

        # Physical transition
        h = Health(state.health[i])
        if a == Action.MAINTAIN:
            if rng.random() < p.p_repair_success:
                state.health[i] = Health.HEALTHY
        else:
            if h == Health.HEALTHY and rng.random() < p.p_degrade:
                state.health[i] = Health.DEGRADED
            elif h == Health.DEGRADED and rng.random() < p.p_fail_from_degraded:
                state.health[i] = Health.FAILED

        # Cyber transition
        c = Cyber(state.cyber[i])
        if a == Action.PATCH:
            if rng.random() < p.p_patch_success:
                state.cyber[i] = Cyber.SECURE
        elif a == Action.REJUVENATE:
            if rng.random() < p.p_rejuvenate_success:
                state.cyber[i] = Cyber.RECOVERING
        else:
            if c == Cyber.SECURE and rng.random() < p.p_suspect:
                state.cyber[i] = Cyber.SUSPECTED
            elif c == Cyber.SUSPECTED and rng.random() < p.p_compromise_from_suspected:
                state.cyber[i] = Cyber.COMPROMISED
            elif c == Cyber.RECOVERING:
                state.cyber[i] = Cyber.SECURE

        # Automation update f(a, h', c', u)
        if a == Action.STOP or state.health[i] == Health.FAILED:
            state.automation[i] = 0
        elif a == Action.ISOLATE:
            state.automation[i] = min(state.automation[i], 2)
        elif a == Action.DOWNGRADE:
            state.automation[i] = max(0, state.automation[i] - 1)
        elif state.cyber[i] == Cyber.COMPROMISED:
            state.automation[i] = min(state.automation[i], 2)
        elif state.health[i] == Health.DEGRADED:
            state.automation[i] = min(state.automation[i], 4)
        elif state.health[i] == Health.HEALTHY and state.cyber[i] == Cyber.SECURE:
            # gradual return to nominal if no explicit bad condition
            state.automation[i] = min(5, state.automation[i] + 1)


def local_cost(state: State, actions: np.ndarray, p: Params) -> float:
    automation_cost = np.sum(p.cost_automation_step * (5 - state.automation))

    phys_cost = 0.0
    phys_cost += np.sum(state.health == Health.DEGRADED) * p.cost_degraded_phys
    phys_cost += np.sum(state.health == Health.FAILED) * p.cost_failed

    cyber_cost = 0.0
    cyber_cost += np.sum(state.cyber == Cyber.SUSPECTED) * p.cost_suspected
    cyber_cost += np.sum(state.cyber == Cyber.COMPROMISED) * p.cost_compromised

    action_cost = 0.0
    action_cost += np.sum(actions == Action.MAINTAIN) * p.cost_maintain
    action_cost += np.sum(actions == Action.PATCH) * p.cost_patch
    action_cost += np.sum(actions == Action.REJUVENATE) * p.cost_rejuvenate
    action_cost += np.sum(actions == Action.ISOLATE) * p.cost_isolate
    action_cost += np.sum(actions == Action.STOP) * p.cost_stop
    action_cost += np.sum(actions == Action.DOWNGRADE) * p.cost_downgrade

    return float(automation_cost + phys_cost + cyber_cost + action_cost)


def interaction_cost(state: State, actions: np.ndarray, p: Params) -> float:
    profile = p.interaction_profile
    strength = p.interaction_strength

    if profile == "none" or strength == 0:
        return 0.0

    # performance proxy in [0,1]
    perf = state.automation / 5.0
    perf = np.where(state.health == Health.FAILED, 0.0, perf)

    if profile == "convoy":
        fleet_perf = np.min(perf)
        return float(strength * np.sum(perf - fleet_perf))

    if profile == "sequential":
        # bottleneck throughput relative to nominal
        throughput = np.min(perf)
        return float(strength * p.n_tractors * (1.0 - throughput))

    if profile == "shared_resource":
        # multiple simultaneous interventions create contention
        n_interventions = np.sum(actions != Action.NOOP)
        overload = max(0, n_interventions - 1)
        return float(strength * overload)

    if profile == "coordination":
        # heterogeneity in automation levels creates coordination losses
        return float(strength * np.var(state.automation))

    raise ValueError(f"Unknown interaction profile: {profile}")


def total_cost(state: State, actions: np.ndarray, p: Params) -> float:
    return local_cost(state, actions, p) + interaction_cost(state, actions, p)


Policy = Callable[[State, np.ndarray, Params], np.ndarray]


def reactive_maintenance_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_tractors, Action.NOOP, dtype=int)
    actions[state.health == Health.FAILED] = Action.MAINTAIN
    return actions


def reactive_cyber_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_tractors, Action.NOOP, dtype=int)
    actions[alerts == 1] = Action.PATCH
    actions[state.belief_compromised > p.belief_rejuvenate_threshold] = Action.REJUVENATE
    return actions


def immediate_intervention_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_tractors, Action.NOOP, dtype=int)
    for i in range(p.n_tractors):
        if state.health[i] in (Health.DEGRADED, Health.FAILED):
            actions[i] = Action.MAINTAIN
        elif alerts[i] or state.belief_compromised[i] > p.belief_patch_threshold:
            actions[i] = Action.PATCH
    return actions


def separated_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    """Independent cyber and physical decisions, with fixed priority resolution."""
    phys = reactive_maintenance_policy(state, alerts, p)
    cyber = reactive_cyber_policy(state, alerts, p)
    actions = np.full(p.n_tractors, Action.NOOP, dtype=int)

    for i in range(p.n_tractors):
        if phys[i] != Action.NOOP:
            actions[i] = phys[i]
        elif cyber[i] != Action.NOOP:
            actions[i] = cyber[i]

    return actions


def joint_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    """
    Heuristic placeholder for a coordinated MOMDP/PBVI policy.

    Replace this function by a PBVI policy lookup:
        action = pbvi_policy(observable_state, belief)
    """
    actions = np.full(p.n_tractors, Action.NOOP, dtype=int)

    # Bundle maintenance and cyber recovery when possible.
    for i in range(p.n_tractors):
        high_belief = state.belief_compromised[i] > p.belief_rejuvenate_threshold
        mid_belief = state.belief_compromised[i] > p.belief_patch_threshold

        if state.health[i] == Health.FAILED:
            actions[i] = Action.MAINTAIN
        elif state.health[i] == Health.DEGRADED and (alerts[i] or mid_belief):
            # one site intervention handles both physical and cyber risk in practice;
            # modeled here by prioritizing maintenance and relying on belief update/transition
            actions[i] = Action.MAINTAIN
        elif high_belief:
            actions[i] = Action.REJUVENATE
        elif mid_belief or alerts[i]:
            actions[i] = Action.PATCH
        elif state.health[i] == Health.DEGRADED:
            actions[i] = Action.MAINTAIN

    # If interaction profile is convoy/sequential, recover bottleneck aggressively.
    if p.interaction_profile in ("convoy", "sequential"):
        bottleneck = int(np.argmin(state.automation))
        if state.automation[bottleneck] < 4 and actions[bottleneck] == Action.NOOP:
            if state.health[bottleneck] == Health.DEGRADED:
                actions[bottleneck] = Action.MAINTAIN
            elif state.belief_compromised[bottleneck] > 0.25:
                actions[bottleneck] = Action.PATCH

    return actions


POLICIES: Dict[str, Policy] = {
    "RM": reactive_maintenance_policy,
    "RC": reactive_cyber_policy,
    "II": immediate_intervention_policy,
    "SEP": separated_policy,
    "JOINT": joint_policy,
}


def simulate_once(policy_name: str, p: Params, run_seed: int) -> Dict[str, float]:
    rng = np.random.default_rng(run_seed)
    state = initial_state(p)

    discounted_cost = 0.0
    raw_cost = 0.0
    failsoft_epochs = 0
    safestop_epochs = 0
    intervention_count = 0
    automation_sum = 0.0

    policy = POLICIES[policy_name]

    # initial observations
    alerts = observe_alerts(state, p, rng)
    update_belief_simple(state, np.full(p.n_tractors, Action.NOOP), alerts, p)

    for t in range(p.horizon):
        actions = policy(state, alerts, p)
        c = total_cost(state, actions, p)

        discounted_cost += (p.gamma ** t) * c
        raw_cost += c
        failsoft_epochs += int(np.sum((state.automation > 0) & (state.automation < 5)))
        safestop_epochs += int(np.sum(state.automation == 0))
        intervention_count += int(np.sum(actions != Action.NOOP))
        automation_sum += float(np.mean(state.automation))

        transition(state, actions, p, rng)
        alerts = observe_alerts(state, p, rng)
        update_belief_simple(state, actions, alerts, p)

    return {
        "policy": policy_name,
        "discounted_cost": discounted_cost,
        "raw_cost": raw_cost,
        "failsoft_epochs": failsoft_epochs,
        "safestop_epochs": safestop_epochs,
        "interventions": intervention_count,
        "avg_automation": automation_sum / p.horizon,
    }


def run_condition(label: str, p: Params, policies: List[str]) -> pd.DataFrame:
    rows = []
    start = time.time()
    for pol in policies:
        for r in range(p.n_runs):
            rows.append(simulate_once(pol, p, p.seed + 100000 * hash(label) % 10000 + r))
    df = pd.DataFrame(rows)
    df["condition"] = label
    df["n_tractors"] = p.n_tractors
    df["false_positive"] = p.false_positive
    df["false_negative"] = p.false_negative
    df["interaction_profile"] = p.interaction_profile
    df["interaction_strength"] = p.interaction_strength
    df["runtime_seconds"] = time.time() - start
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["discounted_cost", "raw_cost", "failsoft_epochs", "safestop_epochs", "interventions", "avg_automation"]
    return (
        df.groupby(["condition", "policy", "n_tractors", "false_positive", "false_negative", "interaction_profile", "interaction_strength"])
        [metrics]
        .agg(["mean", "std"])
        .reset_index()
    )


def run_all(base: Params) -> pd.DataFrame:
    policies = ["RM", "RC", "SEP", "II", "JOINT"]
    all_runs = []

    # RQ1: baseline comparison
    all_runs.append(run_condition("baseline_comparison", base, policies))

    # RQ2: IDS quality sensitivity
    for fp, fn in [(0.01, 0.05), (0.05, 0.15), (0.10, 0.25), (0.20, 0.35)]:
        p = replace(base, false_positive=fp, false_negative=fn)
        all_runs.append(run_condition(f"ids_fp{fp}_fn{fn}", p, policies))

    # RQ3: interaction profile/strength
    for profile in ["none", "convoy", "sequential", "shared_resource", "coordination"]:
        for strength in [0.0, 50.0, 100.0, 200.0]:
            p = replace(base, interaction_profile=profile, interaction_strength=strength)
            all_runs.append(run_condition(f"interaction_{profile}_{strength}", p, policies))

    # RQ4: fleet size scaling
    for n in [1, 3, 5, 10, 20]:
        p = replace(base, n_tractors=n, n_runs=max(50, min(base.n_runs, 200)))
        all_runs.append(run_condition(f"fleet_size_{n}", p, policies))

    return pd.concat(all_runs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=300)
    parser.add_argument("--horizon", type=int, default=365)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results_summary.csv")
    parser.add_argument("--raw-output", default="results_raw.csv")
    args = parser.parse_args()

    base = Params(n_runs=args.runs, horizon=args.horizon, seed=args.seed)
    raw = run_all(base)
    summary = summarize(raw)
    raw.to_csv(args.raw_output, index=False)
    summary.to_csv(args.output, index=False)

    print(f"Wrote raw runs to {args.raw_output}")
    print(f"Wrote summary to {args.output}")
    print(summary.head(20).to_string())


if __name__ == "__main__":
    main()
