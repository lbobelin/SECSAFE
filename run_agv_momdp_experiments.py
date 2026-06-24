#!/usr/bin/env python3
"""
AGV resilience orchestration experiments with an offline belief-grid/PBVI-style
MOMDP solver for small fleets and heuristic baselines for larger fleets.

This script extends the earlier scaffold by:
  - renaming tractors to AGVs;
  - using AGV-oriented state/action/cost parameters;
  - adding a 4-state cyber observation model P(alert | cyber_state);
  - enforcing technician (alpha) and monitoring (beta) capacity constraints;
  - adding an offline approximate MOMDP solver over representative beliefs;
  - using the offline policy as a lookup during Monte Carlo experiments.

Important modeling note
-----------------------
The offline solver is a finite belief-grid value-iteration implementation in the
spirit of PBVI. It enumerates all observable states for small fleets, evaluates a
finite set of representative beliefs, and stores a policy table
    (observable_state, nearest_belief_point) -> joint_action.
This is intended for small fleets (default: 2 AGVs). For larger fleets, use the
heuristic JOINT policy and report scalability limitations.
"""

from __future__ import annotations

from dataclasses import dataclass, replace, field
from enum import IntEnum
from functools import lru_cache
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import argparse
import itertools
import pickle
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
    MONITOR = 7  # trust degradation / enhanced monitoring


HEALTH_STATES = tuple(int(h) for h in Health)
CYBER_STATES = tuple(int(c) for c in Cyber)
AUTOMATION_LEVELS = tuple(range(6))  # 0 stopped, 5 nominal autonomous
ACTIONS = tuple(int(a) for a in Action)


@dataclass(frozen=True)
class Params:
    # Fleet / simulation
    n_agvs: int = 3
    horizon: int = 365
    gamma: float = 0.99
    n_runs: int = 300
    seed: int = 42

    # Resource constraints
    alpha_technicians: int = 1       # max simultaneous heavy physical actions
    beta_monitored_agvs: int = 2     # max simultaneous trust-monitoring actions

    # Physical transitions per epoch
    p_degrade: float = 0.03
    p_fail_from_degraded: float = 0.07
    p_repair_success: float = 0.90

    # Cyber transitions per epoch
    p_suspect: float = 0.02
    p_compromise_from_suspected: float = 0.15
    p_recovering_to_secure: float = 0.80
    p_patch_success: float = 0.70
    p_rejuvenate_success: float = 0.90
    p_monitor_suspected_to_secure: float = 0.15

    # Observation model P(alert=1 | cyber_state)
    p_alert_secure: float = 0.02
    p_alert_suspected: float = 0.40
    p_alert_compromised: float = 0.85
    p_alert_recovering: float = 0.10

    # Normalized costs per epoch/action. Values are relative to one unit of
    # degraded-operation cost; tune and run sensitivity analysis for the paper.
    cost_automation_step: float = 1.0
    cost_degraded_phys: float = 1.5
    cost_failed: float = 12.0
    cost_suspected: float = 2.0
    cost_compromised: float = 8.0
    cost_monitor: float = 0.5
    cost_patch: float = 2.0
    cost_rejuvenate: float = 3.0
    cost_maintain: float = 5.0
    cost_isolate: float = 8.0
    cost_stop: float = 10.0
    cost_downgrade: float = 0.5

    # Interaction costs
    interaction_profile: str = "convoy"  # none, convoy, sequential, shared_resource, coordination
    interaction_strength: float = 3.0
    resource_overload_penalty: float = 20.0

    # Heuristic policy thresholds
    belief_patch_threshold: float = 0.35
    belief_rejuvenate_threshold: float = 0.65

    # Offline solver settings
    pbvi_n_agvs: int = 2
    pbvi_iterations: int = 12
    pbvi_belief_points: int = 20
    pbvi_seed: int = 7
    use_pbvi_for_baseline: bool = True


@dataclass
class State:
    automation: np.ndarray  # length n_agvs, values 0..5
    health: np.ndarray      # Health values
    cyber: np.ndarray       # Cyber values (true hidden state in simulation)
    belief_joint: Optional[np.ndarray] = None  # distribution over joint cyber states
    belief_compromised: Optional[np.ndarray] = None  # marginals, derived from belief_joint


@dataclass
class OfflinePolicy:
    n_agvs: int
    hidden_states: List[Tuple[int, ...]]
    obs_states: List[Tuple[int, ...]]
    belief_points: np.ndarray
    policy: Dict[Tuple[Tuple[int, ...], int], Tuple[int, ...]]
    values: Dict[Tuple[Tuple[int, ...], int], float]
    metadata: Dict[str, float] = field(default_factory=dict)


_CURRENT_OFFLINE_POLICY: Optional[OfflinePolicy] = None


def alert_prob(c: int, p: Params) -> float:
    if c == Cyber.SECURE:
        return p.p_alert_secure
    if c == Cyber.SUSPECTED:
        return p.p_alert_suspected
    if c == Cyber.COMPROMISED:
        return p.p_alert_compromised
    if c == Cyber.RECOVERING:
        return p.p_alert_recovering
    raise ValueError(f"Unknown cyber state: {c}")


def initial_joint_belief(n: int) -> np.ndarray:
    # Prior per AGV: mostly secure, small probability suspected/compromised.
    local = np.array([0.97, 0.02, 0.01, 0.0], dtype=float)
    b = local.copy()
    for _ in range(n - 1):
        b = np.kron(b, local)
    return b / b.sum()


def cyber_marginals_from_joint(belief: np.ndarray, hidden_states: List[Tuple[int, ...]], n: int) -> np.ndarray:
    marg = np.zeros(n, dtype=float)
    for prob, c_tuple in zip(belief, hidden_states):
        for i, c in enumerate(c_tuple):
            if c == Cyber.COMPROMISED:
                marg[i] += prob
    return marg


def initial_state(p: Params, hidden_states: Optional[List[Tuple[int, ...]]] = None) -> State:
    belief = None
    marg = None
    if hidden_states is not None:
        belief = initial_joint_belief(p.n_agvs)
        marg = cyber_marginals_from_joint(belief, hidden_states, p.n_agvs)
    return State(
        automation=np.full(p.n_agvs, 5, dtype=int),
        health=np.full(p.n_agvs, Health.HEALTHY, dtype=int),
        cyber=np.full(p.n_agvs, Cyber.SECURE, dtype=int),
        belief_joint=belief,
        belief_compromised=marg if marg is not None else np.full(p.n_agvs, 0.01, dtype=float),
    )


def obs_tuple_from_state(state: State) -> Tuple[int, ...]:
    # Flattened as (a_0,h_0,a_1,h_1,...)
    out: List[int] = []
    for a, h in zip(state.automation, state.health):
        out.extend([int(a), int(h)])
    return tuple(out)


def split_obs_tuple(obs: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    a = tuple(obs[0::2])
    h = tuple(obs[1::2])
    return a, h


def make_state_from_obs_hidden(obs: Tuple[int, ...], c_tuple: Tuple[int, ...], actions: Optional[Tuple[int, ...]] = None) -> State:
    a, h = split_obs_tuple(obs)
    n = len(a)
    return State(
        automation=np.array(a, dtype=int),
        health=np.array(h, dtype=int),
        cyber=np.array(c_tuple, dtype=int),
        belief_joint=None,
        belief_compromised=np.array([1.0 if c == Cyber.COMPROMISED else 0.0 for c in c_tuple]),
    )


def action_counts(actions: Sequence[int]) -> Tuple[int, int]:
    heavy_physical = sum(1 for a in actions if a == Action.MAINTAIN)
    monitored = sum(1 for a in actions if a == Action.MONITOR)
    return heavy_physical, monitored


def action_feasible(actions: Sequence[int], p: Params) -> bool:
    heavy, monitored = action_counts(actions)
    return heavy <= p.alpha_technicians and monitored <= p.beta_monitored_agvs


def enumerate_actions(n: int, p: Params, reduced: bool = False) -> List[Tuple[int, ...]]:
    # For PBVI, reduced=True avoids exploding action spaces while keeping main recovery options.
    local_actions = (
        (Action.NOOP, Action.MAINTAIN, Action.PATCH, Action.REJUVENATE, Action.ISOLATE, Action.MONITOR)
        if reduced else tuple(Action)
    )
    all_actions = []
    for u in itertools.product([int(a) for a in local_actions], repeat=n):
        if action_feasible(u, p):
            all_actions.append(tuple(int(x) for x in u))
    return all_actions


def enumerate_obs_states(n: int) -> List[Tuple[int, ...]]:
    local_obs = list(itertools.product(AUTOMATION_LEVELS, HEALTH_STATES))
    return [tuple(itertools.chain.from_iterable(items)) for items in itertools.product(local_obs, repeat=n)]


def enumerate_hidden_states(n: int) -> List[Tuple[int, ...]]:
    return [tuple(x) for x in itertools.product(CYBER_STATES, repeat=n)]


def sample_observable_reachable_states(n: int) -> List[Tuple[int, ...]]:
    # For offline value iteration, full enumeration is okay for n<=2/3. This helper is kept for extension.
    return enumerate_obs_states(n)


def local_health_transition(h: int, u: int, p: Params) -> List[Tuple[int, float]]:
    if u == Action.MAINTAIN:
        if h == Health.HEALTHY:
            return [(Health.HEALTHY, 1.0)]
        return [(Health.HEALTHY, p.p_repair_success), (h, 1.0 - p.p_repair_success)]
    if h == Health.HEALTHY:
        return [(Health.DEGRADED, p.p_degrade), (Health.HEALTHY, 1.0 - p.p_degrade)]
    if h == Health.DEGRADED:
        return [(Health.FAILED, p.p_fail_from_degraded), (Health.DEGRADED, 1.0 - p.p_fail_from_degraded)]
    return [(Health.FAILED, 1.0)]


def local_cyber_transition(c: int, u: int, p: Params) -> List[Tuple[int, float]]:
    if u == Action.PATCH:
        if c in (Cyber.SUSPECTED, Cyber.COMPROMISED):
            return [(Cyber.SECURE, p.p_patch_success), (c, 1.0 - p.p_patch_success)]
        return [(Cyber.SECURE, 1.0)]
    if u == Action.REJUVENATE:
        if c in (Cyber.SUSPECTED, Cyber.COMPROMISED, Cyber.RECOVERING):
            return [(Cyber.RECOVERING, 1.0 - p.p_rejuvenate_success), (Cyber.SECURE, p.p_rejuvenate_success)]
        return [(Cyber.SECURE, 1.0)]
    if u == Action.MONITOR:
        if c == Cyber.SUSPECTED:
            return [(Cyber.SECURE, p.p_monitor_suspected_to_secure), (Cyber.SUSPECTED, 1.0 - p.p_monitor_suspected_to_secure)]
        return [(c, 1.0)]
    if u == Action.ISOLATE:
        # Isolation prevents propagation but does not clean the AGV by itself.
        return [(c, 1.0)]

    # Passive cyber dynamics.
    if c == Cyber.SECURE:
        return [(Cyber.SUSPECTED, p.p_suspect), (Cyber.SECURE, 1.0 - p.p_suspect)]
    if c == Cyber.SUSPECTED:
        return [(Cyber.COMPROMISED, p.p_compromise_from_suspected), (Cyber.SUSPECTED, 1.0 - p.p_compromise_from_suspected)]
    if c == Cyber.RECOVERING:
        return [(Cyber.SECURE, p.p_recovering_to_secure), (Cyber.RECOVERING, 1.0 - p.p_recovering_to_secure)]
    return [(Cyber.COMPROMISED, 1.0)]


def automation_update(a: int, h_next: int, c_next: int, u: int) -> int:
    if u == Action.STOP or h_next == Health.FAILED:
        return 0
    if u == Action.ISOLATE:
        return min(a, 2)
    if u == Action.DOWNGRADE:
        return max(0, a - 1)
    if u == Action.MAINTAIN and h_next == Health.HEALTHY and c_next == Cyber.SECURE:
        return min(5, a + 1)
    if u == Action.PATCH and c_next == Cyber.SECURE and h_next == Health.HEALTHY:
        return min(5, a + 1)
    if c_next == Cyber.COMPROMISED:
        return min(a, 2)
    if c_next == Cyber.SUSPECTED:
        return min(a, 3)
    if h_next == Health.DEGRADED:
        return min(a, 4)
    if h_next == Health.HEALTHY and c_next in (Cyber.SECURE, Cyber.RECOVERING):
        return min(5, a + 1)
    return a


@lru_cache(maxsize=None)
def _local_transition_cached(a: int, h: int, c: int, u: int, params_tuple: Tuple) -> Tuple[Tuple[int, int, int, float], ...]:
    # Rehydrate minimal Params-like values from params_tuple is cumbersome, so this cache is not used.
    return tuple()


def local_transition_dist(a: int, h: int, c: int, u: int, p: Params) -> List[Tuple[Tuple[int, int, int], float]]:
    out: Dict[Tuple[int, int, int], float] = {}
    for h_next, ph in local_health_transition(h, u, p):
        for c_next, pc in local_cyber_transition(c, u, p):
            a_next = automation_update(a, h_next, c_next, u)
            key = (int(a_next), int(h_next), int(c_next))
            out[key] = out.get(key, 0.0) + ph * pc
    return [(k, v) for k, v in out.items() if v > 0]


def global_transition_dist(obs: Tuple[int, ...], c_tuple: Tuple[int, ...], actions: Tuple[int, ...], p: Params) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]:
    a_tuple, h_tuple = split_obs_tuple(obs)
    local_dists = []
    for i in range(len(a_tuple)):
        local_dists.append(local_transition_dist(a_tuple[i], h_tuple[i], c_tuple[i], actions[i], p))

    dist: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
    for combo in itertools.product(*local_dists):
        prob = 1.0
        next_obs_flat: List[int] = []
        next_c: List[int] = []
        for (a_next, h_next, c_next), pr in combo:
            prob *= pr
            next_obs_flat.extend([a_next, h_next])
            next_c.append(c_next)
        key = (tuple(next_obs_flat), tuple(next_c))
        dist[key] = dist.get(key, 0.0) + prob
    return dist


def observe_alerts(state: State, p: Params, rng: np.random.Generator) -> np.ndarray:
    alerts = np.zeros(p.n_agvs, dtype=int)
    for i, c in enumerate(state.cyber):
        alerts[i] = int(rng.random() < alert_prob(int(c), p))
    return alerts


def alert_likelihood(alerts: Tuple[int, ...], c_tuple: Tuple[int, ...], p: Params) -> float:
    prob = 1.0
    for z, c in zip(alerts, c_tuple):
        pa = alert_prob(int(c), p)
        prob *= pa if z else (1.0 - pa)
    return prob


def all_alert_vectors(n: int) -> List[Tuple[int, ...]]:
    return [tuple(z) for z in itertools.product((0, 1), repeat=n)]


def transition(state: State, actions: np.ndarray, p: Params, rng: np.random.Generator) -> None:
    for i in range(p.n_agvs):
        a, h, c, u = int(state.automation[i]), int(state.health[i]), int(state.cyber[i]), int(actions[i])
        dist = local_transition_dist(a, h, c, u, p)
        probs = np.array([pr for _, pr in dist], dtype=float)
        idx = int(rng.choice(len(dist), p=probs / probs.sum()))
        (a_next, h_next, c_next), _ = dist[idx]
        state.automation[i] = a_next
        state.health[i] = h_next
        state.cyber[i] = c_next


def local_cost_from_arrays(automation: Sequence[int], health: Sequence[int], cyber: Sequence[int], actions: Sequence[int], p: Params) -> float:
    automation_arr = np.array(automation)
    health_arr = np.array(health)
    cyber_arr = np.array(cyber)
    actions_arr = np.array(actions)

    cost = float(np.sum(p.cost_automation_step * (5 - automation_arr)))
    cost += float(np.sum(health_arr == Health.DEGRADED) * p.cost_degraded_phys)
    cost += float(np.sum(health_arr == Health.FAILED) * p.cost_failed)
    cost += float(np.sum(cyber_arr == Cyber.SUSPECTED) * p.cost_suspected)
    cost += float(np.sum(cyber_arr == Cyber.COMPROMISED) * p.cost_compromised)
    cost += float(np.sum(actions_arr == Action.MONITOR) * p.cost_monitor)
    cost += float(np.sum(actions_arr == Action.PATCH) * p.cost_patch)
    cost += float(np.sum(actions_arr == Action.REJUVENATE) * p.cost_rejuvenate)
    cost += float(np.sum(actions_arr == Action.MAINTAIN) * p.cost_maintain)
    cost += float(np.sum(actions_arr == Action.ISOLATE) * p.cost_isolate)
    cost += float(np.sum(actions_arr == Action.STOP) * p.cost_stop)
    cost += float(np.sum(actions_arr == Action.DOWNGRADE) * p.cost_downgrade)
    return cost


def interaction_cost_from_arrays(automation: Sequence[int], health: Sequence[int], actions: Sequence[int], p: Params) -> float:
    profile = p.interaction_profile
    strength = p.interaction_strength
    if profile == "none" or strength == 0:
        base = 0.0
    else:
        perf = np.array(automation, dtype=float) / 5.0
        perf = np.where(np.array(health) == Health.FAILED, 0.0, perf)
        if profile == "convoy":
            fleet_perf = np.min(perf)
            base = float(strength * np.sum(perf - fleet_perf))
        elif profile == "sequential":
            throughput = np.min(perf)
            base = float(strength * len(automation) * (1.0 - throughput))
        elif profile == "shared_resource":
            n_interventions = sum(1 for a in actions if a != Action.NOOP)
            overload = max(0, n_interventions - 1)
            base = float(strength * overload)
        elif profile == "coordination":
            base = float(strength * np.var(np.array(automation)))
        else:
            raise ValueError(f"Unknown interaction profile: {profile}")

    heavy, monitored = action_counts(actions)
    overload_penalty = p.resource_overload_penalty * max(0, heavy - p.alpha_technicians)
    overload_penalty += p.resource_overload_penalty * max(0, monitored - p.beta_monitored_agvs)
    return base + overload_penalty


def cost_from_obs_hidden_action(obs: Tuple[int, ...], c_tuple: Tuple[int, ...], actions: Tuple[int, ...], p: Params) -> float:
    automation, health = split_obs_tuple(obs)
    return local_cost_from_arrays(automation, health, c_tuple, actions, p) + interaction_cost_from_arrays(automation, health, actions, p)


def total_cost(state: State, actions: np.ndarray, p: Params) -> float:
    return cost_from_obs_hidden_action(obs_tuple_from_state(state), tuple(int(c) for c in state.cyber), tuple(int(a) for a in actions), p)


def generate_belief_points(hidden_states: List[Tuple[int, ...]], n_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_hidden = len(hidden_states)
    points: List[np.ndarray] = []

    # Prior and uniform.
    n = len(hidden_states[0])
    points.append(initial_joint_belief(n))
    points.append(np.full(n_hidden, 1.0 / n_hidden))

    # Concentrated beliefs on secure, suspected, compromised, recovering vectors if present.
    for cyber_state in CYBER_STATES:
        b = np.zeros(n_hidden)
        target = tuple([cyber_state] * n)
        if target in hidden_states:
            b[hidden_states.index(target)] = 1.0
            points.append(b)

    # Random Dirichlet beliefs.
    while len(points) < n_points:
        # sparse-ish beliefs to mimic reachable distributions
        alpha = np.full(n_hidden, 0.2)
        points.append(rng.dirichlet(alpha))

    # Deduplicate approximately.
    unique: List[np.ndarray] = []
    for b in points:
        if not any(np.linalg.norm(b - u) < 1e-8 for u in unique):
            unique.append(b)
        if len(unique) >= n_points:
            break
    return np.vstack(unique)


def nearest_belief_index(b: np.ndarray, belief_points: np.ndarray) -> int:
    return int(np.argmin(np.sum((belief_points - b.reshape(1, -1)) ** 2, axis=1)))


def expected_cost(obs: Tuple[int, ...], belief: np.ndarray, actions: Tuple[int, ...], hidden_states: List[Tuple[int, ...]], p: Params) -> float:
    return float(sum(prob * cost_from_obs_hidden_action(obs, c_tuple, actions, p) for prob, c_tuple in zip(belief, hidden_states)))


def solve_offline_belief_grid(p: Params) -> OfflinePolicy:
    if p.n_agvs > 3:
        raise ValueError("Offline belief-grid solver is intended for small fleets (<=3 AGVs).")

    start = time.time()
    hidden_states = enumerate_hidden_states(p.n_agvs)
    obs_states = sample_observable_reachable_states(p.n_agvs)
    actions = enumerate_actions(p.n_agvs, p, reduced=True)
    alerts = all_alert_vectors(p.n_agvs)
    belief_points = generate_belief_points(hidden_states, p.pbvi_belief_points, p.pbvi_seed)

    V: Dict[Tuple[Tuple[int, ...], int], float] = {(obs, bi): 0.0 for obs in obs_states for bi in range(len(belief_points))}
    policy: Dict[Tuple[Tuple[int, ...], int], Tuple[int, ...]] = {}

    # Precompute transitions for speed: (obs, hidden_idx, action) -> dist
    trans_cache: Dict[Tuple[Tuple[int, ...], int, Tuple[int, ...]], Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]] = {}

    hidden_index = {c: i for i, c in enumerate(hidden_states)}

    for it in range(p.pbvi_iterations):
        V_new: Dict[Tuple[Tuple[int, ...], int], float] = {}
        for obs in obs_states:
            for bi, belief in enumerate(belief_points):
                best_q = float("inf")
                best_u = actions[0]

                for u in actions:
                    q = expected_cost(obs, belief, u, hidden_states, p)
                    future = 0.0

                    # Distribution over next observable states and hidden states before alert observation.
                    joint_next: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float] = {}
                    for ci, c_tuple in enumerate(hidden_states):
                        if belief[ci] == 0:
                            continue
                        key = (obs, ci, u)
                        if key not in trans_cache:
                            trans_cache[key] = global_transition_dist(obs, c_tuple, u, p)
                        for next_key, pr in trans_cache[key].items():
                            joint_next[next_key] = joint_next.get(next_key, 0.0) + belief[ci] * pr

                    # Split by observed next obs and alert vector; update belief over cyber states.
                    by_next_obs: Dict[Tuple[int, ...], np.ndarray] = {}
                    for (next_obs, next_c), pr in joint_next.items():
                        if next_obs not in by_next_obs:
                            by_next_obs[next_obs] = np.zeros(len(hidden_states))
                        by_next_obs[next_obs][hidden_index[next_c]] += pr

                    for next_obs, pred_b_unnormalized in by_next_obs.items():
                        for z in alerts:
                            weights = np.zeros(len(hidden_states))
                            for ci, c_tuple in enumerate(hidden_states):
                                if pred_b_unnormalized[ci] > 0:
                                    weights[ci] = pred_b_unnormalized[ci] * alert_likelihood(z, c_tuple, p)
                            prob_obs = float(weights.sum())
                            if prob_obs <= 0:
                                continue
                            b_next = weights / prob_obs
                            bj = nearest_belief_index(b_next, belief_points)
                            future += prob_obs * V[(next_obs, bj)]

                    q += p.gamma * future
                    if q < best_q:
                        best_q = q
                        best_u = u

                V_new[(obs, bi)] = best_q
                policy[(obs, bi)] = best_u
        V = V_new

    metadata = {
        "solve_seconds": time.time() - start,
        "n_obs_states": len(obs_states),
        "n_hidden_states": len(hidden_states),
        "n_belief_points": len(belief_points),
        "n_actions": len(actions),
        "iterations": p.pbvi_iterations,
    }
    return OfflinePolicy(
        n_agvs=p.n_agvs,
        hidden_states=hidden_states,
        obs_states=obs_states,
        belief_points=belief_points,
        policy=policy,
        values=V,
        metadata=metadata,
    )


def save_offline_policy(pol: OfflinePolicy, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(pol, f)


def load_offline_policy(path: str) -> OfflinePolicy:
    with open(path, "rb") as f:
        return pickle.load(f)


def update_joint_belief_exact(state: State, prev_obs: Tuple[int, ...], actions: Tuple[int, ...], alerts: Tuple[int, ...], pol: OfflinePolicy, p: Params) -> None:
    if state.belief_joint is None:
        state.belief_joint = initial_joint_belief(p.n_agvs)

    hidden_states = pol.hidden_states
    hidden_index = {c: i for i, c in enumerate(hidden_states)}
    pred = np.zeros(len(hidden_states))
    observed_next_obs = obs_tuple_from_state(state)

    for ci, c_tuple in enumerate(hidden_states):
        bprob = state.belief_joint[ci]
        if bprob == 0:
            continue
        dist = global_transition_dist(prev_obs, c_tuple, actions, p)
        for (next_obs, next_c), pr in dist.items():
            if next_obs == observed_next_obs:
                pred[hidden_index[next_c]] += bprob * pr

    weights = np.zeros(len(hidden_states))
    for ci, c_tuple in enumerate(hidden_states):
        weights[ci] = pred[ci] * alert_likelihood(alerts, c_tuple, p)
    if weights.sum() <= 0:
        # Fallback if the observed transition was impossible under the model.
        weights = initial_joint_belief(p.n_agvs)
    else:
        weights = weights / weights.sum()

    state.belief_joint = weights
    state.belief_compromised = cyber_marginals_from_joint(weights, hidden_states, p.n_agvs)


def update_belief_simple(state: State, actions: np.ndarray, alerts: np.ndarray, p: Params) -> None:
    # Marginal approximation used by heuristic baselines and larger fleets.
    b = state.belief_compromised.copy() if state.belief_compromised is not None else np.full(p.n_agvs, 0.01)
    for i, a in enumerate(actions):
        b[i] = b[i] + (1.0 - b[i]) * p.p_suspect * p.p_compromise_from_suspected
        if a == Action.PATCH:
            b[i] *= (1.0 - p.p_patch_success)
        elif a == Action.REJUVENATE:
            b[i] *= (1.0 - p.p_rejuvenate_success)
        elif a == Action.MONITOR:
            b[i] *= 0.85

    for i, alert in enumerate(alerts):
        if alert:
            like_c = p.p_alert_compromised
            like_not = (p.p_alert_secure + p.p_alert_suspected + p.p_alert_recovering) / 3.0
        else:
            like_c = 1.0 - p.p_alert_compromised
            like_not = 1.0 - (p.p_alert_secure + p.p_alert_suspected + p.p_alert_recovering) / 3.0
        denom = like_c * b[i] + like_not * (1.0 - b[i])
        if denom > 0:
            b[i] = (like_c * b[i]) / denom
    state.belief_compromised = np.clip(b, 0.0, 1.0)


Policy = Callable[[State, np.ndarray, Params], np.ndarray]


def reactive_maintenance_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_agvs, Action.NOOP, dtype=int)
    failed = np.where(state.health == Health.FAILED)[0]
    for i in failed[: p.alpha_technicians]:
        actions[i] = Action.MAINTAIN
    return actions


def reactive_cyber_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_agvs, Action.NOOP, dtype=int)
    belief = state.belief_compromised if state.belief_compromised is not None else np.zeros(p.n_agvs)
    for i in range(p.n_agvs):
        if belief[i] > p.belief_rejuvenate_threshold:
            actions[i] = Action.REJUVENATE
        elif alerts[i] == 1:
            actions[i] = Action.PATCH
    return actions


def immediate_intervention_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_agvs, Action.NOOP, dtype=int)
    belief = state.belief_compromised if state.belief_compromised is not None else np.zeros(p.n_agvs)
    tech_used = 0
    monitor_used = 0
    for i in range(p.n_agvs):
        if state.health[i] in (Health.DEGRADED, Health.FAILED) and tech_used < p.alpha_technicians:
            actions[i] = Action.MAINTAIN
            tech_used += 1
        elif belief[i] > p.belief_rejuvenate_threshold:
            actions[i] = Action.REJUVENATE
        elif alerts[i] or belief[i] > p.belief_patch_threshold:
            actions[i] = Action.PATCH
        elif monitor_used < p.beta_monitored_agvs and belief[i] > 0.10:
            actions[i] = Action.MONITOR
            monitor_used += 1
    return actions


def separated_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    phys = reactive_maintenance_policy(state, alerts, p)
    cyber = reactive_cyber_policy(state, alerts, p)
    actions = np.full(p.n_agvs, Action.NOOP, dtype=int)
    for i in range(p.n_agvs):
        actions[i] = phys[i] if phys[i] != Action.NOOP else cyber[i]
    return actions


def joint_heuristic_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_agvs, Action.NOOP, dtype=int)
    belief = state.belief_compromised if state.belief_compromised is not None else np.zeros(p.n_agvs)
    tech_used = 0
    monitor_used = 0

    # Prioritize failed/stopped bottlenecks, then bundle physical/cyber risk.
    order = list(np.argsort(state.automation))
    for i in order:
        high_belief = belief[i] > p.belief_rejuvenate_threshold
        mid_belief = belief[i] > p.belief_patch_threshold
        if state.health[i] == Health.FAILED and tech_used < p.alpha_technicians:
            actions[i] = Action.MAINTAIN
            tech_used += 1
        elif state.health[i] == Health.DEGRADED and (alerts[i] or mid_belief) and tech_used < p.alpha_technicians:
            actions[i] = Action.MAINTAIN
            tech_used += 1
        elif high_belief:
            actions[i] = Action.REJUVENATE
        elif mid_belief or alerts[i]:
            actions[i] = Action.PATCH
        elif state.health[i] == Health.DEGRADED and tech_used < p.alpha_technicians:
            actions[i] = Action.MAINTAIN
            tech_used += 1
        elif belief[i] > 0.10 and monitor_used < p.beta_monitored_agvs:
            actions[i] = Action.MONITOR
            monitor_used += 1
    return actions


def pbvi_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    global _CURRENT_OFFLINE_POLICY
    pol = _CURRENT_OFFLINE_POLICY
    if pol is None or pol.n_agvs != p.n_agvs or state.belief_joint is None:
        return joint_heuristic_policy(state, alerts, p)
    obs = obs_tuple_from_state(state)
    bi = nearest_belief_index(state.belief_joint, pol.belief_points)
    action_tuple = pol.policy.get((obs, bi))
    if action_tuple is None:
        return joint_heuristic_policy(state, alerts, p)
    return np.array(action_tuple, dtype=int)


POLICIES: Dict[str, Policy] = {
    "RM": reactive_maintenance_policy,
    "RC": reactive_cyber_policy,
    "II": immediate_intervention_policy,
    "SEP": separated_policy,
    "JOINT": joint_heuristic_policy,
    "PBVI": pbvi_policy,
}


def simulate_once(policy_name: str, p: Params, run_seed: int, offline_policy: Optional[OfflinePolicy] = None) -> Dict[str, float]:
    rng = np.random.default_rng(run_seed)
    state = initial_state(p, offline_policy.hidden_states if offline_policy is not None and offline_policy.n_agvs == p.n_agvs else None)

    discounted_cost = 0.0
    raw_cost = 0.0
    failsoft_epochs = 0
    safestop_epochs = 0
    intervention_count = 0
    automation_sum = 0.0

    policy = POLICIES[policy_name]
    alerts = observe_alerts(state, p, rng)
    update_belief_simple(state, np.full(p.n_agvs, Action.NOOP), alerts, p)

    for t in range(p.horizon):
        prev_obs = obs_tuple_from_state(state)
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
        if offline_policy is not None and offline_policy.n_agvs == p.n_agvs and state.belief_joint is not None:
            update_joint_belief_exact(state, prev_obs, tuple(int(a) for a in actions), tuple(int(z) for z in alerts), offline_policy, p)
        else:
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


def run_condition(label: str, p: Params, policies: List[str], offline_policy: Optional[OfflinePolicy] = None) -> pd.DataFrame:
    rows = []
    start = time.time()
    for pol in policies:
        for r in range(p.n_runs):
            rows.append(simulate_once(pol, p, p.seed + abs(hash((label, pol))) % 100000 + r, offline_policy))
    df = pd.DataFrame(rows)
    df["condition"] = label
    df["n_agvs"] = p.n_agvs
    df["p_alert_secure"] = p.p_alert_secure
    df["p_alert_suspected"] = p.p_alert_suspected
    df["p_alert_compromised"] = p.p_alert_compromised
    df["p_alert_recovering"] = p.p_alert_recovering
    df["interaction_profile"] = p.interaction_profile
    df["interaction_strength"] = p.interaction_strength
    df["alpha_technicians"] = p.alpha_technicians
    df["beta_monitored_agvs"] = p.beta_monitored_agvs
    df["runtime_seconds"] = time.time() - start
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["discounted_cost", "raw_cost", "failsoft_epochs", "safestop_epochs", "interventions", "avg_automation"]
    return (
        df.groupby(["condition", "policy", "n_agvs", "interaction_profile", "interaction_strength", "alpha_technicians", "beta_monitored_agvs"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )


def run_all(base: Params, offline_policy: Optional[OfflinePolicy]) -> pd.DataFrame:
    small_policies = ["RM", "RC", "SEP", "II", "JOINT"] + (["PBVI"] if offline_policy is not None and offline_policy.n_agvs == base.n_agvs else [])
    all_runs = [run_condition("baseline_comparison", base, small_policies, offline_policy)]

    # RQ2: IDS quality sensitivity around the AGV observation matrix.
    for secure_alert, compromised_alert in [(0.01, 0.90), (0.02, 0.85), (0.05, 0.75), (0.10, 0.65)]:
        p = replace(base, p_alert_secure=secure_alert, p_alert_compromised=compromised_alert)
        all_runs.append(run_condition(f"ids_secure{secure_alert}_comp{compromised_alert}", p, small_policies, offline_policy))

    # RQ3: interaction profile/strength.
    for profile in ["none", "convoy", "sequential", "shared_resource", "coordination"]:
        for strength in [0.0, 1.0, 3.0, 6.0]:
            p = replace(base, interaction_profile=profile, interaction_strength=strength)
            all_runs.append(run_condition(f"interaction_{profile}_{strength}", p, small_policies, offline_policy))

    # RQ4: fleet size scaling. PBVI only if a matching offline policy exists.
    for n in [1, 2, 3, 5, 10, 20]:
        p = replace(base, n_agvs=n, n_runs=max(30, min(base.n_runs, 100)))
        policies = ["RM", "RC", "SEP", "II", "JOINT"]
        matching_policy = offline_policy if offline_policy is not None and offline_policy.n_agvs == n else None
        if matching_policy is not None:
            policies.append("PBVI")
        all_runs.append(run_condition(f"fleet_size_{n}", p, policies, matching_policy))

    return pd.concat(all_runs, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-agvs", type=int, default=2)
    parser.add_argument("--solve-pbvi", action="store_true", help="Solve offline belief-grid policy before running experiments")
    parser.add_argument("--policy-in", default=None, help="Load offline policy from pickle")
    parser.add_argument("--policy-out", default="offline_agv_policy.pkl", help="Where to save solved offline policy")
    parser.add_argument("--pbvi-iterations", type=int, default=10)
    parser.add_argument("--belief-points", type=int, default=16)
    parser.add_argument("--output", default="results_summary.csv")
    parser.add_argument("--raw-output", default="results_raw.csv")
    args = parser.parse_args()

    global _CURRENT_OFFLINE_POLICY
    base = Params(
        n_agvs=args.n_agvs,
        n_runs=args.runs,
        horizon=args.horizon,
        seed=args.seed,
        pbvi_iterations=args.pbvi_iterations,
        pbvi_belief_points=args.belief_points,
    )

    offline_policy = None
    if args.policy_in:
        offline_policy = load_offline_policy(args.policy_in)
        print(f"Loaded offline policy for {offline_policy.n_agvs} AGVs from {args.policy_in}")
    elif args.solve_pbvi:
        if base.n_agvs > 3:
            print("Offline solver skipped: use --n-agvs <= 3 for the belief-grid solver.")
        else:
            print(f"Solving offline belief-grid policy for {base.n_agvs} AGVs...")
            offline_policy = solve_offline_belief_grid(base)
            save_offline_policy(offline_policy, args.policy_out)
            print(f"Saved offline policy to {args.policy_out}")
            print(f"Solver metadata: {offline_policy.metadata}")

    _CURRENT_OFFLINE_POLICY = offline_policy
    raw = run_all(base, offline_policy)
    summary = summarize(raw)
    raw.to_csv(args.raw_output, index=False)
    summary.to_csv(args.output, index=False)
    print(f"Wrote raw runs to {args.raw_output}")
    print(f"Wrote summary to {args.output}")
    print(summary.head(20).to_string())


if __name__ == "__main__":
    main()
