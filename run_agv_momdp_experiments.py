#!/usr/bin/env python3
"""
AGV resilience orchestration experiments with an offline finite-belief
PBVI-style MOMDP solver for small fleets and heuristic baselines for larger fleets.

This script extends the earlier scaffold by:
  - renaming tractors to AGVs;
  - using AGV-oriented state/action/cost parameters;
  - adding a 4-state cyber observation model P(alert | cyber_state);
  - enforcing technician (alpha) and monitoring (beta) capacity constraints;
  - adding an offline approximate finite-belief PBVI-style solver over representative beliefs;
  - using the offline policy as a lookup during Monte Carlo experiments.

Important modeling note
-----------------------
The offline solver is a finite-belief point-based value-iteration implementation.
It is PBVI-style: it performs Bellman backups on a finite representative belief set
rather than solving the continuous belief space exactly. It enumerates all observable states for small fleets, evaluates a
finite set of representative beliefs, and stores a policy table
    (observable_state, nearest_belief_point) -> joint_action.
This is intended for small fleets (default: 2 AGVs). For larger fleets, use the
scalable IAGR heuristic policy and report scalability limitations.
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
import sys

import numpy as np
import pandas as pd

# Make policy pickles portable when this file is executed as a script.
# Without this alias, pickle records OfflinePolicy as __main__.OfflinePolicy,
# which cannot be loaded from diagnostic scripts that import this module.
if __name__ == "__main__":
    sys.modules.setdefault("run_agv_momdp_experiments", sys.modules[__name__])


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

    # Intervention-cost calibration. The multiplier scales one-shot recovery
    # actions; the downtime penalty approximates temporary operational
    # unavailability while an intervention is performed. These two parameters
    # are deliberately separate so experiments can distinguish expensive
    # interventions from interventions that also disrupt operations.
    intervention_cost_multiplier: float = 1.0
    # Additional one-shot cost per intervention. This captures direct downtime
    # accounting even in the offline PBVI cost model.
    intervention_downtime_penalty: float = 0.0
    # Intervention durations used only to compute the optional one-shot downtime
    # penalty in the cost function. The Monte Carlo simulator intentionally does
    # not force multi-epoch unavailability, so that simulation, belief update,
    # and the offline MOMDP/PBVI transition model remain consistent.
    downtime_monitor_epochs: int = 0
    downtime_patch_epochs: int = 1
    downtime_rejuvenate_epochs: int = 2
    downtime_maintain_epochs: int = 3
    downtime_isolate_epochs: int = 1
    downtime_stop_epochs: int = 0
    downtime_downgrade_epochs: int = 0

    # Interaction costs
    interaction_profile: str = "convoy"  # none, convoy, sequential, shared_resource, coordination
    interaction_strength: float = 3.0
    resource_overload_penalty: float = 20.0

    # Campaign-aware intervention cost model. When enabled, compatible
    # interventions performed on AGVs in the same interaction group share a
    # setup cost: a singleton campaign has the original cost, while additional
    # AGVs in the same campaign are charged at a discounted marginal rate.
    # The cost model applies to every policy; only IAGR_C explicitly optimizes
    # for campaign formation.
    campaign_cost_model: bool = False
    campaign_group_size: int = 5
    campaign_marginal_cost_factor: float = 0.45
    # IAGR-C explicit campaign planner controls. Campaigns are generated inside
    # interaction groups and selected by expected fleet benefit under resource
    # constraints. These parameters deliberately keep the planner lightweight.
    iagrc_campaign_max_size: int = 5
    iagrc_min_campaign_size: int = 2
    iagrc_min_campaign_score: float = 0.0
    iagrc_preventive_patch_belief: float = 0.08
    iagrc_trigger_patch_belief: float = 0.18
    iagrc_rejuvenate_belief: float = 0.45

    # Heuristic policy thresholds
    belief_patch_threshold: float = 0.35
    belief_rejuvenate_threshold: float = 0.65

    # IAGR heuristic weights. IAGR = Interaction-Aware Greedy Recovery.
    # It is an online scalable heuristic combining local cyber/physical risk
    # and estimated system-level interaction benefit.
    iagr_w_local_risk: float = 1.0
    iagr_w_interaction: float = 1.0
    iagr_w_alert: float = 1.0
    iagr_candidate_min_score: float = 0.0
    # Separate admission threshold for physical recovery. Preventive maintenance
    # may have a slightly negative one-step score while still avoiding future
    # failure and interaction bottlenecks; cyber actions keep the default score
    # threshold to avoid overreacting to noisy alerts.
    iagr_physical_min_score: float = -5.0
    # Finite-lookahead multiplier used by IAGR to value avoided future
    # degradation/failure risk and interaction bottleneck relief. This keeps
    # IAGR scalable while making it less myopic than the original one-step score.
    iagr_risk_lookahead: float = 6.0

    # Offline solver settings
    # Deprecated: the offline solver uses n_agvs. Kept for backward compatibility.
    pbvi_n_agvs: int = 2
    pbvi_iterations: int = 12
    pbvi_belief_points: int = 20
    pbvi_seed: int = 7
    # Full enumeration of observable states is slow in pure Python. The offline
    # solver therefore uses a reachable-state subset by default. Set to 0 to
    # enumerate all observable states.
    pbvi_max_obs_states: int = 64
    pbvi_rollouts: int = 200
    use_pbvi_for_baseline: bool = True


@dataclass
class State:
    automation: np.ndarray  # length n_agvs, values 0..5
    health: np.ndarray      # Health values
    cyber: np.ndarray       # Cyber values (true hidden state in simulation)
    # Kept for backward-compatible result metrics. Under the consistent MOMDP
    # simulator this timer remains zero; intervention disruption is represented
    # only through immediate action costs and the optional one-shot downtime
    # penalty.
    intervention_remaining: Optional[np.ndarray] = None
    belief_joint: Optional[np.ndarray] = None  # distribution over joint cyber states
    belief_compromised: Optional[np.ndarray] = None  # marginals, derived from belief_joint


@dataclass
class OfflinePolicy:
    n_agvs: int
    hidden_states: List[Tuple[int, ...]]
    obs_states: List[Tuple[int, ...]]
    belief_points: np.ndarray
    # Finite-grid policy table. For alpha-vector PBVI this is populated only
    # on the representative belief points for diagnostics/backward compatibility.
    policy: Dict[Tuple[Tuple[int, ...], int], Tuple[int, ...]]
    values: Dict[Tuple[Tuple[int, ...], int], float]
    metadata: Dict[str, float] = field(default_factory=dict)
    # True MOMDP-PBVI representation: for each observable state, a list of
    # alpha-vectors over the hidden cyber states and the action attached to each
    # vector. Cost minimization uses min_alpha alpha.dot(b).
    alpha_vectors: Dict[Tuple[int, ...], np.ndarray] = field(default_factory=dict)
    alpha_actions: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = field(default_factory=dict)


# Store policy objects under the importable module name even when this file is
# run as a script, so saved policies can be reloaded by helper scripts.
OfflinePolicy.__module__ = "run_agv_momdp_experiments"

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


def action_downtime_duration(action: int, p: Params) -> int:
    """Return the number of future epochs during which an action makes an AGV unavailable."""
    if action == Action.MONITOR:
        return int(p.downtime_monitor_epochs)
    if action == Action.PATCH:
        return int(p.downtime_patch_epochs)
    if action == Action.REJUVENATE:
        return int(p.downtime_rejuvenate_epochs)
    if action == Action.MAINTAIN:
        return int(p.downtime_maintain_epochs)
    if action == Action.ISOLATE:
        return int(p.downtime_isolate_epochs)
    if action == Action.STOP:
        return int(p.downtime_stop_epochs)
    if action == Action.DOWNGRADE:
        return int(p.downtime_downgrade_epochs)
    return 0


def is_agv_unavailable(state: State, i: int) -> bool:
    # Option A consistency fix: the online simulator no longer imposes explicit
    # multi-epoch downtime. PBVI's transition model does not include an
    # intervention timer, so policy execution must not block AGVs using one.
    return False


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
        intervention_remaining=np.zeros(p.n_agvs, dtype=int),
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
        intervention_remaining=np.zeros(n, dtype=int),
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


def sample_observable_reachable_states(n: int, p: Optional[Params] = None) -> List[Tuple[int, ...]]:
    """Return observable states used by the offline solver.

    Enumerating all states gives (6*3)^n observable states. This is already 324
    states for n=2 and 5832 for n=3. In pure Python, PBVI over all of them is
    slow. The default therefore builds a compact reachable subset by rolling out
    simple policies from the nominal state. This is sufficient for the first
    experiments and still includes the nominal, degraded, failed, and stopped
    states encountered during simulations. Set p.pbvi_max_obs_states=0 to force
    full enumeration.
    """
    if p is None or p.pbvi_max_obs_states == 0:
        return enumerate_obs_states(n)

    rng = np.random.default_rng(p.pbvi_seed)
    seen = set()

    def add(obs):
        if len(seen) < p.pbvi_max_obs_states:
            seen.add(obs)

    # Always include nominal, all stopped, and common degraded states.
    nominal = tuple(itertools.chain.from_iterable([(5, int(Health.HEALTHY)) for _ in range(n)]))
    stopped = tuple(itertools.chain.from_iterable([(0, int(Health.FAILED)) for _ in range(n)]))
    add(nominal)
    add(stopped)
    for i in range(n):
        vals = []
        for j in range(n):
            vals.append((4, int(Health.DEGRADED)) if i == j else (5, int(Health.HEALTHY)))
        add(tuple(itertools.chain.from_iterable(vals)))

    # Roll out from nominal with simple random feasible actions.
    hidden_states = enumerate_hidden_states(n)
    policies_for_sampling = [joint_heuristic_policy, immediate_intervention_policy, separated_policy]
    for r in range(p.pbvi_rollouts):
        st = initial_state(replace(p, n_agvs=n), hidden_states)
        alerts = observe_alerts(st, p, rng)
        for t in range(min(80, p.horizon)):
            add(obs_tuple_from_state(st))
            if len(seen) >= p.pbvi_max_obs_states:
                return list(seen)
            pol = policies_for_sampling[(r + t) % len(policies_for_sampling)]
            # Add noise by occasionally using random feasible action.
            if rng.random() < 0.20:
                acts = enumerate_actions(n, p, reduced=True)
                actions = np.array(acts[int(rng.integers(len(acts)))], dtype=int)
            else:
                actions = pol(st, alerts, p)
            transition(st, actions, p, rng)
            alerts = observe_alerts(st, p, rng)
            update_belief_simple(st, actions, alerts, p)

    return list(seen)


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
    if state.intervention_remaining is None:
        state.intervention_remaining = np.zeros(p.n_agvs, dtype=int)

    # Consistent MOMDP/PBVI simulator: interventions affect the next physical,
    # cyber, and automation state through local_transition_dist, and their
    # disruption is charged as immediate cost. They do not create an additional
    # hidden downtime timer because that timer is absent from the offline MOMDP
    # transition and belief-update model.
    state.intervention_remaining[:] = 0

    for i in range(p.n_agvs):
        effective_action = int(actions[i])

        a, h, c = int(state.automation[i]), int(state.health[i]), int(state.cyber[i])
        dist = local_transition_dist(a, h, c, effective_action, p)
        probs = np.array([pr for _, pr in dist], dtype=float)
        idx = int(rng.choice(len(dist), p=probs / probs.sum()))
        (a_next, h_next, c_next), _ = dist[idx]

        state.automation[i] = a_next
        state.health[i] = h_next
        state.cyber[i] = c_next



CAMPAIGN_ACTIONS = (int(Action.MAINTAIN), int(Action.PATCH), int(Action.REJUVENATE))


def action_base_cost(action: int, p: Params) -> float:
    """Base one-shot action cost before global multiplier/downtime penalty."""
    if action == Action.MONITOR:
        return p.cost_monitor
    if action == Action.PATCH:
        return p.cost_patch
    if action == Action.REJUVENATE:
        return p.cost_rejuvenate
    if action == Action.MAINTAIN:
        return p.cost_maintain
    if action == Action.ISOLATE:
        return p.cost_isolate
    if action == Action.STOP:
        return p.cost_stop
    if action == Action.DOWNGRADE:
        return p.cost_downgrade
    return 0.0


def campaign_groups(n: int, p: Params) -> List[List[int]]:
    """Disjoint interaction groups used for campaign cost sharing.

    The grouping is deliberately simple and transparent: AGVs are partitioned
    into contiguous local groups. A singleton group leaves costs unchanged, so
    the same cost function can be applied to all policies without giving any
    policy private discounts. For profile='none', campaigns are disabled by
    returning singleton groups.
    """
    if (not getattr(p, "campaign_cost_model", False)) or p.interaction_profile == "none":
        return [[i] for i in range(n)]
    g = max(1, int(getattr(p, "campaign_group_size", 5)))
    return [list(range(start, min(n, start + g))) for start in range(0, n, g)]


def campaign_action_base_cost(actions: Sequence[int], p: Params) -> float:
    """Action cost with optional campaign setup/marginal structure.

    For a single AGV receiving an action, the cost is identical to the original
    per-AGV action cost. For k>1 compatible actions of the same type inside the
    same group, the first action pays the full setup cost and the remaining
    k-1 actions pay campaign_marginal_cost_factor times the base cost.
    """
    actions_arr = np.array(actions, dtype=int)
    if not getattr(p, "campaign_cost_model", False):
        return float(sum(action_base_cost(int(a), p) for a in actions_arr))

    total = 0.0
    accounted = np.zeros(len(actions_arr), dtype=bool)
    marginal = float(getattr(p, "campaign_marginal_cost_factor", 0.45))
    for group in campaign_groups(len(actions_arr), p):
        for action in CAMPAIGN_ACTIONS:
            idx = [i for i in group if int(actions_arr[i]) == action]
            if not idx:
                continue
            k = len(idx)
            total += action_base_cost(action, p) * (1.0 + marginal * max(0, k - 1))
            accounted[idx] = True
    # Non-campaign actions, and campaign actions outside valid groups, retain
    # the original independent action cost.
    for i, a in enumerate(actions_arr):
        if not accounted[i]:
            total += action_base_cost(int(a), p)
    return float(total)


def campaign_savings_for_action(action: int, k: int, p: Params) -> float:
    if k <= 1 or action not in CAMPAIGN_ACTIONS or not getattr(p, "campaign_cost_model", False):
        return 0.0
    base = action_base_cost(action, p)
    marginal = float(getattr(p, "campaign_marginal_cost_factor", 0.45))
    independent = k * base
    campaign = base * (1.0 + marginal * (k - 1))
    return p.intervention_cost_multiplier * max(0.0, independent - campaign)


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
    # One-shot recovery/action costs. These costs are scaled by
    # intervention_cost_multiplier so that sensitivity studies can test whether
    # aggressive policies such as II remain attractive when interventions are
    # expensive. A separate downtime penalty approximates temporary
    # unavailability during interventions.
    action_base = campaign_action_base_cost(actions_arr, p)

    downtime_epochs = sum(action_downtime_duration(int(a), p) for a in actions_arr)
    downtime_cost = float(downtime_epochs * p.intervention_downtime_penalty)
    cost += p.intervention_cost_multiplier * action_base + downtime_cost
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


def campaign_execution_stats(state: State, actions: Sequence[int], alerts: np.ndarray, p: Params) -> Dict[str, float]:
    """Diagnostics for campaign use by any policy.

    These metrics are descriptive only.  They do not change the simulator.  A
    campaign is counted when at least two AGVs in the same campaign group execute
    the same campaign-compatible action in the same epoch.  Preventive cyber
    add-ons are campaign members that were not locally required by the standard
    patch/rejuvenation thresholds.
    """
    actions_arr = np.array(actions, dtype=int)
    belief = _belief_array(state, p)
    count = 0
    members = 0
    preventive = 0
    savings = 0.0
    for group in campaign_groups(len(actions_arr), p):
        for action in CAMPAIGN_ACTIONS:
            idx = [i for i in group if int(actions_arr[i]) == int(action)]
            if len(idx) < 2:
                continue
            count += 1
            members += len(idx)
            savings += campaign_savings_for_action(int(action), len(idx), p)
            if int(action) == int(Action.PATCH):
                preventive += sum(1 for i in idx if (not bool(alerts[i])) and belief[i] < p.belief_patch_threshold)
            elif int(action) == int(Action.REJUVENATE):
                preventive += sum(1 for i in idx if belief[i] < p.belief_rejuvenate_threshold)
    return {
        "campaigns": float(count),
        "campaign_members": float(members),
        "campaign_preventive_members": float(preventive),
        "campaign_savings": float(savings),
    }


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


def off_grid_fallback_value(obs: Tuple[int, ...], belief: np.ndarray, actions: Tuple[int, ...], hidden_states: List[Tuple[int, ...]], p: Params) -> float:
    """Conservative continuation value for next observable states outside the sampled PBVI grid.

    The offline solver may use a sampled subset of reachable observable states.
    In the previous implementation, transitions to states outside that subset used
    ``0.0`` as continuation value through ``dict.get``. Because the problem is
    formulated as discounted cost minimization, that made off-grid transitions
    artificially attractive.

    This fallback uses the current expected one-step cost as a stationary
    infinite-horizon proxy. It is deliberately simple, deterministic, and
    non-optimistic. Setting ``pbvi_max_obs_states=0`` still forces full observable
    enumeration and therefore avoids this approximation entirely.
    """
    denom = max(1.0 - p.gamma, 1e-12)
    return expected_cost(obs, belief, actions, hidden_states, p) / denom


def solve_offline_belief_grid(p: Params) -> OfflinePolicy:
    if p.n_agvs > 3:
        raise ValueError("Offline belief-grid solver is intended for small fleets (<=3 AGVs).")

    start = time.time()
    hidden_states = enumerate_hidden_states(p.n_agvs)
    obs_states = sample_observable_reachable_states(p.n_agvs, p)
    actions = enumerate_actions(p.n_agvs, p, reduced=True)
    alerts = all_alert_vectors(p.n_agvs)
    belief_points = generate_belief_points(hidden_states, p.pbvi_belief_points, p.pbvi_seed)

    V: Dict[Tuple[Tuple[int, ...], int], float] = {(obs, bi): 0.0 for obs in obs_states for bi in range(len(belief_points))}
    policy: Dict[Tuple[Tuple[int, ...], int], Tuple[int, ...]] = {}
    print(f"Offline solver grid: obs={len(obs_states)}, hidden={len(hidden_states)}, beliefs={len(belief_points)}, actions={len(actions)}, iterations={p.pbvi_iterations}", flush=True)

    # Precompute transitions for speed: (obs, hidden_idx, action) -> dist
    trans_cache: Dict[Tuple[Tuple[int, ...], int, Tuple[int, ...]], Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], float]] = {}

    hidden_index = {c: i for i, c in enumerate(hidden_states)}

    off_grid_fallback_count = 0

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
                            key = (next_obs, bj)
                            if key in V:
                                future += prob_obs * V[key]
                            else:
                                off_grid_fallback_count += 1
                                future += prob_obs * off_grid_fallback_value(next_obs, b_next, u, hidden_states, p)

                    q += p.gamma * future
                    if q < best_q:
                        best_q = q
                        best_u = u

                V_new[(obs, bi)] = best_q
                policy[(obs, bi)] = best_u
        V = V_new
        print(f"  PBVI iteration {it + 1}/{p.pbvi_iterations} done", flush=True)

    metadata = {
        "solve_seconds": time.time() - start,
        "n_obs_states": len(obs_states),
        "n_hidden_states": len(hidden_states),
        "n_belief_points": len(belief_points),
        "n_actions": len(actions),
        "iterations": p.pbvi_iterations,
        "solver_kind": "finite_belief_grid_pbvi",
        "off_grid_fallback_count": off_grid_fallback_count,
        "off_grid_fallback": "expected_one_step_cost_over_1_minus_gamma",
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



def _pbvi_best_alpha(obs: Tuple[int, ...], belief: np.ndarray, gamma_sets: Dict[Tuple[int, ...], np.ndarray]) -> Tuple[np.ndarray, float, int]:
    """Return the minimum-cost alpha-vector for a belief at an observable state."""
    alphas = gamma_sets.get(obs)
    if alphas is None or len(alphas) == 0:
        # This should only happen with sampled observable grids. A zero vector is
        # non-conservative, so callers should avoid this path in validated runs
        # by using pbvi_max_obs_states=0.
        z = np.zeros_like(belief, dtype=float)
        return z, 0.0, 0
    vals = alphas @ belief
    idx = int(np.argmin(vals))
    return alphas[idx], float(vals[idx]), idx


def _prune_alpha_set(
    candidates: List[Tuple[np.ndarray, Tuple[int, ...]]],
    belief_points: np.ndarray,
    atol: float = 1e-10,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """PBVI-style pruning: keep vectors that are optimal for at least one sampled belief.

    This is not full LP-based dominance pruning; it is the standard lightweight
    PBVI pruning sufficient for a point-based approximation and deterministic
    diagnostics. Duplicate vectors/actions are removed first.
    """
    unique: List[Tuple[np.ndarray, Tuple[int, ...]]] = []
    for alpha, action in candidates:
        if not any(action == a2 and np.allclose(alpha, a2v, atol=atol, rtol=0.0) for a2v, a2 in unique):
            unique.append((alpha, action))

    keep_indices = set()
    mat = np.vstack([a for a, _ in unique])
    for b in belief_points:
        keep_indices.add(int(np.argmin(mat @ b)))

    kept = [unique[i] for i in sorted(keep_indices)]
    return np.vstack([a for a, _ in kept]), [act for _, act in kept]



def solve_momdp_alpha_pbvi(p: Params) -> OfflinePolicy:
    """Solve a small-fleet MOMDP with true alpha-vector PBVI.

    The observable component is (automation, health), and alpha-vectors are over
    the hidden joint cyber state. For each observable state o, the solver stores
    a set Gamma[o] of alpha-vectors. Online execution evaluates the continuous
    belief directly; it does not use nearest-neighbor belief lookup.

    This is a cost-minimization variant of PBVI:
        V(o, b) = min_{alpha in Gamma[o]} alpha · b.

    Step-3 implementation note: the backup precomputes sparse transition /
    observation matrices M_{o,u,o',z}[c,c'] = P(o', c', z | o, c, u), which
    replaces the earlier dictionary-heavy nested loops and makes 2-AGV sampled
    validation practical.
    """
    if p.n_agvs > 3:
        raise ValueError("MOMDP alpha-vector PBVI is intended for small fleets (<=3 AGVs).")

    start = time.time()
    hidden_states = enumerate_hidden_states(p.n_agvs)
    hidden_index = {c: i for i, c in enumerate(hidden_states)}
    obs_states = sample_observable_reachable_states(p.n_agvs, p)
    obs_set = set(obs_states)
    n_hidden = len(hidden_states)

    # Keep the validated core action set tractable. It contains the main
    # recovery decisions used by the planner: wait, physical maintenance,
    # patching, and rejuvenation. Wider action sets can be reintroduced once the
    # alpha backup is fully validated and profiled.
    alpha_local_actions = (int(Action.NOOP), int(Action.MAINTAIN), int(Action.PATCH), int(Action.REJUVENATE))
    actions: List[Tuple[int, ...]] = []
    for prod in itertools.product(alpha_local_actions, repeat=p.n_agvs):
        u = tuple(int(x) for x in prod)
        if action_feasible(u, p):
            actions.append(u)

    alerts = all_alert_vectors(p.n_agvs)
    belief_points = generate_belief_points(hidden_states, p.pbvi_belief_points, p.pbvi_seed)

    print(
        f"MOMDP alpha-PBVI grid: obs={len(obs_states)}, hidden={n_hidden}, "
        f"beliefs={len(belief_points)}, actions={len(actions)}, iterations={p.pbvi_iterations}",
        flush=True,
    )

    nearest_obs_cache: Dict[Tuple[int, ...], Tuple[int, ...]] = {}
    def nearest_grid_obs(x: Tuple[int, ...]) -> Tuple[int, ...]:
        if x in obs_set:
            return x
        cached = nearest_obs_cache.get(x)
        if cached is not None:
            return cached
        arr = np.array(x, dtype=float)
        best = min(obs_states, key=lambda o: float(np.sum((np.array(o, dtype=float) - arr) ** 2)))
        nearest_obs_cache[x] = best
        return best

    # Precompute immediate cost vectors and transition-observation matrices.
    # backup_data[(obs,u)] = (cost_vec, M_stack, next_obs_for_matrix, groups)
    # where M_stack[k,c,c'] = P(next_obs_k, c', z_k | obs, c, u).
    backup_data = {}
    off_grid_count = 0
    pre_start = time.time()
    for obs in obs_states:
        for u in actions:
            cost_vec = np.array([cost_from_obs_hidden_action(obs, c_tuple, u, p) for c_tuple in hidden_states], dtype=float)
            mats: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], np.ndarray] = {}
            for ci, c_tuple in enumerate(hidden_states):
                dist = global_transition_dist(obs, c_tuple, u, p)
                for (next_obs, next_c), pr in dist.items():
                    grid_obs = nearest_grid_obs(next_obs)
                    if grid_obs != next_obs:
                        off_grid_count += 1
                    nci = hidden_index[next_c]
                    for z in alerts:
                        key = (grid_obs, z)
                        M = mats.get(key)
                        if M is None:
                            M = np.zeros((n_hidden, n_hidden), dtype=float)
                            mats[key] = M
                        M[ci, nci] += float(pr) * alert_likelihood(z, next_c, p)
            mat_items = [(go, z, M) for (go, z), M in mats.items() if np.any(M)]
            if mat_items:
                next_obs_for_matrix = [go for (go, _z, _M) in mat_items]
                M_stack = np.stack([M for (_go, _z, M) in mat_items], axis=0)
            else:
                next_obs_for_matrix = []
                M_stack = np.zeros((0, n_hidden, n_hidden), dtype=float)
            groups: Dict[Tuple[int, ...], np.ndarray] = {}
            for go in set(next_obs_for_matrix):
                groups[go] = np.array([i for i, x in enumerate(next_obs_for_matrix) if x == go], dtype=int)
            backup_data[(obs, u)] = (cost_vec, M_stack, next_obs_for_matrix, groups)
    print(f"  precomputed backup matrices in {time.time() - pre_start:.2f}s", flush=True)

    gamma_sets: Dict[Tuple[int, ...], np.ndarray] = {obs: np.zeros((1, n_hidden), dtype=float) for obs in obs_states}
    gamma_actions: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {obs: [tuple([int(Action.NOOP)] * p.n_agvs)] for obs in obs_states}

    eps = 1e-12
    for it in range(p.pbvi_iterations):
        new_gamma_sets: Dict[Tuple[int, ...], np.ndarray] = {}
        new_gamma_actions: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}

        for obs in obs_states:
            # Step-4 optimization: batch all representative beliefs for a given
            # observable state/action. This removes a costly inner Python loop
            # and lets NumPy compute continuation-alpha selection and alpha
            # reconstruction for the whole belief set at once.
            B = belief_points
            B_select = B + eps
            B_select = B_select / B_select.sum(axis=1, keepdims=True)
            n_beliefs = B.shape[0]
            best_alphas_by_b = np.zeros((n_beliefs, n_hidden), dtype=float)
            best_actions_by_b: List[Optional[Tuple[int, ...]]] = [None] * n_beliefs
            best_values_by_b = np.full(n_beliefs, np.inf, dtype=float)

            for u in actions:
                cost_vec, M_stack, next_obs_for_matrix, groups = backup_data[(obs, u)]
                if M_stack.shape[0] == 0:
                    candidate_alphas = np.repeat(cost_vec.reshape(1, -1), n_beliefs, axis=0)
                else:
                    # weights[b,k,c'] = sum_c B_select[b,c] M[k,c,c']
                    weights = np.einsum("bc,kcj->bkj", B_select, M_stack, optimize=True)
                    probs = weights.sum(axis=2)
                    alpha_next = np.zeros((n_beliefs, M_stack.shape[0], n_hidden), dtype=float)

                    for next_grid_obs, idxs in groups.items():
                        alphas_next = gamma_sets.get(next_grid_obs)
                        if alphas_next is None or len(alphas_next) == 0:
                            continue
                        # Normalize only rows with positive probability. Use an
                        # epsilon denominator for vectorized safety; rows with
                        # zero probability remain irrelevant in the final sum.
                        denom = np.maximum(probs[:, idxs], eps)
                        b_nexts = weights[:, idxs, :] / denom[:, :, None]
                        # vals[b,row,alpha] = alpha · b_next[b,row]
                        vals = np.einsum("bkh,ah->bka", b_nexts, alphas_next, optimize=True)
                        best = np.argmin(vals, axis=2)
                        alpha_next[:, idxs, :] = alphas_next[best]

                    cont = np.einsum("kij,bkj->bi", M_stack, alpha_next, optimize=True)
                    candidate_alphas = cost_vec.reshape(1, -1) + p.gamma * cont

                candidate_values = np.einsum("bh,bh->b", candidate_alphas, B, optimize=True)
                improved = candidate_values < best_values_by_b
                if np.any(improved):
                    best_values_by_b[improved] = candidate_values[improved]
                    best_alphas_by_b[improved] = candidate_alphas[improved]
                    for bi in np.where(improved)[0]:
                        best_actions_by_b[int(bi)] = u

            obs_candidates: List[Tuple[np.ndarray, Tuple[int, ...]]] = []
            for bi in range(n_beliefs):
                assert best_actions_by_b[bi] is not None
                obs_candidates.append((best_alphas_by_b[bi].copy(), best_actions_by_b[bi]))

            pruned_alphas, pruned_actions = _prune_alpha_set(obs_candidates, belief_points)
            new_gamma_sets[obs] = pruned_alphas
            new_gamma_actions[obs] = pruned_actions

        gamma_sets = new_gamma_sets
        gamma_actions = new_gamma_actions
        n_alpha = sum(len(v) for v in gamma_sets.values())
        print(f"  alpha-PBVI iteration {it + 1}/{p.pbvi_iterations} done; alpha_vectors={n_alpha}", flush=True)

    table_policy: Dict[Tuple[Tuple[int, ...], int], Tuple[int, ...]] = {}
    table_values: Dict[Tuple[Tuple[int, ...], int], float] = {}
    for obs in obs_states:
        alphas = gamma_sets[obs]
        acts = gamma_actions[obs]
        for bi, b in enumerate(belief_points):
            vals = alphas @ b
            idx = int(np.argmin(vals))
            table_values[(obs, bi)] = float(vals[idx])
            table_policy[(obs, bi)] = acts[idx]

    metadata = {
        "solve_seconds": time.time() - start,
        "n_obs_states": len(obs_states),
        "n_hidden_states": n_hidden,
        "n_belief_points": len(belief_points),
        "n_actions": len(actions),
        "iterations": p.pbvi_iterations,
        "solver_kind": "momdp_alpha_pbvi",
        "n_alpha_vectors": sum(len(v) for v in gamma_sets.values()),
        "off_grid_fallback_count": off_grid_count,
        "value_representation": "V(obs,b)=min_alpha alpha_dot_b",
        "backup_precomputation": "transition_observation_matrices",
    }
    return OfflinePolicy(
        n_agvs=p.n_agvs,
        hidden_states=hidden_states,
        obs_states=obs_states,
        belief_points=belief_points,
        policy=table_policy,
        values=table_values,
        metadata=metadata,
        alpha_vectors=gamma_sets,
        alpha_actions=gamma_actions,
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
    pred_any_obs = np.zeros(len(hidden_states))
    observed_next_obs = obs_tuple_from_state(state)

    for ci, c_tuple in enumerate(hidden_states):
        bprob = state.belief_joint[ci]
        if bprob == 0:
            continue
        dist = global_transition_dist(prev_obs, c_tuple, actions, p)
        for (next_obs, next_c), pr in dist.items():
            pred_any_obs[hidden_index[next_c]] += bprob * pr
            if next_obs == observed_next_obs:
                pred[hidden_index[next_c]] += bprob * pr

    weights = np.zeros(len(hidden_states))
    for ci, c_tuple in enumerate(hidden_states):
        weights[ci] = pred[ci] * alert_likelihood(alerts, c_tuple, p)
    if weights.sum() <= 0:
        # Numerical/model fallback: if the exact observable transition has zero
        # likelihood, keep the cyber prediction and update only with alert
        # likelihood rather than resetting the belief to the prior.
        for ci, c_tuple in enumerate(hidden_states):
            weights[ci] = pred_any_obs[ci] * alert_likelihood(alerts, c_tuple, p)
    if weights.sum() <= 0:
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
    failed = [i for i in np.where(state.health == Health.FAILED)[0] if not is_agv_unavailable(state, int(i))]
    for i in failed[: p.alpha_technicians]:
        actions[i] = Action.MAINTAIN
    return actions


def reactive_cyber_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    actions = np.full(p.n_agvs, Action.NOOP, dtype=int)
    belief = state.belief_compromised if state.belief_compromised is not None else np.zeros(p.n_agvs)
    for i in range(p.n_agvs):
        if is_agv_unavailable(state, i):
            continue
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
        if is_agv_unavailable(state, i):
            continue
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


def _belief_array(state: State, p: Params) -> np.ndarray:
    return state.belief_compromised if state.belief_compromised is not None else np.zeros(p.n_agvs)


def _interaction_cost_for_proxy(automation: np.ndarray, health: np.ndarray, actions: np.ndarray, p: Params) -> float:
    return interaction_cost_from_arrays(
        tuple(int(a) for a in automation),
        tuple(int(h) for h in health),
        tuple(int(u) for u in actions),
        p,
    )


def _local_risk_score(i: int, state: State, alerts: np.ndarray, p: Params) -> float:
    """Local cyber/physical risk used by IAGR.

    This is not a value function. It is a scalable online priority score:
    physical degradation, automation loss, cyber belief, and current alerts
    increase the priority of recovering AGV i.
    """
    belief = _belief_array(state, p)
    score = 0.0
    score += p.cost_automation_step * max(0, 5 - int(state.automation[i]))
    if state.health[i] == Health.DEGRADED:
        score += p.cost_degraded_phys
    elif state.health[i] == Health.FAILED:
        score += p.cost_failed
    score += belief[i] * p.cost_compromised
    score += p.iagr_w_alert * int(alerts[i]) * p.cost_suspected
    return float(score)


def _interaction_benefit_score(i: int, state: State, p: Params) -> float:
    """Estimate system-level benefit if AGV i is restored to nominal capability.

    This approximates the reduction of C_int obtained by recovering the AGV.
    It captures bottleneck/convoy/sequential/coordination effects without
    solving the full MOMDP online.
    """
    noop = np.full(p.n_agvs, Action.NOOP, dtype=int)
    current = _interaction_cost_for_proxy(state.automation, state.health, noop, p)

    a2 = state.automation.copy()
    h2 = state.health.copy()
    a2[i] = 5
    h2[i] = Health.HEALTHY
    improved = _interaction_cost_for_proxy(a2, h2, noop, p)
    return float(max(0.0, current - improved))


def _candidate_actions_for_agv(i: int, state: State, alerts: np.ndarray, p: Params) -> List[int]:
    """Return feasible local recovery candidates for IAGR before global capacities."""
    if is_agv_unavailable(state, i):
        return [int(Action.NOOP)]
    belief = _belief_array(state, p)
    candidates = [int(Action.NOOP)]

    if state.health[i] in (Health.DEGRADED, Health.FAILED):
        candidates.append(int(Action.MAINTAIN))
    if alerts[i] or belief[i] > p.belief_patch_threshold:
        candidates.append(int(Action.PATCH))
    if belief[i] > p.belief_rejuvenate_threshold:
        candidates.append(int(Action.REJUVENATE))
    if belief[i] > 0.10:
        candidates.append(int(Action.MONITOR))
    if belief[i] > 0.75 or state.cyber[i] == Cyber.COMPROMISED:
        candidates.append(int(Action.ISOLATE))
    if state.automation[i] > 0 and (state.health[i] == Health.FAILED or belief[i] > 0.90):
        candidates.append(int(Action.STOP))

    # Preserve order while removing duplicates.
    out = []
    for a in candidates:
        if a not in out:
            out.append(a)
    return out


def _scaled_action_cost(action: int, p: Params) -> float:
    if action == Action.NOOP:
        return 0.0
    if action == Action.MAINTAIN:
        base = p.cost_maintain
    elif action == Action.PATCH:
        base = p.cost_patch
    elif action == Action.REJUVENATE:
        base = p.cost_rejuvenate
    elif action == Action.MONITOR:
        base = p.cost_monitor
    elif action == Action.ISOLATE:
        base = p.cost_isolate
    elif action == Action.STOP:
        base = p.cost_stop
    elif action == Action.DOWNGRADE:
        base = p.cost_downgrade
    else:
        base = 0.0
    duration = action_downtime_duration(action, p)
    return p.intervention_cost_multiplier * base + duration * p.intervention_downtime_penalty


def _expected_action_benefit(i: int, action: int, state: State, alerts: np.ndarray, p: Params) -> float:
    """Scalable finite-lookahead proxy benefit of applying action to AGV i.

    The first IAGR version used an almost purely one-step score. That made it
    under-value preventive maintenance: paying the maintenance cost while an AGV
    is only degraded looked unattractive, even though it avoids repeated degraded
    operation, failure risk, and interaction bottlenecks. This version keeps the
    computation linear in the fleet size, but multiplies persistent risk terms by
    a short finite-lookahead factor. It is still a heuristic, not an online MOMDP
    solver.
    """
    belief = _belief_array(state, p)
    local_risk = _local_risk_score(i, state, alerts, p)
    interaction_benefit = _interaction_benefit_score(i, state, p)
    lookahead = max(1.0, float(getattr(p, "iagr_risk_lookahead", 1.0)))
    auto_gap = max(0, 5 - int(state.automation[i]))

    if action == Action.NOOP:
        return 0.0

    if action == Action.MAINTAIN:
        if state.health[i] == Health.FAILED:
            # Repair relieves persistent failed-state cost, automation loss, and
            # interaction bottlenecks.
            benefit = lookahead * (p.cost_failed + p.cost_automation_step * auto_gap)
        elif state.health[i] == Health.DEGRADED:
            # Preventive maintenance avoids persistent degradation and the
            # probability of entering the much more expensive failed state.
            avoided_failure = p.p_fail_from_degraded * p.cost_failed * lookahead
            avoided_degraded = p.cost_degraded_phys * lookahead
            automation_relief = p.cost_automation_step * auto_gap * min(lookahead, 3.0)
            benefit = avoided_degraded + avoided_failure + automation_relief
        else:
            benefit = 0.0
        benefit += p.iagr_w_interaction * interaction_benefit * min(lookahead, 4.0)
        return benefit - _scaled_action_cost(action, p)

    if action == Action.PATCH:
        # Patching is cheap and useful for alert-driven or moderately risky AGVs.
        cyber_risk = belief[i] * p.cost_compromised + int(alerts[i]) * p.cost_suspected
        latent_progression = (1.0 - belief[i]) * int(alerts[i]) * p.p_compromise_from_suspected * p.cost_compromised
        benefit = (cyber_risk + latent_progression) * p.p_patch_success * min(lookahead, 3.0)
        benefit += 0.20 * p.iagr_w_interaction * interaction_benefit
        return benefit - _scaled_action_cost(action, p)

    if action == Action.REJUVENATE:
        cyber_risk = belief[i] * p.cost_compromised + int(alerts[i]) * p.cost_suspected
        benefit = cyber_risk * p.p_rejuvenate_success * min(lookahead, 4.0)
        benefit += 0.20 * p.iagr_w_interaction * interaction_benefit
        return benefit - _scaled_action_cost(action, p)

    if action == Action.MONITOR:
        benefit = belief[i] * p.cost_compromised * p.p_monitor_suspected_to_secure * min(lookahead, 3.0)
        return benefit - _scaled_action_cost(action, p)

    if action == Action.ISOLATE:
        # Safety-driven action: reduce cyber exposure but may lower automation and interaction performance.
        benefit = 0.5 * belief[i] * p.cost_compromised + 0.5 * int(alerts[i]) * p.cost_suspected
        return benefit - _scaled_action_cost(action, p)

    if action == Action.STOP:
        benefit = local_risk
        return benefit - _scaled_action_cost(action, p)

    if action == Action.DOWNGRADE:
        benefit = 0.25 * belief[i] * p.cost_compromised
        return benefit - _scaled_action_cost(action, p)

    return -float("inf")


def interaction_aware_greedy_recovery_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    """Interaction-Aware Greedy Recovery (IAGR).

    IAGR is a scalable interaction-aware refinement of the Immediate
    Intervention (II) baseline.  Earlier versions tried to decide both *whether*
    to intervene and *where* to allocate scarce resources from a single myopic
    score.  Calibration showed that this can make IAGR unnecessarily worse than
    II in weakly coupled cases.  The tuned version therefore uses II as a robust
    local fallback and improves the scarce-resource allocation step.

    Concretely, cyber actions remain governed by the same alert/belief logic as
    II, while the scarce technician budget is allocated to the AGVs with the
    largest interaction-aware maintenance priority.  This preserves II's strong
    local behavior and makes IAGR differ from II mainly when several AGVs compete
    for maintenance resources.
    """
    # Robust local fallback: this avoids making IAGR worse than II merely
    # because a proxy score was too conservative or too aggressive.
    actions = immediate_intervention_policy(state, alerts, p).copy()

    # No need to reallocate if there is no scarce maintenance conflict.
    needy = [
        i for i in range(p.n_agvs)
        if (not is_agv_unavailable(state, i))
        and state.health[i] in (Health.DEGRADED, Health.FAILED)
    ]
    if len(needy) <= p.alpha_technicians:
        return actions

    # Remove the arbitrary index-order maintenance choices made by II, then
    # reassign the technician(s) to the most system-critical AGVs.
    for i in range(p.n_agvs):
        if actions[i] == Action.MAINTAIN:
            actions[i] = Action.NOOP

    scored: List[Tuple[float, int]] = []
    for i in needy:
        score = _expected_action_benefit(i, int(Action.MAINTAIN), state, alerts, p)
        # Tie-break toward lower automation and failed health because these are
        # observable and directly affect interaction bottlenecks.
        score += 0.01 * max(0, 5 - int(state.automation[i]))
        score += 0.1 * int(state.health[i] == Health.FAILED)
        scored.append((float(score), i))
    scored.sort(reverse=True, key=lambda x: x[0])

    for _, i in scored[: p.alpha_technicians]:
        actions[i] = Action.MAINTAIN

    # For non-maintained AGVs that II would have maintained only because it
    # visited them earlier in index order, allow cyber remediation if warranted.
    belief = _belief_array(state, p)
    for _, i in scored[p.alpha_technicians:]:
        if belief[i] > p.belief_rejuvenate_threshold:
            actions[i] = Action.REJUVENATE
        elif alerts[i] or belief[i] > p.belief_patch_threshold:
            actions[i] = Action.PATCH
        else:
            actions[i] = Action.NOOP

    return actions





def _campaign_action_cost(action: int, k: int, p: Params) -> float:
    """Cost of a k-AGV same-action campaign, including downtime penalties.

    For k=1 this equals the ordinary individual action cost.  For k>1, the
    one-shot action cost follows the shared setup/marginal campaign model while
    downtime remains per-AGV.  This mirrors total_cost(), where campaign
    accounting affects action_base_cost() but not the explicit downtime term.
    """
    if k <= 0:
        return 0.0
    base = action_base_cost(action, p)
    duration = action_downtime_duration(action, p)
    if not getattr(p, "campaign_cost_model", False) or action not in CAMPAIGN_ACTIONS:
        setup_part = k * base
    else:
        marginal = float(getattr(p, "campaign_marginal_cost_factor", 0.45))
        setup_part = base * (1.0 + marginal * max(0, k - 1))
    return p.intervention_cost_multiplier * setup_part + k * duration * p.intervention_downtime_penalty


def _gross_action_benefit(i: int, action: int, state: State, alerts: np.ndarray, p: Params) -> float:
    """Benefit proxy before subtracting the individual action cost."""
    return _expected_action_benefit(i, action, state, alerts, p) + _scaled_action_cost(action, p)


def _campaign_members_for_action(group: Sequence[int], action: int, state: State, alerts: np.ndarray, p: Params) -> List[Tuple[float, int, bool]]:
    """Return candidate members as (gross_benefit, agv, preventive_flag)."""
    belief = _belief_array(state, p)
    group = [i for i in group if not is_agv_unavailable(state, i)]
    out: List[Tuple[float, int, bool]] = []

    if action == int(Action.MAINTAIN):
        for i in group:
            if state.health[i] in (Health.DEGRADED, Health.FAILED):
                out.append((_gross_action_benefit(i, action, state, alerts, p), i, False))
        return out

    if action == int(Action.PATCH):
        # A cyber campaign is triggered by at least one alert or moderately risky
        # AGV.  Once the setup exists, IAGR-C may include compatible low-risk
        # neighbours preventively when their marginal campaign cost is low.
        trigger = any(bool(alerts[i]) or belief[i] >= getattr(p, "iagrc_trigger_patch_belief", 0.18) for i in group)
        if not trigger:
            return out
        preventive_thr = float(getattr(p, "iagrc_preventive_patch_belief", 0.08))
        for i in group:
            is_direct = bool(alerts[i]) or belief[i] >= p.belief_patch_threshold
            is_preventive = (not is_direct) and belief[i] >= preventive_thr
            if is_direct or is_preventive:
                gross = _gross_action_benefit(i, action, state, alerts, p)
                # Make preventive patching visible to the campaign optimizer but
                # still conservative: a very small latent-risk term prevents it
                # from choosing completely safe AGVs just for a discount.
                if is_preventive:
                    gross += 0.5 * belief[i] * p.cost_compromised * min(float(getattr(p, "iagr_risk_lookahead", 1.0)), 3.0)
                out.append((gross, i, bool(is_preventive)))
        return out

    if action == int(Action.REJUVENATE):
        thr = float(getattr(p, "iagrc_rejuvenate_belief", 0.45))
        for i in group:
            if belief[i] >= thr or (bool(alerts[i]) and belief[i] >= 0.25):
                out.append((_gross_action_benefit(i, action, state, alerts, p), i, False))
        return out

    return out


def _campaign_score(members: Sequence[Tuple[float, int, bool]], action: int, p: Params) -> Tuple[float, Tuple[int, ...], int]:
    """Compute score and member tuple for a same-action campaign."""
    if not members:
        return -float("inf"), tuple(), 0
    members = sorted(members, reverse=True, key=lambda x: x[0])
    max_size = max(1, int(getattr(p, "iagrc_campaign_max_size", 5)))
    members = members[:max_size]
    best_score = -float("inf")
    best_members: Tuple[int, ...] = tuple()
    best_preventive = 0
    min_size = max(1, int(getattr(p, "iagrc_min_campaign_size", 2)))
    for k in range(1, len(members) + 1):
        chosen = members[:k]
        if k < min_size:
            # Singleton campaigns are handled by IAGR fallback; IAGR-C should
            # be justified by actual grouping unless the user lowers min size.
            continue
        gross = sum(float(x[0]) for x in chosen)
        cost = _campaign_action_cost(action, k, p)
        score = gross - cost
        if score > best_score:
            best_score = score
            best_members = tuple(int(x[1]) for x in chosen)
            best_preventive = sum(1 for x in chosen if bool(x[2]))
    return best_score, best_members, best_preventive



def _selection_proxy_value(actions: Sequence[int], state: State, alerts: np.ndarray, p: Params) -> float:
    """Proxy value of an action vector for campaign-aware local search."""
    actions_arr = np.array(actions, dtype=int)
    gross = 0.0
    for i, a in enumerate(actions_arr):
        if int(a) == int(Action.NOOP):
            continue
        gross += _gross_action_benefit(i, int(a), state, alerts, p)
    action_cost = p.intervention_cost_multiplier * campaign_action_base_cost(actions_arr, p)
    downtime_cost = sum(action_downtime_duration(int(a), p) for a in actions_arr) * p.intervention_downtime_penalty
    return float(gross - action_cost - downtime_cost)


def interaction_aware_campaign_recovery_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    """Explicit campaign-planning IAGR-C with safe improvement over IAGR.

    The policy starts from tuned IAGR, then evaluates campaign-level edits.  A
    campaign edit may add preventive members or replace isolated same-group
    actions, but it is accepted only if it improves a common proxy objective that
    accounts for shared campaign costs.  This keeps IAGR-C from becoming worse
    than IAGR merely because the campaign generator is over-eager, while still
    allowing it to deliberately include preventive patch/rejuvenation actions
    that SEP/II would only obtain accidentally.
    """
    n = p.n_agvs
    base = interaction_aware_greedy_recovery_policy(state, alerts, p).copy()
    actions = base.copy()
    current_value = _selection_proxy_value(actions, state, alerts, p)

    # Build candidate campaign edits from interaction groups.  We allow cyber
    # campaigns to add preventive members.  Maintenance campaigns are constrained
    # by the technician budget and therefore mainly reallocate scarce repairs.
    edits: List[Tuple[float, np.ndarray, Tuple[int, ...], int, int]] = []
    for group in campaign_groups(n, p):
        group_set = set(group)
        for action in (int(Action.PATCH), int(Action.REJUVENATE), int(Action.MAINTAIN)):
            # IAGR-C is a campaign *planner*, but it is still anchored in the
            # robust tuned IAGR decision.  A campaign must contain at least one
            # intervention that IAGR already considered necessary in the same
            # group; the campaign planner then decides whether compatible
            # preventive members should be added at low marginal cost.
            if action == int(Action.PATCH):
                has_anchor = any(int(base[i]) in (int(Action.PATCH), int(Action.REJUVENATE)) for i in group)
            else:
                has_anchor = any(int(base[i]) == action for i in group)
            if not has_anchor:
                continue
            cand_members = _campaign_members_for_action(group, action, state, alerts, p)
            if action == int(Action.MAINTAIN):
                cand_members = sorted(cand_members, reverse=True, key=lambda x: x[0])[: max(1, p.alpha_technicians)]
            else:
                cand_members = sorted(cand_members, reverse=True, key=lambda x: x[0])[: max(1, int(getattr(p, "iagrc_campaign_max_size", 5)))]
            if len(cand_members) < max(2, int(getattr(p, "iagrc_min_campaign_size", 2))):
                continue

            # Try prefixes: adding every candidate is not always best.
            for k in range(max(2, int(getattr(p, "iagrc_min_campaign_size", 2))), len(cand_members) + 1):
                members = tuple(int(x[1]) for x in cand_members[:k])
                preventive_count = sum(1 for x in cand_members[:k] if bool(x[2]))
                candidate = actions.copy()

                # Do not overwrite a heavy physical repair with a cyber action;
                # this avoids losing scarce physical recovery just to form a
                # cyber campaign.  Other cyber-to-cyber replacements are allowed
                # if the proxy says they are beneficial.
                valid = True
                for i in members:
                    if action != int(Action.MAINTAIN) and candidate[i] == Action.MAINTAIN:
                        valid = False
                        break
                    candidate[i] = action
                if not valid:
                    continue

                if action == int(Action.MAINTAIN) and int(np.sum(candidate == Action.MAINTAIN)) > p.alpha_technicians:
                    continue

                value = _selection_proxy_value(candidate, state, alerts, p)
                delta = value - current_value
                if delta > float(getattr(p, "iagrc_min_campaign_score", 0.0)):
                    edits.append((float(delta), candidate, members, action, int(preventive_count)))

    # Apply non-overlapping positive edits greedily by improvement per involved AGV.
    edits.sort(reverse=True, key=lambda x: (x[0] / max(1, len(x[2])), x[0]))
    used: set[int] = set()
    for delta, candidate, members, action, preventive_count in edits:
        if any(i in used for i in members):
            continue
        # Re-evaluate against the current action vector because previous edits
        # can change campaign costs for the same action type.
        if action == int(Action.MAINTAIN) and int(np.sum(candidate == Action.MAINTAIN)) > p.alpha_technicians:
            continue
        value = _selection_proxy_value(candidate, state, alerts, p)
        if value > current_value + float(getattr(p, "iagrc_min_campaign_score", 0.0)):
            actions = candidate
            current_value = value
            used.update(members)

    return actions


# Backward-compatible alias: older scripts may still refer to JOINT.
def joint_heuristic_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    return interaction_aware_greedy_recovery_policy(state, alerts, p)


def pbvi_policy(state: State, alerts: np.ndarray, p: Params) -> np.ndarray:
    global _CURRENT_OFFLINE_POLICY
    pol = _CURRENT_OFFLINE_POLICY
    if pol is None or pol.n_agvs != p.n_agvs or state.belief_joint is None:
        return joint_heuristic_policy(state, alerts, p)
    obs = obs_tuple_from_state(state)

    # True MOMDP alpha-vector PBVI: evaluate the continuous belief directly.
    if pol.metadata.get("solver_kind") == "momdp_alpha_pbvi" and pol.alpha_vectors:
        alphas = pol.alpha_vectors.get(obs)
        actions = pol.alpha_actions.get(obs)
        if alphas is None or actions is None or len(actions) == 0:
            return joint_heuristic_policy(state, alerts, p)
        vals = alphas @ state.belief_joint
        idx = int(np.argmin(vals))
        return np.array(actions[idx], dtype=int)

    # Finite belief-grid fallback/reference solver.
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
    "IAGR": interaction_aware_greedy_recovery_policy,
    "IAGR_C": interaction_aware_campaign_recovery_policy,
    # Backward-compatible alias. Use IAGR in plots/paper text.
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
    unavailable_epochs = 0
    automation_sum = 0.0
    campaign_count_sum = 0.0
    campaign_member_sum = 0.0
    campaign_preventive_sum = 0.0
    campaign_savings_sum = 0.0

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
        if state.intervention_remaining is not None:
            unavailable_epochs += int(np.sum(state.intervention_remaining > 0))
        automation_sum += float(np.mean(state.automation))
        camp = campaign_execution_stats(state, actions, alerts, p)
        campaign_count_sum += camp["campaigns"]
        campaign_member_sum += camp["campaign_members"]
        campaign_preventive_sum += camp["campaign_preventive_members"]
        campaign_savings_sum += camp["campaign_savings"]

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
        "unavailable_epochs": unavailable_epochs,
        "avg_automation": automation_sum / p.horizon,
        "campaigns": campaign_count_sum,
        "campaign_members": campaign_member_sum,
        "campaign_preventive_members": campaign_preventive_sum,
        "campaign_savings": campaign_savings_sum,
        "mean_campaign_size": (campaign_member_sum / campaign_count_sum) if campaign_count_sum > 0 else 0.0,
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
    metrics = ["discounted_cost", "raw_cost", "failsoft_epochs", "safestop_epochs", "interventions", "unavailable_epochs", "avg_automation", "campaigns", "campaign_members", "campaign_preventive_members", "campaign_savings", "mean_campaign_size"]
    return (
        df.groupby(["condition", "policy", "n_agvs", "interaction_profile", "interaction_strength", "alpha_technicians", "beta_monitored_agvs"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )


def run_all(base: Params, offline_policy: Optional[OfflinePolicy]) -> pd.DataFrame:
    small_policies = ["RM", "RC", "SEP", "II", "IAGR"] + (["PBVI"] if offline_policy is not None and offline_policy.n_agvs == base.n_agvs else [])
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
        policies = ["RM", "RC", "SEP", "II", "IAGR"]
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
    parser.add_argument("--solve-pbvi", action="store_true", help="Solve offline PBVI policy before running experiments")
    parser.add_argument("--solver", choices=["momdp-pbvi", "belief-grid"], default="momdp-pbvi", help="Offline solver: true alpha-vector MOMDP-PBVI or finite belief-grid reference")
    parser.add_argument("--policy-in", default=None, help="Load offline policy from pickle")
    parser.add_argument("--policy-out", default="offline_agv_policy.pkl", help="Where to save solved offline policy")
    parser.add_argument("--pbvi-iterations", type=int, default=10)
    parser.add_argument("--belief-points", type=int, default=16)
    parser.add_argument("--pbvi-max-obs-states", type=int, default=64, help="Max reachable observable states for offline solver; 0 means full enumeration")
    parser.add_argument("--pbvi-rollouts", type=int, default=200, help="Rollouts used to collect reachable observable states")
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
        pbvi_max_obs_states=args.pbvi_max_obs_states,
        pbvi_rollouts=args.pbvi_rollouts,
    )

    offline_policy = None
    if args.policy_in:
        offline_policy = load_offline_policy(args.policy_in)
        print(f"Loaded offline policy for {offline_policy.n_agvs} AGVs from {args.policy_in}")
    elif args.solve_pbvi:
        if base.n_agvs > 3:
            print("Offline solver skipped: use --n-agvs <= 3 for offline PBVI solvers.")
        else:
            if args.solver == "momdp-pbvi":
                print(f"Solving true alpha-vector MOMDP-PBVI policy for {base.n_agvs} AGVs...")
                offline_policy = solve_momdp_alpha_pbvi(base)
            else:
                print(f"Solving finite belief-grid reference policy for {base.n_agvs} AGVs...")
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
