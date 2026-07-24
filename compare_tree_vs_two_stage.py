"""
compare_tree_vs_two_stage.py — Multistage (tree) vs Static / MNP / MRP, same scenarios
=========================================================================================

Solves the same underlying demand distribution four ways and compares expected
total cost — a "value of the multistage solution" (VMS) analysis, the
multistage analogue of VSS:

  1. ScenarioTreeModel.ScenarioTreeVehicleAllocationModel — genuine multistage
     recourse: the fleet/subcontracting/rebalancing plan for each season is
     chosen only after that season's demand is revealed, but before future
     seasons are known.

  2. Model.py's VehicleAllocationModel, three two-stage-family formulations,
     from least to most flexible:
       * build_model_static — no time variation at all: fleet + subcontracting
         are single numbers sized to cover the single worst-case peak demand
         across every week and every scenario at once (not a probability-
         weighted expectation like the other three).
       * build_model_MNP — time-varying planned + corrective subcontracting,
         but no rebalancing between hubs.
       * build_model_MRP — full two-stage recourse: fleet size (X) and the
         per-week allocation/planned-subcontracting plan (x, s) are fixed
         once, without seeing any scenario; only corrective subcontracting
         (s_corr) and rebalancing (y) adapt, and they adapt to the *entire*
         realized path at once rather than week by week.

To make the four objective values comparable, all are solved on the same
probability space: every root-to-leaf path of the scenario tree becomes one
full-horizon scenario o for the Model.py models, its probability is the
leaf's actual tree probability (NOT the uniform 1/|O| Model.py assumes by
default).

Implementation note: static/MNP/MRP all live on the SAME VehicleAllocationModel
instance, and each solve_*() call rebuilds self.model from scratch — so
whatever you need from a solve (objective, cost breakdown, fleet, resource
levels) must be extracted into a plain dict immediately after that solve,
before the next solve_*() call overwrites it. See _extract_two_stage_result().
"""

import csv
import os
import pickle
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ScenarioTreeModel import ScenarioTreeVehicleAllocationModel, build_toy_scenario_tree, ScenarioTree
from Model import VehicleAllocationModel
from real_hub_data import build_real_data_scenario_tree, build_distance_based_alpha, save_hub_data_used


# ---------------------------------------------------------------------------
# Flatten the tree into full-horizon scenarios for the Model.py models
# ---------------------------------------------------------------------------

def leaf_paths(tree):
    """List of (leaf_id, leaf_prob, [root, ..., leaf]) for every leaf of the tree."""
    paths = []

    def walk(node_id, ancestry):
        node = tree.nodes[node_id]
        ancestry = ancestry + [node_id]
        if not node.children:
            paths.append((node_id, node.prob, ancestry))
        else:
            for c in node.children:
                walk(c, ancestry)

    walk(tree.root_id, [])
    return paths


def build_full_horizon_scenarios(tree, N):
    """
    Concatenate every root-to-leaf path's per-season demand into one
    contiguous weekly series.

    Returns
        d_real     : dict {(i, t, o): demand}, t 0-indexed over the whole horizon
        leaf_prob  : dict {o: probability}
        total_weeks: int
    """
    paths = leaf_paths(tree)
    total_weeks = sum(tree.e[b] for b in sorted(tree.e))
    d_real = {}
    leaf_prob = {}
    for o, (leaf_id, prob, ancestry) in enumerate(paths):
        leaf_prob[o] = prob
        week_offset = 0
        for node_id in ancestry:
            node = tree.nodes[node_id]
            if node.stage == 0:
                continue
            block_len = tree.e[node.stage]
            for i in N:
                for t_local in range(1, block_len + 1):
                    d_real[i, week_offset + t_local - 1, o] = node.demand[i, t_local]
            week_offset += block_len
    return d_real, leaf_prob, total_weeks


def build_d_pred(d_real, leaf_prob, N, total_weeks):
    """Probability-weighted mean demand — the forecast the two-stage models'
    scenario-blind first stage (x, s) is sized against."""
    d_pred = {}
    for i in N:
        for t in range(total_weeks):
            d_pred[i, t] = sum(leaf_prob[o] * d_real[i, t, o] for o in leaf_prob)
    return d_pred


def save_flat_scenarios(d_real, leaf_prob, N, total_weeks, output_path, label=""):
    """
    Write the flattened full-horizon scenario set (the same d_real/leaf_prob
    every two-stage model is built from — see build_full_horizon_scenarios)
    to a CSV, one row per (scenario, hub, week): scenario, probability, hub,
    week, demand.

    label : optional prefix (e.g. an instance tag) put in front of the
        confirmation print, so concurrent runs' output stays distinguishable.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "probability", "hub", "week", "demand"])
        for o in sorted(leaf_prob):
            prob = leaf_prob[o]
            for i in N:
                for t in range(total_weeks):
                    writer.writerow([o, prob, i, t, d_real[i, t, o]])
    tag = f"[{label}] " if label else ""
    print(f"{tag}Flat scenario data saved to {output_path}")


# ---------------------------------------------------------------------------
# Cost/capacity parameters — single source of truth for ALL models
# ---------------------------------------------------------------------------
#
# The tree model and the Model.py models must price vehicles identically for
# the comparison to mean anything, so both builders below pull from this one
# function instead of hardcoding their own copies. Pass `overrides` (e.g. from
# an INSTANCES dict's "costs" key) to change q/beta/alpha/gamma/gamma_corr/
# theta/S/g for a given run; each key you supply replaces that parameter's
# dict wholesale (not merged per vehicle type).

def build_cost_params(N, K, overrides=None):
    q = {0: 720, 1: 2000, 2: 2000}                       # cargo bike, e-van, d-van
    beta = {0: 10538, 1: 32000, 2: 30000}
    alpha = {(i, j, 0): 10 for i in N for j in N}
    alpha.update({(i, j, 1): 50 for i in N for j in N})
    alpha.update({(i, j, 2): 50 for i in N for j in N})
    gamma = {0: 300, 1: 800, 2: 732}
    gamma_corr = {k: 1.5 * v for k, v in gamma.items()}
    theta = {i: 0.3 for i in N}
    S = {0: 100, 1: 100, 2: 100}
    g = {0: 1, 1: 1, 2: 0}                              # 1 = green vehicle type

    params = dict(q=q, beta=beta, alpha=alpha, gamma=gamma, gamma_corr=gamma_corr,
                   theta=theta, S=S, g=g)
    if overrides:
        params.update(overrides)
    return params


def _apply_cost_params(m, params):
    m.q = params["q"]
    m.beta = params["beta"]
    m.alpha = params["alpha"]
    m.gamma = params["gamma"]
    m.gamma_corr = params["gamma_corr"]
    m.theta = params["theta"]
    m.S = params["S"]
    m.g = params["g"]
    return m


def save_cost_params(params, output_dir):
    """Writes the resolved cost params (exactly what build_cost_params
    returned, including any cost_overrides -- e.g. real_hub_data's
    distance-based alpha when demand_source="real") to two CSVs:
      - cost_params.csv: the scalar-keyed params (q, beta, gamma,
        gamma_corr, theta, S, g) -- columns parameter, key, value.
      - cost_params_alpha.csv: alpha specifically (relational i,j,k) --
        columns hub_from, hub_to, type, alpha_eur.
    """
    os.makedirs(output_dir, exist_ok=True)
    scalar_path = os.path.join(output_dir, "cost_params.csv")
    with open(scalar_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "key", "value"])
        for name in ("q", "beta", "gamma", "gamma_corr", "theta", "S", "g"):
            for key, value in params[name].items():
                writer.writerow([name, key, value])
    print(f"Cost params saved to {scalar_path}")

    alpha_path = os.path.join(output_dir, "cost_params_alpha.csv")
    with open(alpha_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hub_from", "hub_to", "type", "alpha_eur"])
        for (i, j, k), value in params["alpha"].items():
            writer.writerow([i, j, k, value])
    print(f"Alpha (transfer cost) params saved to {alpha_path}")


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_tree_model(N, K, tree, seed=42, cost_overrides=None):
    m = ScenarioTreeVehicleAllocationModel(N, K, tree, seed=seed)
    params = build_cost_params(N, K, cost_overrides)
    _apply_cost_params(m, params)
    m.K_green = [k for k in m.K if params["g"].get(k, 0) == 1]
    return m


def build_flat_tree(tree, N):
    """Re-express an existing multistage tree's O scenarios as O
    independent, non-branching chains through the SAME season structure
    (tree.e unchanged) -- i.e. the identical scenario set static/MNP/MRP
    already solve on, just re-packaged so build_model(s_first_stage=True)
    naturally recovers MRP: s[i,k,b] ends up shared across every chain at
    season b (first-stage), while x/s_tilde/y stay fully independent per
    chain (second-stage, no anticipativity restriction beyond the root).

    NOT a depth-1 collapse (one leaf per scenario spanning the whole
    horizon as a single "season") -- that would force MRP's s[i,k,b] down
    to one shared value for the entire horizon instead of B independently
    -chosen seasonal values, a strictly more restrictive formulation than
    real MRP, not an equivalent one.
    """
    d_real, leaf_prob, total_weeks = build_full_horizon_scenarios(tree, N)
    B = sorted(tree.e)
    flat = ScenarioTree(dict(tree.e))
    flat.add_root()
    for o in sorted(leaf_prob):
        parent_id = "root"
        week_offset = 0
        for b in B:
            L = tree.e[b]
            demand = {(i, t_local): d_real[i, week_offset + t_local - 1, o]
                      for i in N for t_local in range(1, L + 1)}
            node_id = f"{o}_{b}"
            flat.add_child(node_id, parent_id, stage=b, prob=leaf_prob[o], demand=demand)
            parent_id = node_id
            week_offset += L
    flat.validate(N)
    return flat


def build_mrp_tree_model(N, K, tree, seed=42, cost_overrides=None):
    """MRP re-expressed as a scenario tree (see build_flat_tree): same
    demand/leaf-probabilities and costs as the flat two-stage MRP, but
    built through the SAME constraint-generating code as the multistage
    tree model (ScenarioTreeVehicleAllocationModel.build_model with
    s_first_stage=True) instead of Model.py's separately-coded
    build_model_MRP. Kept alongside, not replacing, the original MRP -- for
    cross-validation. Solve with m.solve(params=..., s_first_stage=True)."""
    flat_tree = build_flat_tree(tree, N)
    m = ScenarioTreeVehicleAllocationModel(N, K, flat_tree, seed=seed)
    params = build_cost_params(N, K, cost_overrides)
    _apply_cost_params(m, params)
    m.K_green = [k for k in m.K if params["g"].get(k, 0) == 1]
    return m


def build_two_stage_model(N, K, tree, seed=42, cost_overrides=None):
    d_real, leaf_prob, total_weeks = build_full_horizon_scenarios(tree, N)
    d_pred = build_d_pred(d_real, leaf_prob, N, total_weeks)

    m = VehicleAllocationModel(N=len(N), K=len(K), T=total_weeks, O=len(leaf_prob), seed=seed)
    m.Ki = {i: m.K for i in m.N}
    _apply_cost_params(m, build_cost_params(N, K, cost_overrides))
    m.l = {(i, k): 1 for i in m.N for k in m.K}
    m.M1 = m.S

    m.d_pred = d_pred
    m.d_real = d_real

    # Use the tree's actual leaf probabilities instead of the default uniform 1/|O|
    m.p_omega = lambda o, _lp=leaf_prob: _lp[o]

    # Real season boundaries, derived from the tree (season b = tree stage b,
    # weeks [offset, offset + tree.e[b]) in the flattened global-week index —
    # same offset logic as build_full_horizon_scenarios). MRP/MNP's planned
    # subcontracting (s[i,k,b]) is committed once per season, not per week;
    # see build_model_MRP's docstring for the season_of_week fallback when
    # this isn't set (e.g. run_parallel.py's generate_data() path).
    m.B = sorted(tree.e)
    season_of_week = {}
    offset = 0
    for b in m.B:
        for t_local in range(tree.e[b]):
            season_of_week[offset + t_local] = b
        offset += tree.e[b]
    m.season_of_week = season_of_week

    return m, leaf_prob, total_weeks


# ---------------------------------------------------------------------------
# Season helper — MNP/MRP's planned subcontracting s[i,k,b] is one value per
# SEASON b (see build_model_MRP/build_model_MNP's docstrings), not per week.
# Mirrors the same optional m.B/m.season_of_week attributes (falling back to
# one season per week if unset) that Model.py itself falls back to, so this
# stays correct even for a model built outside build_two_stage_model.
# ---------------------------------------------------------------------------

def _season_weeks(m):
    """Returns (B, season_of_week, weeks_in_season)."""
    B = getattr(m, "B", None) or list(m.T)
    season_of_week = getattr(m, "season_of_week", None) or {t: t for t in m.T}
    weeks_in_season = {b: [] for b in B}
    for t in m.T:
        weeks_in_season[season_of_week[t]].append(t)
    return B, season_of_week, weeks_in_season


def _seasonal_planned_sub_cost(m):
    """gamma[k] * s[i,k,b] * (weeks in season b), summed over i, k, b."""
    B, _, weeks_in_season = _season_weeks(m)
    return sum(m.gamma[k] * m._get_val(f"s[{i},{k},{b}]") * len(weeks_in_season[b])
               for i in m.N for k in m.K for b in B)


def _seasonal_planned_sub_quantity(m):
    """Raw s[i,k,b] * (weeks in season b), summed over i, k, b (vehicle-weeks,
    no cost weighting) — total across all hubs/types."""
    B, _, weeks_in_season = _season_weeks(m)
    return sum(m._get_val(f"s[{i},{k},{b}]") * len(weeks_in_season[b])
               for i in m.N for k in m.K for b in B)


def _seasonal_planned_sub_quantity_by_type(m):
    """{k: raw s[i,k,b] * (weeks in season b), summed over i, b}."""
    B, _, weeks_in_season = _season_weeks(m)
    return {k: sum(m._get_val(f"s[{i},{k},{b}]") * len(weeks_in_season[b]) for i in m.N for b in B)
            for k in m.K}


# ---------------------------------------------------------------------------
# Cost breakdowns (purchase / planned sub / redeployment / corrective sub / rebalancing)
# ---------------------------------------------------------------------------
#
# All four functions below return the same 5-key shape so report() and the
# plotting functions can treat tree/static/MNP/MRP uniformly; components that
# don't exist in a given formulation (e.g. static has no corrective
# subcontracting or rebalancing at all) are simply fixed at 0.0.
#
# "Redeployment" vs "rebalancing" is a tree-only distinction (same alpha
# price, different meaning — MNP/MRP have no equivalent stage-boundary
# structure, so they always report redeployment=0.0). A node n's block runs
# weeks 1..L. The transfer y[n,...,L] (the LAST week) is the only one that
# feeds the CHILD's opening inventory (see the init constraint in
# ScenarioTreeModel.py) — it's attributed to the receiving/child stage as
# "redeployment". Transfers in weeks 1..L-1 never cross a stage boundary —
# those are "rebalancing". Consequently stage-1 nodes have zero redeployment
# (their opening inventory comes from Delta, not a transfer) while leaf nodes
# do have a redeployment cost (inherited from their parent's last week),
# even though a leaf's own last-week transfer is always 0 (nothing
# downstream ever references it, so the solver drives it to 0).

def tree_cost_breakdown(m):
    tree = m.tree
    nodes_ge1 = tree.nodes_with_stage_ge1()
    purchase = sum(m.beta[k] * m._get_val(f"Delta[{i},{k}]") for i in m.N for k in m.K)

    # MRP_tree (m.s_first_stage=True) has no staged-revelation story to tell
    # -- it's a two-stage model re-expressed through the tree's constraint
    # -generating code, not a genuine multistage plan. So every transfer,
    # including a season's last week (which would be "redeployment" for the
    # real multistage tree), is reported as "rebalancing", matching flat
    # MRP's own convention (mrp_cost_breakdown always reports redeployment=0).
    planned_sub = corrective_sub = rebalancing = redeployment = 0.0
    for n in nodes_ge1:
        node = tree.nodes[n]
        prob = node.prob
        L = tree.block_length(n)
        weeks = range(1, L + 1)
        weeks_within = weeks if m.s_first_stage else range(1, L)  # excludes the last week (that's redeployment)
        planned_sub += prob * len(weeks) * sum(m.gamma[k] * m._s_val(n, i, k)
                                                for i in m.N for k in m.K)
        corrective_sub += prob * sum(m.gamma_corr[k] * m._get_val(f"stilde[{n},{i},{k},{t}]")
                                      for i in m.N for k in m.K for t in weeks)
        rebalancing += prob * sum(m.alpha[i, j, k] * m._get_val(f"y[{n},{i},{j},{k},{t}]")
                                   for (i, j) in m.A for k in m.K for t in weeks_within)
        if node.children and not m.s_first_stage:  # leaves have nothing to redeploy into
            redeployment += prob * sum(m.alpha[i, j, k] * m._get_val(f"y[{n},{i},{j},{k},{L}]")
                                        for (i, j) in m.A for k in m.K)
    return {"purchase": purchase, "planned_subcontracting": planned_sub,
            "redeployment": redeployment, "corrective_subcontracting": corrective_sub,
            "rebalancing": rebalancing}


def static_cost_breakdown(m):
    """build_model_static's objective: beta*X + gamma*len(T)*s (single, time-
    and scenario-independent numbers — see the module docstring's caveat)."""
    purchase = sum(m.beta[k] * m._get_val(f"X[{k}]") for k in m.K)
    planned_sub = sum(m.gamma[k] * len(m.T) * m._get_val(f"s[{i},{k}]")
                       for i in m.N for k in m.K)
    return {"purchase": purchase, "planned_subcontracting": planned_sub, "redeployment": 0.0,
            "corrective_subcontracting": 0.0, "rebalancing": 0.0}


def mnp_cost_breakdown(m):
    """build_model_MNP's objective: beta*X + gamma*s (seasonal, scenario-blind)
    + expected gamma_corr*s_corr. No rebalancing in this formulation."""
    purchase = sum(m.beta[k] * m._get_val(f"X[{k}]") for k in m.K)
    planned_sub = _seasonal_planned_sub_cost(m)
    corrective_sub = sum(
        m.p_omega(o) * m.gamma_corr[k] * m._get_val(f"s_corr[{i},{k},{t},{o}]")
        for i in m.N for k in m.K for t in m.T for o in m.O
    )
    return {"purchase": purchase, "planned_subcontracting": planned_sub, "redeployment": 0.0,
            "corrective_subcontracting": corrective_sub, "rebalancing": 0.0}


def mrp_cost_breakdown(m):
    """build_model_MRP's objective: beta*X + gamma*s (seasonal, scenario-blind)
    + expected gamma_corr*s_corr + expected alpha*y (rebalancing). x is now
    genuine second-stage recourse (scenario-dependent), but purchase/planned
    subcontracting stay first-stage (no o)."""
    purchase = sum(m.beta[k] * m._get_val(f"X[{k}]") for k in m.K)
    planned_sub = _seasonal_planned_sub_cost(m)

    corrective_sub = rebalancing = 0.0
    for o in m.O:
        p = m.p_omega(o)
        corrective_sub += p * sum(m.gamma_corr[k] * m._get_val(f"s_corr[{i},{k},{t},{o}]")
                                   for i in m.N for k in m.K for t in m.T)
        rebalancing += p * sum(m.alpha[i, j, k] * m._get_val(f"y[{i},{j},{k},{t},{o}]")
                                for i in m.N for j in m.N for k in m.K for t in m.T)
    return {"purchase": purchase, "planned_subcontracting": planned_sub, "redeployment": 0.0,
            "corrective_subcontracting": corrective_sub, "rebalancing": rebalancing}


# ---------------------------------------------------------------------------
# Per-scenario cost breakdowns (realized, NOT probability-weighted)
# ---------------------------------------------------------------------------
#
# The *_cost_breakdown() functions above give the expected (probability-
# weighted) cost — useful for comparing overall plans, but it hides how much
# the recourse cost actually varies from one demand realization to another.
# These functions instead return {o: {component: value}} for every scenario
# o, using the REALIZED cost under that one specific path (no p_omega/prob
# weighting). Scenario o is indexed the same way in all four — the same
# enumeration order as leaf_paths()/build_full_horizon_scenarios() — so
# scenario o always means the same underlying demand realization across
# tree/static/MNP/MRP.

def tree_scenario_cost_breakdown(m):
    """Realized cost along each root-to-leaf path (sum of that path's node
    contributions, not weighted by leaf probability)."""
    tree = m.tree
    purchase = sum(m.beta[k] * m._get_val(f"Delta[{i},{k}]") for i in m.N for k in m.K)
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        planned_sub = corrective_sub = rebalancing = redeployment = 0.0
        for idx, n in enumerate(ancestry):
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            L = tree.block_length(n)
            weeks = range(1, L + 1)
            weeks_within = weeks if m.s_first_stage else range(1, L)  # excludes the last week (that's redeployment)
            planned_sub += len(weeks) * sum(m.gamma[k] * m._s_val(n, i, k)
                                             for i in m.N for k in m.K)
            corrective_sub += sum(m.gamma_corr[k] * m._get_val(f"stilde[{n},{i},{k},{t}]")
                                   for i in m.N for k in m.K for t in weeks)
            rebalancing += sum(m.alpha[i, j, k] * m._get_val(f"y[{n},{i},{j},{k},{t}]")
                                for (i, j) in m.A for k in m.K for t in weeks_within)
            # Redeployment INTO n: the parent's last-week transfer (previous
            # node in the ancestry), attributed to n (the receiving stage).
            # Stage-1 nodes' parent is the root, which has no y variables.
            # Skipped entirely for MRP_tree -- see tree_cost_breakdown's note.
            parent_id = ancestry[idx - 1]
            if tree.nodes[parent_id].stage >= 1 and not m.s_first_stage:
                parent_L = tree.block_length(parent_id)
                redeployment += sum(
                    m.alpha[i, j, k] * m._get_val(f"y[{parent_id},{i},{j},{k},{parent_L}]")
                    for (i, j) in m.A for k in m.K
                )
        result[o] = {"purchase": purchase, "planned_subcontracting": planned_sub,
                      "redeployment": redeployment, "corrective_subcontracting": corrective_sub,
                      "rebalancing": rebalancing}
    return result


def static_scenario_cost_breakdown(m):
    """Static has no scenario dependence at all — same values for every o."""
    base = static_cost_breakdown(m)
    return {o: dict(base) for o in m.O}


def mnp_scenario_cost_breakdown(m):
    purchase = sum(m.beta[k] * m._get_val(f"X[{k}]") for k in m.K)
    planned_sub = _seasonal_planned_sub_cost(m)
    result = {}
    for o in m.O:
        corrective_sub = sum(m.gamma_corr[k] * m._get_val(f"s_corr[{i},{k},{t},{o}]")
                              for i in m.N for k in m.K for t in m.T)
        result[o] = {"purchase": purchase, "planned_subcontracting": planned_sub, "redeployment": 0.0,
                      "corrective_subcontracting": corrective_sub, "rebalancing": 0.0}
    return result


def mrp_scenario_cost_breakdown(m):
    purchase = sum(m.beta[k] * m._get_val(f"X[{k}]") for k in m.K)
    planned_sub = _seasonal_planned_sub_cost(m)
    result = {}
    for o in m.O:
        corrective_sub = sum(m.gamma_corr[k] * m._get_val(f"s_corr[{i},{k},{t},{o}]")
                              for i in m.N for k in m.K for t in m.T)
        rebalancing = sum(m.alpha[i, j, k] * m._get_val(f"y[{i},{j},{k},{t},{o}]")
                           for i in m.N for j in m.N for k in m.K for t in m.T)
        result[o] = {"purchase": purchase, "planned_subcontracting": planned_sub, "redeployment": 0.0,
                      "corrective_subcontracting": corrective_sub, "rebalancing": rebalancing}
    return result


# ---------------------------------------------------------------------------
# Per-scenario subcontracting QUANTITIES (vehicle units, not dollars)
# ---------------------------------------------------------------------------
#
# Same realized/per-scenario idea as *_scenario_cost_breakdown(), but summing
# the raw s/stilde/s_corr variable values directly (no gamma/gamma_corr price
# multiplier), aggregated across hubs and vehicle types into one number per
# scenario. Returns {o: {"planned": qty, "corrective": qty}}.

def tree_scenario_subcontracting_quantities(m):
    tree = m.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        planned = corrective = 0.0
        for n in ancestry:
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            L = tree.block_length(n)
            weeks = range(1, L + 1)
            planned += L * sum(m._s_val(n, i, k) for i in m.N for k in m.K)
            corrective += sum(m._get_val(f"stilde[{n},{i},{k},{t}]")
                               for i in m.N for k in m.K for t in weeks)
        result[o] = {"planned": planned, "corrective": corrective}
    return result


def static_scenario_subcontracting_quantities(m):
    """Static has no scenario dependence and no corrective subcontracting at all."""
    planned = sum(m._get_val(f"s[{i},{k}]") for i in m.N for k in m.K)
    return {o: {"planned": planned, "corrective": 0.0} for o in m.O}


def mnp_scenario_subcontracting_quantities(m):
    planned = _seasonal_planned_sub_quantity(m)
    result = {}
    for o in m.O:
        corrective = sum(m._get_val(f"s_corr[{i},{k},{t},{o}]") for i in m.N for k in m.K for t in m.T)
        result[o] = {"planned": planned, "corrective": corrective}
    return result


def mrp_scenario_subcontracting_quantities(m):
    planned = _seasonal_planned_sub_quantity(m)
    result = {}
    for o in m.O:
        corrective = sum(m._get_val(f"s_corr[{i},{k},{t},{o}]") for i in m.N for k in m.K for t in m.T)
        result[o] = {"planned": planned, "corrective": corrective}
    return result


# ---------------------------------------------------------------------------
# Per-scenario, PER VEHICLE TYPE quantities: purchase, planned sub, corrective sub
# ---------------------------------------------------------------------------
#
# Same idea as the aggregated *_scenario_subcontracting_quantities() above,
# but broken out by vehicle type k instead of summed across types, and
# including the purchase quantity (fleet bought, per type — a first-stage
# decision, so constant across every scenario o for a given model). Returns
# {(o, k): {"purchase": qty, "planned": qty, "corrective": qty}}.

def tree_scenario_quantities_by_type(m):
    tree = m.tree
    purchase = {k: sum(m._get_val(f"Delta[{i},{k}]") for i in m.N) for k in m.K}
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        planned = {k: 0.0 for k in m.K}
        corrective = {k: 0.0 for k in m.K}
        for n in ancestry:
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            L = tree.block_length(n)
            for k in m.K:
                planned[k] += L * sum(m._s_val(n, i, k) for i in m.N)
                corrective[k] += sum(m._get_val(f"stilde[{n},{i},{k},{t}]")
                                      for i in m.N for t in range(1, L + 1))
        for k in m.K:
            result[o, k] = {"purchase": purchase[k], "planned": planned[k], "corrective": corrective[k]}
    return result


def static_scenario_quantities_by_type(m):
    """Static has no scenario dependence and no corrective subcontracting at all."""
    purchase = {k: m._get_val(f"X[{k}]") for k in m.K}
    planned = {k: sum(m._get_val(f"s[{i},{k}]") for i in m.N) for k in m.K}
    return {(o, k): {"purchase": purchase[k], "planned": planned[k], "corrective": 0.0}
            for o in m.O for k in m.K}


def mnp_scenario_quantities_by_type(m):
    purchase = {k: m._get_val(f"X[{k}]") for k in m.K}
    planned = _seasonal_planned_sub_quantity_by_type(m)
    result = {}
    for o in m.O:
        for k in m.K:
            corrective = sum(m._get_val(f"s_corr[{i},{k},{t},{o}]") for i in m.N for t in m.T)
            result[o, k] = {"purchase": purchase[k], "planned": planned[k], "corrective": corrective}
    return result


def mrp_scenario_quantities_by_type(m):
    purchase = {k: m._get_val(f"X[{k}]") for k in m.K}
    planned = _seasonal_planned_sub_quantity_by_type(m)
    result = {}
    for o in m.O:
        for k in m.K:
            corrective = sum(m._get_val(f"s_corr[{i},{k},{t},{o}]") for i in m.N for t in m.T)
            result[o, k] = {"purchase": purchase[k], "planned": planned[k], "corrective": corrective}
    return result


def save_scenario_quantities_by_type(results, K, output_path, label=""):
    """
    Write per-scenario, per-vehicle-type purchase/planned/corrective
    quantities for all 4 models to a CSV: model, scenario, type, purchase,
    planned, corrective.
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scenario", "type", "purchase", "planned", "corrective"])
        for m in MODEL_ORDER:
            qbt = results[m]["scenario_quantities_by_type"]
            n_scenarios = len({o for o, k in qbt})
            for o in range(n_scenarios):
                for k in K:
                    q = qbt[o, k]
                    writer.writerow([MODEL_LABELS[m], o, k, q["purchase"], q["planned"], q["corrective"]])
    tag = f"[{label}] " if label else ""
    print(f"{tag}Per-scenario, per-type quantities saved to {output_path}")


# ---------------------------------------------------------------------------
# Per-scenario subcontracting BY PERIOD (planned + corrective, per week),
# used to find which week(s) had the heaviest subcontracting load.
# ---------------------------------------------------------------------------
#
# Returns {(o, t, k): qty}, t 0-indexed over range(total_weeks). "qty" is
# planned + corrective subcontracted units of type k active that week,
# summed across hubs (planned is constant within a season/tree-block; only
# corrective genuinely varies week to week).

def tree_scenario_subcontracting_by_period(m):
    tree = m.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        week_offset = 0
        for n in ancestry:
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            L = tree.block_length(n)
            for k in m.K:
                planned = sum(m._s_val(n, i, k) for i in m.N)
                for t_local in range(1, L + 1):
                    global_week = week_offset + t_local - 1
                    corrective = sum(m._get_val(f"stilde[{n},{i},{k},{t_local}]") for i in m.N)
                    result[o, global_week, k] = planned + corrective
            week_offset += L
    return result


def static_scenario_subcontracting_by_period(m, total_weeks):
    """Static's subcontracting has no time/scenario dependence -- the same
    planned level is maintained every week, with no corrective term."""
    planned = {k: sum(m._get_val(f"s[{i},{k}]") for i in m.N) for k in m.K}
    return {(o, t, k): planned[k] for o in m.O for t in range(total_weeks) for k in m.K}


def mnp_scenario_subcontracting_by_period(m, total_weeks):
    _, season_of_week, _ = _season_weeks(m)
    result = {}
    for k in m.K:
        planned_by_t = {t: sum(m._get_val(f"s[{i},{k},{season_of_week[t]}]") for i in m.N)
                         for t in range(total_weeks)}
        for o in m.O:
            for t in range(total_weeks):
                corrective = sum(m._get_val(f"s_corr[{i},{k},{t},{o}]") for i in m.N)
                result[o, t, k] = planned_by_t[t] + corrective
    return result


def mrp_scenario_subcontracting_by_period(m, total_weeks):
    """MRP's s/s_corr have the identical shape/index convention as MNP's --
    only x (fleet position, not queried here) gains the extra rebalancing
    lever -- so the same formula applies unchanged."""
    return mnp_scenario_subcontracting_by_period(m, total_weeks)


def subcontracting_period_extremes(by_period, K, total_weeks):
    """Given {(o, t, k): qty} (see *_scenario_subcontracting_by_period above),
    returns, per scenario o:
      - total_by_type: {k: total qty of type k subcontracted across the
        WHOLE horizon (cumulative, summed over every week)}
      - total_all_types: total_by_type summed over k (cumulative, whole
        horizon, every type)
      - peak_week_total: the single highest WEEK's total qty (summed across
        all types that week only -- NOT the same as total_all_types)
      - max_period_overall: 1-indexed week(s) tied for peak_week_total (None
        if every week has the same total -- nothing to single out)
      - max_period_by_type: {k: 1-indexed week(s) tied for type k's own
        highest single week (None if constant across every week)}
    """
    n_scenarios = len({o for o, t, k in by_period})
    result = {}
    for o in range(n_scenarios):
        total_by_type = {k: sum(by_period[o, t, k] for t in range(total_weeks)) for k in K}
        total_by_period = [sum(by_period[o, t, k] for k in K) for t in range(total_weeks)]
        peak_week_total = max(total_by_period)
        if peak_week_total == min(total_by_period):
            max_period_overall = None
        else:
            max_period_overall = [t + 1 for t, v in enumerate(total_by_period) if v == peak_week_total]
        max_period_by_type = {}
        for k in K:
            vals = [by_period[o, t, k] for t in range(total_weeks)]
            mx = max(vals)
            max_period_by_type[k] = None if mx == min(vals) else [t + 1 for t, v in enumerate(vals) if v == mx]
        result[o] = {
            "total_by_type": total_by_type,
            "total_all_types": sum(total_by_type.values()),
            "peak_week_total": peak_week_total,
            "max_period_overall": max_period_overall,
            "max_period_by_type": max_period_by_type,
        }
    return result


def _fmt_periods(periods):
    return "constant (n/a)" if periods is None else ";".join(str(p) for p in periods)


def save_subcontracting_extremes_table(results, K, output_path, label=""):
    """Per scenario, per model, per vehicle type: total subcontracted
    (planned + corrective) vehicles across the whole horizon, the week(s)
    with that type's own highest single-week subcontracted quantity, and
    the week(s) with the highest single-week TOTAL subcontracted quantity
    across all types. 'constant (n/a)' in a max-period column means the
    quantity never changed across the horizon (e.g. Static, or a scenario
    with zero subcontracting throughout). total_all_types/peak_week_total
    are repeated on every type row of a given (model, scenario) for
    convenience -- they don't vary by type."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scenario", "type", "subcontracted_qty_type",
                          "max_period_this_type", "subcontracted_qty_all_types_total",
                          "peak_week_total_qty", "max_period_overall"])
        for model_key in MODEL_ORDER:
            extremes = results[model_key]["scenario_subcontracting_extremes"]
            for o in sorted(extremes):
                e = extremes[o]
                for k in K:
                    writer.writerow([
                        MODEL_LABELS[model_key], o, k,
                        e["total_by_type"][k], _fmt_periods(e["max_period_by_type"][k]),
                        e["total_all_types"], e["peak_week_total"],
                        _fmt_periods(e["max_period_overall"]),
                    ])
    tag = f"[{label}] " if label else ""
    print(f"{tag}Subcontracting period-extremes table saved to {output_path}")


# ---------------------------------------------------------------------------
# Per-scenario rebalancing / redeployment MOVEMENT quantities (raw vehicle
# units, not cost-weighted) -- outbound counts only: y[i,j,...] already
# represents a move OUT of hub i, so summing over all (i, j, k, t) counts
# each unit-move exactly once.
# ---------------------------------------------------------------------------

def tree_scenario_rebalancing_quantities(m):
    """{o: {"rebalancing": qty, "redeployment": qty}}, mirroring the
    rebalancing/redeployment split in tree_scenario_cost_breakdown but
    without the alpha price multiplier. For MRP_tree (m.s_first_stage=True)
    redeployment is always 0 and everything counts as rebalancing -- see
    tree_cost_breakdown's note."""
    tree = m.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        rebalancing = redeployment = 0.0
        for idx, n in enumerate(ancestry):
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            L = tree.block_length(n)
            weeks_within = range(1, L + 1) if m.s_first_stage else range(1, L)
            rebalancing += sum(m._get_val(f"y[{n},{i},{j},{k},{t}]")
                                for (i, j) in m.A for k in m.K for t in weeks_within)
            parent_id = ancestry[idx - 1]
            if tree.nodes[parent_id].stage >= 1 and not m.s_first_stage:
                parent_L = tree.block_length(parent_id)
                redeployment += sum(
                    m._get_val(f"y[{parent_id},{i},{j},{k},{parent_L}]")
                    for (i, j) in m.A for k in m.K
                )
        result[o] = {"rebalancing": rebalancing, "redeployment": redeployment}
    return result


def static_scenario_rebalancing_quantities(m):
    """Static has no rebalancing lever at all."""
    return {o: {"rebalancing": 0.0, "redeployment": 0.0} for o in m.O}


def mnp_scenario_rebalancing_quantities(m):
    """MNP has no rebalancing lever at all."""
    return {o: {"rebalancing": 0.0, "redeployment": 0.0} for o in m.O}


def mrp_scenario_rebalancing_quantities(m):
    """MRP has no stage-boundary concept, so all movement is "rebalancing";
    redeployment is always 0 (matching mrp_scenario_cost_breakdown)."""
    result = {}
    for o in m.O:
        rebalancing = sum(m._get_val(f"y[{i},{j},{k},{t},{o}]")
                           for i in m.N for j in m.N for k in m.K for t in m.T)
        result[o] = {"rebalancing": rebalancing, "redeployment": 0.0}
    return result


def save_rebalancing_movement_table(results, output_path, label=""):
    """Per scenario, per model: total outbound rebalancing and redeployment
    vehicle-unit movements (raw counts, not cost-weighted). Static/MNP
    always report 0 for both -- neither model has a rebalancing lever."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "scenario", "rebalancing_qty", "redeployment_qty", "total_movement_qty"])
        for model_key in MODEL_ORDER:
            qty = results[model_key]["scenario_rebalancing_quantities"]
            for o in sorted(qty):
                r, d = qty[o]["rebalancing"], qty[o]["redeployment"]
                writer.writerow([MODEL_LABELS[model_key], o, r, d, r + d])
    tag = f"[{label}] " if label else ""
    print(f"{tag}Rebalancing/redeployment movement table saved to {output_path}")


# ---------------------------------------------------------------------------
# EXPECTED (probability-weighted) breakdowns BY HUB AND SEASON -- coarser
# than the per-week/per-scenario tables above: one number per (hub, season),
# averaged over every scenario, for two operationally-focused metrics:
#   - outbound vehicles: total rebalancing + redeployment vehicle-units
#     leaving hub i during season b (0 for Static/MNP -- neither has a
#     rebalancing lever)
#   - corrective subcontracting: total corrective vehicle-units used at hub
#     i during season b (0 for Static -- no second-stage recourse at all)
# Mirrors *_cost_breakdown()'s probability-weighting convention (node.prob
# for tree, m.p_omega(o) for the two-stage models) rather than the realized/
# per-scenario convention used by the *_scenario_* functions above.
# ---------------------------------------------------------------------------

def tree_outbound_by_hub_season(m):
    """{(i, b): expected total outbound vehicle-units (rebalancing +
    redeployment together) from hub i during season b}."""
    tree = m.tree
    B = sorted(tree.e)
    result = {(i, b): 0.0 for i in m.N for b in B}
    for n in tree.nodes_with_stage_ge1():
        node = tree.nodes[n]
        b = node.stage
        L = tree.block_length(n)
        for i in m.N:
            total = sum(m._get_val(f"y[{n},{i},{j},{k},{t}]")
                        for j in m.N if (i, j) in m.A
                        for k in m.K for t in range(1, L + 1))
            result[i, b] += node.prob * total
    return result


def static_outbound_by_hub_season(m, total_weeks):
    """Static has no rebalancing lever at all."""
    B, _, _ = _season_weeks(m)
    return {(i, b): 0.0 for i in m.N for b in B}


def mnp_outbound_by_hub_season(m, total_weeks):
    """MNP has no rebalancing lever at all."""
    B, _, _ = _season_weeks(m)
    return {(i, b): 0.0 for i in m.N for b in B}


def mrp_outbound_by_hub_season(m, total_weeks):
    B, season_of_week, _ = _season_weeks(m)
    result = {(i, b): 0.0 for i in m.N for b in B}
    for i in m.N:
        for t in range(total_weeks):
            b = season_of_week[t]
            for o in m.O:
                result[i, b] += m.p_omega(o) * sum(
                    m._get_val(f"y[{i},{j},{k},{t},{o}]") for j in m.N for k in m.K)
    return result


def tree_corrective_by_hub_season(m):
    """{(i, b): expected corrective-subcontracted vehicle-units at hub i
    during season b}."""
    tree = m.tree
    B = sorted(tree.e)
    result = {(i, b): 0.0 for i in m.N for b in B}
    for n in tree.nodes_with_stage_ge1():
        node = tree.nodes[n]
        b = node.stage
        L = tree.block_length(n)
        for i in m.N:
            total = sum(m._get_val(f"stilde[{n},{i},{k},{t}]") for k in m.K for t in range(1, L + 1))
            result[i, b] += node.prob * total
    return result


def static_corrective_by_hub_season(m, total_weeks):
    """Static has no corrective subcontracting at all."""
    B, _, _ = _season_weeks(m)
    return {(i, b): 0.0 for i in m.N for b in B}


def mnp_corrective_by_hub_season(m, total_weeks):
    B, season_of_week, _ = _season_weeks(m)
    result = {(i, b): 0.0 for i in m.N for b in B}
    for i in m.N:
        for t in range(total_weeks):
            b = season_of_week[t]
            for o in m.O:
                result[i, b] += m.p_omega(o) * sum(
                    m._get_val(f"s_corr[{i},{k},{t},{o}]") for k in m.K)
    return result


def mrp_corrective_by_hub_season(m, total_weeks):
    """MRP's s_corr has the identical shape/index convention as MNP's."""
    return mnp_corrective_by_hub_season(m, total_weeks)


def save_hub_season_table(by_model, N, B, output_path, value_label="qty", label=""):
    """by_model: {model_key: {(i, b): qty}} (see *_outbound_by_hub_season /
    *_corrective_by_hub_season above). Writes model, hub, season, <value_label>."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "hub", "season", value_label])
        for model_key in MODEL_ORDER:
            d = by_model[model_key]
            for i in N:
                for b in B:
                    writer.writerow([MODEL_LABELS[model_key], i, b, d[i, b]])
    tag = f"[{label}] " if label else ""
    print(f"{tag}{value_label} by hub/season saved to {output_path}")


# ---------------------------------------------------------------------------
# PER-SCENARIO (realized, NOT probability-weighted) breakdowns BY HUB AND
# SEASON -- same two metrics as *_outbound_by_hub_season / *_corrective_by_
# hub_season above, but keeping every scenario o separate instead of
# collapsing to an expectation. Returns {(i, b, o): qty}.
# ---------------------------------------------------------------------------

def tree_scenario_outbound_by_hub_season(m):
    """{(i, b, o): total outbound vehicle-units (rebalancing + redeployment
    together) from hub i during season b, realized along scenario o's own
    root-to-leaf path (no probability weighting)."""
    tree = m.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        for n in ancestry:
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            b = node.stage
            L = tree.block_length(n)
            for i in m.N:
                total = sum(m._get_val(f"y[{n},{i},{j},{k},{t}]")
                            for j in m.N if (i, j) in m.A
                            for k in m.K for t in range(1, L + 1))
                result[i, b, o] = total
    return result


def static_scenario_outbound_by_hub_season(m, total_weeks):
    B, _, _ = _season_weeks(m)
    return {(i, b, o): 0.0 for i in m.N for b in B for o in m.O}


def mnp_scenario_outbound_by_hub_season(m, total_weeks):
    """MNP has no rebalancing lever at all."""
    B, _, _ = _season_weeks(m)
    return {(i, b, o): 0.0 for i in m.N for b in B for o in m.O}


def mrp_scenario_outbound_by_hub_season(m, total_weeks):
    B, season_of_week, _ = _season_weeks(m)
    result = {(i, b, o): 0.0 for i in m.N for b in B for o in m.O}
    for i in m.N:
        for t in range(total_weeks):
            b = season_of_week[t]
            for o in m.O:
                result[i, b, o] += sum(m._get_val(f"y[{i},{j},{k},{t},{o}]") for j in m.N for k in m.K)
    return result


def tree_scenario_corrective_by_hub_season(m):
    """{(i, b, o): corrective-subcontracted vehicle-units at hub i during
    season b, realized along scenario o's own root-to-leaf path."""
    tree = m.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        for n in ancestry:
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            b = node.stage
            L = tree.block_length(n)
            for i in m.N:
                total = sum(m._get_val(f"stilde[{n},{i},{k},{t}]") for k in m.K for t in range(1, L + 1))
                result[i, b, o] = total
    return result


def static_scenario_corrective_by_hub_season(m, total_weeks):
    B, _, _ = _season_weeks(m)
    return {(i, b, o): 0.0 for i in m.N for b in B for o in m.O}


def mnp_scenario_corrective_by_hub_season(m, total_weeks):
    B, season_of_week, _ = _season_weeks(m)
    result = {(i, b, o): 0.0 for i in m.N for b in B for o in m.O}
    for i in m.N:
        for t in range(total_weeks):
            b = season_of_week[t]
            for o in m.O:
                result[i, b, o] += sum(m._get_val(f"s_corr[{i},{k},{t},{o}]") for k in m.K)
    return result


def mrp_scenario_corrective_by_hub_season(m, total_weeks):
    """MRP's s_corr has the identical shape/index convention as MNP's."""
    return mnp_scenario_corrective_by_hub_season(m, total_weeks)


def save_hub_season_scenario_table(by_model, N, B, output_path, value_label="qty", label=""):
    """by_model: {model_key: {(i, b, o): qty}} (see *_scenario_outbound_by_hub_season /
    *_scenario_corrective_by_hub_season above). Writes model, hub, season, scenario, <value_label>."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "hub", "season", "scenario", value_label])
        for model_key in MODEL_ORDER:
            d = by_model[model_key]
            n_scenarios = len({o for i, b, o in d})
            for i in N:
                for b in B:
                    for o in range(n_scenarios):
                        writer.writerow([MODEL_LABELS[model_key], i, b, o, d[i, b, o]])
    tag = f"[{label}] " if label else ""
    print(f"{tag}{value_label} by hub/season/scenario saved to {output_path}")


def _parse_var_index(token):
    try:
        return int(token)
    except ValueError:
        return token


def save_all_variables(m, output_path, label=""):
    """
    Dump every decision variable's solved value for this model into a nested
    dict keyed by variable prefix then parsed index (e.g. "x[5,2,1,3]" ->
    variables["x"][(5, 2, 1, 3)]; single-index names collapse to a bare
    scalar key, e.g. "X[2]" -> variables["X"][2]), pickled to output_path.
    Mirrors how _get_val(f"x[{n},{i},{k},{t}]") already looks values up
    elsewhere in this file, so it's directly usable for further analysis
    without re-parsing name strings on reload.

    MUST be called immediately after solving: static/MNP/MRP share one
    mutable model instance, and each solve_*() call rebuilds self.model from
    scratch, discarding the previous formulation's variables entirely (same
    eager-extraction requirement as _extract_two_stage_result() -- see the
    module docstring).

    label : optional instance tag put in front of the confirmation print,
        combined with the model name derived from output_path's basename
        (e.g. ".../variables/tree.pkl" -> "tree"), so concurrent solves'
        output stays distinguishable in the terminal.
    """
    model_name = os.path.splitext(os.path.basename(output_path))[0]
    tag = f"[{label}:{model_name}] " if label else f"[{model_name}] "
    status = m.model.status
    if status not in (2, 9):  # GRB.OPTIMAL, GRB.SUBOPTIMAL/TIME_LIMIT-with-incumbent
        print(f"{tag}Skipping variable dump for {output_path}: not solved (status {status})")
        return
    variables = {}
    for var in m.model.getVars():
        name = var.VarName
        if "[" in name:
            prefix, rest = name.split("[", 1)
            idx = tuple(_parse_var_index(p) for p in rest.rstrip("]").split(","))
            if len(idx) == 1:
                idx = idx[0]
        else:
            prefix, idx = name, None
        variables.setdefault(prefix, {})[idx] = var.X
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(variables, f)
    print(f"{tag}All variable values saved to {output_path}")


# ---------------------------------------------------------------------------
# Gurobi solve log — status / objective / gap / solve time, per model
# ---------------------------------------------------------------------------

GRB_STATUS_NAMES = {
    1: "LOADED", 2: "OPTIMAL", 3: "INFEASIBLE", 4: "INF_OR_UNBD", 5: "UNBOUNDED",
    6: "CUTOFF", 7: "ITERATION_LIMIT", 8: "NODE_LIMIT", 9: "TIME_LIMIT",
    10: "SOLUTION_LIMIT", 11: "INTERRUPTED", 12: "NUMERIC", 13: "SUBOPTIMAL",
    14: "INPROGRESS", 15: "USER_OBJ_LIMIT",
}


def _params_with_log_file(params, log_path):
    """Returns a copy of `params` with Gurobi's LogFile parameter set to
    log_path -- Gurobi writes the full solver log there independently of
    OutputFlag (so it's captured even with console output suppressed).
    Also sets LogToConsole=0: whenever we're saving a log file, the raw
    solver output (B&B progress, MIP gap updates, ...) is redundant on the
    terminal and can't be reliably prefixed with an instance/model tag line
    by line -- especially once multiple solves run concurrently on
    different threads/processes and their console output interleaves.
    Copies rather than mutates, since the same params dict is normally
    reused across all 4 solves and each needs its own LogFile."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    merged = dict(params) if params else {}
    merged["LogFile"] = log_path
    merged["LogToConsole"] = 0
    return merged


def save_solve_log(exp_dir, results, solve_times, label=""):
    """
    Saves each model's Gurobi solve outcome (status, objective, MIP gap,
    solve time) both as one CSV per model (<exp_dir>/solve_log/<model>.csv,
    one row) and as a single combined <exp_dir>/solve_summary.csv (one row
    per model), so a run's outcome can be checked at a glance without
    unpickling result.pkl.
    """
    log_dir = os.path.join(exp_dir, "solve_log")
    os.makedirs(log_dir, exist_ok=True)
    fieldnames = ["label", "model", "status", "status_desc", "obj", "gap", "solve_time_sec"]

    rows = []
    for model_key in MODEL_ORDER:
        r = results[model_key]
        status = r.get("status")
        row = {
            "label": label,
            "model": model_key,
            "status": status,
            "status_desc": GRB_STATUS_NAMES.get(status, f"STATUS_{status}"),
            "obj": r.get("obj"),
            "gap": r.get("gap"),
            "solve_time_sec": solve_times.get(model_key),
        }
        rows.append(row)
        with open(os.path.join(log_dir, f"{model_key}.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(row)

    summary_path = os.path.join(exp_dir, "solve_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tag = f"[{label}] " if label else ""
    print(f"{tag}Solve log saved to {log_dir}/ and {summary_path}")


# ---------------------------------------------------------------------------
# Green-vehicle coverage check (theta constraint), per (hub, week, scenario)
# ---------------------------------------------------------------------------
#
# Each model has a hard constraint: green-only vehicle capacity delivered at
# (hub i, period t[, scenario o]) must be >= theta[i] * demand there. These
# functions replicate that EXACT formula (not a "cleaner" reinterpretation of
# it) so the reported ratio genuinely reflects what the solver enforced. All
# return {(i, t, o): {"coverage", "required", "ratio", "margin"}}; ratio is
# nan when required == 0 (no demand to cover), so violation checks use
# margin (= coverage - required, always defined) instead of ratio.

def tree_green_coverage_ratios(m):
    """Mirrors ScenarioTreeModel.py's green constraint exactly: coverage[k] =
    x[n,i,k,t] + s[n,i,k] + stilde[n,i,k,t], green requirement =
    sum(q[k]*coverage[k] for k in K_green) >= theta[i]*demand."""
    tree = m.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        week_offset = 0
        for n in ancestry:
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            L = tree.block_length(n)
            for t_local in range(1, L + 1):
                global_week = week_offset + t_local - 1
                for i in m.N:
                    coverage = 0.0
                    for k in m.K_green:
                        cov_k = (m._get_val(f"x[{n},{i},{k},{t_local}]") + m._s_val(n, i, k)
                                 + m._get_val(f"stilde[{n},{i},{k},{t_local}]"))
                        coverage += m.q[k] * cov_k
                    demand = node.demand[i, t_local]
                    required = round(m.theta[i] * demand)
                    ratio = coverage / required if required > 0 else float("nan")
                    result[i, global_week, o] = {"coverage": coverage, "required": required,
                                                  "ratio": ratio, "margin": coverage - required}
            week_offset += L
    return result


def static_green_coverage_ratios(m):
    """Static's coverage is a single constant per hub; re-evaluated here
    against every period's ACTUAL realized demand (not just the d_max the
    model was sized against) for a uniform per-scenario comparison."""
    coverage = {i: sum(m.g[k] * m.q[k] * m._get_val(f"x[{i},{k}]") for k in m.K)
                   + sum(m.g[k] * m.q[k] * m._get_val(f"s[{i},{k}]") for k in m.K)
                for i in m.N}
    result = {}
    for i in m.N:
        for t in m.T:
            for o in m.O:
                demand = m.d_real[i, t, o]
                required = round(m.theta[i] * demand)
                cov = coverage[i]
                ratio = cov / required if required > 0 else float("nan")
                result[i, t, o] = {"coverage": cov, "required": required,
                                    "ratio": ratio, "margin": cov - required}
    return result


def mnp_green_coverage_ratios(m):
    """Mirrors build_model_MNP's green constraint exactly: no rebalancing
    term, s is seasonal (s[i,k,season_of_week[t]])."""
    _, season_of_week, _ = _season_weeks(m)
    result = {}
    for i in m.N:
        for t in m.T:
            b = season_of_week[t]
            for o in m.O:
                coverage = sum(
                    m.g[k] * m.q[k] * (m._get_val(f"x[{i},{k}]") + m._get_val(f"s[{i},{k},{b}]")
                                        + m._get_val(f"s_corr[{i},{k},{t},{o}]"))
                    for k in m.K
                )
                demand = m.d_real[i, t, o]
                required = round(m.theta[i] * demand)
                ratio = coverage / required if required > 0 else float("nan")
                result[i, t, o] = {"coverage": coverage, "required": required,
                                    "ratio": ratio, "margin": coverage - required}
    return result


def mrp_green_coverage_ratios(m):
    """Mirrors build_model_MRP's green constraint exactly: no separate flow
    term — x[i,k,t,o] already carries that period's net rebalancing per type
    via the precedence constraint, so coverage is just x + s + s_corr,
    filtered to green k (same shape as the tree model's demand constraint).
    x is now genuine second-stage recourse; s is seasonal (first-stage)."""
    _, season_of_week, _ = _season_weeks(m)
    result = {}
    for i in m.N:
        for t in m.T:
            b = season_of_week[t]
            for o in m.O:
                coverage = sum(
                    m.g[k] * m.q[k] * (m._get_val(f"x[{i},{k},{t},{o}]") + m._get_val(f"s[{i},{k},{b}]")
                                        + m._get_val(f"s_corr[{i},{k},{t},{o}]"))
                    for k in m.K
                )
                demand = m.d_real[i, t, o]
                required = round(m.theta[i] * demand)
                ratio = coverage / required if required > 0 else float("nan")
                result[i, t, o] = {"coverage": coverage, "required": required,
                                    "ratio": ratio, "margin": coverage - required}
    return result


# ---------------------------------------------------------------------------
# Per-scenario demand coverage BY SOURCE (purchased incl. rebalancing /
# planned subcontracting / corrective subcontracting), aggregated across all
# hubs & weeks. Returns {o: {"demand", "purchased", "planned", "corrective"}}
# -- q-weighted capacity from each source, plus that scenario's total
# realized demand, so a caller can express each source as a fraction of
# demand (e.g. purchased/demand = "the portion of demand met by owned fleet").
# "purchased" already reflects rebalancing for tree/MRP, since x[...,t] (or
# x[...,t,o] for MRP) carries that period's net rebalancing per type via the
# flow-balance/precedence constraint (see build_model_MRP's docstring) --
# there's no separate rebalancing term to add on top.
# ---------------------------------------------------------------------------

def tree_scenario_demand_coverage(m):
    tree = m.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        demand = purchased = planned = corrective = 0.0
        for n in ancestry:
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            L = tree.block_length(n)
            for t_local in range(1, L + 1):
                for i in m.N:
                    demand += node.demand[i, t_local]
                    for k in m.K:
                        purchased += m.q[k] * m._get_val(f"x[{n},{i},{k},{t_local}]")
                        corrective += m.q[k] * m._get_val(f"stilde[{n},{i},{k},{t_local}]")
            planned += L * sum(m.q[k] * m._s_val(n, i, k) for i in m.N for k in m.K)
        result[o] = {"demand": demand, "purchased": purchased, "planned": planned, "corrective": corrective}
    return result


def static_scenario_demand_coverage(m):
    """Static's x/s have no time index -- the same standing capacity applies
    every week, so its per-week contribution is multiplied by len(T) to
    match demand summed over the same weeks."""
    purchased = sum(m.q[k] * m._get_val(f"x[{i},{k}]") for i in m.N for k in m.K) * len(m.T)
    planned = sum(m.q[k] * m._get_val(f"s[{i},{k}]") for i in m.N for k in m.K) * len(m.T)
    result = {}
    for o in m.O:
        demand = sum(m.d_real[i, t, o] for i in m.N for t in m.T)
        result[o] = {"demand": demand, "purchased": purchased, "planned": planned, "corrective": 0.0}
    return result


def mnp_scenario_demand_coverage(m):
    _, season_of_week, _ = _season_weeks(m)
    purchased = sum(m.q[k] * m._get_val(f"x[{i},{k}]") for i in m.N for k in m.K) * len(m.T)
    result = {}
    for o in m.O:
        demand = sum(m.d_real[i, t, o] for i in m.N for t in m.T)
        planned = sum(m.q[k] * m._get_val(f"s[{i},{k},{season_of_week[t]}]")
                      for i in m.N for k in m.K for t in m.T)
        corrective = sum(m.q[k] * m._get_val(f"s_corr[{i},{k},{t},{o}]")
                          for i in m.N for k in m.K for t in m.T)
        result[o] = {"demand": demand, "purchased": purchased, "planned": planned, "corrective": corrective}
    return result


def mrp_scenario_demand_coverage(m):
    _, season_of_week, _ = _season_weeks(m)
    result = {}
    for o in m.O:
        demand = sum(m.d_real[i, t, o] for i in m.N for t in m.T)
        purchased = sum(m.q[k] * m._get_val(f"x[{i},{k},{t},{o}]") for i in m.N for k in m.K for t in m.T)
        planned = sum(m.q[k] * m._get_val(f"s[{i},{k},{season_of_week[t]}]")
                      for i in m.N for k in m.K for t in m.T)
        corrective = sum(m.q[k] * m._get_val(f"s_corr[{i},{k},{t},{o}]")
                          for i in m.N for k in m.K for t in m.T)
        result[o] = {"demand": demand, "purchased": purchased, "planned": planned, "corrective": corrective}
    return result


# ---------------------------------------------------------------------------
# Result extraction — snapshot before the next solve_*() overwrites self.model
# ---------------------------------------------------------------------------

def _flat_resource(m, N, K):
    """x[i,k] has no time index (static/MNP) — one value per (hub, type)."""
    return {(i, k): m._get_val(f"x[{i},{k}]") for i in N for k in K}


def _mrp_resource(m, N, K, total_weeks):
    """Expected x[i,k,t] across scenarios. x is now genuine second-stage
    recourse (indexed by o — see build_model_MRP), reduced to an expectation
    here since callers (e.g. plot_compare_resource_over_time) expect one
    value per (hub, type, week), matching the tree model's own 'expected'
    resource-over-time curve."""
    return {(i, k, t): sum(m.p_omega(o) * m._get_val(f"x[{i},{k},{t},{o}]") for o in m.O)
            for i in N for k in K for t in range(total_weeks)}


def _extract_two_stage_result(m, N, K, total_weeks, cost_fn, resource_fn, scenario_cost_fn, scenario_qty_fn,
                               green_ratio_fn, qty_by_type_fn, demand_coverage_fn,
                               subcontracting_by_period_fn, rebalancing_qty_fn,
                               outbound_hub_season_fn, corrective_hub_season_fn,
                               scenario_outbound_hub_season_fn, scenario_corrective_hub_season_fn):
    status = m.model.status
    solved = status in (2, 9)  # GRB.OPTIMAL, GRB.SUBOPTIMAL/TIME_LIMIT-with-incumbent
    return {
        "status": status,
        "obj": m.model.ObjVal if solved else None,
        "gap": m.model.MIPGap if solved else None,
        "fleet": {k: m._get_val(f"X[{k}]") for k in K},
        "costs": cost_fn(m),
        "resource": resource_fn(m, N, K),
        "scenario_costs": scenario_cost_fn(m),
        "scenario_quantities": scenario_qty_fn(m),
        "green_ratios": green_ratio_fn(m),
        "scenario_quantities_by_type": qty_by_type_fn(m),
        "demand_coverage": demand_coverage_fn(m),
        "scenario_subcontracting_extremes": subcontracting_period_extremes(
            subcontracting_by_period_fn(m), K, total_weeks),
        "scenario_rebalancing_quantities": rebalancing_qty_fn(m),
        "outbound_by_hub_season": outbound_hub_season_fn(m, total_weeks),
        "corrective_by_hub_season": corrective_hub_season_fn(m, total_weeks),
        "scenario_outbound_by_hub_season": scenario_outbound_hub_season_fn(m, total_weeks),
        "scenario_corrective_by_hub_season": scenario_corrective_hub_season_fn(m, total_weeks),
    }


def _tree_result(tree_model, N, K, total_weeks):
    status = tree_model.model.status
    solved = status in (2, 9)
    return {
        "status": status,
        "obj": tree_model.model.ObjVal if solved else None,
        "gap": tree_model.model.MIPGap if solved else None,
        "fleet": {k: sum(tree_model._get_val(f"Delta[{i},{k}]") for i in N) for k in K},
        "costs": tree_cost_breakdown(tree_model),
        "scenario_costs": tree_scenario_cost_breakdown(tree_model),
        "scenario_quantities": tree_scenario_subcontracting_quantities(tree_model),
        "green_ratios": tree_green_coverage_ratios(tree_model),
        "scenario_quantities_by_type": tree_scenario_quantities_by_type(tree_model),
        "scenario_subcontracting_extremes": subcontracting_period_extremes(
            tree_scenario_subcontracting_by_period(tree_model), K, total_weeks),
        "scenario_rebalancing_quantities": tree_scenario_rebalancing_quantities(tree_model),
        "outbound_by_hub_season": tree_outbound_by_hub_season(tree_model),
        "corrective_by_hub_season": tree_corrective_by_hub_season(tree_model),
        "scenario_outbound_by_hub_season": tree_scenario_outbound_by_hub_season(tree_model),
        "scenario_corrective_by_hub_season": tree_scenario_corrective_by_hub_season(tree_model),
        "demand_coverage": tree_scenario_demand_coverage(tree_model),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

MODEL_LABELS = {"tree": "Multistage (tree)", "static": "Static", "mnp": "MNP", "mrp": "MRP"}
MODEL_ORDER = ["tree", "static", "mnp", "mrp"]

# Vehicle type names, matching build_cost_params' q/g ordering (see comment there).
K_LABELS = {0: "E-cargo bike", 1: "E-van", 2: "D-van"}

# Cost components that actually vary by scenario for each model (static is
# omitted entirely — none of its components vary). Used to keep per-scenario
# tables/plots focused on signal, not constant first-stage decisions repeated
# on every row (e.g. MNP/MRP's planned subcontracting is scenario-blind).
RELEVANT_SCENARIO_COMPONENTS = {
    "tree": ["planned_subcontracting", "redeployment", "corrective_subcontracting", "rebalancing"],
    "mnp": ["corrective_subcontracting"],
    "mrp": ["corrective_subcontracting", "rebalancing"],
}
SCENARIO_COMPONENT_ABBR = {
    "planned_subcontracting": "planned", "redeployment": "redeploy",
    "corrective_subcontracting": "corrective", "rebalancing": "rebalance",
}


def report(results, K):
    tree_obj = results["tree"]["obj"]

    print("\n" + "=" * 78)
    print("EXPECTED TOTAL COST")
    print("=" * 78)
    for key in MODEL_ORDER:
        obj = results[key]["obj"]
        print(f"  {MODEL_LABELS[key]:<20}: {obj:>15,.0f}" if obj is not None
              else f"  {MODEL_LABELS[key]:<20}: {'N/A':>15}")

    print()
    for key in ["static", "mnp", "mrp"]:
        obj = results[key]["obj"]
        if obj is None or tree_obj is None:
            continue
        vms = obj - tree_obj
        vms_pct = 100 * vms / tree_obj if tree_obj else float("nan")
        print(f"  Value of multistage vs {MODEL_LABELS[key]:<18}: {vms:>15,.0f}  ({vms_pct:+.2f}% of tree cost)")
        if vms < 0:
            print(f"    NOTE: {MODEL_LABELS[key]} came out cheaper — check MIPGap/TimeLimit; with equal")
            print("          gaps this would indicate the models aren't on a fair common ground.")

    print("\n" + "=" * 78)
    print("FLEET PURCHASED BY TYPE")
    print("=" * 78)
    header = "  " + f"{'Type':<6}" + "".join(f"{MODEL_LABELS[k]:>18}" for k in MODEL_ORDER)
    print(header)
    for k in K:
        row = "  " + f"{k:<6}" + "".join(f"{results[m]['fleet'][k]:>18,.0f}" for m in MODEL_ORDER)
        print(row)

    print("\n" + "=" * 78)
    print("COST BREAKDOWN (expected; Static is a worst-case hedge, not an expectation)")
    print("=" * 78)
    print("  " + f"{'Component':<26}" + "".join(f"{MODEL_LABELS[k]:>18}" for k in MODEL_ORDER))
    for label, ckey in [("Purchase", "purchase"),
                        ("Planned subcontracting", "planned_subcontracting"),
                        ("Redeployment", "redeployment"),
                        ("Corrective subcontracting", "corrective_subcontracting"),
                        ("Rebalancing", "rebalancing")]:
        row = "  " + f"{label:<26}" + "".join(f"{results[m]['costs'][ckey]:>18,.0f}" for m in MODEL_ORDER)
        print(row)
    total_row = "  " + f"{'TOTAL':<26}" + "".join(
        f"{sum(results[m]['costs'].values()):>18,.0f}" for m in MODEL_ORDER)
    print(total_row)


def report_scenario_costs(results):
    """
    Per-scenario cost breakdown — how much planned/corrective subcontracting
    and rebalancing actually cost under EACH demand realization (not the
    probability-weighted expectation report() prints). Printed as summary
    statistics across scenarios since the full per-scenario table can be long;
    see plot_compare_scenario_costs / cost_summary.txt for the full detail.
    """
    print("\n" + "=" * 78)
    print("PER-SCENARIO COST BREAKDOWN (realized cost under each demand path)")
    print("=" * 78)
    for label, ckey in [("Planned subcontracting", "planned_subcontracting"),
                        ("Redeployment", "redeployment"),
                        ("Corrective subcontracting", "corrective_subcontracting"),
                        ("Rebalancing", "rebalancing")]:
        print(f"\n  {label}:")
        print("    " + f"{'Model':<20}{'mean':>14}{'std':>14}{'min':>14}{'max':>14}")
        for m in MODEL_ORDER:
            vals = np.array([sc[ckey] for sc in results[m]["scenario_costs"].values()])
            print(f"    {MODEL_LABELS[m]:<20}{vals.mean():>14,.0f}{vals.std():>14,.0f}"
                  f"{vals.min():>14,.0f}{vals.max():>14,.0f}")


def report_green_constraint(results, tol=1e-6):
    """
    Verifies the green-vehicle coverage (theta) constraint holds at every
    (hub, week, scenario) for all 4 models, and reports how much slack there
    is. Since this is a hard MIP constraint, violations should be exactly
    zero for any feasible solve — margin < -tol (not ratio, which is
    undefined when required demand is 0) is used to detect them.
    """
    print("\n" + "=" * 78)
    print("GREEN VEHICLE COVERAGE CHECK (coverage vs theta * demand)")
    print("=" * 78)
    print("  " + f"{'Model':<20}{'violations':>12}{'min ratio':>12}{'mean ratio':>12}{'max ratio':>12}")
    for m in MODEL_ORDER:
        cells = results[m]["green_ratios"].values()
        violations = sum(1 for c in cells if c["margin"] < -tol)
        ratios = np.array([c["ratio"] for c in cells if not np.isnan(c["ratio"])])
        if len(ratios) == 0:
            print(f"  {MODEL_LABELS[m]:<20}{violations:>12}{'N/A':>12}{'N/A':>12}{'N/A':>12}")
            continue
        print(f"  {MODEL_LABELS[m]:<20}{violations:>12}{ratios.min():>12.3f}"
              f"{ratios.mean():>12.3f}{ratios.max():>12.3f}")
        if violations:
            print(f"    WARNING: {violations} (hub, week, scenario) combos violate the green requirement.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compare(seasons=(1, 2, 3, 4), weeks_per_season=13, branching=2,
            n_hubs=3, n_types=3, solver_params=None, seed=42, cost_overrides=None,
            hub_correlation=None, season_drift=None, sibling_drift_correlation=None,
            noise_frac=None, make_plots=False, plot_dir="compare_plots", mrp_variant="flat",
            demand_source="synthetic", data_dir="true_data",
            tree_solver_params=None, mrp_solver_params=None):
    """
    seed : int or None. An int (the default, 42) reproduces the exact same
        instance every call — same demand tree, same everything. Pass
        seed=None to get a fresh, genuinely random instance each call (drawn
        from OS entropy); the actual seed used is printed so you can pass it
        back in later to reproduce that specific run.

    tree_solver_params, mrp_solver_params : optional dicts merged ON TOP of
        solver_params for just that one model's solve (e.g. mrp_solver_params=
        {"MIPGap": 0.05} to give MRP a looser gap than the tree/static/MNP
        share, since MRP's flattened scenario set is typically a much harder
        MIP to close to a tight gap -- see the tree-vs-MRP solve-log
        discussion this was added for). Static/MNP always use solver_params
        unmodified.

    season_drift, sibling_drift_correlation, noise_frac : passed straight
        through to build_toy_scenario_tree — see its docstring in
        ScenarioTreeModel.py. Each defaults to None here, which leaves that
        function's own default in place.

    mrp_variant : "flat" (default) solves MRP the original way, via
        Model.py's build_model_MRP on the shared two-stage VehicleAllocationModel
        `m`. "tree" instead solves it via build_mrp_tree_model -- the same
        two-stage problem re-expressed through the tree's constraint
        -generating code (build_model(s_first_stage=True)) -- for
        cross-validation against the flat formulation. Both report/plot/save
        results the same way (the "mrp" slot in `results`, MODEL_ORDER, every
        CSV, and every plot -- plots_tree_vs_two_stage.py auto-detects which
        kind of model mrp_model is and dispatches to the matching
        extraction functions, see its _mrp_is_tree_shaped).

    demand_source : "synthetic" (default) builds the demand tree via
        ScenarioTreeModel.build_toy_scenario_tree (random base level per
        hub). "real" instead builds it via
        real_hub_data.build_real_data_scenario_tree -- real per-hub mean
        demand, real inter-hub correlation, and real per-hub weekly-noise
        scale, from data_dir's TRUE_negative_pairs_SOLID_*.csv, using the
        first n_hubs hubs listed there (capped at 21 -- that's all the data
        covers). The season_drift/sibling_drift_correlation mechanism is
        unchanged either way. hub_correlation/noise_frac are ignored (with
        a printed note if explicitly set) when demand_source="real", since
        real mode derives both from the data instead.
    """
    if mrp_variant not in ("flat", "tree"):
        raise ValueError(f"mrp_variant must be 'flat' or 'tree', got {mrp_variant!r}")
    if demand_source not in ("synthetic", "real"):
        raise ValueError(f"demand_source must be 'synthetic' or 'real', got {demand_source!r}")
    if seed is None:
        seed = random.SystemRandom().randrange(2**31)
        print(f"No seed given — using randomly generated seed={seed} "
              f"(pass seed={seed} to reproduce this exact instance later)")

    N = list(range(n_hubs))
    K = list(range(n_types))
    solver_params = solver_params or {"TimeLimit": 600, "MIPGap": 0.01, "OutputFlag": 1}

    real_hub_data_used = None  # set below when demand_source="real"; used later for saving
    if demand_source == "real":
        if hub_correlation is not None or noise_frac is not None:
            print("Note: hub_correlation/noise_frac are ignored with demand_source='real' "
                  "-- both are derived from the real data instead.")
        real_kwargs = dict(seasons=seasons, branching=branching, weeks_per_season=weeks_per_season, seed=seed)
        if season_drift is not None:
            real_kwargs["season_drift"] = season_drift
        if sibling_drift_correlation is not None:
            real_kwargs["sibling_drift_correlation"] = sibling_drift_correlation
        tree, base_demand, corr_matrix, cv_by_hub, hub_meta = build_real_data_scenario_tree(
            n_hubs, data_dir=data_dir, **real_kwargs)
        real_hub_data_used = (base_demand, corr_matrix, cv_by_hub, hub_meta)
        hub_list = ", ".join(f"{i}={hub_meta[i]['site']}" for i in hub_meta)
        print(f"Using real hub data: {hub_list}")

        distance_alpha = build_distance_based_alpha(hub_meta, N, K)
        if cost_overrides and "alpha" in cost_overrides:
            print("Note: cost_overrides['alpha'] is ignored with demand_source='real' "
                  "-- using real distance-based transfer costs instead.")
        cost_overrides = {**(cost_overrides or {}), "alpha": distance_alpha}
    else:
        tree_kwargs = dict(seasons=seasons, branching=branching, weeks_per_season=weeks_per_season,
                            hub_correlation=hub_correlation, seed=seed)
        if season_drift is not None:
            tree_kwargs["season_drift"] = season_drift
        if sibling_drift_correlation is not None:
            tree_kwargs["sibling_drift_correlation"] = sibling_drift_correlation
        if noise_frac is not None:
            tree_kwargs["noise_frac"] = noise_frac
        tree = build_toy_scenario_tree(N, **tree_kwargs)
    tree.validate(N)
    n_leaves = branching ** len(seasons)
    total_weeks = len(seasons) * weeks_per_season
    print(f"Scenario tree: {len(seasons)} stages x {weeks_per_season} weeks, "
          f"branching {branching} -> {n_leaves} full-horizon paths ({total_weeks} weeks total)")

    # Tree is often the single slowest solve, so it's built now and solved in
    # a background thread, overlapping with the static/MNP/MRP chain below.
    # tree_model is a fully separate Gurobi Model/Env from `m` (the two-stage
    # model) -- no shared mutable state to race on -- and Gurobi releases the
    # GIL during optimize(), so this is genuine parallelism, not just
    # concurrency.
    print("\n--- Solving multistage scenario-tree model (in background) and "
          "Static / MNP / MRP concurrently ---")
    tree_model = build_tree_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)

    def _solve_tree_model():
        t0 = time.time()
        tree_base_params = {**solver_params, **(tree_solver_params or {})}
        tree_params = (_params_with_log_file(tree_base_params, os.path.join(plot_dir, "gurobi_log", "tree.log"))
                       if make_plots else tree_base_params)
        tree_model.solve(params=tree_params)
        if make_plots:
            save_all_variables(tree_model, os.path.join(plot_dir, "variables", "tree.pkl"))
        return time.time() - t0

    tree_executor = ThreadPoolExecutor(max_workers=1)
    tree_future = tree_executor.submit(_solve_tree_model)

    m, leaf_prob, total_weeks = build_two_stage_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)

    print("Static...")
    t0 = time.time()
    static_params = (_params_with_log_file(solver_params, os.path.join(plot_dir, "gurobi_log", "static.log"))
                      if make_plots else solver_params)
    m.solve_static(params=static_params)
    t_static = time.time() - t0
    static_result = _extract_two_stage_result(
        m, N, K, total_weeks, static_cost_breakdown, _flat_resource, static_scenario_cost_breakdown,
        static_scenario_subcontracting_quantities, static_green_coverage_ratios,
        static_scenario_quantities_by_type, static_scenario_demand_coverage,
        lambda mm: static_scenario_subcontracting_by_period(mm, total_weeks),
        static_scenario_rebalancing_quantities,
        static_outbound_by_hub_season, static_corrective_by_hub_season,
        static_scenario_outbound_by_hub_season, static_scenario_corrective_by_hub_season)
    if make_plots:
        save_all_variables(m, os.path.join(plot_dir, "variables", "static.pkl"))

    print("MNP...")
    t0 = time.time()
    mnp_params = (_params_with_log_file(solver_params, os.path.join(plot_dir, "gurobi_log", "mnp.log"))
                  if make_plots else solver_params)
    m.solve_MNP(params=mnp_params)
    t_mnp = time.time() - t0
    mnp_result = _extract_two_stage_result(
        m, N, K, total_weeks, mnp_cost_breakdown, _flat_resource, mnp_scenario_cost_breakdown,
        mnp_scenario_subcontracting_quantities, mnp_green_coverage_ratios,
        mnp_scenario_quantities_by_type, mnp_scenario_demand_coverage,
        lambda mm: mnp_scenario_subcontracting_by_period(mm, total_weeks),
        mnp_scenario_rebalancing_quantities,
        mnp_outbound_by_hub_season, mnp_corrective_by_hub_season,
        mnp_scenario_outbound_by_hub_season, mnp_scenario_corrective_by_hub_season)
    if make_plots:
        save_all_variables(m, os.path.join(plot_dir, "variables", "mnp.pkl"))

    print(f"MRP ({mrp_variant})...")
    t0 = time.time()
    mrp_base_params = {**solver_params, **(mrp_solver_params or {})}
    mrp_params = (_params_with_log_file(mrp_base_params, os.path.join(plot_dir, "gurobi_log", "mrp.log"))
                  if make_plots else mrp_base_params)
    if mrp_variant == "tree":
        mrp_tree_model = build_mrp_tree_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)
        mrp_tree_model.solve(params=mrp_params, s_first_stage=True)
        t_mrp = time.time() - t0
        mrp_result = _tree_result(mrp_tree_model, N, K, total_weeks)
        if make_plots:
            save_all_variables(mrp_tree_model, os.path.join(plot_dir, "variables", "mrp.pkl"))
        mrp_model_for_plots = mrp_tree_model
    else:
        m.solve_MRP(params=mrp_params)
        t_mrp = time.time() - t0
        mrp_result = _extract_two_stage_result(
            m, N, K, total_weeks, mrp_cost_breakdown, lambda mm, NN, KK: _mrp_resource(mm, NN, KK, total_weeks),
            mrp_scenario_cost_breakdown, mrp_scenario_subcontracting_quantities, mrp_green_coverage_ratios,
            mrp_scenario_quantities_by_type, mrp_scenario_demand_coverage,
            lambda mm: mrp_scenario_subcontracting_by_period(mm, total_weeks),
            mrp_scenario_rebalancing_quantities,
            mrp_outbound_by_hub_season, mrp_corrective_by_hub_season,
            mrp_scenario_outbound_by_hub_season, mrp_scenario_corrective_by_hub_season
        )
        if make_plots:
            save_all_variables(m, os.path.join(plot_dir, "variables", "mrp.pkl"))
        # m is now left in its MRP-solved state — the rebalancing plots rely on
        # being able to query it live, right after this call.
        mrp_model_for_plots = m

    # Join the background tree solve (usually already done by now, since it
    # started before static/MNP/MRP and often takes comparably long).
    t_tree = tree_future.result()
    tree_executor.shutdown()

    results = {
        "tree": _tree_result(tree_model, N, K, total_weeks),
        "static": static_result,
        "mnp": mnp_result,
        "mrp": mrp_result,
    }
    solve_times = {"tree": t_tree, "static": t_static, "mnp": t_mnp, "mrp": t_mrp}

    report(results, K)
    report_scenario_costs(results)
    report_green_constraint(results)

    if make_plots:
        import matplotlib
        matplotlib.use("Agg")
        from plots_tree_vs_two_stage import plot_all_comparisons
        plot_all_comparisons(tree_model, mrp_model_for_plots, results, N, K, total_weeks, output_dir=plot_dir)

        d_real, _, _ = build_full_horizon_scenarios(tree, N)
        save_flat_scenarios(d_real, leaf_prob, N, total_weeks,
                             os.path.join(plot_dir, "flat_scenarios.csv"))
        save_scenario_quantities_by_type(results, K, os.path.join(plot_dir, "scenario_quantities.csv"))
        save_subcontracting_extremes_table(results, K, os.path.join(plot_dir, "subcontracting_extremes.csv"))
        save_rebalancing_movement_table(results, os.path.join(plot_dir, "rebalancing_movements.csv"))
        B = sorted(tree.e)
        save_hub_season_table({mk: results[mk]["outbound_by_hub_season"] for mk in MODEL_ORDER}, N, B,
                               os.path.join(plot_dir, "outbound_by_hub_season.csv"),
                               value_label="expected_outbound_qty")
        save_hub_season_table({mk: results[mk]["corrective_by_hub_season"] for mk in MODEL_ORDER}, N, B,
                               os.path.join(plot_dir, "corrective_by_hub_season.csv"),
                               value_label="expected_corrective_qty")
        save_hub_season_scenario_table(
            {mk: results[mk]["scenario_outbound_by_hub_season"] for mk in MODEL_ORDER}, N, B,
            os.path.join(plot_dir, "outbound_by_hub_season_scenario.csv"), value_label="outbound_qty")
        save_hub_season_scenario_table(
            {mk: results[mk]["scenario_corrective_by_hub_season"] for mk in MODEL_ORDER}, N, B,
            os.path.join(plot_dir, "corrective_by_hub_season_scenario.csv"), value_label="corrective_qty")
        save_solve_log(plot_dir, results, solve_times)

        final_cost_params = build_cost_params(N, K, cost_overrides)
        save_cost_params(final_cost_params, plot_dir)
        if real_hub_data_used is not None:
            base_demand, corr_matrix, cv_by_hub, hub_meta = real_hub_data_used
            save_hub_data_used(base_demand, corr_matrix, cv_by_hub, hub_meta, N, plot_dir)

    return tree_model, m, results



if __name__ == "__main__":

    # hub_correlation = [
    # [ 1.0, -0.8,  0.0],
    # [-0.8,  1.0,  0.0],
    # [ 0.0,  0.0,  1.0],
    # ]
    # hub_correlation= None
    seed = None
    season_drift = (0.20,0.25)
    sibling_drift_correlation = 1
    noise_frac = 0.05

    compare(seasons=(1, 2, 3, 4), weeks_per_season= 13, branching=2,
            n_hubs=2, n_types=3, 
            solver_params={"TimeLimit": 1200, "MIPGap": 0.01},
            make_plots=True, plot_dir="compare_plots", seed=seed, season_drift=season_drift,
            sibling_drift_correlation=sibling_drift_correlation,
            noise_frac=noise_frac, mrp_variant="tree", demand_source="real")
