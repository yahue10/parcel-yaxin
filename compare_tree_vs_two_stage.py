"""
compare_tree_vs_two_stage.py — Multistage (tree) vs two-stage (MRP), same scenarios
=====================================================================================

Solves the same underlying demand distribution two ways and compares expected
total cost — a "value of the multistage solution" (VMS) analysis, the
multistage analogue of VSS:

  1. ScenarioTreeModel.ScenarioTreeVehicleAllocationModel — genuine multistage
     recourse: the fleet/subcontracting/rebalancing plan for each season is
     chosen only after that season's demand is revealed, but before future
     seasons are known.

  2. Model.py's VehicleAllocationModel.build_model_MRP — two-stage recourse:
     fleet size (X) and the per-week allocation/planned-subcontracting plan
     (x, s) are fixed once, without seeing any scenario; only corrective
     subcontracting (s_corr) and rebalancing (y) adapt, and they adapt to the
     *entire* realized path at once rather than week by week.

To make the two objective values comparable, both models are solved on the
same probability space: every root-to-leaf path of the scenario tree becomes
one full-horizon scenario o for the two-stage model, its probability is the
leaf's actual tree probability (NOT the uniform 1/|O| the two-stage model
assumes by default), and cost coefficients are copied verbatim from
ScenarioTreeVehicleAllocationModel.generate_costs() (same hardcoded values as
Model.py's generate_data()).

Caveat: the two formulations are not perfectly nested. The tree model's root
decision (Delta[i,k]) commits fleet directly to hubs, while the two-stage
model separates an aggregate purchase X[k] from a time-varying (but
scenario-blind) hub allocation x[i,k,t] — an extra degree of freedom that
works in the two-stage model's favor. So any gap observed here is, if
anything, an underestimate of the true value of adapting stage by stage.
"""

import numpy as np

from ScenarioTreeModel import ScenarioTreeVehicleAllocationModel, build_toy_scenario_tree
from Model import VehicleAllocationModel


# ---------------------------------------------------------------------------
# Flatten the tree into full-horizon scenarios for the two-stage model
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
    """Probability-weighted mean demand — the forecast the two-stage model's
    scenario-blind first stage (x, s) is sized against."""
    d_pred = {}
    for i in N:
        for t in range(total_weeks):
            d_pred[i, t] = sum(leaf_prob[o] * d_real[i, t, o] for o in leaf_prob)
    return d_pred


# ---------------------------------------------------------------------------
# Model builders (shared cost parameters, same values in both files)
# ---------------------------------------------------------------------------

def build_tree_model(N, K, tree, seed=42):
    m = ScenarioTreeVehicleAllocationModel(N, K, tree, seed=seed)
    m.generate_costs()
    return m


def build_two_stage_model(N, K, tree, seed=42):
    d_real, leaf_prob, total_weeks = build_full_horizon_scenarios(tree, N)
    d_pred = build_d_pred(d_real, leaf_prob, N, total_weeks)

    m = VehicleAllocationModel(N=len(N), K=len(K), T=total_weeks, O=len(leaf_prob), seed=seed)

    # Same cost parameters as ScenarioTreeVehicleAllocationModel.generate_costs()
    m.Ki = {i: m.K for i in m.N}
    m.q = {0: 650, 1: 910, 2: 910}
    m.beta = {0: 44000, 1: 49000, 2: 51000}
    m.alpha = {(i, j, 0): 10 for i in m.N for j in m.N}
    m.alpha.update({(i, j, 1): 50 for i in m.N for j in m.N})
    m.alpha.update({(i, j, 2): 50 for i in m.N for j in m.N})
    m.gamma = {0: 3 * 650, 1: 3 * 910, 2: 3 * 910}
    m.gamma_corr = {0: 3.3 * 650, 1: 3.3 * 910, 2: 3.3 * 910}
    m.theta = {i: 0.3 for i in m.N}
    m.l = {(i, k): 1 for i in m.N for k in m.K}
    m.g = {0: 1, 1: 1, 2: 0}
    m.M1 = {0: 6170, 1: 6170, 2: 6170}
    m.S = m.M1

    m.d_pred = d_pred
    m.d_real = d_real

    # Use the tree's actual leaf probabilities instead of the default uniform 1/|O|
    m.p_omega = lambda o, _lp=leaf_prob: _lp[o]

    return m, leaf_prob, total_weeks


# ---------------------------------------------------------------------------
# Cost breakdowns (purchase / planned sub / corrective sub / rebalancing)
# ---------------------------------------------------------------------------

def tree_cost_breakdown(m):
    tree = m.tree
    nodes_ge1 = tree.nodes_with_stage_ge1()
    purchase = sum(m.beta[k] * m._get_val(f"Delta[{i},{k}]") for i in m.N for k in m.K)

    planned_sub = corrective_sub = rebalancing = 0.0
    for n in nodes_ge1:
        prob = tree.nodes[n].prob
        weeks = range(1, tree.block_length(n) + 1)
        planned_sub += prob * len(weeks) * sum(m.gamma[k] * m._get_val(f"s[{n},{i},{k}]")
                                                for i in m.N for k in m.K)
        corrective_sub += prob * sum(m.gamma_corr[k] * m._get_val(f"stilde[{n},{i},{k},{t}]")
                                      for i in m.N for k in m.K for t in weeks)
        rebalancing += prob * sum(m.alpha[i, j, k] * m._get_val(f"y[{n},{i},{j},{k},{t}]")
                                   for (i, j) in m.A for k in m.K for t in weeks)
    return {"purchase": purchase, "planned_subcontracting": planned_sub,
            "corrective_subcontracting": corrective_sub, "rebalancing": rebalancing}


def two_stage_cost_breakdown(m):
    purchase = sum(m.beta[k] * m._get_val(f"X[{k}]") for k in m.K)
    planned_sub = sum(m.gamma[k] * m._get_val(f"s[{i},{k},{t}]")
                       for i in m.N for k in m.K for t in m.T)

    corrective_sub = rebalancing = 0.0
    for o in m.O:
        p = m.p_omega(o)
        corrective_sub += p * sum(m.gamma_corr[k] * m._get_val(f"s_corr[{i},{k},{t},{o}]")
                                   for i in m.N for k in m.K for t in m.T)
        rebalancing += p * sum(m.alpha[i, j, k] * m._get_val(f"y[{i},{j},{k},{t},{o}]")
                                for i in m.N for j in m.N for k in m.K for t in m.T)
    return {"purchase": purchase, "planned_subcontracting": planned_sub,
            "corrective_subcontracting": corrective_sub, "rebalancing": rebalancing}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(tree_model, two_stage_model, N, K):
    tree_obj = tree_model.model.ObjVal
    two_stage_obj = two_stage_model.model.ObjVal
    vms = two_stage_obj - tree_obj
    vms_pct = 100 * vms / tree_obj if tree_obj else float("nan")

    print("\n" + "=" * 70)
    print("EXPECTED TOTAL COST")
    print("=" * 70)
    print(f"  Multistage (scenario tree) : {tree_obj:>15,.0f}")
    print(f"  Two-stage  (MRP)           : {two_stage_obj:>15,.0f}")
    print(f"  Value of multistage (VMS)  : {vms:>15,.0f}  ({vms_pct:+.2f}% of tree cost)")
    if vms < 0:
        print("  NOTE: two-stage came out cheaper — check MIPGap/TimeLimit; with equal")
        print("        gaps this would indicate the models aren't on a fair common ground.")

    print("\n" + "=" * 70)
    print("FLEET PURCHASED BY TYPE")
    print("=" * 70)
    print(f"  {'Type':<6}{'Tree: sum_i Delta[i,k]':>26}{'Two-stage: X[k]':>20}")
    for k in K:
        tree_fleet = sum(tree_model._get_val(f"Delta[{i},{k}]") for i in N)
        two_stage_fleet = two_stage_model._get_val(f"X[{k}]")
        print(f"  {k:<6}{tree_fleet:>26,.0f}{two_stage_fleet:>20,.0f}")

    tb = tree_cost_breakdown(tree_model)
    sb = two_stage_cost_breakdown(two_stage_model)

    print("\n" + "=" * 70)
    print("COST BREAKDOWN (expected)")
    print("=" * 70)
    print(f"  {'Component':<28}{'Tree':>18}{'Two-stage':>18}")
    for label, key in [("Purchase", "purchase"),
                       ("Planned subcontracting", "planned_subcontracting"),
                       ("Corrective subcontracting", "corrective_subcontracting"),
                       ("Rebalancing", "rebalancing")]:
        print(f"  {label:<28}{tb[key]:>18,.0f}{sb[key]:>18,.0f}")
    print(f"  {'TOTAL':<28}{sum(tb.values()):>18,.0f}{sum(sb.values()):>18,.0f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compare(seasons=(1, 2, 3, 4), weeks_per_season=13, branching=2,
            n_hubs=3, n_types=3, solver_params=None, seed=42):
    N = list(range(n_hubs))
    K = list(range(n_types))
    solver_params = solver_params or {"TimeLimit": 600, "MIPGap": 0.01, "OutputFlag": 1}

    tree = build_toy_scenario_tree(N, seasons=seasons, branching=branching,
                                    weeks_per_season=weeks_per_season, seed=seed)
    tree.validate(N)
    n_leaves = branching ** len(seasons)
    total_weeks = len(seasons) * weeks_per_season
    print(f"Scenario tree: {len(seasons)} stages x {weeks_per_season} weeks, "
          f"branching {branching} -> {n_leaves} full-horizon paths ({total_weeks} weeks total)")

    print("\n--- Solving multistage scenario-tree model ---")
    tree_model = build_tree_model(N, K, tree, seed=seed)
    tree_model.solve(params=solver_params)

    print("\n--- Solving two-stage (MRP) model on the same scenario paths ---")
    two_stage_model, leaf_prob, _ = build_two_stage_model(N, K, tree, seed=seed)
    two_stage_model.solve_MRP(params=solver_params)

    report(tree_model, two_stage_model, N, K)
    return tree_model, two_stage_model


if __name__ == "__main__":
    compare(seasons=(1, 2, 3, 4), weeks_per_season=13, branching=2,
            n_hubs=3, n_types=3,
            solver_params={"TimeLimit": 600, "MIPGap": 0.01})
