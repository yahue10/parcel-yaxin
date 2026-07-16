"""
run_parallel_tree_vs_two_stage.py — Batch comparison across multiple instances
==================================================================================

Runs compare_tree_vs_two_stage.py's multistage-vs-Static-vs-MNP-vs-MRP
comparison across several scenario-tree configurations concurrently — same
pattern as run_parallel.py.

Each worker builds its OWN scenario tree and derives the flattened
full-horizon scenarios for the Static/MNP/MRP models from that same tree
(mirroring where run_parallel.py calls M.generate_data() inside
solve_instance): the instance and its scenario set live entirely inside the
worker, alongside its own Gurobi environment, never shared across processes.
Static/MNP/MRP are solved sequentially on one shared VehicleAllocationModel
instance, with results extracted into plain dicts immediately after each
solve (see compare_tree_vs_two_stage._extract_two_stage_result) since each
solve_*() call overwrites the instance's underlying Gurobi model.

You can configure:
    - INSTANCES: list of dicts defining each tree configuration
      (seasons, weeks_per_season, branching, n_hubs, n_types, seed, label,
      optionally "costs" — overrides for q/beta/alpha/gamma/gamma_corr/
      theta/S/g, see compare_tree_vs_two_stage.build_cost_params, and
      optionally "hub_correlation" — an n_hubs x n_hubs correlation matrix
      applied to both the season-drift and weekly-noise demand shocks, see
      ScenarioTreeModel.build_toy_scenario_tree)
    - MAX_WORKERS: number of parallel processes (defaults to CPU count)
    - GUROBI_OPTIONS: WLS license credentials shared across all workers
    - SOLVER_PARAMS: Gurobi solver parameters (TimeLimit, MIPGap, Threads, ...)
"""

import os
import time
import json
import pickle
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Gurobi license: uses the local server license (gurobi.lic / token server).
# No WLS options needed — Gurobi will auto-detect the license on the machine.
GUROBI_OPTIONS = None

SOLVER_PARAMS = {
    "TimeLimit": 18000,
    "MIPGap": 0.01,
    "Threads": 8,
    "Presolve": 2,
}

MAX_WORKERS = 6

# Define the tree configurations to compare.
# Each dict is passed to one worker. Vary branching, weeks_per_season, seed, etc.
# Optional "costs" key overrides q/beta/alpha/gamma/gamma_corr/theta/S/g for that
# instance (see compare_tree_vs_two_stage.build_cost_params) — each key you
# supply replaces that parameter's dict wholesale, e.g. "costs": {"S": {0: 3000,
# 1: 3000, 2: 3000}} to test a tighter purchase cap. Omit "costs" to use defaults.
# Optional "hub_correlation" key: an n_hubs x n_hubs correlation matrix (symmetric,
# unit diagonal) applied to both season-drift and weekly-noise demand shocks, e.g.
# "hub_correlation": [[1, -0.8, 0], [-0.8, 1, 0], [0, 0, 1]] for hubs 0/1 strongly
# opposed and hub 2 independent. Omit for independent hubs (today's default).
# Optional "season_drift" key: float, dict {season: drift}, or (low, high)
# tuple — see build_toy_scenario_tree's docstring in ScenarioTreeModel.py.
# Omit to use that function's own default.
# Optional "sibling_drift_correlation" key: float in [0, 1] controlling how
# much siblings' season-drift shocks move together (0=independent,
# 1=lock-step). Optional "noise_frac" key: within-season weekly noise scale.
# Both omit to that function's own defaults — see its docstring.
INSTANCES = [
    {"label": "hub10_corr1_b2_seed42", "seasons": (1, 2, 3, 4), "weeks_per_season": 13,
     "branching": 2, "n_hubs": 10, "n_types": 3, "seed": 42, "season_drift": (0.15, 0.20), 
     "sibling_drift_correlation": 1, "noise_frac": 0.05, 
     "hub_correlation": [
    [ 1.0, -0.7,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [-0.7,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.6,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.6,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  1.0,  -0.3,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  -0.3,  1.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -0.5,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0,  0.4],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.4,  1.0],
        ]
        },

    {"label": "hub8_corr1_b2_seed40", "seasons": (1, 2, 3, 4), "weeks_per_season": 13,
     "branching": 2, "n_hubs": 8, "n_types": 3, "seed": 40, "season_drift": (0.15, 0.20), 
     "sibling_drift_correlation": 1, "noise_frac": 0.05,
     "hub_correlation": [
    [ 1.0, -0.7,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [-0.7,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.6,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.6,  1.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  1.0,  -0.3,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  -0.3,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -0.5],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,  1.0],
        ]},
    {"label": "hub8_corr2_b2_seed40", "seasons": (1, 2, 3, 4), "weeks_per_season": 13,
     "branching": 2, "n_hubs": 8, "n_types": 3, "seed": 40, "season_drift": (0.15, 0.20), 
     "sibling_drift_correlation": 1, "noise_frac": 0.05,
     "hub_correlation": [
    [ 1.0, -0.7,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [-0.7,  1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  1.0,  0.6,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.6,  1.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  1.0,  0.3,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.3,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  1.0, -0.5],
    [ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5,  1.0],
        ]},
]


# ---------------------------------------------------------------------------
# Worker function — runs in a separate process
# ---------------------------------------------------------------------------

def solve_instance(instance_cfg):
    """
    Build one scenario tree + its derived two-stage scenario set, solve the
    multistage tree model and the Static / MNP / MRP models, and return a
    results dict. This function runs in its own process with its own Gurobi
    environment.
    """
    import matplotlib
    matplotlib.use("Agg")

    from ScenarioTreeModel import build_toy_scenario_tree
    from compare_tree_vs_two_stage import (
        build_tree_model, build_two_stage_model, build_full_horizon_scenarios,
        save_flat_scenarios, save_scenario_quantities_by_type, save_all_variables,
        static_cost_breakdown, mnp_cost_breakdown, mrp_cost_breakdown,
        static_scenario_cost_breakdown, mnp_scenario_cost_breakdown, mrp_scenario_cost_breakdown,
        static_scenario_subcontracting_quantities, mnp_scenario_subcontracting_quantities,
        mrp_scenario_subcontracting_quantities,
        static_green_coverage_ratios, mnp_green_coverage_ratios, mrp_green_coverage_ratios,
        static_scenario_quantities_by_type, mnp_scenario_quantities_by_type,
        mrp_scenario_quantities_by_type,
        static_scenario_demand_coverage, mnp_scenario_demand_coverage, mrp_scenario_demand_coverage,
        _extract_two_stage_result, _tree_result, _flat_resource, _mrp_resource,
    )
    from plots_tree_vs_two_stage import plot_all_comparisons

    label = instance_cfg["label"]
    seasons = instance_cfg["seasons"]
    weeks_per_season = instance_cfg["weeks_per_season"]
    branching = instance_cfg["branching"]
    seed = instance_cfg["seed"]
    cost_overrides = instance_cfg.get("costs")
    hub_correlation = instance_cfg.get("hub_correlation")
    season_drift = instance_cfg.get("season_drift")
    sibling_drift_correlation = instance_cfg.get("sibling_drift_correlation")
    noise_frac = instance_cfg.get("noise_frac")
    N = list(range(instance_cfg["n_hubs"]))
    K = list(range(instance_cfg["n_types"]))

    print(f"[{label}] Starting: seasons={seasons}, weeks/season={weeks_per_season}, "
          f"branching={branching}, hubs={len(N)}, types={len(K)}, seed={seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments_tree_vs_two_stage", f"{label}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Save the exact instance config used, before any solving starts, so a
    # crashed/partial run still leaves a record of what was attempted, and
    # so results can be traced back to their settings even after INSTANCES
    # has since changed. Tuples round-trip as JSON arrays (informational
    # only -- not meant to be fed back into INSTANCES verbatim).
    with open(os.path.join(exp_dir, "instance_config.json"), "w") as f:
        json.dump(instance_cfg, f, indent=2, default=str)

    # --- Build the instance: the scenario tree, then the scenarios derived from it ---
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

    meta = {
        "label": label, "seasons": seasons, "weeks_per_season": weeks_per_season,
        "branching": branching, "n_hubs": len(N), "n_types": len(K),
        "seed": seed, "exp_dir": exp_dir,
    }

    # --- Multistage scenario-tree model ---
    t0 = time.time()
    tree_model = build_tree_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)
    tree_model.solve(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_tree = time.time() - t0
    print(f"[{label}] Tree model done in {t_tree:.1f}s "
          f"obj={tree_model.model.ObjVal if tree_model.model.status in (2, 9) else 'N/A'}")
    save_all_variables(tree_model, os.path.join(exp_dir, "variables", "tree.pkl"))

    # --- Static / MNP / MRP, sequentially on the SAME shared model instance ---
    m, leaf_prob, total_weeks = build_two_stage_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)

    t0 = time.time()
    m.solve_static(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    static_result = _extract_two_stage_result(
        m, N, K, static_cost_breakdown, _flat_resource, static_scenario_cost_breakdown,
        static_scenario_subcontracting_quantities, static_green_coverage_ratios,
        static_scenario_quantities_by_type, static_scenario_demand_coverage)
    save_all_variables(m, os.path.join(exp_dir, "variables", "static.pkl"))
    t_static = time.time() - t0
    print(f"[{label}] Static done in {t_static:.1f}s obj={static_result['obj']}")

    t0 = time.time()
    m.solve_MNP(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    mnp_result = _extract_two_stage_result(
        m, N, K, mnp_cost_breakdown, _flat_resource, mnp_scenario_cost_breakdown,
        mnp_scenario_subcontracting_quantities, mnp_green_coverage_ratios,
        mnp_scenario_quantities_by_type, mnp_scenario_demand_coverage)
    save_all_variables(m, os.path.join(exp_dir, "variables", "mnp.pkl"))
    t_mnp = time.time() - t0
    print(f"[{label}] MNP done in {t_mnp:.1f}s obj={mnp_result['obj']}")

    t0 = time.time()
    m.solve_MRP(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    mrp_result = _extract_two_stage_result(
        m, N, K, mrp_cost_breakdown, lambda mm, NN, KK: _mrp_resource(mm, NN, KK, total_weeks),
        mrp_scenario_cost_breakdown, mrp_scenario_subcontracting_quantities, mrp_green_coverage_ratios,
        mrp_scenario_quantities_by_type, mrp_scenario_demand_coverage
    )
    save_all_variables(m, os.path.join(exp_dir, "variables", "mrp.pkl"))
    t_mrp = time.time() - t0
    print(f"[{label}] MRP done in {t_mrp:.1f}s obj={mrp_result['obj']}")
    # m is now left in its MRP-solved state — plot_all_comparisons' rebalancing
    # plots rely on being able to query it live, right after this call.

    results = {
        "tree": _tree_result(tree_model, N, K),
        "static": static_result,
        "mnp": mnp_result,
        "mrp": mrp_result,
    }

    result = dict(meta)
    result["results"] = results
    result["n_leaves"] = len(leaf_prob)
    result["total_weeks"] = total_weeks
    result["solve_times"] = {"tree": t_tree, "static": t_static, "mnp": t_mnp, "mrp": t_mrp}
    result["total_time"] = t_tree + t_static + t_mnp + t_mrp

    tree_obj = results["tree"]["obj"]
    mrp_obj = results["mrp"]["obj"]
    if tree_obj is not None and mrp_obj is not None:
        result["vms_vs_mrp"] = mrp_obj - tree_obj
        result["vms_vs_mrp_pct"] = 100 * result["vms_vs_mrp"] / tree_obj if tree_obj else float("nan")

    print(f"[{label}] Done in {result['total_time']:.1f}s "
          f"(tree={t_tree:.1f}s, static={t_static:.1f}s, mnp={t_mnp:.1f}s, mrp={t_mrp:.1f}s)")

    if all(results[k]["obj"] is not None for k in ("tree", "static", "mnp", "mrp")):
        plot_all_comparisons(tree_model, m, results, N, K, total_weeks, output_dir=exp_dir)

    d_real, _, _ = build_full_horizon_scenarios(tree, N)
    save_flat_scenarios(d_real, leaf_prob, N, total_weeks, os.path.join(exp_dir, "flat_scenarios.csv"))
    save_scenario_quantities_by_type(results, K, os.path.join(exp_dir, "scenario_quantities.csv"))

    with open(os.path.join(exp_dir, "result.pkl"), "wb") as f:
        pickle.dump(result, f)

    return result


# ---------------------------------------------------------------------------
# Main — launches workers
# ---------------------------------------------------------------------------

def main():
    print(f"Launching {len(INSTANCES)} instances with up to {MAX_WORKERS} parallel workers")
    print("=" * 70)

    all_results = []
    start_all = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_label = {
            executor.submit(solve_instance, cfg): cfg["label"]
            for cfg in INSTANCES
        }

        for future in as_completed(future_to_label):
            label = future_to_label[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                print(f"[{label}] FAILED with exception:")
                traceback.print_exception(type(exc), exc, exc.__traceback__)
                all_results.append({"label": label, "error": str(exc)})

    total_time = time.time() - start_all

    # --- Summary ---
    print("\n" + "=" * 70)
    print(f"ALL DONE in {total_time:.1f}s")
    print("=" * 70)
    print(f"{'Label':<14} {'Tree':>13} {'Static':>13} {'MNP':>13} {'MRP':>13} {'VMS%':>8} {'Time (s)':>9}  Dir")
    print("-" * 115)
    for r in sorted(all_results, key=lambda x: x.get("label", "")):
        if "error" in r:
            print(f"{r['label']:<14} {'ERROR':>13}")
            continue
        res = r.get("results", {})

        def fmt(m):
            obj = res.get(m, {}).get("obj")
            return f"{obj:>13,.0f}" if obj is not None else f"{'N/A':>13}"

        vms_pct = f"{r['vms_vs_mrp_pct']:>7.2f}%" if r.get("vms_vs_mrp_pct") is not None else f"{'N/A':>8}"
        t = f"{r.get('total_time', 0):>9.1f}"
        print(f"{r['label']:<14} {fmt('tree')} {fmt('static')} {fmt('mnp')} {fmt('mrp')} "
              f"{vms_pct} {t}  {r.get('exp_dir', '')}")

    # Save combined results
    os.makedirs("experiments_tree_vs_two_stage", exist_ok=True)
    summary_path = os.path.join("experiments_tree_vs_two_stage", "all_results.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nCombined results saved to {summary_path}")


if __name__ == "__main__":
    main()
