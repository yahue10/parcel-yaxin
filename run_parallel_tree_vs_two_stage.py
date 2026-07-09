"""
run_parallel_tree_vs_two_stage.py — Batch VMS comparison across multiple instances
====================================================================================

Runs compare_tree_vs_two_stage.py's multistage-vs-two-stage comparison across
several scenario-tree configurations concurrently — same pattern as
run_parallel.py.

Each worker builds its OWN scenario tree and derives the flattened
full-horizon scenarios for the two-stage model from that same tree (mirroring
where run_parallel.py calls M.generate_data() inside solve_instance): the
instance and its scenario set live entirely inside the worker, alongside its
own Gurobi environment, never shared across processes.

You can configure:
    - INSTANCES: list of dicts defining each tree configuration
      (seasons, weeks_per_season, branching, n_hubs, n_types, seed, label,
      and optionally "costs" — overrides for q/beta/alpha/gamma/gamma_corr/
      theta/S/g, see compare_tree_vs_two_stage.build_cost_params)
    - MAX_WORKERS: number of parallel processes (defaults to CPU count)
    - GUROBI_OPTIONS: WLS license credentials shared across all workers
    - SOLVER_PARAMS: Gurobi solver parameters (TimeLimit, MIPGap, Threads, ...)
"""

import os
import time
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
    "TimeLimit": 3600,
    "MIPGap": 0.01,
    "Threads": 8,
    "Presolve": 2,
}

MAX_WORKERS = 2

# Define the tree configurations to compare.
# Each dict is passed to one worker. Vary branching, weeks_per_season, seed, etc.
# Optional "costs" key overrides q/beta/alpha/gamma/gamma_corr/theta/S/g for that
# instance (see compare_tree_vs_two_stage.build_cost_params) — each key you
# supply replaces that parameter's dict wholesale, e.g. "costs": {"S": {0: 3000,
# 1: 3000, 2: 3000}} to test a tighter purchase cap. Omit "costs" to use defaults.
INSTANCES = [
    {"label": "b2_seed42", "seasons": (1, 2, 3, 4), "weeks_per_season": 13,
     "branching": 2, "n_hubs": 3, "n_types": 3, "seed": 42},
    {"label": "b2_seed7", "seasons": (1, 2, 3, 4), "weeks_per_season": 13,
     "branching": 2, "n_hubs": 3, "n_types": 3, "seed": 7},
]


# ---------------------------------------------------------------------------
# Worker function — runs in a separate process
# ---------------------------------------------------------------------------

def solve_instance(instance_cfg):
    """
    Build one scenario tree + its derived two-stage scenario set, solve both
    the multistage and two-stage models, and return a results dict.
    This function runs in its own process with its own Gurobi environment.
    """
    import matplotlib
    matplotlib.use("Agg")

    from ScenarioTreeModel import build_toy_scenario_tree
    from compare_tree_vs_two_stage import (
        build_tree_model, build_two_stage_model,
        tree_cost_breakdown, two_stage_cost_breakdown,
    )
    from plots_tree_vs_two_stage import plot_all_comparisons

    label = instance_cfg["label"]
    seasons = instance_cfg["seasons"]
    weeks_per_season = instance_cfg["weeks_per_season"]
    branching = instance_cfg["branching"]
    seed = instance_cfg["seed"]
    cost_overrides = instance_cfg.get("costs")
    N = list(range(instance_cfg["n_hubs"]))
    K = list(range(instance_cfg["n_types"]))

    print(f"[{label}] Starting: seasons={seasons}, weeks/season={weeks_per_season}, "
          f"branching={branching}, hubs={len(N)}, types={len(K)}, seed={seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments_tree_vs_two_stage", f"{label}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # --- Build the instance: the scenario tree, then the scenarios derived from it ---
    tree = build_toy_scenario_tree(N, seasons=seasons, branching=branching,
                                    weeks_per_season=weeks_per_season, seed=seed)
    tree.validate(N)

    result = {
        "label": label, "seasons": seasons, "weeks_per_season": weeks_per_season,
        "branching": branching, "n_hubs": len(N), "n_types": len(K),
        "seed": seed, "exp_dir": exp_dir,
    }

    # --- Multistage scenario-tree model ---
    t0 = time.time()
    tree_model = build_tree_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)
    tree_model.solve(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_tree = time.time() - t0

    if tree_model.model.status in (2, 9):
        result["tree_obj"] = tree_model.model.ObjVal
        result["tree_gap"] = tree_model.model.MIPGap
        result["tree_costs"] = tree_cost_breakdown(tree_model)
        print(f"[{label}] Tree model done in {t_tree:.1f}s  obj={result['tree_obj']:.0f}")
    else:
        result["tree_obj"] = None
        print(f"[{label}] Tree model ended with status {tree_model.model.status}")
    result["tree_time"] = t_tree

    # --- Two-stage (MRP) model, built on the SAME tree's flattened scenarios ---
    t0 = time.time()
    two_stage_model, leaf_prob, total_weeks = build_two_stage_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)
    two_stage_model.solve_MRP(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_two_stage = time.time() - t0

    if two_stage_model.model.status in (2, 9):
        result["two_stage_obj"] = two_stage_model.model.ObjVal
        result["two_stage_gap"] = two_stage_model.model.MIPGap
        result["two_stage_costs"] = two_stage_cost_breakdown(two_stage_model)
        print(f"[{label}] Two-stage model done in {t_two_stage:.1f}s  obj={result['two_stage_obj']:.0f}")
    else:
        result["two_stage_obj"] = None
        print(f"[{label}] Two-stage model ended with status {two_stage_model.model.status}")
    result["two_stage_time"] = t_two_stage
    result["n_leaves"] = len(leaf_prob)
    result["total_weeks"] = total_weeks

    if result.get("tree_obj") is not None and result.get("two_stage_obj") is not None:
        result["vms"] = result["two_stage_obj"] - result["tree_obj"]
        result["vms_pct"] = 100 * result["vms"] / result["tree_obj"] if result["tree_obj"] else float("nan")

    result["total_time"] = t_tree + t_two_stage
    print(f"[{label}] Done in {result['total_time']:.1f}s "
          f"(tree={t_tree:.1f}s, two_stage={t_two_stage:.1f}s)")

    if result.get("tree_obj") is not None and result.get("two_stage_obj") is not None:
        plot_all_comparisons(tree_model, two_stage_model, N, K, total_weeks, output_dir=exp_dir)

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
    print(f"{'Label':<16} {'Tree obj':>14} {'Two-stage obj':>16} {'VMS %':>9} {'Time (s)':>10}  Dir")
    print("-" * 100)
    for r in sorted(all_results, key=lambda x: x.get("label", "")):
        if "error" in r:
            print(f"{r['label']:<16} {'ERROR':>14}")
            continue
        tree_obj = f"{r['tree_obj']:>14,.0f}" if r.get("tree_obj") is not None else f"{'N/A':>14}"
        two_stage_obj = f"{r['two_stage_obj']:>16,.0f}" if r.get("two_stage_obj") is not None else f"{'N/A':>16}"
        vms_pct = f"{r['vms_pct']:>8.2f}%" if r.get("vms_pct") is not None else f"{'N/A':>9}"
        t = f"{r.get('total_time', 0):>10.1f}"
        print(f"{r['label']:<16} {tree_obj} {two_stage_obj} {vms_pct} {t}  {r.get('exp_dir', '')}")

    # Save combined results
    os.makedirs("experiments_tree_vs_two_stage", exist_ok=True)
    summary_path = os.path.join("experiments_tree_vs_two_stage", "all_results.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nCombined results saved to {summary_path}")


if __name__ == "__main__":
    main()
