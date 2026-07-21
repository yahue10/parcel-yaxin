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
import sys
import time
import json
import pickle
import traceback
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Gurobi license: uses the local server license (gurobi.lic / token server).
# No WLS options needed — Gurobi will auto-detect the license on the machine.
GUROBI_OPTIONS = None

SOLVER_PARAMS = {
    "TimeLimit": 2592000,
    "MIPGap": 0.01,
    "Threads": 32,
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
# Optional "mrp_variant" key: "flat" (default) solves MRP the original way
# (Model.py's build_model_MRP); "tree" instead solves it via
# compare_tree_vs_two_stage.build_mrp_tree_model -- the same two-stage
# problem re-expressed through the tree's constraint-generating code, for
# cross-validation. Both produce the same results shape everywhere EXCEPT:
# with "tree", the rebalancing-over-time/heatmap and resource-decomposition
# plots are skipped (they're hardcoded for the flat model's variable names).
# Optional "demand_source" key: "synthetic" (default) builds demand via
# ScenarioTreeModel.build_toy_scenario_tree (random base level per hub).
# "real" instead builds it via real_hub_data.build_real_data_scenario_tree --
# real per-hub mean demand, real (repaired) inter-hub correlation, and real
# per-hub weekly-noise scale, using the first n_hubs hubs listed in
# data/negative_pairs_overlap20_within80km_first20hubs*.csv (capped at 20 --
# that's all the data covers). season_drift/sibling_drift_correlation still
# apply; hub_correlation/noise_frac are ignored with "real" (both come from
# the data instead). Optional "data_dir" key overrides the "data" folder
# path (only relevant with demand_source="real").
INSTANCES = [
    # demand_source="real" derives base demand, inter-hub correlation, and
    # per-hub noise scale from data/negative_pairs_overlap20_within80km_first20hubs*.csv
    # -- no more hand-picked hub_correlation matrix or noise_frac needed
    # (both are ignored if given, since real data provides them instead).
    # season_drift/sibling_drift_correlation still apply on top of the real
    # base level. n_hubs picks the first N hubs listed in the data (capped
    # at 20).
    {"label": "hub16_real_seed40", "seasons": (1, 2, 3, 4), "weeks_per_season": 13,
     "branching": 3, "n_hubs": 16, "n_types": 3, "seed": 40,
     "season_drift": (0.20, 0.25), "sibling_drift_correlation": 1,
     "demand_source": "real"},

    # Same n_hubs=6 shape twice, differentiated by seed (different
    # season-drift/weekly-noise draws) instead of noise_frac/hub_correlation
    # -- those aren't independent knobs anymore under demand_source="real".
    {"label": "hub16_real_seed40", "seasons": (1, 2, 3, 4), "weeks_per_season": 13,
     "branching": 4, "n_hubs": 16, "n_types": 3, "seed": 43,
     "season_drift": (0.20, 0.25), "sibling_drift_correlation": 1,
     "demand_source": "real"},

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
    from real_hub_data import build_real_data_scenario_tree
    from compare_tree_vs_two_stage import (
        build_tree_model, build_two_stage_model, build_mrp_tree_model, build_full_horizon_scenarios,
        save_flat_scenarios, save_scenario_quantities_by_type, save_all_variables,
        static_cost_breakdown, mnp_cost_breakdown, mrp_cost_breakdown,
        static_scenario_cost_breakdown, mnp_scenario_cost_breakdown, mrp_scenario_cost_breakdown,
        static_scenario_subcontracting_quantities, mnp_scenario_subcontracting_quantities,
        mrp_scenario_subcontracting_quantities,
        static_green_coverage_ratios, mnp_green_coverage_ratios, mrp_green_coverage_ratios,
        static_scenario_quantities_by_type, mnp_scenario_quantities_by_type,
        mrp_scenario_quantities_by_type,
        static_scenario_demand_coverage, mnp_scenario_demand_coverage, mrp_scenario_demand_coverage,
        static_scenario_subcontracting_by_period, mnp_scenario_subcontracting_by_period,
        mrp_scenario_subcontracting_by_period,
        static_scenario_rebalancing_quantities, mnp_scenario_rebalancing_quantities,
        mrp_scenario_rebalancing_quantities,
        static_outbound_by_hub_season, mnp_outbound_by_hub_season, mrp_outbound_by_hub_season,
        static_corrective_by_hub_season, mnp_corrective_by_hub_season, mrp_corrective_by_hub_season,
        static_scenario_outbound_by_hub_season, mnp_scenario_outbound_by_hub_season,
        mrp_scenario_outbound_by_hub_season,
        static_scenario_corrective_by_hub_season, mnp_scenario_corrective_by_hub_season,
        mrp_scenario_corrective_by_hub_season,
        save_subcontracting_extremes_table, save_rebalancing_movement_table, save_hub_season_table,
        save_hub_season_scenario_table,
        _extract_two_stage_result, _tree_result, _flat_resource, _mrp_resource, save_solve_log,
        _params_with_log_file,
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
    mrp_variant = instance_cfg.get("mrp_variant", "flat")
    if mrp_variant not in ("flat", "tree"):
        raise ValueError(f"[{label}] mrp_variant must be 'flat' or 'tree', got {mrp_variant!r}")
    demand_source = instance_cfg.get("demand_source", "synthetic")
    if demand_source not in ("synthetic", "real"):
        raise ValueError(f"[{label}] demand_source must be 'synthetic' or 'real', got {demand_source!r}")
    data_dir = instance_cfg.get("data_dir", "data")
    N = list(range(instance_cfg["n_hubs"]))
    K = list(range(instance_cfg["n_types"]))

    print(f"[{label}] Starting: seasons={seasons}, weeks/season={weeks_per_season}, "
          f"branching={branching}, hubs={len(N)}, types={len(K)}, seed={seed}, "
          f"mrp_variant={mrp_variant}, demand_source={demand_source}")

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
    if demand_source == "real":
        if hub_correlation is not None or noise_frac is not None:
            print(f"[{label}] Note: hub_correlation/noise_frac are ignored with "
                  "demand_source='real' -- both are derived from the real data instead.")
        real_kwargs = dict(seasons=seasons, branching=branching, weeks_per_season=weeks_per_season, seed=seed)
        if season_drift is not None:
            real_kwargs["season_drift"] = season_drift
        if sibling_drift_correlation is not None:
            real_kwargs["sibling_drift_correlation"] = sibling_drift_correlation
        tree, hub_meta = build_real_data_scenario_tree(len(N), data_dir=data_dir, **real_kwargs)
        hub_list = ", ".join(f"{i}={hub_meta[i]['site']}" for i in hub_meta)
        print(f"[{label}] Using real hub data: {hub_list}")
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

    meta = {
        "label": label, "seasons": seasons, "weeks_per_season": weeks_per_season,
        "branching": branching, "n_hubs": len(N), "n_types": len(K),
        "seed": seed, "exp_dir": exp_dir,
    }

    wall_clock_start = time.time()

    # --- Multistage scenario-tree model: build now, solve in a background
    # thread so it overlaps with the static/MNP/MRP chain below. tree_model
    # is a fully separate Gurobi Model/Env from `m` (the two-stage model),
    # so there's no shared mutable state to race on, and Gurobi releases the
    # GIL during optimize() -- this is genuine parallelism, not just
    # concurrency. Tree is often the single slowest solve, so overlapping it
    # removes it from the critical path entirely instead of adding to it.
    tree_model = build_tree_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)

    def _solve_tree_model():
        t0 = time.time()
        tree_params = _params_with_log_file(SOLVER_PARAMS, os.path.join(exp_dir, "gurobi_log", "tree.log"))
        tree_model.solve(params=tree_params, options=GUROBI_OPTIONS, label=f"{label}:tree")
        save_all_variables(tree_model, os.path.join(exp_dir, "variables", "tree.pkl"), label=label)
        return time.time() - t0

    tree_executor = ThreadPoolExecutor(max_workers=1)
    tree_future = tree_executor.submit(_solve_tree_model)

    # --- Static / MNP / MRP, sequentially on the SAME shared model instance,
    # running concurrently with the tree solve above ---
    m, leaf_prob, total_weeks = build_two_stage_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)

    t0 = time.time()
    static_params = _params_with_log_file(SOLVER_PARAMS, os.path.join(exp_dir, "gurobi_log", "static.log"))
    m.solve_static(params=static_params, options=GUROBI_OPTIONS, label=f"{label}:static")
    static_result = _extract_two_stage_result(
        m, N, K, total_weeks, static_cost_breakdown, _flat_resource, static_scenario_cost_breakdown,
        static_scenario_subcontracting_quantities, static_green_coverage_ratios,
        static_scenario_quantities_by_type, static_scenario_demand_coverage,
        lambda mm: static_scenario_subcontracting_by_period(mm, total_weeks),
        static_scenario_rebalancing_quantities,
        static_outbound_by_hub_season, static_corrective_by_hub_season,
        static_scenario_outbound_by_hub_season, static_scenario_corrective_by_hub_season)
    save_all_variables(m, os.path.join(exp_dir, "variables", "static.pkl"), label=label)
    t_static = time.time() - t0
    print(f"[{label}] Static done in {t_static:.1f}s obj={static_result['obj']}")

    t0 = time.time()
    mnp_params = _params_with_log_file(SOLVER_PARAMS, os.path.join(exp_dir, "gurobi_log", "mnp.log"))
    m.solve_MNP(params=mnp_params, options=GUROBI_OPTIONS, label=f"{label}:mnp")
    mnp_result = _extract_two_stage_result(
        m, N, K, total_weeks, mnp_cost_breakdown, _flat_resource, mnp_scenario_cost_breakdown,
        mnp_scenario_subcontracting_quantities, mnp_green_coverage_ratios,
        mnp_scenario_quantities_by_type, mnp_scenario_demand_coverage,
        lambda mm: mnp_scenario_subcontracting_by_period(mm, total_weeks),
        mnp_scenario_rebalancing_quantities,
        mnp_outbound_by_hub_season, mnp_corrective_by_hub_season,
        mnp_scenario_outbound_by_hub_season, mnp_scenario_corrective_by_hub_season)
    save_all_variables(m, os.path.join(exp_dir, "variables", "mnp.pkl"), label=label)
    t_mnp = time.time() - t0
    print(f"[{label}] MNP done in {t_mnp:.1f}s obj={mnp_result['obj']}")

    t0 = time.time()
    mrp_params = _params_with_log_file(SOLVER_PARAMS, os.path.join(exp_dir, "gurobi_log", "mrp.log"))
    if mrp_variant == "tree":
        mrp_tree_model = build_mrp_tree_model(N, K, tree, seed=seed, cost_overrides=cost_overrides)
        mrp_tree_model.solve(params=mrp_params, options=GUROBI_OPTIONS, label=f"{label}:mrp", s_first_stage=True)
        mrp_result = _tree_result(mrp_tree_model, N, K, total_weeks)
        save_all_variables(mrp_tree_model, os.path.join(exp_dir, "variables", "mrp.pkl"), label=label)
        mrp_model_for_plots = mrp_tree_model
    else:
        m.solve_MRP(params=mrp_params, options=GUROBI_OPTIONS, label=f"{label}:mrp")
        mrp_result = _extract_two_stage_result(
            m, N, K, total_weeks, mrp_cost_breakdown, lambda mm, NN, KK: _mrp_resource(mm, NN, KK, total_weeks),
            mrp_scenario_cost_breakdown, mrp_scenario_subcontracting_quantities, mrp_green_coverage_ratios,
            mrp_scenario_quantities_by_type, mrp_scenario_demand_coverage,
            lambda mm: mrp_scenario_subcontracting_by_period(mm, total_weeks),
            mrp_scenario_rebalancing_quantities,
            mrp_outbound_by_hub_season, mrp_corrective_by_hub_season,
            mrp_scenario_outbound_by_hub_season, mrp_scenario_corrective_by_hub_season
        )
        save_all_variables(m, os.path.join(exp_dir, "variables", "mrp.pkl"), label=label)
        # m is now left in its MRP-solved state — plot_all_comparisons' rebalancing
        # plots rely on being able to query it live, right after this call.
        mrp_model_for_plots = m
    t_mrp = time.time() - t0
    print(f"[{label}] MRP done in {t_mrp:.1f}s obj={mrp_result['obj']}")

    # Join the background tree solve (usually already done by now, since it
    # started before static/MNP/MRP and often takes comparably long).
    t_tree = tree_future.result()
    tree_executor.shutdown()
    print(f"[{label}] Tree model done in {t_tree:.1f}s "
          f"obj={tree_model.model.ObjVal if tree_model.model.status in (2, 9) else 'N/A'}")

    results = {
        "tree": _tree_result(tree_model, N, K, total_weeks),
        "static": static_result,
        "mnp": mnp_result,
        "mrp": mrp_result,
    }

    result = dict(meta)
    result["results"] = results
    result["n_leaves"] = len(leaf_prob)
    result["total_weeks"] = total_weeks
    result["solve_times"] = {"tree": t_tree, "static": t_static, "mnp": t_mnp, "mrp": t_mrp}
    # Wall-clock time for this instance, not the sum of the 4 individual solve
    # times above -- tree now runs concurrently with static/MNP/MRP, so the
    # sum would double-count the overlapped portion.
    result["total_time"] = time.time() - wall_clock_start

    tree_obj = results["tree"]["obj"]
    mrp_obj = results["mrp"]["obj"]
    if tree_obj is not None and mrp_obj is not None:
        result["vms_vs_mrp"] = mrp_obj - tree_obj
        result["vms_vs_mrp_pct"] = 100 * result["vms_vs_mrp"] / tree_obj if tree_obj else float("nan")

    print(f"[{label}] Done in {result['total_time']:.1f}s "
          f"(tree={t_tree:.1f}s, static={t_static:.1f}s, mnp={t_mnp:.1f}s, mrp={t_mrp:.1f}s)")

    if all(results[k]["obj"] is not None for k in ("tree", "static", "mnp", "mrp")):
        plot_all_comparisons(tree_model, mrp_model_for_plots, results, N, K, total_weeks, output_dir=exp_dir,
                              instance_label=label)

    d_real, _, _ = build_full_horizon_scenarios(tree, N)
    save_flat_scenarios(d_real, leaf_prob, N, total_weeks, os.path.join(exp_dir, "flat_scenarios.csv"), label=label)
    save_scenario_quantities_by_type(results, K, os.path.join(exp_dir, "scenario_quantities.csv"), label=label)
    save_subcontracting_extremes_table(results, K, os.path.join(exp_dir, "subcontracting_extremes.csv"), label=label)
    save_rebalancing_movement_table(results, os.path.join(exp_dir, "rebalancing_movements.csv"), label=label)
    B = sorted(tree.e)
    save_hub_season_table({mk: results[mk]["outbound_by_hub_season"] for mk in ("tree", "static", "mnp", "mrp")},
                           N, B, os.path.join(exp_dir, "outbound_by_hub_season.csv"),
                           value_label="expected_outbound_qty", label=label)
    save_hub_season_table({mk: results[mk]["corrective_by_hub_season"] for mk in ("tree", "static", "mnp", "mrp")},
                           N, B, os.path.join(exp_dir, "corrective_by_hub_season.csv"),
                           value_label="expected_corrective_qty", label=label)
    save_hub_season_scenario_table(
        {mk: results[mk]["scenario_outbound_by_hub_season"] for mk in ("tree", "static", "mnp", "mrp")},
        N, B, os.path.join(exp_dir, "outbound_by_hub_season_scenario.csv"), value_label="outbound_qty", label=label)
    save_hub_season_scenario_table(
        {mk: results[mk]["scenario_corrective_by_hub_season"] for mk in ("tree", "static", "mnp", "mrp")},
        N, B, os.path.join(exp_dir, "corrective_by_hub_season_scenario.csv"), value_label="corrective_qty", label=label)
    save_solve_log(exp_dir, results, result["solve_times"], label=label)

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
    interrupted = False

    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
    try:
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
    except KeyboardInterrupt:
        # Gurobi installs its own SIGINT handler while optimize() is running,
        # so a single Ctrl+C typically only stops whichever solve is active
        # in each worker at that instant (as an early "interrupted" status)
        # -- the surrounding Python code then just carries on to the next
        # solve/instance, which is why it looked like only some experiments
        # stopped. Force-killing every worker process with SIGTERM instead
        # bypasses Gurobi's handler entirely (it only customizes SIGINT) and
        # kills the whole process outright, including its background tree
        # thread, rather than waiting for ProcessPoolExecutor's default
        # graceful shutdown (which would otherwise block here until every
        # busy worker finishes on its own).
        interrupted = True
        procs = list(executor._processes.values())
        print(f"\nInterrupted -- terminating {len(procs)} running worker process(es) now "
              "(not waiting for in-progress solves to finish)...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)

    if interrupted:
        print(f"Stopped after {len(all_results)}/{len(INSTANCES)} instance(s) completed.")

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

    if interrupted:
        sys.exit(1)


if __name__ == "__main__":
    main()
