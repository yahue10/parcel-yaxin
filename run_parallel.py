"""
run_parallel.py — Run multiple VehicleAllocationModel instances in parallel
=============================================================================

Usage (Windows or any OS):
    python run_parallel.py

This script solves multiple problem instances concurrently using Python's
multiprocessing via ProcessPoolExecutor. Each worker process gets its own
Gurobi environment and license token.

You can configure:
    - INSTANCES: list of dicts defining each instance (N, K, T, O, seed, ...)
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

# Solver parameters passed to Gurobi
SOLVER_PARAMS = {
    "TimeLimit": 72000,
    "MIPGap": 0.01,
    "Threads": 32,   # ~total_cores / MAX_WORKERS; sweet spot for Gurobi MIP
    "Presolve": 2,
}

MAX_WORKERS = 2

# Define the instances to solve.
# Each dict is passed to one worker. Vary N, O, seed, etc. as needed.
INSTANCES = [
    {"N": 20, "K": 3, "T": 52, "O": 100, "seed": 42,  "label": "xlarge_s42"},
    {"N": 20, "K": 3, "T": 52, "O": 100, "seed": 123, "label": "xlarge_s123"},
]


# ---------------------------------------------------------------------------
# Worker function — runs in a separate process
# ---------------------------------------------------------------------------

def solve_instance(instance_cfg):
    """
    Solve one instance (static + MNP + MRP) and return a results dict.
    This function runs in its own process with its own Gurobi environment.
    """
    import matplotlib
    matplotlib.use("Agg")

    from Model import VehicleAllocationModel
    from plots import (extract_static, extract_MNP, extract_MNP_costs,
                       extract_MRP, extract_MRP_costs,
                       plot_compare_subcontracting_3way,
                       plot_compare_resource_3way,
                       plot_compare_costs_3way)

    label = instance_cfg["label"]
    N = instance_cfg["N"]
    K = instance_cfg["K"]
    T = instance_cfg["T"]
    O = instance_cfg["O"]
    seed = instance_cfg["seed"]

    print(f"[{label}] Starting: N={N}, K={K}, T={T}, O={O}, seed={seed}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments_parallel", f"{label}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    M = VehicleAllocationModel(N, K, T, O, seed=seed)
    M.generate_data()
    M.save_instance(os.path.join(exp_dir, "instance.pkl"))

    result = {
        "label": label,
        "N": N, "K": K, "T": T, "O": O, "seed": seed,
        "exp_dir": exp_dir,
    }

    # --- Static ---
    t0 = time.time()
    M.solve_static(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_static = time.time() - t0

    static_x = static_s = static_obj = None
    if M.model.status in (2, 9):
        static_obj = M.model.ObjVal
        static_x, static_s = extract_static(M)
        result["static_obj"] = static_obj
        result["static_gap"] = M.model.MIPGap
        result["static_time"] = t_static
        print(f"[{label}] Static done in {t_static:.1f}s  obj={static_obj:.0f}")
    else:
        result["static_obj"] = None
        result["static_time"] = t_static
        print(f"[{label}] Static ended with status {M.model.status}")

    # --- MNP ---
    t0 = time.time()
    M.solve_MNP(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_mnp = time.time() - t0

    mnp_x = mnp_s = mnp_costs = None
    if M.model.status in (2, 9):
        mnp_obj = M.model.ObjVal
        mnp_x, mnp_s = extract_MNP(M)
        mnp_costs = extract_MNP_costs(M)
        result["mnp_obj"] = mnp_obj
        result["mnp_gap"] = M.model.MIPGap
        result["mnp_time"] = t_mnp
        print(f"[{label}] MNP done in {t_mnp:.1f}s  obj={mnp_obj:.0f}")
    else:
        result["mnp_obj"] = None
        result["mnp_time"] = t_mnp
        print(f"[{label}] MNP ended with status {M.model.status}")

    # --- MRP ---
    t0 = time.time()
    M.solve_MRP(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_mrp = time.time() - t0

    mrp_x = mrp_s = mrp_costs = None
    if M.model.status in (2, 9):
        mrp_obj = M.model.ObjVal
        mrp_x, mrp_s, mrp_y = extract_MRP(M)
        mrp_costs = extract_MRP_costs(M)
        result["mrp_obj"] = mrp_obj
        result["mrp_gap"] = M.model.MIPGap
        result["mrp_time"] = t_mrp
        print(f"[{label}] MRP done in {t_mrp:.1f}s  obj={mrp_obj:.0f}")

        with open(os.path.join(exp_dir, "mrp_y.pkl"), "wb") as f:
            pickle.dump(mrp_y, f)
    else:
        result["mrp_obj"] = None
        result["mrp_time"] = t_mrp
        print(f"[{label}] MRP ended with status {M.model.status}")

    # --- 3-way plots ---
    if all(v is not None for v in [static_obj, mnp_costs, mrp_costs]):
        plot_compare_subcontracting_3way(M, mrp_s, mnp_s, static_s, output_dir=exp_dir)
        plot_compare_resource_3way(M, mrp_x, mnp_x, static_x, output_dir=exp_dir)
        plot_compare_costs_3way(M, mrp_costs, mnp_costs, static_obj, output_dir=exp_dir)

    result["total_time"] = t_static + t_mnp + t_mrp
    print(f"[{label}] Done in {result['total_time']:.1f}s "
          f"(static={t_static:.1f}s, MNP={t_mnp:.1f}s, MRP={t_mrp:.1f}s)")

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
        # Submit all instances
        future_to_label = {
            executor.submit(solve_instance, cfg): cfg["label"]
            for cfg in INSTANCES
        }

        # Collect results as they complete
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
    print(f"{'Label':<20} {'Static':>12} {'MNP':>12} {'MRP':>12} {'Time (s)':>10}  Dir")
    print("-" * 90)
    for r in sorted(all_results, key=lambda x: x.get("label", "")):
        if "error" in r:
            print(f"{r['label']:<20} {'ERROR':>12}")
            continue
        static = f"{r['static_obj']:>12.0f}" if r.get('static_obj') else f"{'N/A':>12}"
        mnp    = f"{r['mnp_obj']:>12.0f}"    if r.get('mnp_obj')    else f"{'N/A':>12}"
        mrp    = f"{r['mrp_obj']:>12.0f}"    if r.get('mrp_obj')    else f"{'N/A':>12}"
        t      = f"{r.get('total_time', 0):>10.1f}"
        print(f"{r['label']:<20} {static} {mnp} {mrp} {t}  {r.get('exp_dir', '')}")

    # Save combined results
    os.makedirs("experiments_parallel", exist_ok=True) summary_path = os.path.join("experiments_parallel", "all_results.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nCombined results saved to {summary_path}")


if __name__ == "__main__":
    main()
