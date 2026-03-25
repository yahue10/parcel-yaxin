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

# Gurobi WLS license options (shared by all workers)
GUROBI_OPTIONS = {
    'WLSACCESSID': "30bca212-81df-41cc-a94e-a0269b14a3ec",
    'WLSSECRET': "215eee4c-3130-4a8b-8156-898521b84f16",
    'LICENSEID': 2738996,
    'WLSTOKENDURATION': 10,  # minutes
}

# Solver parameters passed to Gurobi
SOLVER_PARAMS = {
    "TimeLimit": 1600,
    "MIPGap": 0.01,
    # Limit threads per solve so parallel instances don't fight for CPU.
    # Rule of thumb: total_cores / MAX_WORKERS
    "Threads": 2,
}

# Number of parallel workers. Set to None to use all CPUs.
# On Windows with WLS license, keep this <= your license's concurrent-use limit.
MAX_WORKERS = 4

# Define the instances to solve.
# Each dict is passed to one worker. Vary N, O, seed, etc. as needed.
INSTANCES = [
    {"N": 3,  "K": 2, "T": 52, "O": 100, "seed": 42,  "label": "small_s42"},
    {"N": 3,  "K": 2, "T": 52, "O": 100, "seed": 123, "label": "small_s123"},
    {"N": 5,  "K": 2, "T": 52, "O": 100, "seed": 42,  "label": "med_s42"},
    {"N": 5,  "K": 2, "T": 52, "O": 100, "seed": 123, "label": "med_s123"},
    {"N": 10, "K": 2, "T": 52, "O": 100, "seed": 42,  "label": "large_s42"},
    {"N": 10, "K": 2, "T": 52, "O": 100, "seed": 123, "label": "large_s123"},
]


# ---------------------------------------------------------------------------
# Worker function — runs in a separate process
# ---------------------------------------------------------------------------

def solve_instance(instance_cfg):
    """
    Solve one instance (static + ST) and return a results dict.
    This function runs in its own process with its own Gurobi environment.
    """
    # Import here so each subprocess has its own module state
    from Model import VehicleAllocationModel
    from plots import extract_ST, extract_static, extract_ST_costs

    label = instance_cfg["label"]
    N = instance_cfg["N"]
    K = instance_cfg["K"]
    T = instance_cfg["T"]
    O = instance_cfg["O"]
    seed = instance_cfg["seed"]

    print(f"[{label}] Starting: N={N}, K={K}, T={T}, O={O}, seed={seed}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experiments_parallel", f"{label}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Build model and generate data
    M = VehicleAllocationModel(N, K, T, O, seed=seed)
    M.generate_data()
    M.save_instance(os.path.join(exp_dir, "instance.pkl"))

    result = {
        "label": label,
        "N": N, "K": K, "T": T, "O": O, "seed": seed,
        "exp_dir": exp_dir,
    }

    # --- Solve static model ---
    t0 = time.time()
    M.solve_static(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_static = time.time() - t0

    if M.model.status in (2, 9):  # OPTIMAL or TIME_LIMIT
        static_obj = M.model.ObjVal
        static_gap = M.model.MIPGap
        static_x, static_s = extract_static(M)
        result["static_obj"] = static_obj
        result["static_gap"] = static_gap
        result["static_time"] = t_static
    else:
        result["static_obj"] = None
        result["static_time"] = t_static
        print(f"[{label}] Static model ended with status {M.model.status}")

    # --- Solve ST model ---
    t0 = time.time()
    M.solve_ST(params=SOLVER_PARAMS, options=GUROBI_OPTIONS)
    t_st = time.time() - t0

    if M.model.status in (2, 9):
        st_obj = M.model.ObjVal
        st_gap = M.model.MIPGap
        st_x, st_s, st_y = extract_ST(M)
        st_costs = extract_ST_costs(M)
        result["st_obj"] = st_obj
        result["st_gap"] = st_gap
        result["st_time"] = t_st

        # Save rebalancing solution
        with open(os.path.join(exp_dir, "st_y.pkl"), "wb") as f:
            pickle.dump(st_y, f)

        # Generate plots (non-interactive, safe for subprocesses)
        import matplotlib
        matplotlib.use("Agg")
        from plots import (plot_compare_subcontracting,
                           plot_compare_resource, plot_compare_costs)
        if result.get("static_obj") is not None:
            plot_compare_subcontracting(M, st_s, static_s, output_dir=exp_dir)
            plot_compare_resource(M, st_x, static_x, output_dir=exp_dir)
            plot_compare_costs(M, st_costs, static_obj, output_dir=exp_dir)
    else:
        result["st_obj"] = None
        result["st_time"] = t_st
        print(f"[{label}] ST model ended with status {M.model.status}")

    result["total_time"] = t_static + t_st
    print(f"[{label}] Done in {result['total_time']:.1f}s "
          f"(static={t_static:.1f}s, ST={t_st:.1f}s)")

    # Save result summary
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
    print(f"{'Label':<20} {'Static Obj':>12} {'ST Obj':>12} {'Time (s)':>10} {'Dir'}")
    print("-" * 70)
    for r in sorted(all_results, key=lambda x: x.get("label", "")):
        if "error" in r:
            print(f"{r['label']:<20} {'ERROR':>12}")
            continue
        static = f"{r.get('static_obj', 'N/A'):>12.0f}" if r.get('static_obj') else f"{'N/A':>12}"
        st = f"{r.get('st_obj', 'N/A'):>12.0f}" if r.get('st_obj') else f"{'N/A':>12}"
        t = f"{r.get('total_time', 0):>10.1f}"
        print(f"{r['label']:<20} {static} {st} {t} {r.get('exp_dir', '')}")

    # Save combined results
    os.makedirs("experiments_parallel", exist_ok=True)
    summary_path = os.path.join("experiments_parallel", "all_results.pkl")
    with open(summary_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nCombined results saved to {summary_path}")


if __name__ == "__main__":
    main()
