"""
real_hub_data.py — Calibrates demand-scenario generation to real hub data
==========================================================================

Loads real per-hub weekly-demand mean/std, location, and pairwise
correlation from data/negative_pairs_overlap20_within80km_first20hubs*.csv,
and builds a scenario tree from it via ScenarioTreeModel.build_toy_scenario_tree
-- real base demand, real (repaired) inter-hub correlation, and real
per-hub weekly-noise scale substituted in, while the season-drift mechanism,
sibling_drift_correlation, and tree structure are all unchanged.

The real correlation matrix (both the full 20x20 and any subset checked
down to 3 hubs) is NOT a valid correlation matrix on its own -- its minimum
eigenvalue is negative (~-1.89 for the full matrix), almost certainly
because n_weeks_observed varies hugely per hub (96 to 567 weeks), so
pairwise correlations were computed on inconsistent, only-partially
-overlapping observation windows. Each pairwise number can be individually
plausible while the full set is mutually inconsistent (some implied
combination of hub demands would need negative variance, which is
impossible). np.linalg.cholesky() -- used directly by build_toy_scenario_tree
-- would raise on this data as-is, so it's repaired once via eigenvalue
clipping before use (see _nearest_psd_correlation).
"""

import os

import numpy as np
import pandas as pd

from ScenarioTreeModel import build_toy_scenario_tree

DEFAULT_DATA_DIR = "data"
DEFAULT_INFO_CSV = "negative_pairs_overlap20_within80km_first20hubs.csv"
DEFAULT_CORR_CSV = "negative_pairs_overlap20_within80km_first20hubs_corr_matrix.csv"


def _nearest_psd_correlation(corr, epsilon=1e-6):
    """Eigenvalue-clipping repair: clips negative eigenvalues to epsilon,
    reconstructs, rescales back to unit diagonal. Not the exact Frobenius
    -nearest matrix (that's Higham's iterative algorithm) but a standard,
    simple approximation adequate for simulation input -- measured impact
    on this dataset: max single-pair change 0.41, mean change 0.095,
    concentrated in the pairs driving the inconsistency."""
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.clip(eigvals, epsilon, None)
    fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(fixed))
    fixed = fixed / np.outer(d, d)
    np.fill_diagonal(fixed, 1.0)
    return fixed


def load_hub_data(n_hubs, data_dir=DEFAULT_DATA_DIR, info_csv=DEFAULT_INFO_CSV, corr_csv=DEFAULT_CORR_CSV):
    """
    Returns (base_demand, corr_matrix, cv_by_hub, hub_meta):
      base_demand : {i: mean_weekly_demand} for i in range(n_hubs)
      corr_matrix : n_hubs x n_hubs PSD-repaired correlation matrix (ndarray)
      cv_by_hub   : {i: std_weekly_demand / mean_weekly_demand} for i in range(n_hubs)
      hub_meta    : {i: {"regate", "site", "lat", "lng", "n_weeks_observed"}} --
                    reference/plotting only, not used by the model itself.

    Hubs are the first n_hubs ROWS of info_csv, in file order (not
    resorted), integer-labeled 0..n_hubs-1 to match this codebase's
    N = list(range(n_hubs)) convention used everywhere else. info_csv and
    corr_csv list hubs in DIFFERENT orders, so hubs are matched by real ID
    (the "regate" column), not row position.
    """
    info_path = os.path.join(data_dir, info_csv)
    corr_path = os.path.join(data_dir, corr_csv)
    info = pd.read_csv(info_path)
    if n_hubs > len(info):
        raise ValueError(f"Requested n_hubs={n_hubs} but only {len(info)} hubs available in {info_path}")
    chosen = info.iloc[:n_hubs].reset_index(drop=True)
    hub_ids = chosen["regate"].astype(str).tolist()

    corr_df = pd.read_csv(corr_path, index_col=0)
    corr_df.columns = [c.split(" ")[0] for c in corr_df.columns]
    corr_df.index = corr_df.columns
    # Repair the FULL matrix once -- any principal submatrix of a PSD matrix
    # is itself guaranteed PSD, so slicing after repair is safe for any N,
    # without needing to re-repair per N.
    full_fixed = _nearest_psd_correlation(corr_df.values)
    full_fixed_df = pd.DataFrame(full_fixed, index=corr_df.index, columns=corr_df.columns)
    corr_matrix = full_fixed_df.loc[hub_ids, hub_ids].values

    base_demand = {i: float(chosen.loc[i, "mean_weekly_demand"]) for i in range(n_hubs)}
    cv_by_hub = {i: float(chosen.loc[i, "std_weekly_demand"] / chosen.loc[i, "mean_weekly_demand"])
                 for i in range(n_hubs)}
    hub_meta = {i: {"regate": chosen.loc[i, "regate"], "site": chosen.loc[i, "site"],
                     "lat": chosen.loc[i, "lat"], "lng": chosen.loc[i, "lng"],
                     "n_weeks_observed": chosen.loc[i, "n_weeks_observed"]}
                for i in range(n_hubs)}
    return base_demand, corr_matrix, cv_by_hub, hub_meta


def build_real_data_scenario_tree(n_hubs, seasons=(1, 2, 3, 4), branching=2, weeks_per_season=13,
                                   season_drift=0.3, sibling_drift_correlation=1, seed=42,
                                   data_dir=DEFAULT_DATA_DIR):
    """
    Builds a tree via ScenarioTreeModel.build_toy_scenario_tree, with real
    base demand / real (repaired) correlation / real per-hub coefficient of
    variation substituted in -- season_drift mechanism,
    sibling_drift_correlation, and tree structure are all UNCHANGED from
    the synthetic generator.

    Returns (tree, hub_meta). N = list(range(n_hubs)) throughout, matching
    every other builder in this codebase; hub_meta maps each integer label
    back to the real hub's site ID/name/location for reference/plotting.
    """
    N = list(range(n_hubs))
    base_demand, corr_matrix, cv_by_hub, hub_meta = load_hub_data(n_hubs, data_dir=data_dir)
    tree = build_toy_scenario_tree(
        N, seasons=seasons, branching=branching, weeks_per_season=weeks_per_season,
        base_demand=base_demand, hub_correlation=corr_matrix, noise_frac=cv_by_hub,
        season_drift=season_drift, sibling_drift_correlation=sibling_drift_correlation, seed=seed,
    )
    return tree, hub_meta
