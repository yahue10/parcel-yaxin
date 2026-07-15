"""
plots.py — Visualization of Gurobi solution results for VehicleAllocationModel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _season_weeks(model):
    """{b: [weeks in season b]} for this model, using model.B/model.season_of_week
    if set (real seasons), else one season per week -- see build_model_MRP's
    docstring in Model.py for the same fallback."""
    B = getattr(model, "B", None) or list(model.T)
    season_of_week = getattr(model, "season_of_week", None) or {t: t for t in model.T}
    weeks_in_season = {b: [] for b in B}
    for t in model.T:
        weeks_in_season[season_of_week[t]].append(t)
    return B, season_of_week, weeks_in_season


def extract_MRP(model):
    """Extract x[i,k,t] (expected across scenarios -- x is now genuine
    second-stage recourse, see build_model_MRP's docstring in Model.py),
    s[i,k,t] (expanded from its seasonal s[i,k,b] into one value per week,
    repeated across each week of its season), and y[i,j,k,t,o] from a solved
    MRP model."""
    _, season_of_week, _ = _season_weeks(model)
    x = {}
    s = {}
    for k in model.K:
        mat_x = np.zeros((len(model.N), len(model.T)))
        mat_s = np.zeros((len(model.N), len(model.T)))
        for idx_i, i in enumerate(model.N):
            for idx_t, t in enumerate(model.T):
                mat_x[idx_i, idx_t] = sum(model.p_omega(o) * model._get_val(f"x[{i},{k},{t},{o}]")
                                           for o in model.O)
                mat_s[idx_i, idx_t] = model._get_val(f"s[{i},{k},{season_of_week[t]}]")
        x[k] = mat_x
        s[k] = mat_s

    # Save rebalancing values y[i,j,k,t,o]
    y = {}
    for i in model.N:
        for j in model.N:
            if i == j:
                continue
            for k in model.K:
                for t in model.T:
                    for o in model.O:
                        val = model._get_val(f"y[{i},{j},{k},{t},{o}]")
                        if val > 0.1:
                            y[i, j, k, t, o] = val
    return x, s, y


def extract_MNP(model):
    """Extract x[i,k] (constant over time -- MNP has no rebalancing lever to
    adapt it with) and s[i,k,t] (expanded from its seasonal s[i,k,b]) from a
    solved MNP model."""
    _, season_of_week, _ = _season_weeks(model)
    x = {}
    s = {}
    for k in model.K:
        mat_x = np.zeros((len(model.N), len(model.T)))
        mat_s = np.zeros((len(model.N), len(model.T)))
        for idx_i, i in enumerate(model.N):
            x_val = model._get_val(f"x[{i},{k}]")
            for idx_t, t in enumerate(model.T):
                mat_x[idx_i, idx_t] = x_val  # constant across time
                mat_s[idx_i, idx_t] = model._get_val(f"s[{i},{k},{season_of_week[t]}]")
        x[k] = mat_x
        s[k] = mat_s
    return x, s


def extract_MNP_costs(model):
    """
    Compute the total cost for each scenario o in the MNP model.
    Per scenario: beta*X + gamma*s + gamma_corr*s_corr(o)  (no rebalancing term)
    s is seasonal (s[i,k,b]), cost = gamma[k] * s[i,k,b] * (weeks in season b).
    """
    fleet_cost = sum(model.beta[k] * model._get_val(f"X[{k}]") for k in model.K)
    B, _, weeks_in_season = _season_weeks(model)
    sub_cost = sum(model.gamma[k] * model._get_val(f"s[{i},{k},{b}]") * len(weeks_in_season[b])
                   for i in model.N for k in model.K for b in B)
    fixed = fleet_cost + sub_cost

    costs = []
    for o in model.O:
        corr_cost = sum(model.gamma_corr[k] * model._get_val(f"s_corr[{i},{k},{t},{o}]")
                        for i in model.N for k in model.K for t in model.T)
        costs.append(fixed + corr_cost)
    return np.array(costs)


def extract_static(model):
    """Extract x[i,k] and s[i,k] (both constant over time -- static has no
    time index on either variable) from a solved static model."""
    x = {}
    s = {}
    for k in model.K:
        mat_x = np.zeros((len(model.N), len(model.T)))
        mat_s = np.zeros((len(model.N), len(model.T)))
        for idx_i, i in enumerate(model.N):
            x_val = model._get_val(f"x[{i},{k}]")
            s_val = model._get_val(f"s[{i},{k}]")
            for idx_t, t in enumerate(model.T):
                mat_x[idx_i, idx_t] = x_val  # constant across time
                mat_s[idx_i, idx_t] = s_val  # constant across time
        x[k] = mat_x
        s[k] = mat_s
    return x, s


def plot_compare_subcontracting(model, st_s, static_s, save=True, output_dir="."):
    """
    Compare subcontracting levels s[i,k,t] between MRP and static models.
    One subplot per hub, one line per (type, model).
    """
    n_hubs = len(model.N)
    n_types = len(model.K)
    colors = plt.cm.Accent(np.linspace(0, 1, n_types))

    fig, axes = plt.subplots(n_hubs, 1, figsize=(12, 3.5 * n_hubs), sharex=True)
    if n_hubs == 1:
        axes = [axes]

    for idx_i, i in enumerate(model.N):
        ax = axes[idx_i]
        for idx_k, k in enumerate(model.K):
            ax.plot(model.T, st_s[k][idx_i, :],
                    marker='o', markersize=3, linewidth=1.5,
                    color=colors[idx_k], label=f"Type {k} — MRP")
            ax.plot(model.T, static_s[k][idx_i, :],
                    marker='x', markersize=3, linewidth=1.5, linestyle='--',
                    color=colors[idx_k], alpha=0.6, label=f"Type {k} — Static")
        ax.set_ylabel("Subcontracting (s)")
        ax.set_title(f"Hub {i}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time period (t)")
    fig.suptitle("Subcontracting Comparison: MRP vs Static", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_subcontracting.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_resource(model, st_x, static_x, save=True, output_dir="."):
    """
    Compare resource levels x between MRP (x[i,k,t]) and static (x[i,k], flat line).
    One subplot per hub, one line per (type, model).
    """
    n_hubs = len(model.N)
    n_types = len(model.K)
    colors = plt.cm.Accent(np.linspace(0, 1, n_types))

    fig, axes = plt.subplots(n_hubs, 1, figsize=(12, 3.5 * n_hubs), sharex=True)
    if n_hubs == 1:
        axes = [axes]

    for idx_i, i in enumerate(model.N):
        ax = axes[idx_i]
        for idx_k, k in enumerate(model.K):
            ax.plot(model.T, st_x[k][idx_i, :],
                    marker='o', markersize=3, linewidth=1.5,
                    color=colors[idx_k], label=f"Type {k} — MRP")
            ax.plot(model.T, static_x[k][idx_i, :],
                    marker='', linewidth=2, linestyle='--',
                    color=colors[idx_k], alpha=0.6, label=f"Type {k} — Static")
        ax.set_ylabel("Resource level (x)")
        ax.set_title(f"Hub {i}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time period (t)")
    fig.suptitle("Resource Level Comparison: MRP vs Static", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_resource.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def extract_MRP_costs(model):
    """
    Compute the total cost for each scenario o in the MRP model.
    Per scenario: beta*X + gamma*s + gamma_corr*s_corr(o) + alpha*y(o)
    s is seasonal (s[i,k,b]), cost = gamma[k] * s[i,k,b] * (weeks in season b).
    """
    # Fixed costs (same across all scenarios)
    fleet_cost = sum(model.beta[k] * model._get_val(f"X[{k}]") for k in model.K)
    B, _, weeks_in_season = _season_weeks(model)
    sub_cost = sum(model.gamma[k] * model._get_val(f"s[{i},{k},{b}]") * len(weeks_in_season[b])
                   for i in model.N for k in model.K for b in B)
    fixed = fleet_cost + sub_cost

    costs = []
    for o in model.O:
        corr_cost = sum(model.gamma_corr[k] * model._get_val(f"s_corr[{i},{k},{t},{o}]")
                        for i in model.N for k in model.K for t in model.T)
        rebal_cost = sum(model.alpha[i, j, k] * model._get_val(f"y[{i},{j},{k},{t},{o}]")
                         for i in model.N for j in model.N for k in model.K for t in model.T)
        costs.append(fixed + corr_cost + rebal_cost)
    return np.array(costs)


def plot_compare_costs(model, st_costs, static_obj, save=True, output_dir="."):
    """
    Box plot of MRP per-scenario costs vs static objective.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(st_costs, positions=[1], widths=0.4, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                    medianprops=dict(color='white', linewidth=2))
    ax.scatter([2], [static_obj], color='#C44E52', s=150, zorder=5,
               marker='D', label=f"Static = {static_obj:.0f}")
    ax.axhline(y=static_obj, color='#C44E52', linewidth=1, linestyle='--', alpha=0.5)

    st_median = np.median(st_costs)
    ax.annotate(f"MRP median = {st_median:.0f}", xy=(1, st_median),
                xytext=(1.3, st_median), fontsize=9, va='center')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["MRP (per scenario)", "Static"])
    ax.set_ylabel("Total cost")
    ax.set_title("Cost Comparison: MRP vs Static")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_costs.png"), dpi=150, bbox_inches="tight")

        # Save cost summary to text file
        summary_path = os.path.join(output_dir, "cost_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Cost Comparison: MRP vs Static\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'MRP Model (per-scenario costs)':}\n")
            f.write(f"  Mean:    {np.mean(st_costs):.2f}\n")
            f.write(f"  Median:  {np.median(st_costs):.2f}\n")
            f.write(f"  Min:     {np.min(st_costs):.2f}\n")
            f.write(f"  Max:     {np.max(st_costs):.2f}\n")
            f.write(f"  Std:     {np.std(st_costs):.2f}\n")
            f.write(f"  Q25:     {np.percentile(st_costs, 25):.2f}\n")
            f.write(f"  Q75:     {np.percentile(st_costs, 75):.2f}\n\n")
            f.write(f"Static Model\n")
            f.write(f"  Objective: {static_obj:.2f}\n\n")
            f.write(f"Difference (Static - MRP mean): {static_obj - np.mean(st_costs):.2f}\n")
            f.write(f"Ratio (MRP mean / Static):      {np.mean(st_costs) / static_obj:.4f}\n")
        print(f"Cost summary saved to {summary_path}")

    plt.close(fig)


def plot_compare_subcontracting_3way(model, st_s, mnp_s, static_s, save=True, output_dir="."):
    """
    Compare subcontracting s[i,k,t] across MRP, MNP, and Static models.
    One subplot per hub.
    """
    n_hubs = len(model.N)
    n_types = len(model.K)
    colors = plt.cm.Accent(np.linspace(0, 1, n_types))

    fig, axes = plt.subplots(n_hubs, 1, figsize=(12, 3.5 * n_hubs), sharex=True)
    if n_hubs == 1:
        axes = [axes]

    for idx_i, i in enumerate(model.N):
        ax = axes[idx_i]
        for idx_k, k in enumerate(model.K):
            ax.plot(model.T, st_s[k][idx_i, :],
                    marker='o', markersize=3, linewidth=1.5,
                    color=colors[idx_k], label=f"Type {k} — MRP")
            ax.plot(model.T, mnp_s[k][idx_i, :],
                    marker='s', markersize=3, linewidth=1.5, linestyle='-.',
                    color=colors[idx_k], alpha=0.8, label=f"Type {k} — MNP")
            ax.plot(model.T, static_s[k][idx_i, :],
                    marker='x', markersize=3, linewidth=1.5, linestyle='--',
                    color=colors[idx_k], alpha=0.5, label=f"Type {k} — Static")
        ax.set_ylabel("Subcontracting (s)")
        ax.set_title(f"Hub {i}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time period (t)")
    fig.suptitle("Subcontracting Comparison: MRP vs MNP vs Static", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_subcontracting_3way.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_resource_3way(model, st_x, mnp_x, static_x, save=True, output_dir="."):
    """
    Compare resource allocation x across MRP, MNP, and Static models.
    One subplot per hub.
    """
    n_hubs = len(model.N)
    n_types = len(model.K)
    colors = plt.cm.Accent(np.linspace(0, 1, n_types))

    fig, axes = plt.subplots(n_hubs, 1, figsize=(12, 3.5 * n_hubs), sharex=True)
    if n_hubs == 1:
        axes = [axes]

    for idx_i, i in enumerate(model.N):
        ax = axes[idx_i]
        for idx_k, k in enumerate(model.K):
            ax.plot(model.T, st_x[k][idx_i, :],
                    marker='o', markersize=3, linewidth=1.5,
                    color=colors[idx_k], label=f"Type {k} — MRP")
            ax.plot(model.T, mnp_x[k][idx_i, :],
                    marker='', linewidth=2, linestyle='-.',
                    color=colors[idx_k], alpha=0.8, label=f"Type {k} — MNP")
            ax.plot(model.T, static_x[k][idx_i, :],
                    marker='', linewidth=2, linestyle='--',
                    color=colors[idx_k], alpha=0.5, label=f"Type {k} — Static")
        ax.set_ylabel("Resource level (x)")
        ax.set_title(f"Hub {i}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time period (t)")
    fig.suptitle("Resource Level Comparison: MRP vs MNP vs Static", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_resource_3way.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_costs_3way(model, st_costs, mnp_costs, static_obj, save=True, output_dir="."):
    """
    Box plot of per-scenario costs for MRP and MNP, vs the static objective value.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.boxplot(st_costs, positions=[1], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor='#4C72B0', alpha=0.7),
               medianprops=dict(color='white', linewidth=2))
    ax.boxplot(mnp_costs, positions=[2], widths=0.4, patch_artist=True,
               boxprops=dict(facecolor='#55A868', alpha=0.7),
               medianprops=dict(color='white', linewidth=2))
    ax.scatter([3], [static_obj], color='#C44E52', s=150, zorder=5,
               marker='D', label=f"Static = {static_obj:.0f}")
    ax.axhline(y=static_obj, color='#C44E52', linewidth=1, linestyle='--', alpha=0.5)

    st_median = np.median(st_costs)
    mnp_median = np.median(mnp_costs)
    ax.annotate(f"MRP med = {st_median:.0f}", xy=(1, st_median),
                xytext=(1.15, st_median), fontsize=9, va='center')
    ax.annotate(f"MNP med = {mnp_median:.0f}", xy=(2, mnp_median),
                xytext=(2.15, mnp_median), fontsize=9, va='center')

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["MRP (per scenario)", "MNP (per scenario)", "Static"])
    ax.set_ylabel("Total cost")
    ax.set_title("Cost Comparison: MRP vs MNP vs Static")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_costs_3way.png"), dpi=150, bbox_inches="tight")

        summary_path = os.path.join(output_dir, "cost_summary_3way.txt")
        with open(summary_path, "w") as f:
            f.write("Cost Comparison: MRP vs MNP vs Static\n")
            f.write("=" * 50 + "\n\n")
            for label, costs in [("MRP", st_costs), ("MNP", mnp_costs)]:
                f.write(f"{label} Model (per-scenario costs):\n")
                f.write(f"  Mean:    {np.mean(costs):.2f}\n")
                f.write(f"  Median:  {np.median(costs):.2f}\n")
                f.write(f"  Min:     {np.min(costs):.2f}\n")
                f.write(f"  Max:     {np.max(costs):.2f}\n")
                f.write(f"  Std:     {np.std(costs):.2f}\n\n")
            f.write(f"Static Model\n  Objective: {static_obj:.2f}\n\n")
            f.write(f"Difference Static - MRP mean:  {static_obj - np.mean(st_costs):.2f}\n")
            f.write(f"Difference Static - MNP mean: {static_obj - np.mean(mnp_costs):.2f}\n")
            f.write(f"Difference MNP mean - MRP mean: {np.mean(mnp_costs) - np.mean(st_costs):.2f}\n")
        print(f"3-way cost summary saved to {summary_path}")

    plt.close(fig)


if __name__ == "__main__":
    from Model import VehicleAllocationModel

    M = VehicleAllocationModel(3, 2, 10, 5, seed=111)
    M.generate_data()

    M.solve_MRP(params={"TimeLimit": 500, "MIPGap": 0.01})
    st_x, st_s = extract_MRP(M)

    M.solve_static(params={"TimeLimit": 500, "MIPGap": 0.01})
    static_x, static_s = extract_static(M)

    plot_compare_subcontracting(M, st_s, static_s)
    plot_compare_resource(M, st_x, static_x)
