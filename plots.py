"""
plots.py — Visualization of Gurobi solution results for VehicleAllocationModel
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def extract_ST(model):
    """Extract x[i,k,t], s[i,k,t], and y[i,j,k,t,o] from a solved ST model."""
    x = {}
    s = {}
    for k in model.K:
        mat_x = np.zeros((len(model.N), len(model.T)))
        mat_s = np.zeros((len(model.N), len(model.T)))
        for idx_i, i in enumerate(model.N):
            for idx_t, t in enumerate(model.T):
                mat_x[idx_i, idx_t] = model._get_val(f"x[{i},{k},{t}]")
                mat_s[idx_i, idx_t] = model._get_val(f"s[{i},{k},{t}]")
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


def extract_static(model):
    """Extract x[i,k] (constant over time) and s[i,k,t] from a solved static model."""
    x = {}
    s = {}
    for k in model.K:
        mat_x = np.zeros((len(model.N), len(model.T)))
        mat_s = np.zeros((len(model.N), len(model.T)))
        for idx_i, i in enumerate(model.N):
            x_val = model._get_val(f"x[{i},{k}]")
            for idx_t, t in enumerate(model.T):
                mat_x[idx_i, idx_t] = x_val  # constant across time
                mat_s[idx_i, idx_t] = model._get_val(f"s[{i},{k},{t}]")
        x[k] = mat_x
        s[k] = mat_s
    return x, s


def plot_compare_subcontracting(model, st_s, static_s, save=True, output_dir="."):
    """
    Compare subcontracting levels s[i,k,t] between ST and static models.
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
                    color=colors[idx_k], label=f"Type {k} — ST")
            ax.plot(model.T, static_s[k][idx_i, :],
                    marker='x', markersize=3, linewidth=1.5, linestyle='--',
                    color=colors[idx_k], alpha=0.6, label=f"Type {k} — Static")
        ax.set_ylabel("Subcontracting (s)")
        ax.set_title(f"Hub {i}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time period (t)")
    fig.suptitle("Subcontracting Comparison: ST vs Static", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_subcontracting.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare_resource(model, st_x, static_x, save=True, output_dir="."):
    """
    Compare resource levels x between ST (x[i,k,t]) and static (x[i,k], flat line).
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
                    color=colors[idx_k], label=f"Type {k} — ST")
            ax.plot(model.T, static_x[k][idx_i, :],
                    marker='', linewidth=2, linestyle='--',
                    color=colors[idx_k], alpha=0.6, label=f"Type {k} — Static")
        ax.set_ylabel("Resource level (x)")
        ax.set_title(f"Hub {i}")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time period (t)")
    fig.suptitle("Resource Level Comparison: ST vs Static", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_resource.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def extract_ST_costs(model):
    """
    Compute the total cost for each scenario o in the ST model.
    Per scenario: beta*X + gamma*s + gamma_corr*s_corr(o) + alpha*y(o)
    """
    # Fixed costs (same across all scenarios)
    fleet_cost = sum(model.beta[k] * model._get_val(f"X[{k}]") for k in model.K)
    sub_cost = sum(model.gamma[k] * model._get_val(f"s[{i},{k},{t}]")
                   for i in model.N for k in model.K for t in model.T)
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
    Box plot of ST per-scenario costs vs static objective.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.boxplot(st_costs, positions=[1], widths=0.4, patch_artist=True,
                    boxprops=dict(facecolor='#4C72B0', alpha=0.7),
                    medianprops=dict(color='white', linewidth=2))
    ax.scatter([2], [static_obj], color='#C44E52', s=150, zorder=5,
               marker='D', label=f"Static = {static_obj:.0f}")
    ax.axhline(y=static_obj, color='#C44E52', linewidth=1, linestyle='--', alpha=0.5)

    st_median = np.median(st_costs)
    ax.annotate(f"ST median = {st_median:.0f}", xy=(1, st_median),
                xytext=(1.3, st_median), fontsize=9, va='center')

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["ST (per scenario)", "Static"])
    ax.set_ylabel("Total cost")
    ax.set_title("Cost Comparison: ST vs Static")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_costs.png"), dpi=150, bbox_inches="tight")

        # Save cost summary to text file
        summary_path = os.path.join(output_dir, "cost_summary.txt")
        with open(summary_path, "w") as f:
            f.write("Cost Comparison: ST vs Static\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'ST Model (per-scenario costs)':}\n")
            f.write(f"  Mean:    {np.mean(st_costs):.2f}\n")
            f.write(f"  Median:  {np.median(st_costs):.2f}\n")
            f.write(f"  Min:     {np.min(st_costs):.2f}\n")
            f.write(f"  Max:     {np.max(st_costs):.2f}\n")
            f.write(f"  Std:     {np.std(st_costs):.2f}\n")
            f.write(f"  Q25:     {np.percentile(st_costs, 25):.2f}\n")
            f.write(f"  Q75:     {np.percentile(st_costs, 75):.2f}\n\n")
            f.write(f"Static Model\n")
            f.write(f"  Objective: {static_obj:.2f}\n\n")
            f.write(f"Difference (Static - ST mean): {static_obj - np.mean(st_costs):.2f}\n")
            f.write(f"Ratio (ST mean / Static):      {np.mean(st_costs) / static_obj:.4f}\n")
        print(f"Cost summary saved to {summary_path}")

    plt.close(fig)


if __name__ == "__main__":
    from Model import VehicleAllocationModel

    M = VehicleAllocationModel(3, 2, 10, 5, seed=111)
    M.generate_data()

    M.solve_ST(params={"TimeLimit": 500, "MIPGap": 0.01})
    st_x, st_s = extract_ST(M)

    M.solve_static(params={"TimeLimit": 500, "MIPGap": 0.01})
    static_x, static_s = extract_static(M)

    plot_compare_subcontracting(M, st_s, static_s)
    plot_compare_resource(M, st_x, static_x)
