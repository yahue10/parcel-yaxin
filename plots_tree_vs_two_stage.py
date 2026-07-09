"""
plots_tree_vs_two_stage.py — Visualization for the multistage vs two-stage comparison
=========================================================================================

Mirrors plots.py's style (same color conventions, same save-to-PNG pattern) but
for the two models compared in compare_tree_vs_two_stage.py:
  * the multistage scenario-tree model (ScenarioTreeVehicleAllocationModel)
  * the two-stage MRP model (VehicleAllocationModel.build_model_MRP), solved
    on the same flattened scenario paths

Reuses tree_cost_breakdown() / two_stage_cost_breakdown() from
compare_tree_vs_two_stage.py as the single source of truth for cost figures —
these plotting functions only visualize, they don't recompute costs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from compare_tree_vs_two_stage import tree_cost_breakdown, two_stage_cost_breakdown

TREE_COLOR = "#4C72B0"
TWO_STAGE_COLOR = "#C44E52"


def plot_compare_objective(tree_model, two_stage_model, save=True, output_dir="."):
    """Bar chart of expected total cost, tree vs two-stage, with the VMS gap annotated."""
    tree_obj = tree_model.model.ObjVal
    two_stage_obj = two_stage_model.model.ObjVal
    vms = two_stage_obj - tree_obj
    vms_pct = 100 * vms / tree_obj if tree_obj else float("nan")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([0, 1], [tree_obj, two_stage_obj],
                  color=[TREE_COLOR, TWO_STAGE_COLOR], width=0.5)
    for b, v in zip(bars, [tree_obj, two_stage_obj]):
        ax.annotate(f"{v:,.0f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                    textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Multistage\n(tree)", "Two-stage\n(MRP)"])
    ax.set_ylabel("Expected total cost")
    ax.set_title(f"Value of Multistage (VMS): {vms:,.0f}  ({vms_pct:+.2f}%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_objective.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_cost_breakdown(tree_model, two_stage_model, save=True, output_dir="."):
    """Stacked bar chart: purchase / planned sub / corrective sub / rebalancing, tree vs two-stage."""
    tb = tree_cost_breakdown(tree_model)
    sb = two_stage_cost_breakdown(two_stage_model)
    components = ["purchase", "planned_subcontracting", "corrective_subcontracting", "rebalancing"]
    labels = ["Purchase", "Planned sub.", "Corrective sub.", "Rebalancing"]
    colors = plt.cm.Accent(np.linspace(0, 1, len(components)))

    fig, ax = plt.subplots(figsize=(7, 6))
    bottom_tree = bottom_two = 0.0
    for comp, label, color in zip(components, labels, colors):
        ax.bar(0, tb[comp], bottom=bottom_tree, color=color, width=0.5, label=label)
        ax.bar(1, sb[comp], bottom=bottom_two, color=color, width=0.5)
        bottom_tree += tb[comp]
        bottom_two += sb[comp]

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Multistage\n(tree)", "Two-stage\n(MRP)"])
    ax.set_ylabel("Expected cost")
    ax.set_title("Cost Breakdown: Multistage vs Two-Stage")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_cost_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_fleet(tree_model, two_stage_model, N, K, save=True, output_dir="."):
    """Grouped bar chart: fleet purchased per vehicle type, tree (sum_i Delta[i,k]) vs two-stage (X[k])."""
    tree_fleet = [sum(tree_model._get_val(f"Delta[{i},{k}]") for i in N) for k in K]
    two_stage_fleet = [two_stage_model._get_val(f"X[{k}]") for k in K]

    x = np.arange(len(K))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, tree_fleet, width, color=TREE_COLOR, label="Multistage (tree)")
    ax.bar(x + width / 2, two_stage_fleet, width, color=TWO_STAGE_COLOR, label="Two-stage (MRP)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Type {k}" for k in K])
    ax.set_ylabel("Vehicles purchased")
    ax.set_title("Fleet Purchased by Type")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_fleet.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_resource_over_time(tree_model, two_stage_model, N, K, total_weeks,
                                     save=True, output_dir="."):
    """
    Resource level x over the horizon, one subplot per hub.

    The tree model has no single time series (its state branches every
    season), so its curve is the *expected* fleet position at each week:
    E[x(t)] = sum over stage-b(t) nodes of node.prob * x[n,i,k,t_local], sampled
    every week within a block (not just at season starts) so within-season
    rebalancing shows up here too. The two-stage model's x[i,k,t] is a genuine
    weekly series, plotted as-is for comparison.
    """
    tree = tree_model.tree
    nodes_ge1 = tree.nodes_with_stage_ge1()
    stages = sorted(set(tree.nodes[n].stage for n in nodes_ge1))

    week_offset = {}
    offset = 0
    for b in stages:
        week_offset[b] = offset
        offset += tree.e[b]

    n_hubs = len(N)
    n_types = len(K)
    colors = plt.cm.Accent(np.linspace(0, 1, n_types))

    fig, axes = plt.subplots(n_hubs, 1, figsize=(12, 3.5 * n_hubs), sharex=True)
    if n_hubs == 1:
        axes = [axes]

    for idx_i, i in enumerate(N):
        ax = axes[idx_i]
        for idx_k, k in enumerate(K):
            tree_weeks, tree_vals = [], []
            for b in stages:
                nodes_b = [n for n in nodes_ge1 if tree.nodes[n].stage == b]
                for t_local in range(1, tree.e[b] + 1):
                    exp_val = sum(tree.nodes[n].prob * tree_model._get_val(f"x[{n},{i},{k},{t_local}]")
                                  for n in nodes_b)
                    tree_weeks.append(week_offset[b] + t_local - 1)
                    tree_vals.append(exp_val)
            ax.plot(tree_weeks, tree_vals, marker="o", markersize=3, linewidth=2,
                    color=colors[idx_k], label=f"Type {k} — Multistage (expected)")

            two_stage_vals = [two_stage_model._get_val(f"x[{i},{k},{t}]") for t in range(total_weeks)]
            ax.plot(range(total_weeks), two_stage_vals, linestyle="--", linewidth=1.5,
                    alpha=0.7, color=colors[idx_k], label=f"Type {k} — Two-stage")

        ax.set_ylabel("Resource level (x)")
        ax.set_title(f"Hub {i}")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Week")
    fig.suptitle("Resource Level Over Time: Multistage (expected) vs Two-Stage", fontsize=14, y=1.01)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_resource_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Rebalancing (y) — hub-to-hub transfers
# ---------------------------------------------------------------------------
#
# y has no natural "one line per hub" reading like x does — it's inherently a
# (from-hub, to-hub, week) quantity. Two views: total moved volume per week
# (to see WHEN rebalancing happens) and a hub x hub heatmap of total volume
# (to see WHERE it flows).

def _tree_rebalancing_by_week(tree_model):
    """Expected total rebalancing volume (all hub pairs & types) per global week."""
    tree = tree_model.tree
    nodes_ge1 = tree.nodes_with_stage_ge1()
    stages = sorted(set(tree.nodes[n].stage for n in nodes_ge1))

    week_offset = {}
    offset = 0
    for b in stages:
        week_offset[b] = offset
        offset += tree.e[b]
    total_weeks = offset

    weekly = {}
    for b in stages:
        nodes_b = [n for n in nodes_ge1 if tree.nodes[n].stage == b]
        for t_local in range(1, tree.e[b] + 1):
            w = week_offset[b] + t_local - 1
            weekly[w] = sum(
                tree.nodes[n].prob * tree_model._get_val(f"y[{n},{i},{j},{k},{t_local}]")
                for n in nodes_b for (i, j) in tree_model.A for k in tree_model.K
            )
    return weekly, total_weeks


def _two_stage_rebalancing_by_week(two_stage_model):
    """Expected total rebalancing volume (all hub pairs & types) per week."""
    weekly = {}
    for t in two_stage_model.T:
        weekly[t] = sum(
            two_stage_model.p_omega(o) * two_stage_model._get_val(f"y[{i},{j},{k},{t},{o}]")
            for o in two_stage_model.O
            for i in two_stage_model.N for j in two_stage_model.N if i != j
            for k in two_stage_model.K
        )
    return weekly


def plot_compare_rebalancing_over_time(tree_model, two_stage_model, save=True, output_dir="."):
    """Total vehicles moved (summed over hub pairs & types) per week, tree vs two-stage."""
    tree_weekly, total_weeks = _tree_rebalancing_by_week(tree_model)
    two_stage_weekly = _two_stage_rebalancing_by_week(two_stage_model)

    tree_weeks = sorted(tree_weekly)
    tree_vals = [tree_weekly[w] for w in tree_weeks]
    two_stage_weeks = sorted(two_stage_weekly)
    two_stage_vals = [two_stage_weekly[w] for w in two_stage_weeks]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(tree_weeks, tree_vals, marker="o", markersize=4, linewidth=1.5,
            color=TREE_COLOR, label="Multistage (expected)")
    ax.plot(two_stage_weeks, two_stage_vals, linestyle="--", linewidth=1.5,
            alpha=0.8, color=TWO_STAGE_COLOR, label="Two-stage (expected)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Vehicles moved (all hub pairs & types)")
    ax.set_title("Rebalancing Volume Over Time: Multistage vs Two-Stage")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_rebalancing_over_time.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def _tree_rebalancing_matrix(tree_model, N):
    """Expected total rebalancing volume (all weeks & types) per (from, to) hub pair."""
    tree = tree_model.tree
    nodes_ge1 = tree.nodes_with_stage_ge1()
    idx = {i: p for p, i in enumerate(N)}
    mat = np.zeros((len(N), len(N)))
    for n in nodes_ge1:
        prob = tree.nodes[n].prob
        weeks = range(1, tree.block_length(n) + 1)
        for (i, j) in tree_model.A:
            total = sum(tree_model._get_val(f"y[{n},{i},{j},{k},{t}]")
                        for k in tree_model.K for t in weeks)
            mat[idx[i], idx[j]] += prob * total
    return mat


def _two_stage_rebalancing_matrix(two_stage_model, N):
    """Expected total rebalancing volume (all periods & types) per (from, to) hub pair."""
    idx = {i: p for p, i in enumerate(N)}
    mat = np.zeros((len(N), len(N)))
    for i in N:
        for j in N:
            if i == j:
                continue
            mat[idx[i], idx[j]] = sum(
                two_stage_model.p_omega(o) * two_stage_model._get_val(f"y[{i},{j},{k},{t},{o}]")
                for o in two_stage_model.O for k in two_stage_model.K for t in two_stage_model.T
            )
    return mat


def plot_compare_rebalancing_heatmap(tree_model, two_stage_model, N, save=True, output_dir="."):
    """Hub x hub heatmap of total expected rebalancing volume, tree vs two-stage side by side."""
    tree_mat = _tree_rebalancing_matrix(tree_model, N)
    two_stage_mat = _two_stage_rebalancing_matrix(two_stage_model, N)
    vmax = max(tree_mat.max(), two_stage_mat.max(), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = None
    for ax, mat, title in zip(axes, [tree_mat, two_stage_mat],
                               ["Multistage (expected)", "Two-stage (expected)"]):
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(N)))
        ax.set_yticks(range(len(N)))
        ax.set_xticklabels([f"Hub {j}" for j in N])
        ax.set_yticklabels([f"Hub {i}" for i in N])
        ax.set_xlabel("To")
        ax.set_ylabel("From")
        ax.set_title(title)
        for pi in range(len(N)):
            for pj in range(len(N)):
                if pi == pj:
                    continue
                val = mat[pi, pj]
                color = "white" if val > vmax * 0.6 else "black"
                ax.text(pj, pi, f"{val:,.0f}", ha="center", va="center", fontsize=9, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Vehicles moved (total, all periods & types)")
    fig.suptitle("Rebalancing Volume by Hub Pair: Multistage vs Two-Stage", fontsize=14)
    if save:
        fig.savefig(os.path.join(output_dir, "compare_rebalancing_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_all_comparisons(tree_model, two_stage_model, N, K, total_weeks, output_dir="."):
    """Generate all comparison plots and a text cost summary in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    plot_compare_objective(tree_model, two_stage_model, output_dir=output_dir)
    plot_compare_cost_breakdown(tree_model, two_stage_model, output_dir=output_dir)
    plot_compare_fleet(tree_model, two_stage_model, N, K, output_dir=output_dir)
    plot_compare_resource_over_time(tree_model, two_stage_model, N, K, total_weeks, output_dir=output_dir)
    plot_compare_rebalancing_over_time(tree_model, two_stage_model, output_dir=output_dir)
    plot_compare_rebalancing_heatmap(tree_model, two_stage_model, N, output_dir=output_dir)

    tree_obj = tree_model.model.ObjVal
    two_stage_obj = two_stage_model.model.ObjVal
    vms = two_stage_obj - tree_obj
    tb = tree_cost_breakdown(tree_model)
    sb = two_stage_cost_breakdown(two_stage_model)

    summary_path = os.path.join(output_dir, "cost_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Multistage (Tree) vs Two-Stage (MRP) — Cost Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Multistage objective : {tree_obj:,.2f}\n")
        f.write(f"Two-stage objective  : {two_stage_obj:,.2f}\n")
        f.write(f"VMS (two-stage - tree): {vms:,.2f} "
                f"({100 * vms / tree_obj if tree_obj else float('nan'):+.2f}%)\n\n")
        f.write(f"{'Component':<28}{'Tree':>18}{'Two-stage':>18}\n")
        for label, key in [("Purchase", "purchase"),
                           ("Planned subcontracting", "planned_subcontracting"),
                           ("Corrective subcontracting", "corrective_subcontracting"),
                           ("Rebalancing", "rebalancing")]:
            f.write(f"{label:<28}{tb[key]:>18,.2f}{sb[key]:>18,.2f}\n")
    print(f"Cost summary saved to {summary_path}")

    print(f"Comparison plots saved to {output_dir}/")
