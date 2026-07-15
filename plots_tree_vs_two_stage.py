"""
plots_tree_vs_two_stage.py — Visualization for the multistage vs Static/MNP/MRP comparison
===============================================================================================

Mirrors plots.py's style (same color conventions, same save-to-PNG pattern) but
for the four models compared in compare_tree_vs_two_stage.py:
  * the multistage scenario-tree model (ScenarioTreeVehicleAllocationModel)
  * Model.py's Static / MNP / MRP formulations, solved on the same flattened
    scenario paths

Reuses the *_cost_breakdown() functions and the `results` dict shape from
compare_tree_vs_two_stage.py as the single source of truth for cost figures —
these plotting functions only visualize, they don't recompute costs. Static
and MNP have no `y` (rebalancing) variable at all, so the two rebalancing
plots stay tree-vs-MRP only.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from compare_tree_vs_two_stage import (
    MODEL_LABELS, MODEL_ORDER, K_LABELS, RELEVANT_SCENARIO_COMPONENTS, SCENARIO_COMPONENT_ABBR,
    build_full_horizon_scenarios, leaf_paths, _season_weeks,
)

MODEL_COLORS = {"tree": "#4C72B0", "static": "#DD8452", "mnp": "#55A868", "mrp": "#C44E52"}
MODEL_LINESTYLES = {"tree": "-", "static": ":", "mnp": "-.", "mrp": "--"}

TREE_COLOR = MODEL_COLORS["tree"]
MRP_COLOR = MODEL_COLORS["mrp"]


def plot_compare_objective(results, save=True, output_dir="."):
    """Bar chart of expected total cost across all 4 models, with the VMS-vs-MRP gap annotated."""
    tree_obj = results["tree"]["obj"]
    mrp_obj = results["mrp"]["obj"]
    vms = mrp_obj - tree_obj
    vms_pct = 100 * vms / tree_obj if tree_obj else float("nan")

    objs = [results[m]["obj"] for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    labels = [MODEL_LABELS[m] for m in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(len(MODEL_ORDER)), objs, color=colors, width=0.6)
    for b, v in zip(bars, objs):
        if v is not None:
            ax.annotate(f"{v:,.0f}", xy=(b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Expected total cost")
    ax.set_title(f"Value of Multistage vs MRP: {vms:,.0f}  ({vms_pct:+.2f}%)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_objective.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_cost_breakdown(results, save=True, output_dir="."):
    """Stacked bar chart: purchase / planned sub / redeployment / corrective sub / rebalancing, all 4 models."""
    components = ["purchase", "planned_subcontracting", "redeployment",
                  "corrective_subcontracting", "rebalancing"]
    labels = ["Purchase", "Planned sub.", "Redeployment", "Corrective sub.", "Rebalancing"]
    comp_colors = plt.cm.Accent(np.linspace(0, 1, len(components)))

    fig, ax = plt.subplots(figsize=(8, 6))
    bottoms = [0.0] * len(MODEL_ORDER)
    for comp, label, color in zip(components, labels, comp_colors):
        vals = [results[m]["costs"][comp] for m in MODEL_ORDER]
        ax.bar(range(len(MODEL_ORDER)), vals, bottom=bottoms, color=color, width=0.6, label=label)
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.set_ylabel("Expected cost")
    ax.set_title("Cost Breakdown: Multistage vs Static vs MNP vs MRP")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_cost_breakdown.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_scenario_costs(results, save=True, output_dir="."):
    """
    Box plots of the REALIZED (not probability-weighted) per-scenario cost for
    each recourse component, one subplot per component, one box per model.
    Shows how much planned/corrective subcontracting and rebalancing actually
    vary across demand realizations — information the expected-cost bar
    charts above hide by averaging it away.
    """
    components = ["planned_subcontracting", "redeployment", "corrective_subcontracting", "rebalancing"]
    labels = ["Planned subcontracting", "Redeployment", "Corrective subcontracting", "Rebalancing"]

    fig, axes = plt.subplots(1, len(components), figsize=(19, 5))
    for ax, comp, label in zip(axes, components, labels):
        data = [[sc[comp] for sc in results[m]["scenario_costs"].values()] for m in MODEL_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, m in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_title(label)
        ax.set_ylabel("Realized cost per scenario")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Per-Scenario Cost Distribution (realized, not expected)", fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_scenario_costs.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_subcontracting_quantities(results, save=True, output_dir="."):
    """
    Box plots of the REALIZED subcontracting QUANTITIES (vehicle units, not
    dollars) per scenario, one panel for planned and one for corrective
    subcontracting, one box per model. Complements plot_compare_scenario_costs
    (which shows the same decisions in dollar terms) by showing the actual
    number of vehicles subcontracted.
    """
    panels = ["planned", "corrective"]
    labels = ["Planned subcontracting", "Corrective subcontracting"]

    fig, axes = plt.subplots(1, len(panels), figsize=(11, 5))
    for ax, panel, label in zip(axes, panels, labels):
        data = [[q[panel] for q in results[m]["scenario_quantities"].values()] for m in MODEL_ORDER]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, m in zip(bp["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[m])
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_title(label)
        ax.set_ylabel("Vehicles subcontracted (units)")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Per-Scenario Subcontracting Quantities (realized, not expected)", fontsize=14)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_subcontracting_quantities.png"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_green_ratio(results, save=True, output_dir="."):
    """
    Box plot of the green-vehicle coverage ratio (delivered / required, per
    hub/week/scenario) for each model, with a reference line at ratio=1.0
    (the hard minimum every cell must clear). Shows how much slack each
    model's plan carries above the theta requirement, not just whether it's
    met (it always is, by construction — see report_green_constraint).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [[c["ratio"] for c in results[m]["green_ratios"].values() if not np.isnan(c["ratio"])]
            for m in MODEL_ORDER]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    for patch, m in zip(bp["boxes"], MODEL_ORDER):
        patch.set_facecolor(MODEL_COLORS[m])
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color("black")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Required minimum (ratio = 1.0)")
    ax.set_xticks(range(1, len(MODEL_ORDER) + 1))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Green coverage ratio (delivered / required, log scale)")
    ax.set_title("Green-Vehicle Service-Level Check: coverage vs theta requirement")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3, which="both")
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_green_ratio.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_fleet(results, K, save=True, output_dir="."):
    """Grouped bar chart: fleet purchased per vehicle type, all 4 models."""
    x = np.arange(len(K))
    n_models = len(MODEL_ORDER)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, m in enumerate(MODEL_ORDER):
        vals = [results[m]["fleet"][k] for k in K]
        offset = (idx - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, color=MODEL_COLORS[m], label=MODEL_LABELS[m])

    ax.set_xticks(x)
    ax.set_xticklabels([K_LABELS.get(k, f"Type {k}") for k in K])
    ax.set_ylabel("Vehicles purchased")
    ax.set_title("Fleet Purchased by Type")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(output_dir, "compare_fleet.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_compare_resource_over_time(tree_model, results, N, K, total_weeks,
                                     save=True, output_dir="."):
    """
    Resource level x over the horizon, one FIGURE per vehicle type, one
    subplot per hub within that figure, one color/linestyle per model.

    The tree model has no single time series (its state branches every
    season), so its curve is the *expected* fleet position at each week,
    sampled every week within a block (not just at season starts) so
    within-season rebalancing shows up here too. Static/MNP have no time
    index on x at all, so they're flat lines (results["static"/"mnp"]["resource"]).
    MRP is a genuine weekly series (results["mrp"]["resource"]).

    Saved as compare_resource_over_time_<type>.png, one file per k in K.
    Returns the list of figures created (one per type).
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
    figs = []

    for k in K:
        type_label = K_LABELS.get(k, f"Type {k}")
        fig, axes = plt.subplots(n_hubs, 1, figsize=(max(12, total_weeks * 0.35), 3.5 * n_hubs),
                                  sharex=True)
        if n_hubs == 1:
            axes = [axes]

        for idx_i, i in enumerate(N):
            ax = axes[idx_i]

            tree_weeks, tree_vals = [], []
            for b in stages:
                nodes_b = [n for n in nodes_ge1 if tree.nodes[n].stage == b]
                for t_local in range(1, tree.e[b] + 1):
                    exp_val = sum(tree.nodes[n].prob * tree_model._get_val(f"x[{n},{i},{k},{t_local}]")
                                  for n in nodes_b)
                    tree_weeks.append(week_offset[b] + t_local - 1)
                    tree_vals.append(exp_val)
            ax.plot(tree_weeks, tree_vals, marker="o", markersize=3, linewidth=2,
                    color=MODEL_COLORS["tree"], label=MODEL_LABELS["tree"])

            static_val = results["static"]["resource"][(i, k)]
            ax.plot([0, total_weeks - 1], [static_val, static_val],
                    linestyle=MODEL_LINESTYLES["static"], linewidth=1.5, alpha=0.7,
                    color=MODEL_COLORS["static"], label=MODEL_LABELS["static"])

            mnp_val = results["mnp"]["resource"][(i, k)]
            ax.plot([0, total_weeks - 1], [mnp_val, mnp_val],
                    linestyle=MODEL_LINESTYLES["mnp"], linewidth=1.5, alpha=0.7,
                    color=MODEL_COLORS["mnp"], label=MODEL_LABELS["mnp"])

            mrp_vals = [results["mrp"]["resource"][(i, k, t)] for t in range(total_weeks)]
            ax.plot(range(total_weeks), mrp_vals,
                    linestyle=MODEL_LINESTYLES["mrp"], linewidth=1.5, alpha=0.7,
                    color=MODEL_COLORS["mrp"], label=MODEL_LABELS["mrp"])

            ax.set_ylabel("Resource level (x)")
            ax.set_title(f"Hub {i}")
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Week")
        axes[-1].set_xticks(range(total_weeks))
        fig.suptitle(f"Resource Level Over Time — {type_label}: Multistage vs Static vs MNP vs MRP",
                     fontsize=14, y=1.01)
        plt.tight_layout()
        if save:
            slug = type_label.lower().replace(" ", "_").replace("-", "_")
            fig.savefig(os.path.join(output_dir, f"compare_resource_over_time_{slug}.png"),
                        dpi=150, bbox_inches="tight")
        plt.close(fig)
        figs.append(fig)

    return figs


# ---------------------------------------------------------------------------
# Rebalancing (y) — hub-to-hub transfers. Tree vs MRP only: Static and MNP
# have no y variable at all.
# ---------------------------------------------------------------------------

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


def _mrp_rebalancing_by_week(mrp_model):
    """Expected total rebalancing volume (all hub pairs & types) per week."""
    weekly = {}
    for t in mrp_model.T:
        weekly[t] = sum(
            mrp_model.p_omega(o) * mrp_model._get_val(f"y[{i},{j},{k},{t},{o}]")
            for o in mrp_model.O
            for i in mrp_model.N for j in mrp_model.N if i != j
            for k in mrp_model.K
        )
    return weekly


def plot_compare_rebalancing_over_time(tree_model, mrp_model, save=True, output_dir="."):
    """Total vehicles moved (summed over hub pairs & types) per week, tree vs MRP."""
    tree_weekly, total_weeks = _tree_rebalancing_by_week(tree_model)
    mrp_weekly = _mrp_rebalancing_by_week(mrp_model)

    tree_weeks = sorted(tree_weekly)
    tree_vals = [tree_weekly[w] for w in tree_weeks]
    mrp_weeks = sorted(mrp_weekly)
    mrp_vals = [mrp_weekly[w] for w in mrp_weeks]

    fig, ax = plt.subplots(figsize=(max(12, total_weeks * 0.35), 5))
    ax.plot(tree_weeks, tree_vals, marker="o", markersize=4, linewidth=1.5,
            color=TREE_COLOR, label=MODEL_LABELS["tree"])
    ax.plot(mrp_weeks, mrp_vals, linestyle="--", linewidth=1.5,
            alpha=0.8, color=MRP_COLOR, label=MODEL_LABELS["mrp"])
    ax.set_xlabel("Week")
    ax.set_xticks(range(total_weeks))
    ax.set_ylabel("Vehicles moved (all hub pairs & types)")
    ax.set_title("Rebalancing Volume Over Time: Multistage vs MRP")
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


def _mrp_rebalancing_matrix(mrp_model, N):
    """Expected total rebalancing volume (all periods & types) per (from, to) hub pair."""
    idx = {i: p for p, i in enumerate(N)}
    mat = np.zeros((len(N), len(N)))
    for i in N:
        for j in N:
            if i == j:
                continue
            mat[idx[i], idx[j]] = sum(
                mrp_model.p_omega(o) * mrp_model._get_val(f"y[{i},{j},{k},{t},{o}]")
                for o in mrp_model.O for k in mrp_model.K for t in mrp_model.T
            )
    return mat


def plot_compare_rebalancing_heatmap(tree_model, mrp_model, N, save=True, output_dir="."):
    """Hub x hub heatmap of total expected rebalancing volume, tree vs MRP side by side."""
    tree_mat = _tree_rebalancing_matrix(tree_model, N)
    mrp_mat = _mrp_rebalancing_matrix(mrp_model, N)
    vmax = max(tree_mat.max(), mrp_mat.max(), 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im = None
    for ax, mat, title in zip(axes, [tree_mat, mrp_mat],
                               [MODEL_LABELS["tree"], MODEL_LABELS["mrp"]]):
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
    fig.suptitle("Rebalancing Volume by Hub Pair: Multistage vs MRP", fontsize=14)
    if save:
        fig.savefig(os.path.join(output_dir, "compare_rebalancing_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_demand_by_scenario(d_real, leaf_prob, N, total_weeks, output_dir="."):
    """
    One figure PER scenario, each showing every hub's demand over the full
    horizon for that specific scenario (same underlying data as
    save_flat_scenarios' CSV, just one chart per root-to-leaf path instead of
    one combined table). Saved to <output_dir>/demand_by_scenario/scenario_<o>.png.
    """
    subdir = os.path.join(output_dir, "demand_by_scenario")
    os.makedirs(subdir, exist_ok=True)
    hub_colors = plt.cm.Accent(np.linspace(0, 1, len(N)))

    for o in sorted(leaf_prob):
        fig, ax = plt.subplots(figsize=(max(10, total_weeks * 0.35), 5))
        for idx, i in enumerate(N):
            vals = [d_real[i, t, o] for t in range(total_weeks)]
            ax.plot(range(total_weeks), vals, marker="o", markersize=2, linewidth=1.5,
                    color=hub_colors[idx], label=f"Hub {i}")
        ax.set_xlabel("Week")
        ax.set_xticks(range(total_weeks))
        ax.set_ylabel("Demand")
        ax.set_title(f"Demand Over Time — Scenario {o} (probability = {leaf_prob[o]:.4f})")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(subdir, f"scenario_{o}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Per-scenario demand plots saved to {subdir}/ ({len(leaf_prob)} figures)")


def plot_aggregated_demand_by_scenario(d_real, leaf_prob, N, total_weeks, output_dir="."):
    """
    One figure PER scenario, each showing the AGGREGATED demand (summed
    across all hubs) over the full horizon for that scenario — companion to
    plot_demand_by_scenario's per-hub breakdown. Saved to
    <output_dir>/demand_by_scenario_aggregated/scenario_<o>.png.
    """
    subdir = os.path.join(output_dir, "demand_by_scenario_aggregated")
    os.makedirs(subdir, exist_ok=True)

    for o in sorted(leaf_prob):
        totals = [sum(d_real[i, t, o] for i in N) for t in range(total_weeks)]
        fig, ax = plt.subplots(figsize=(max(10, total_weeks * 0.35), 5))
        ax.plot(range(total_weeks), totals, marker="o", markersize=2, linewidth=1.5, color=TREE_COLOR)
        ax.set_xlabel("Week")
        ax.set_xticks(range(total_weeks))
        ax.set_ylabel("Aggregated demand (all hubs)")
        ax.set_title(f"Aggregated Demand Over Time — Scenario {o} (probability = {leaf_prob[o]:.4f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(os.path.join(subdir, f"scenario_{o}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Per-scenario aggregated demand plots saved to {subdir}/ ({len(leaf_prob)} figures)")


# ---------------------------------------------------------------------------
# Resource-level decomposition, per scenario — Tree & MRP only (Static's
# allocation doesn't vary by week and MNP has no rebalancing, so a per-period
# stacked breakdown isn't informative for those two).
# ---------------------------------------------------------------------------
#
# Each period's resource level is decomposed into how it was built up:
#   base       — level carried over from the previous period (or the initial
#                Delta/x[...,0] purchase, in the very first period)
#   flow       — net redeployment/rebalancing into this period (0 in the
#                first period); can be negative if a hub is a net exporter
#   planned    — planned subcontracting active that period
#   corrective — corrective subcontracting used that period
# base + flow + planned + corrective == the resource level actually used to
# cover that period's demand (x[...,t] + s[...] + stilde/s_corr[...,t]).

COMPONENT_ORDER = ["base", "flow", "planned", "corrective"]
COMPONENT_COLORS = {
    "base": "#B0B0B0",
    "flow": "#4C72B0",
    "planned": "#DD8452",
    "corrective": "#C44E52",
}
COMPONENT_LABELS = {
    "base": "Base (retained from last period)",
    "flow": "Net inflow (redeployment / rebalancing)",
    "planned": "Planned subcontracting",
    "corrective": "Corrective subcontracting",
}
OUTFLOW_LABEL = "Net outflow this period (dashed, already left — not part of bar total)"
RESIDUAL_COLOR = "#2CA02C"
TYPE_HATCHES = ["", "//", "xx", "..", "++"]


def _tree_resource_decomposition(tree_model):
    """{(i, global_week, k, o): {"base", "flow", "planned", "corrective"}}
    for every hub/week/type/scenario, walked leaf by leaf. See module note
    above for what each component means; "flow" is redeployment when it
    crosses a stage boundary (the node's first local week) and rebalancing
    otherwise. Returns (result, n_scenarios)."""
    tree = tree_model.tree
    paths = leaf_paths(tree)
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(paths):
        week_offset = 0
        for idx, n in enumerate(ancestry):
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            parent_id = ancestry[idx - 1]
            L = tree.block_length(n)
            for i in tree_model.N:
                for k in tree_model.K:
                    planned = tree_model._get_val(f"s[{n},{i},{k}]")
                    for t_local in range(1, L + 1):
                        global_week = week_offset + t_local - 1
                        x_val = tree_model._get_val(f"x[{n},{i},{k},{t_local}]")
                        if t_local == 1:
                            if node.stage == 1:
                                base, flow = x_val, 0.0
                            else:
                                parent_L = tree.block_length(parent_id)
                                base = tree_model._get_val(f"x[{parent_id},{i},{k},{parent_L}]")
                                flow = x_val - base
                        else:
                            base = tree_model._get_val(f"x[{n},{i},{k},{t_local - 1}]")
                            flow = x_val - base
                        corrective = tree_model._get_val(f"stilde[{n},{i},{k},{t_local}]")
                        result[i, global_week, k, o] = {
                            "base": base, "flow": flow, "planned": planned, "corrective": corrective,
                        }
            week_offset += L
    return result, len(paths)


def _mrp_resource_decomposition(mrp_model, total_weeks):
    """Same shape as _tree_resource_decomposition, for MRP. x is genuine
    second-stage recourse (see build_model_MRP's docstring), so base/flow
    now vary by scenario too — only planned (s, seasonal/first-stage) stays
    shared across every scenario o. Returns (result, n_scenarios)."""
    _, season_of_week, _ = _season_weeks(mrp_model)
    result = {}
    for i in mrp_model.N:
        for k in mrp_model.K:
            planned_by_t = {t: mrp_model._get_val(f"s[{i},{k},{season_of_week[t]}]")
                             for t in range(total_weeks)}
            for o in mrp_model.O:
                for t in range(total_weeks):
                    x_val = mrp_model._get_val(f"x[{i},{k},{t},{o}]")
                    if t == 0:
                        base, flow = x_val, 0.0
                    else:
                        base = mrp_model._get_val(f"x[{i},{k},{t - 1},{o}]")
                        flow = x_val - base
                    corrective = mrp_model._get_val(f"s_corr[{i},{k},{t},{o}]")
                    result[i, t, k, o] = {
                        "base": base, "flow": flow, "planned": planned_by_t[t], "corrective": corrective,
                    }
    return result, len(mrp_model.O)


def _tree_residual_capacity(tree_model, N, K):
    """{(i, global_week, o): coverage - demand}, mirroring the tree's actual
    demand constraint exactly (ALL vehicle types, including the s[parent]
    netting on stage > 1 nodes — see tree_green_coverage_ratios in
    compare_tree_vs_two_stage.py for the same formula scoped to green types
    only). Always >= 0 for a feasible solve; how much spare capacity, in
    q-weighted units, is left after covering that period's demand."""
    tree = tree_model.tree
    result = {}
    for o, (leaf_id, prob, ancestry) in enumerate(leaf_paths(tree)):
        week_offset = 0
        for idx, n in enumerate(ancestry):
            node = tree.nodes[n]
            if node.stage == 0:
                continue
            parent_id = ancestry[idx - 1]
            L = tree.block_length(n)
            for t_local in range(1, L + 1):
                global_week = week_offset + t_local - 1
                for i in N:
                    coverage = 0.0
                    for k in K:
                        cov_k = (tree_model._get_val(f"x[{n},{i},{k},{t_local}]")
                                 + tree_model._get_val(f"s[{n},{i},{k}]"))
                        if node.stage > 1:
                            cov_k -= tree_model._get_val(f"s[{parent_id},{i},{k}]")
                        cov_k += tree_model._get_val(f"stilde[{n},{i},{k},{t_local}]")
                        coverage += tree_model.q[k] * cov_k
                    result[i, global_week, o] = coverage - node.demand[i, t_local]
            week_offset += L
    return result


def _mrp_residual_capacity(mrp_model, N, K, total_weeks):
    """Same shape as _tree_residual_capacity, mirroring MRP's actual demand
    constraint exactly: no separate flow term — x[i,k,t,o] already carries
    that period's net rebalancing per type via the precedence constraint.
    x is second-stage (per o); s is seasonal/first-stage."""
    _, season_of_week, _ = _season_weeks(mrp_model)
    result = {}
    for i in N:
        for t in range(total_weeks):
            b = season_of_week[t]
            for o in mrp_model.O:
                coverage = sum(
                    mrp_model.q[k] * (mrp_model._get_val(f"x[{i},{k},{t},{o}]")
                                       + mrp_model._get_val(f"s[{i},{k},{b}]")
                                       + mrp_model._get_val(f"s_corr[{i},{k},{t},{o}]"))
                    for k in K
                )
                result[i, t, o] = coverage - mrp_model.d_real[i, t, o]
    return result


def _plot_resource_decomposition_model(model_key, decomp, residual, N, K, total_weeks, n_scenarios, output_dir):
    """One figure per scenario o: one subplot per hub, 3 grouped bars per
    week (one per vehicle type, distinguished by hatch), each bar stacked by
    component (distinguished by color), plus a residual-capacity line
    (secondary axis — total q-weighted capacity across all types minus that
    period's demand, i.e. the demand constraint's slack). Saved to
    <output_dir>/resource_decomposition/<model_key>/scenario_<o>.png."""
    subdir = os.path.join(output_dir, "resource_decomposition", model_key)
    os.makedirs(subdir, exist_ok=True)
    n_hubs = len(N)
    n_types = len(K)
    width = 0.8 / n_types
    weeks = np.arange(total_weeks)

    for o in range(n_scenarios):
        fig, axes = plt.subplots(n_hubs, 1, figsize=(max(12, total_weeks * 0.5), 3.5 * n_hubs),
                                  sharex=True)
        if n_hubs == 1:
            axes = [axes]

        for idx_i, i in enumerate(N):
            ax = axes[idx_i]
            for idx_k, k in enumerate(K):
                offset = (idx_k - (n_types - 1) / 2) * width
                hatch = TYPE_HATCHES[idx_k % len(TYPE_HATCHES)]

                base_arr = np.array([decomp[i, t, k, o]["base"] for t in range(total_weeks)])
                flow_arr = np.array([decomp[i, t, k, o]["flow"] for t in range(total_weeks)])
                planned_arr = np.array([decomp[i, t, k, o]["planned"] for t in range(total_weeks)])
                corrective_arr = np.array([decomp[i, t, k, o]["corrective"] for t in range(total_weeks)])

                # flow can be negative (net outflow). A negative segment
                # stacked directly on a positive one would overpaint rather
                # than shrink the bar, making the total look unchanged. So
                # split flow into non-negative retained/inflow pieces (solid
                # stack, height always == true resource level) and draw net
                # outflow separately as a dashed, unfilled "ghost" segment
                # above the bar that doesn't add to the solid total.
                retained = np.minimum(base_arr, base_arr + flow_arr)
                inflow = np.maximum(flow_arr, 0.0)
                outflow = np.maximum(-flow_arr, 0.0)

                bottoms = np.zeros(total_weeks)
                for comp, heights in [("base", retained), ("flow", inflow),
                                       ("planned", planned_arr), ("corrective", corrective_arr)]:
                    ax.bar(weeks + offset, heights, width, bottom=bottoms,
                           color=COMPONENT_COLORS[comp], hatch=hatch,
                           edgecolor="black", linewidth=0.3)
                    bottoms += heights
                ax.bar(weeks + offset, outflow, width, bottom=bottoms,
                       facecolor="none", edgecolor=COMPONENT_COLORS["flow"],
                       linewidth=1.2, linestyle="--")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_title(f"Hub {i}")
            ax.set_ylabel("Resource level (units)")
            ax.grid(True, axis="y", alpha=0.3)

            residual_vals = [residual[i, t, o] for t in range(total_weeks)]
            ax2 = ax.twinx()
            ax2.plot(weeks, residual_vals, color=RESIDUAL_COLOR, marker="o",
                     markersize=3, linewidth=1.5, zorder=5)
            ax2.axhline(0, color=RESIDUAL_COLOR, linestyle=":", linewidth=0.8, alpha=0.6)
            ax2.set_ylabel("Residual capacity", color=RESIDUAL_COLOR)
            ax2.tick_params(axis="y", labelcolor=RESIDUAL_COLOR)

        axes[-1].set_xlabel("Week")
        axes[-1].set_xticks(weeks)

        component_handles = [mpatches.Patch(facecolor=COMPONENT_COLORS[c], edgecolor="black",
                                             label=COMPONENT_LABELS[c]) for c in COMPONENT_ORDER]
        component_handles.append(mpatches.Patch(facecolor="none", edgecolor=COMPONENT_COLORS["flow"],
                                                 linestyle="--", label=OUTFLOW_LABEL))
        type_handles = [mpatches.Patch(facecolor="white", edgecolor="black",
                                        hatch=TYPE_HATCHES[idx_k % len(TYPE_HATCHES)],
                                        label=K_LABELS.get(k, f"Type {k}"))
                         for idx_k, k in enumerate(K)]
        line_handles = [Line2D([0], [0], color=RESIDUAL_COLOR, marker="o", markersize=4,
                                linewidth=1.5, label="Residual capacity (total capacity − demand)")]
        fig.legend(handles=component_handles, title="Component",
                   loc="upper left", bbox_to_anchor=(1.0, 0.95), fontsize=8)
        fig.legend(handles=type_handles, title="Vehicle type",
                   loc="upper left", bbox_to_anchor=(1.0, 0.55), fontsize=8)
        fig.legend(handles=line_handles, title="Line",
                   loc="upper left", bbox_to_anchor=(1.0, 0.3), fontsize=8)

        fig.suptitle(f"Resource Level Decomposition — {MODEL_LABELS[model_key]}, Scenario {o}", y=1.02)
        plt.tight_layout()
        fig.savefig(os.path.join(subdir, f"scenario_{o}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    return n_scenarios


def plot_resource_decomposition_by_scenario(tree_model, mrp_model, N, K, total_weeks, output_dir="."):
    """Tree & MRP only (see module note above). One figure per (model,
    scenario), saved under resource_decomposition/<tree|mrp>/scenario_<o>.png."""
    tree_decomp, n_tree = _tree_resource_decomposition(tree_model)
    mrp_decomp, n_mrp = _mrp_resource_decomposition(mrp_model, total_weeks)
    tree_residual = _tree_residual_capacity(tree_model, N, K)
    mrp_residual = _mrp_residual_capacity(mrp_model, N, K, total_weeks)

    _plot_resource_decomposition_model("tree", tree_decomp, tree_residual, N, K, total_weeks,
                                        n_tree, output_dir)
    _plot_resource_decomposition_model("mrp", mrp_decomp, mrp_residual, N, K, total_weeks,
                                        n_mrp, output_dir)

    print(f"Resource decomposition plots saved to "
          f"{os.path.join(output_dir, 'resource_decomposition')}/ ({n_tree + n_mrp} figures)")


def plot_all_comparisons(tree_model, mrp_model, results, N, K, total_weeks, output_dir="."):
    """Generate all comparison plots and a text cost summary in output_dir.

    `mrp_model` must be the shared VehicleAllocationModel instance right after
    solve_MRP() was called on it (i.e. still in its MRP-solved state) — the
    rebalancing plots query it live.
    """
    os.makedirs(output_dir, exist_ok=True)

    plot_compare_objective(results, output_dir=output_dir)
    plot_compare_cost_breakdown(results, output_dir=output_dir)
    plot_compare_fleet(results, K, output_dir=output_dir)
    plot_compare_resource_over_time(tree_model, results, N, K, total_weeks, output_dir=output_dir)
    plot_compare_scenario_costs(results, output_dir=output_dir)
    plot_compare_subcontracting_quantities(results, output_dir=output_dir)
    plot_compare_green_ratio(results, output_dir=output_dir)
    plot_compare_rebalancing_over_time(tree_model, mrp_model, output_dir=output_dir)
    plot_compare_rebalancing_heatmap(tree_model, mrp_model, N, output_dir=output_dir)

    d_real, leaf_prob, _ = build_full_horizon_scenarios(tree_model.tree, N)
    plot_demand_by_scenario(d_real, leaf_prob, N, total_weeks, output_dir=output_dir)
    plot_aggregated_demand_by_scenario(d_real, leaf_prob, N, total_weeks, output_dir=output_dir)
    plot_resource_decomposition_by_scenario(tree_model, mrp_model, N, K, total_weeks, output_dir=output_dir)

    tree_obj = results["tree"]["obj"]

    summary_path = os.path.join(output_dir, "cost_summary.txt")
    with open(summary_path, "w") as f:
        f.write("Multistage (Tree) vs Static vs MNP vs MRP — Cost Summary\n")
        f.write("=" * 58 + "\n\n")
        for m in MODEL_ORDER:
            obj = results[m]["obj"]
            f.write(f"{MODEL_LABELS[m]:<20}: {obj:,.2f}\n" if obj is not None
                    else f"{MODEL_LABELS[m]:<20}: N/A\n")
        f.write("\n")
        for m in ["static", "mnp", "mrp"]:
            obj = results[m]["obj"]
            if obj is None or tree_obj is None:
                continue
            vms = obj - tree_obj
            vms_pct = 100 * vms / tree_obj if tree_obj else float("nan")
            f.write(f"VMS vs {MODEL_LABELS[m]:<15}: {vms:,.2f} ({vms_pct:+.2f}%)\n")
        f.write("\n")
        f.write(f"{'Component':<26}" + "".join(f"{MODEL_LABELS[m]:>18}" for m in MODEL_ORDER) + "\n")
        for label, key in [("Purchase", "purchase"),
                           ("Planned subcontracting", "planned_subcontracting"),
                           ("Redeployment", "redeployment"),
                           ("Corrective subcontracting", "corrective_subcontracting"),
                           ("Rebalancing", "rebalancing")]:
            f.write(f"{label:<26}" + "".join(
                f"{results[m]['costs'][key]:>18,.2f}" for m in MODEL_ORDER) + "\n")

        f.write("\n\nPer-scenario cost breakdown (realized, not expected)\n")
        f.write("=" * 58 + "\n")
        for label, key in [("Planned subcontracting", "planned_subcontracting"),
                           ("Redeployment", "redeployment"),
                           ("Corrective subcontracting", "corrective_subcontracting"),
                           ("Rebalancing", "rebalancing")]:
            f.write(f"\n{label}:\n")
            f.write(f"  {'Model':<20}{'mean':>14}{'std':>14}{'min':>14}{'max':>14}\n")
            for m in MODEL_ORDER:
                vals = np.array([sc[key] for sc in results[m]["scenario_costs"].values()])
                f.write(f"  {MODEL_LABELS[m]:<20}{vals.mean():>14,.2f}{vals.std():>14,.2f}"
                        f"{vals.min():>14,.2f}{vals.max():>14,.2f}\n")

        f.write("\nPer-scenario resource cost by model (scenario-varying components only)\n")
        f.write("=" * 78 + "\n")
        col0_w, col_w = 10, 13
        models_shown = [m for m in ["tree", "mnp", "mrp"] if m in RELEVANT_SCENARIO_COMPONENTS]
        header1 = " " * col0_w
        header2 = f"{'o':>{col0_w}}"
        for m in models_shown:
            comps = RELEVANT_SCENARIO_COMPONENTS[m]
            header1 += MODEL_LABELS[m].center(col_w * len(comps))
            for c in comps:
                header2 += f"{SCENARIO_COMPONENT_ABBR[c]:>{col_w}}"
        f.write(header1 + "\n")
        f.write(header2 + "\n")

        n_scenarios = len(results["mrp"]["scenario_costs"])
        for o in range(n_scenarios):
            row = f"{o:>{col0_w}}"
            for m in models_shown:
                for c in RELEVANT_SCENARIO_COMPONENTS[m]:
                    row += f"{results[m]['scenario_costs'][o][c]:>{col_w},.0f}"
            f.write(row + "\n")

        row = f"{'Expected':>{col0_w}}"
        for m in models_shown:
            for c in RELEVANT_SCENARIO_COMPONENTS[m]:
                row += f"{results[m]['costs'][c]:>{col_w},.0f}"
        f.write(row + "\n")

        f.write("\n\nGreen vehicle coverage check (coverage vs theta * demand)\n")
        f.write("=" * 58 + "\n")
        f.write(f"{'Model':<20}{'violations':>12}{'min ratio':>12}{'mean ratio':>12}{'max ratio':>12}\n")
        for m in MODEL_ORDER:
            cells = results[m]["green_ratios"].values()
            violations = sum(1 for c in cells if c["margin"] < -1e-6)
            ratios = np.array([c["ratio"] for c in cells if not np.isnan(c["ratio"])])
            if len(ratios) == 0:
                f.write(f"{MODEL_LABELS[m]:<20}{violations:>12}{'N/A':>12}{'N/A':>12}{'N/A':>12}\n")
            else:
                f.write(f"{MODEL_LABELS[m]:<20}{violations:>12}{ratios.min():>12.3f}"
                        f"{ratios.mean():>12.3f}{ratios.max():>12.3f}\n")
    print(f"Cost summary saved to {summary_path}")

    print(f"Comparison plots saved to {output_dir}/")
