"""
ScenarioTreeModel.py — Multistage (scenario-tree) Vehicle Allocation Model
===========================================================================

Implements the scenario-tree formulation: nodes n in a tree N carry their own
e(b(n))-week block of (x, s~, y) plus a node-level planned-subcontracting
decision s(n). Non-anticipativity is structural (one variable per node) rather
than enforced with explicit NA constraints.

Two independent pieces:
  * ScenarioTree / ScenarioTreeNode — a plain data structure for the tree.
    You don't need real data to use this file: `build_toy_scenario_tree()`
    fabricates a small tree (fake demand, geometric branching) so the Gurobi
    model can be built and solved today. Swap in the real tree later via
    `ScenarioTree.from_records(...)` — same shape, no model code changes.
  * ScenarioTreeVehicleAllocationModel — builds/solves the Gurobi model in
    Model.py's style (`_get_val` lookup by var name, `solve(...)` with a
    Gurobi Env/params dict).

Note on the objective's corrective/rebalancing sums: the write-up sums
t=1..13 explicitly; here we sum t=1..e(b(n)) for the node instead, since
that's the block actually decided at n. If every season really is a fixed
13-week block this is equivalent; adjust `e` if not.
"""

from gurobipy import Model, GRB, Env
import random
import numpy as np


# ---------------------------------------------------------------------------
# Scenario tree data structure
# ---------------------------------------------------------------------------

class ScenarioTreeNode:
    def __init__(self, node_id, parent_id, stage, prob, demand=None):
        self.id = node_id
        self.parent_id = parent_id      # None for the root
        self.stage = stage              # b(n) in {0, 1, ..., B}
        self.prob = prob                # pi(n): unconditional probability of reaching n
        self.demand = demand or {}      # {(i, t): d_it(n)} for t = 1..e(stage)
        self.children = []              # filled in by ScenarioTree.add_child


class ScenarioTree:
    """
    e : dict {season b: number of weeks e(b) in that season's block}, b >= 1.
    """

    def __init__(self, e):
        self.e = e
        self.nodes = {}
        self.root_id = None

    def add_root(self, node_id="root"):
        node = ScenarioTreeNode(node_id, parent_id=None, stage=0, prob=1.0)
        self.nodes[node_id] = node
        self.root_id = node_id
        return node

    def add_child(self, node_id, parent_id, stage, prob, demand):
        parent = self.nodes[parent_id]
        if stage != parent.stage + 1:
            raise ValueError(f"node {node_id}: stage {stage} must be parent stage {parent.stage} + 1")
        node = ScenarioTreeNode(node_id, parent_id, stage, prob, demand)
        self.nodes[node_id] = node
        parent.children.append(node_id)
        return node

    @classmethod
    def from_records(cls, e, records):
        """
        records : iterable of dicts, each with keys
            'id', 'parent_id' (None for root), 'stage', 'prob', 'demand'
            (demand omitted/None for the root). Records must be given in an
            order where each parent appears before its children.
        """
        tree = cls(e)
        for rec in records:
            if rec["parent_id"] is None:
                node = ScenarioTreeNode(rec["id"], None, rec.get("stage", 0), rec.get("prob", 1.0))
                tree.nodes[rec["id"]] = node
                tree.root_id = rec["id"]
            else:
                tree.add_child(rec["id"], rec["parent_id"], rec["stage"], rec["prob"], rec["demand"])
        return tree

    def block_length(self, node_id):
        return self.e[self.nodes[node_id].stage]

    def nodes_with_stage_ge1(self):
        return [nid for nid, nd in self.nodes.items() if nd.stage >= 1]

    def leaves(self):
        return [nid for nid, nd in self.nodes.items() if not nd.children]

    def validate(self, N):
        """Sanity-check structure: stage progression, probabilities, demand coverage."""
        if self.root_id is None or self.nodes[self.root_id].stage != 0:
            raise ValueError("tree has no valid root (stage 0)")
        for nid, nd in self.nodes.items():
            if nd.stage == 0:
                continue
            expected_weeks = set(range(1, self.e[nd.stage] + 1))
            missing = {(i, t) for i in N for t in expected_weeks} - set(nd.demand.keys())
            if missing:
                raise ValueError(f"node {nid}: missing demand entries {sorted(missing)[:5]}...")
        for nid, nd in self.nodes.items():
            if nd.children:
                total = sum(self.nodes[c].prob for c in nd.children)
                if abs(total - nd.prob) > 1e-6:
                    raise ValueError(
                        f"node {nid}: children probabilities sum to {total}, expected {nd.prob}"
                    )
        return True


def build_toy_scenario_tree(N, seasons=(1, 2, 3, 4), branching=2, weeks_per_season=13,
                             base_mean_range=(10000, 30000), season_drift=0.3,
                             noise_frac=0.02, hub_correlation=None,
                             sibling_drift_correlation=1, seed=42):
    """
    Demand per node is a random-walk perturbation of the parent's demand
    level; branching is uniform (equal child probabilities) at every stage.

    hub_correlation : optional len(N) x len(N) correlation matrix (nested
        list/array, indexed by position in N, symmetric, unit diagonal).
        None (default) means hubs move independently, same as before. E.g.
        for N = [0, 1, 2] with hub 0 and 1 strongly opposed and hub 2
        independent of both:
            [[ 1.0, -0.8,  0.0],
             [-0.8,  1.0,  0.0],
             [ 0.0,  0.0,  1.0]]
        Applies to both the season-to-season drift and the within-season
        weekly noise (see module notes). Both shocks are drawn from a
        correlated multivariate normal rather than the previous independent
        uniform/gauss draws, so season_drift/noise_frac are now read as a
        standard-deviation scale rather than a uniform half-width.

    sibling_drift_correlation : float in [0, 1], default 0.0 (no change from
        previous behavior). Controls how much the SEASON-DRIFT shock (the
        shock that sets a child's season-level demand relative to its
        parent — not the within-season weekly noise) is shared across
        siblings, i.e. children of the same parent. Each child's drift shock
        is built as sqrt(rho)*shared + sqrt(1-rho)*own, where `shared` is
        drawn once per parent and `own` is drawn independently per child (both
        already hub-correlated via hub_correlation). rho=0: siblings drift
        fully independently (today's behavior) — one may trend up while
        another trends down. rho=1: siblings get the identical drift shock
        (same direction AND magnitude that season; they only start to differ
        via the still-independent weekly noise). Values in between make
        siblings *tend* to move the same direction without forcing it.

    season_drift : float, dict, or (low, high) tuple.
        - float (default): the same standard-deviation scale is used for
          every season's drift shock, as before.
        - dict {season b: drift value}: an explicit, different drift
          magnitude per season, e.g. {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.5} for a
          system that grows more volatile in later seasons. Must have an
          entry for every season in `seasons`.
        - (low, high) tuple: draws ONE random drift value per season,
          uniformly from [low, high), using this function's own seeded RNG
          (so it's reproducible for a given seed, but varies season to
          season) — e.g. (0.1, 0.5).
    """
    rng = random.Random(seed)
    rng_np = np.random.default_rng(seed)

    if isinstance(season_drift, dict):
        missing = set(seasons) - set(season_drift)
        if missing:
            raise ValueError(f"season_drift dict is missing entries for season(s) {sorted(missing)}")
        drift_by_season = dict(season_drift)
    elif isinstance(season_drift, tuple):
        lo, hi = season_drift
        drift_by_season = {b: rng.uniform(lo, hi) for b in seasons}
    else:
        drift_by_season = {b: season_drift for b in seasons}

    n = len(N)
    corr = np.eye(n) if hub_correlation is None else np.asarray(hub_correlation, dtype=float)
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        raise ValueError("hub_correlation must be a valid (symmetric, positive-semidefinite) "
                          "correlation matrix")

    def correlated_shocks():
        """One correlated standard-normal shock per hub, in N's order."""
        return chol @ rng_np.standard_normal(n)

    rho = sibling_drift_correlation

    e = {b: weeks_per_season for b in seasons}
    tree = ScenarioTree(e)
    root = tree.add_root()

    base = {i: rng.uniform(*base_mean_range) for i in N}
    frontier = [(root.id, base)]
    next_id = 1

    for b in seasons:
        new_frontier = []
        for parent_id, parent_level in frontier:
            parent = tree.nodes[parent_id]
            child_prob = parent.prob / branching
            shared_drift = correlated_shocks() if rho > 0 else None
            for _ in range(branching):
                node_id = next_id
                next_id += 1
                own_drift = correlated_shocks()
                z_drift = (own_drift if rho <= 0
                           else np.sqrt(rho) * shared_drift + np.sqrt(1 - rho) * own_drift)
                level = {i: max(0.0, parent_level[i] * (1 + drift_by_season[b] * z_drift[idx]))
                         for idx, i in enumerate(N)}
                demand = {}
                for t in range(1, weeks_per_season + 1):
                    z_noise = correlated_shocks()
                    for idx, i in enumerate(N):
                        demand[i, t] = int(round(max(0.0, level[i] + noise_frac * level[i] * z_noise[idx])))
                tree.add_child(node_id, parent_id, stage=b, prob=child_prob, demand=demand)
                new_frontier.append((node_id, level))
        frontier = new_frontier

    return tree


# ---------------------------------------------------------------------------
# Gurobi model
# ---------------------------------------------------------------------------

class ScenarioTreeVehicleAllocationModel:
    def __init__(self, N, K, tree, K_green=None, A=None, Ki=None,
                 vtype=GRB.INTEGER, seed=42):
        self.N = list(N)
        self.K = list(K)
        self.tree = tree
        self.Ki = Ki if Ki is not None else {i: self.K for i in self.N}
        self.K_green = list(K_green) if K_green is not None else list(self.K)
        self.A = list(A) if A is not None else [(i, j) for i in self.N for j in self.N if i != j]
        self.vtype = vtype
        self.seed = seed
        random.seed(self.seed)
        self.model = None
        self.vars = {}
        self.s_first_stage = False

    def generate_costs(self):
        """Cost parameters — same scale/shape as Model.py's generate_data()."""
        self.q = {0: 650, 1: 910, 2: 910}                       # cargo bike, e-van, d-van
        self.beta = {0: 44000, 1: 49000, 2: 51000}
        self.alpha = {(i, j, 0): 10 for i in self.N for j in self.N}
        self.alpha.update({(i, j, 1): 50 for i in self.N for j in self.N})
        self.alpha.update({(i, j, 2): 50 for i in self.N for j in self.N})
        self.gamma = {0: 3 * 650, 1: 3 * 910, 2: 3 * 910}
        self.gamma_corr = {0: 3.3 * 650, 1: 3.3 * 910, 2: 3.3 * 910}
        self.theta = {i: 0.3 for i in self.N}
        self.S = {0: 6170, 1: 6170, 2: 6170}
        g = {0: 1, 1: 1, 2: 0}                                  # 1 = green vehicle type
        self.K_green = [k for k in self.K if g.get(k, 0) == 1]

    def build_model(self, env=None, s_first_stage=False):
        """
        s_first_stage : if True, planned subcontracting s is committed once
            per SEASON, before anything is known (indexed by (hub, type,
            season) only, shared by every node at that season) instead of
            today's default: freshly re-decided at every node (adapting to
            everything observed so far). This is what turns the tree model
            into the depth-B analogue of two-stage MRP -- see
            compare_tree_vs_two_stage.build_mrp_tree_model.
        """
        name = "ScenarioTreeVehicleAllocation"
        self.model = Model(env=env, name=name) if env is not None else Model(name=name)
        m = self.model
        tree = self.tree
        self.s_first_stage = s_first_stage
        nodes_ge1 = tree.nodes_with_stage_ge1()
        weeks = {n: list(range(1, tree.block_length(n) + 1)) for n in nodes_ge1}

        # --- VARIABLES ---
        delta_idx = [(i, k) for i in self.N for k in self.K]
        if s_first_stage:
            B = sorted({tree.nodes[n].stage for n in nodes_ge1})
            s_idx = [(i, k, b) for i in self.N for k in self.K for b in B]
        else:
            s_idx = [(n, i, k) for n in nodes_ge1 for i in self.N for k in self.K]
        x_idx = [(n, i, k, t) for n in nodes_ge1 for i in self.N for k in self.K for t in weeks[n]]
        stilde_idx = x_idx
        y_idx = [(n, i, j, k, t) for n in nodes_ge1 for (i, j) in self.A
                 for k in self.K for t in weeks[n]]

        Delta = m.addVars(delta_idx, vtype=self.vtype, name="Delta")
        s = m.addVars(s_idx, vtype=self.vtype, name="s")
        x = m.addVars(x_idx, vtype=self.vtype, name="x")
        s_tilde = m.addVars(stilde_idx, vtype=self.vtype, name="stilde")
        y = m.addVars(y_idx, vtype=self.vtype, name="y")
        m.update()

        self.vars = dict(Delta=Delta, s=s, x=x, s_tilde=s_tilde, y=y)

        def s_of(n, i, k):
            return s[i, k, tree.nodes[n].stage] if s_first_stage else s[n, i, k]

        # --- OBJECTIVE ---
        purchase_cost = sum(self.beta[k] * Delta[i, k] for i in self.N for k in self.K)

        if s_first_stage:
            # First-stage cost: paid once, unconditionally -- not weighted by
            # node.prob (that would double-count it across every scenario
            # sharing a season, since s is now the SAME variable for all of them).
            s_cost = sum(self.gamma[k] * s[i, k, b] * tree.e[b]
                         for i in self.N for k in self.K for b in B)
        else:
            s_cost = sum(
                tree.nodes[n].prob * len(weeks[n]) * sum(self.gamma[k] * s[n, i, k]
                                                          for i in self.N for k in self.K)
                for n in nodes_ge1
            )

        recourse_cost = s_cost + sum(
            tree.nodes[n].prob * (
                sum(self.gamma_corr[k] * s_tilde[n, i, k, t]
                    for i in self.N for k in self.K for t in weeks[n])
                + sum(self.alpha[i, j, k] * y[n, i, j, k, t]
                      for (i, j) in self.A for k in self.K for t in weeks[n])
            )
            for n in nodes_ge1
        )

        m.setObjective(purchase_cost + recourse_cost, GRB.MINIMIZE)

        # --- CONSTRAINTS ---
        for n in nodes_ge1:
            nd = tree.nodes[n]
            parent = nd.parent_id

            # vehicle initialization / redeployment (eq:tree_init)
            if nd.stage == 1:
                for i in self.N:
                    for k in self.K:
                        m.addConstr(x[n, i, k, 1] == Delta[i, k], name=f"init_{n}_{i}_{k}")
            else:
                L = tree.block_length(parent)
                for i in self.N:
                    for k in self.K:
                        inflow = sum(y[parent, j, i, k, L] for j in self.N if (j, i) in self.A)
                        outflow = sum(y[parent, i, j, k, L] for j in self.N if (i, j) in self.A)
                        m.addConstr(
                            x[n, i, k, 1] == x[parent, i, k, L] + inflow - outflow,
                            name=f"init_{n}_{i}_{k}"
                        )

            # weekly rebalancing dynamics (eq:tree_flow), t = 2..e(b(n))
            for t in weeks[n][1:]:
                for i in self.N:
                    for k in self.K:
                        inflow = sum(y[n, j, i, k, t - 1] for j in self.N if (j, i) in self.A)
                        outflow = sum(y[n, i, j, k, t - 1] for j in self.N if (i, j) in self.A)
                        m.addConstr(
                            x[n, i, k, t] == x[n, i, k, t - 1] + inflow - outflow,
                            name=f"flow_{n}_{i}_{k}_{t}"
                        )

            # demand satisfaction + green service level (eq:tree_demand, eq:tree_green)
            for t in weeks[n]:
                for i in self.N:
                    coverage = {
                        k: x[n, i, k, t] + s_of(n, i, k) + s_tilde[n, i, k, t]
                        for k in self.Ki[i]
                    }
                    m.addConstr(
                        sum(self.q[k] * coverage[k] for k in self.Ki[i]) >= nd.demand[i, t],
                        name=f"demand_{n}_{i}_{t}"
                    )
                    green_k = [k for k in self.Ki[i] if k in self.K_green]
                    m.addConstr(
                        sum(self.q[k] * coverage[k] for k in green_k) >= round(self.theta[i] * nd.demand[i, t]),
                        name=f"green_{n}_{i}_{t}"
                    )

        # purchase capacity, root only (eq:tree_cap)
        for k in self.K:
            m.addConstr(sum(Delta[i, k] for i in self.N) <= self.S[k], name=f"cap_{k}")

        return m

    def solve(self, params=None, options=None, label="", s_first_stage=False):
        """
        label : optional prefix (e.g. an instance/model tag) put in front of
            this solve's outcome line, so concurrent solves' output stays
            distinguishable in the terminal.
        s_first_stage : forwarded to build_model() -- see its docstring.
        """
        env = Env(params=options) if options is not None else Env()
        self.build_model(env=env, s_first_stage=s_first_stage)

        if params:
            for name, value in params.items():
                self.model.setParam(name, value)

        self.model.optimize()

        tag = f"[{label}] " if label else ""
        if self.model.status == GRB.OPTIMAL:
            print(f"{tag}Optimal solution found: {self.model.ObjVal}")
        elif self.model.status == GRB.TIME_LIMIT:
            print(f"{tag}Time limit reached. Best objective: {self.model.ObjVal}, Gap: {self.model.MIPGap:.2%}")
        elif self.model.status == GRB.INFEASIBLE:
            print(f"{tag}Model is infeasible.")
        else:
            print(f"{tag}Optimization ended with status {self.model.status}")

    def _get_val(self, var_name):
        var = self.model.getVarByName(var_name)
        if var is None:
            return 0.0
        try:
            return var.X
        except AttributeError:
            return 0.0

    def _s_val(self, n, i, k):
        """Looks up planned subcontracting s for node n, hub i, type k --
        transparently handling both s_first_stage=True (s[i,k,b], shared by
        every node at that season) and the default False (s[n,i,k], per
        node). When False this is byte-for-byte the same lookup as calling
        _get_val directly -- a pure no-op for every existing caller."""
        if self.s_first_stage:
            b = self.tree.nodes[n].stage
            return self._get_val(f"s[{i},{k},{b}]")
        return self._get_val(f"s[{n},{i},{k}]")


if __name__ == "__main__":
    N = list(range(3))
    K = list(range(3))


    tree = build_toy_scenario_tree(N, seasons=(1, 2, 3, 4), branching=2, weeks_per_season=13, seed=42)
    tree.validate(N)

    M = ScenarioTreeVehicleAllocationModel(N, K, tree, seed=42)
    M.generate_costs()
    M.solve(params={"TimeLimit": 300, "MIPGap": 0.01})

    print("\nFleet purchased (Delta[i,k]):")
    for i in N:
        for k in K:
            val = M._get_val(f"Delta[{i},{k}]")
            if val > 0.1:
                print(f"  Hub {i}, Type {k}: {val:.0f}")
