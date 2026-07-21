"""
===========================================================================================
Model.py — Core Vehicle Allocation Model used throughout the SAA pipeline
===========================================================================================

This file defines the baseline **VehicleAllocationModel** class and all the data
generation utilities required for the *entire* Sample Average Approximation
pipeline implemented in `SAA.py`.

IMPORTANT
---------
Every step of the SAA workflow (scenario sampling, replications, stability
analysis, statistical gap computation, out-of-sample evaluation) relies on this
model implementation. The SAA code never rebuilds its own optimization model: it
always instantiates and manipulates the model defined here.

Thus, this file is the *foundation* of the SAA pipeline.


-------------------------------------------------------------------------------------------
1. Purpose of this module
-------------------------------------------------------------------------------------------
`Model.py` defines a generic, lightweight and fully randomizable version of the 
stochastic vehicle allocation model. It includes:
    • Hub sets (N), vehicle types (K), periods (T), scenarios (O)
    • Cost parameters (beta, gamma, gamma_corr, alpha)
    • Vehicle capacities (q)
    • Predicted demand d_pred and scenario-based demand d_real
    • Geographical placeholders (lat/lon)
    • Demand perturbation logic for building scenarios

This version is *not* the “La Poste calibrated model”; instead, it provides a
clean generic skeleton used by:
    - SAA.py
    - runners_for_saa.py
    - SAA_statgap.py
    - peak_scen.py
    - ModelafterSAA.py

These scripts all expect the structure defined here.


-------------------------------------------------------------------------------------------
2. Data generation & scenario generation
-------------------------------------------------------------------------------------------
This file provides two scenario generators:

• generate_scenarios()
    For each hub i and period t, it perturbs d_pred[i][t] by adding a 
    standard-deviation based noise (std of the weekly profile). Produces a
    dictionary-like `self.d_real[o][i][t]`.

• generate_scenarios_from_dict()
    Same as above but using the flattened dict `self.d_pred[i,t]`.
    Output: `self.d_real[i,t,o]`.

Both are used internally by SAA to build empirical scenario sets when running
multiple replications.

`generate_data()` prepares all cost parameters and predicted demand before
calling the scenario generator.


-------------------------------------------------------------------------------------------
3. Minimal embedded test instances
-------------------------------------------------------------------------------------------
`DataManoTest(rebal=True/False)` builds a small handcrafted model instance for 
manual debugging of the first-stage/second-stage relationship. It is often used 
in notebooks before running an actual SAA solve.


-------------------------------------------------------------------------------------------
6. Notes
-------------------------------------------------------------------------------------------
This simplified version of the model is intentionally generic.
The more calibrated “La Poste operational model” is implemented in the
dedicated scripts used in production ( ModelAva.py),
but SAA.py always relies on the clean, minimal interface defined here.

===========================================================================================
"""





from gurobipy import Model, GRB, quicksum, Env

import random
import time
import numpy as np
import os
import pickle


class VehicleAllocationModel:
    def __init__(self, N=0, K=0, T=0, O=0, seed=42):
        self.N = list(range(N))  # Hubs
        self.K = list(range(K))  # Vehicle types
        self.T = list(range(T))  # Time periods
        self.O = list(range(O))  # Scenarios
        self.g = 0
        self.seed = seed
        random.seed(self.seed)
        self.model = Model("Vehicle Allocation")

    def p_omega(self, omega):
        return 1 / len(self.O)

    def generate_data(self):
        # part fix with K = 3
        self.Ki = {i: self.K for i in self.N}

        self.q = {0:650, 1:910, 2:910} # cargo bike ,e van, d van
        self.beta = {0:44000, 1:49000, 2:51000}
        self.alpha = {(i, j, 0): 10 for i in self.N for j in self.N}
        self.alpha.update({(i, j, 1): 50 for i in self.N for j in self.N})
        self.alpha.update({(i, j, 2): 50 for i in self.N for j in self.N})
        self.gamma = {0: 3*650, 1: 3*910, 2: 3*910}
        self.gamma_corr = {0: 3.3*650, 1: 3.3*910, 2: 3.3*910}

        # # Poisson demand with rough negative correlation across hubs
        # # Each hub has its own mean; a shared per-period shift pushes some hubs up and others down
        # self.d_pred = {}
        # mean_demand = {i: random.randint(200, 400) for i in self.N}
        # sign = {i: 1 if i % 2 == 0 else -1 for i in self.N}  # alternating +/-
        # for t in self.T:
        #     shift = random.randint(30, 80)
        #     for i in self.N:
        #         base = np.random.poisson(lam=mean_demand[i])
        #         self.d_pred[i, t] = max(0, base + sign[i] * shift)
        # for i in self.N:
        #     mean_demand = {i: random.randint(200, 400) for i in self.N}
        #     self.d_pred = {(i, t): np.random.poisson(lam=mean_demand[i]) for i in self.N for t in self.T}
        
        mean = {i: random.randint(300000, 400000) for i in self.N}
        self.d_pred = {(i, t): random.randint(mean[i]-75000, mean[i]+75000) for i in self.N for t in self.T}

        self.theta = {i: 0.3 for i in self.N}
        self.l = {(i, k): 1 for i in self.N for k in self.K}
        self.g = {0:1, 1:1, 2:0}

        self.M1 = {0:6170, 1:6170, 2:6170}
        self.S = self.M1



        # random
        # self.Ki = self.K

        # self.q = {k: random.randint(30, 50) for k in self.K}
        # self.beta = {k: random.randint(100, 150) for k in self.K}
        # self.alpha = {(i, j, k): random.randint(0, 2) for i in self.N for j in self.N for k in self.K}
        # self.gamma = {k: random.randint(5, 10) for k in self.K}
        # self.gamma_corr = {k: self.gamma[k] + random.randint(5, 10) for k in self.K}
        # # self.gamma_corr = {k: 25 for k in self.K}

        # self.d_pred = {(i, t): random.randint(40, 80) for i in self.N for t in self.T}
        # self.theta = {i: round(random.uniform(0.3, 0.6), 2) for i in self.N}
        # self.l = {(i, k): 1 for i in self.N for k in self.K}
        # self.g = {k: 1 for k in self.K}

        # self.lat = {i: random.uniform(43.0, 50.0) for i in self.N}
        # self.lon = {i: random.uniform(0.5, 6.0) for i in self.N}

        # self.M1 = {k: 7 for k in self.K}
        # self.S = self.M1
        # self.M2 = {k: 100 for k in self.K}
        # self.M3 = {k: 100 for k in self.K}
        # self.M4 = {k: 100 for k in self.K}

        self.generate_scenarios_from_dict()


    # def generate_scenarios(self):
    #     self.d_real = []
    #     for o in self.O:
    #         do = []
    #         for i in self.N:
    #             doi = []
    #             di = [self.d_pred[i][t] for t in self.T]
    #             #print(di)
    #             for t in self.T:
    #                 noise = np.std(di)
    #                 doi.append( max(0, self.d_pred[i][t] + noise))
    #             do.append(doi)
    #         self.d_real.append(do)
    #     print(np.shape(self.d_real))


        
    def generate_scenarios_from_dict(self):
        self.d_real = {}
        for o in self.O:
            for i in self.N:
                di = [self.d_pred[i, t] for t in self.T]
                std_dev = np.std(di)
                for t in self.T:
                    noise = std_dev * random.gauss(0, 1)
                    perturbed = int(round(max(0, self.d_pred[i, t] + noise)))
                    self.d_real[i, t, o] = perturbed


####################################################################################################"""
    """def LaPosteData(self,inputcsv ):
        self.K = 3
        self.N = inputcsv['']
        self.T = 20
        self.O=Omega

        self.Ki = self.K

        self.q = {0:120,1:80,2:80}
        self.beta = {0:878.14,1:1000,2:1000}
        self.alpha = 
        self.gamma = {0:2160.14,1:2160,2:2160}
        self.gamma_corr = {k: 1.5*self.gamma[k]  for k in self.K}

        self.d_pred = 
        self.theta = {i: 0 for i in self.N}
        self.l = {(i, k): 1 for i in self.N for k in self.K}
        self.g = {k: 1 for k in self.K}

        self.lat = {i: random.uniform(43.0, 50.0) for i in self.N}
        self.lon = {i: random.uniform(0.5, 6.0) for i in self.N}

        self.M1 = {k: 10 for k in self.K}
        self.S = self.M1
        self.M2 = {k: 100 for k in self.K}
        self.M3 = {k: 100 for k in self.K}
        self.M4 = {k: 100 for k in self.K}"""


    def DataManoTest(self,rebal):
        if rebal : 
            self.N = list(range(3))  # Hubs
            self.K = list(range(2))  # Vehicle types
            self.Ki = self.K
            self.T = list(range(4))  # Low-frequency periods
            self.O = list(range(1))  # Scenarios
            random.seed(self.seed)
            self.model = Model("Vehicle Allocation")

            self.q = {0:1, 1:8}
            self.beta = {0:1.5, 1:10}
            self.alpha = {(i, j, k): 10 for i in self.N for j in self.N for k in self.K }
            self.gamma = {0:20, 1:100}#{0:1000000000, 1:10000000000000}
            self.gamma_corr = {0:40, 1:200}
            self.d_pred = {(0, 0):7, (0,1):16, (0,2):14, (0,3):30,(1,0):20,(1,1):16, (1,2):12, (1,3):13, (2,0):4, (2,1):8, (2,2):22, (2,3):7 }
            self.d_real = {(0, 0,0):7, (0,1,0):19, (0,2,0):12, (0,3,0):27,(1,0,0):20,(1,1,0):24, (1,2,0):12, (1,3,0):20, (2,0,0):4, (2,1,0):11, (2,2,0):22, (2,3,0):8 }
            self.theta = {i: 1 for i in self.N}
            self.l = {(i, k): 1 for i in self.N for k in self.K}
            self.g = {k: 1 for k in self.K}



            self.S = [5,5]
            """self.M1 = self.S
            self.M2 = {k: 10000000 for k in self.K}
            self.M3 = {k: 10000000 for k in self.K}
            self.M4 = {k: 10000000 for k in self.K}
            self.M5 = {k: 10000000 for k in self.K}"""
        else:
            self.N = list(range(3))  # Hubs
            self.K = list(range(2))  # Vehicle types
            self.Ki = self.K
            self.T = list(range(4))  # Low-frequency periods
            self.O = list(range(1))  # Scenarios
            random.seed(self.seed)
            self.model = Model("Vehicle Allocation")

            self.q = {0:1, 1:8}
            self.beta = {0:1.5, 1:10}
            self.alpha = {(i, j, k): 100000000000 for i in self.N for j in self.N for k in self.K }
            self.gamma = {0:20, 1:100}#{0:1000000000, 1:10000000000000}
            self.gamma_corr = {0:40, 1:200}
            self.d_pred = {(0, 0):7, (0,1):16, (0,2):14, (0,3):30,(1,0):20,(1,1):16, (1,2):12, (1,3):13, (2,0):4, (2,1):8, (2,2):22, (2,3):7 }
            self.d_real = {(0, 0,0):7, (0,1,0):19, (0,2,0):12, (0,3,0):27,(1,0,0):20,(1,1,0):24, (1,2,0):12, (1,3,0):20, (2,0,0):4, (2,1,0):11, (2,2,0):22, (2,3,0):8 }
            self.theta = {i: 1 for i in self.N}
            self.l = {(i, k): 1 for i in self.N for k in self.K}
            self.g = {k: 1 for k in self.K}
            self.S = [5,5]



    def build_model_MRP(self, env=None):
        """
        Two-stage MRP: v[i,k] (fleet acquired + positioned at hub i, first
        stage) and s[i,k,b] (planned subcontracting, first stage, one value
        per SEASON b -- not per week) are fixed before any scenario is known.
        x[i,k,t,o] (fleet position), s_corr[i,k,t,o] (corrective
        subcontracting) and y[i,j,k,t,o] (rebalancing) are second-stage
        recourse, adapting to the realized scenario o.

        Season structure: self.B (list of season ids) / self.season_of_week
        (dict t -> b) are optional -- if unset, every week is its own season
        (b(t) = t), exactly reproducing the old per-week s[i,k,t] granularity
        for callers that don't have a real season concept (e.g. run_parallel.py's
        generate_data() path). compare_tree_vs_two_stage.build_two_stage_model()
        attaches the tree's real season boundaries.
        """
        if env is not None:
            self.model = Model(env=env, name="DynamicVehicleAllocation")
        else:
            self.model = Model(name="DynamicVehicleAllocation")

        B = getattr(self, "B", None) or list(self.T)
        season_of_week = getattr(self, "season_of_week", None) or {t: t for t in self.T}
        weeks_in_season = {b: [t for t in self.T if season_of_week[t] == b] for b in B}

        # VARIABLES
        v = self.model.addVars(self.N, self.K, vtype=GRB.INTEGER, name="v")
        X = self.model.addVars(self.K, vtype=GRB.INTEGER, name="X")
        x = self.model.addVars(self.N, self.K, self.T, self.O, vtype=GRB.INTEGER, name="x")
        s = self.model.addVars(self.N, self.K, B, vtype=GRB.INTEGER, name="s")
        s_corr = self.model.addVars(self.N, self.K, self.T, self.O, vtype=GRB.INTEGER, name="s_corr")
        y = self.model.addVars(self.N, self.N, self.K, self.T, self.O, vtype=GRB.INTEGER, name="y")

        # OBJECTIVE
        self.model.setObjective(
            quicksum(self.beta[k] * X[k] for k in self.K) +
            quicksum(self.gamma[k] * s[i, k, b] * len(weeks_in_season[b])
                     for i in self.N for k in self.K for b in B) +
            quicksum(
                self.p_omega(o) * (
                    quicksum(self.gamma_corr[k] * s_corr[i, k, t, o]
                            for i in self.N for k in self.K for t in self.T) +
                    quicksum(self.alpha[i, j, k] * y[i, j, k, t, o]
                            for i in self.N for j in self.N for k in self.K for t in self.T)
                ) for o in self.O
            ),
            GRB.MINIMIZE
        )

        # CONSTRAINTS

        # Fleet acquisition + positioning (first stage)
        for k in self.K:
            self.model.addConstr(X[k] == quicksum(v[i, k] for i in self.N), name=f"stock_def_{k}")
            self.model.addConstr(X[k] <= self.S[k], name=f"stock_max_{k}")

        # Initial fleet position: x[i,k,t=first,o] == v[i,k], for every scenario
        t0 = self.T[0]
        for i in self.N:
            for k in self.K:
                for o in self.O:
                    self.model.addConstr(x[i, k, t0, o] == v[i, k], name=f"init_{i}_{k}_{o}")

        # Real demand satisfaction and green constraint
        #
        # No separate inflow/outflow term here: x[i,k,t,o] already carries
        # that period's net rebalancing per type via the precedence
        # constraint below. Adding a flow term again on top of x would
        # double-count it -- see the tree model's demand constraint, which
        # has no separate flow term for the same reason.
        for i in self.N:
            for t in self.T:
                b = season_of_week[t]
                for o in self.O:
                    self.model.addConstr(
                        quicksum(self.q[k] * (
                            x[i, k, t, o] + s[i, k, b] + s_corr[i, k, t, o]) for k in self.Ki[i])
                         >= self.d_real[i, t, o],
                        name=f"real_demand_{i}_{k}_{t}_{o}"
                    )
                    self.model.addConstr(
                        quicksum(self.g[k] * self.q[k] * (
                            x[i, k, t, o] + s[i, k, b] + s_corr[i, k, t, o]) for k in self.Ki[i])
                         >= round(self.theta[i] * self.d_real[i, t, o]),
                        name=f"green_real_{i}_{k}_{t}_{o}"
                    )

        ##precedence
        for i in self.N:
            for k in self.K:
                for t in self.T:
                    if t > t0:
                        for o in self.O:
                            inflow_k = quicksum(y[j, i, k, t - 1, o] for j in self.N)
                            outflow_k = quicksum(y[i, j, k, t - 1, o] for j in self.N)
                            self.model.addConstr(x[i, k, t, o] == x[i, k, t - 1, o] + inflow_k - outflow_k,
                                                  name="precedence")

    def build_model_static(self, env=None):

        if env is not None:
            self.model = Model(env=env, name="StaticVehicleAllocation")
        else:
            self.model = Model(name="StaticVehicleAllocation")

        # Max demand per hub across all time periods
        d_max = {i: max(self.d_real[i, t, o] for t in self.T for o in self.O) for i in self.N}

        # VARIABLES
        X = self.model.addVars(self.K, vtype=GRB.INTEGER, name="X")
        x = self.model.addVars(self.N, self.K, vtype=GRB.INTEGER, name="x")
        s = self.model.addVars(self.N, self.K, vtype=GRB.INTEGER, name="s")

        # OBJECTIVE
        self.model.setObjective(
            quicksum(self.beta[k] * X[k] for k in self.K) +
            quicksum(self.gamma[k] * len(self.T) * s[i, k] for i in self.N for k in self.K) ,
            GRB.MINIMIZE
        )

        # CONSTRAINTS

        # Planification support
        for k in self.K:
            self.model.addConstr(X[k] <= self.S[k], name=f"stock_max_{k}")
            self.model.addConstr(quicksum(x[i, k] for i in self.N) <= X[k], name=f"stock_sum_{k}")

        # Demand coverage using d_max (static allocation must cover peak demand)
        for i in self.N:
            self.model.addConstr(
                quicksum(self.q[k] * (x[i, k]+s[i, k]) for k in self.Ki[i]) >= d_max[i],
                name=f"peak_demand_{i}"
            )
            self.model.addConstr(
                quicksum(self.g[k] * self.q[k] * (x[i, k]+s[i, k]) for k in self.Ki[i])
                >= round(self.theta[i] * d_max[i]),
                name=f"green_peak_{i}"
            )


    def build_model_MNP(self, env=None):
        """
        MNP: static fleet allocation (x time- and scenario-independent, since
        there's no rebalancing lever to adapt it with) with seasonal planned
        subcontracting (s[i,k,b], one value per season b, not per week) and
        corrective subcontracting (s_corr[i,k,t,o]).

        Season structure: self.B / self.season_of_week, same optional
        attributes as build_model_MRP (defaults to one season per week if
        unset -- see that method's docstring).
        """
        if env is not None:
            self.model = Model(env=env, name="MNPVehicleAllocation")
        else:
            self.model = Model(name="MNPVehicleAllocation")

        B = getattr(self, "B", None) or list(self.T)
        season_of_week = getattr(self, "season_of_week", None) or {t: t for t in self.T}
        weeks_in_season = {b: [t for t in self.T if season_of_week[t] == b] for b in B}

        # VARIABLES
        X = self.model.addVars(self.K, vtype=GRB.INTEGER, name="X")
        x = self.model.addVars(self.N, self.K, vtype=GRB.INTEGER, name="x")
        s = self.model.addVars(self.N, self.K, B, vtype=GRB.INTEGER, name="s")
        s_corr = self.model.addVars(self.N, self.K, self.T, self.O, vtype=GRB.INTEGER, name="s_corr")

        # OBJECTIVE
        self.model.setObjective(
            quicksum(self.beta[k] * X[k] for k in self.K) +
            quicksum(self.gamma[k] * s[i, k, b] * len(weeks_in_season[b])
                     for i in self.N for k in self.K for b in B) +
            quicksum(
                self.p_omega(o) * (
                    quicksum(self.gamma_corr[k] * s_corr[i, k, t, o]
                            for i in self.N for k in self.K for t in self.T)
                ) for o in self.O
            ),
            GRB.MINIMIZE
        )

        # CONSTRAINTS

        # Planification support
        for k in self.K:
            self.model.addConstr(X[k] <= self.S[k], name=f"stock_max_{k}")
            self.model.addConstr(quicksum(x[i, k] for i in self.N) <= X[k], name=f"stock_sum_{k}")

        # Real demand satisfaction and green constraint
        for i in self.N:
            for t in self.T:
                b = season_of_week[t]
                for o in self.O:
                    self.model.addConstr(
                        quicksum(self.q[k] * (
                            x[i, k] + s[i, k, b] + s_corr[i, k, t, o]) for k in self.Ki[i])
                         >= self.d_real[i, t, o],
                        name=f"real_demand_{i}_{k}_{t}_{o}"
                    )
                    self.model.addConstr(
                        quicksum(self.g[k] * self.q[k] * (
                            x[i, k] + s[i, k, b] + s_corr[i, k, t, o]) for k in self.Ki[i])
                         >= round(self.theta[i] * self.d_real[i, t, o]),
                        name=f"green_real_{i}_{k}_{t}_{o}"
                    )


    
        

    
        
    def solve_MRP(self, params=None, options=None, label=""):
        """
        Build and solve the model.

        params : dict – Gurobi parameters, e.g.
            {"TimeLimit": 500, "MIPGap": 0.01, "Threads": 4,
             "OutputFlag": 0, "LogFile": "gurobi_log.txt"}
        label : optional prefix (e.g. an instance/model tag) put in front of
            this solve's outcome line, so concurrent solves' output stays
            distinguishable in the terminal.
        """
        if options is not None:
            env = Env(params=options)
        else:
            env = Env()

        self.build_model_MRP(env=env)

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

    def solve_static(self, params=None, options=None, label=""):
        """
        Build and solve the model.

        params : dict – Gurobi parameters, e.g.
            {"TimeLimit": 500, "MIPGap": 0.01, "Threads": 4,
             "OutputFlag": 0, "LogFile": "gurobi_log.txt"}
        label : optional prefix (e.g. an instance/model tag) put in front of
            this solve's outcome line, so concurrent solves' output stays
            distinguishable in the terminal.
        """
        if options is not None:
            env = Env(params=options)
        else:
            env = Env()
        self.build_model_static(env=env)

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

    def solve_MNP(self, params=None, options=None, label=""):
        """
        Build and solve the MNP model (no rebalancing, static x, time-varying s).

        params : dict – Gurobi parameters, e.g.
            {"TimeLimit": 500, "MIPGap": 0.01, "Threads": 4,
             "OutputFlag": 0, "LogFile": "gurobi_log.txt"}
        label : optional prefix (e.g. an instance/model tag) put in front of
            this solve's outcome line, so concurrent solves' output stays
            distinguishable in the terminal.
        """
        if options is not None:
            env = Env(params=options)
        else:
            env = Env()
        self.build_model_MNP(env=env)

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


    def set_fleet_cap_from_static(self):
        """
        After solving the static model, read the optimal X[k] values,
        compute Xstatic = sum_k X[k], then set M1[k] = S[k] = Xstatic / |K|
        for every vehicle type k.  Call this before solving MNP or MRP.
        """
        xstatic = sum(self._get_val(f"X[{k}]") for k in self.K)
        cap = int(round(xstatic / len(self.K)))
        self.M1 = {k: cap for k in self.K}
        self.S  = self.M1
        return xstatic, cap

    def _get_val(self, var_name):
        var = self.model.getVarByName(var_name)
        if var is None:
            return 0.0
        try:
            return var.X
        except AttributeError:
            return 0.0



    
    def export_solution_summaryuiui(self, filename="solution_summary.txt"): 
        with open(filename, "w") as f:
            def write(line=""):
                f.write(line + "\n")

            write("=== Vehicle Allocation Model Solution Summary ===\n")

            write("🔹 Total Vehicles Allocated to the System (X_k):")
            for k in self.K:
                val = self._get_val(f"X[{k}]")
                write(f"  Type {k}: {val:.0f}")
            write("")

            write("🔹 Allocation per Hub, Time and Scenario (x[i,k,t,o]):")
            for t in self.T:
                for k in self.K:
                    for i in self.N:
                        for o in self.O:
                            val = self._get_val(f"x[{i},{k},{t},{o}]")
                            if val > 0.1:
                                write(f"  Hub {i}, Type {k}, Time {t}, Scenario {o}: {val:.0f}")
                write("\n")
            write("")

            B = getattr(self, "B", None) or list(self.T)
            write("🔹 Anticipated Subcontracting (s[i,k,b], one value per season):")
            for b in B:
                for k in self.K:
                    for i in self.N:
                        val = self._get_val(f"s[{i},{k},{b}]")
                        if val > 0.1:
                            write(f"  Hub {i}, Type {k}, Season {b}: {val:.0f}")
                write("\n")
            write("")

            write("🔹 Corrective Subcontracting (s_corr[i,k,t,o]):")
            for i in self.N:
                for k in self.K:
                    for t in self.T:
                        for o in self.O:
                            val = self._get_val(f"s_corr[{i},{k},{t},{o}]")
                            if val > 0.1:
                                write(f"  Hub {i}, Type {k}, Time {t}, Scenario {o}: {val:.0f}")
                write("\n")
            write("")

            write("🔹 Rebalancing Transfers (y[i,j,k,t,o]):")
            for t in self.T:
                for i in self.N:
                    for j in self.N:
                        if i == j:
                            continue
                        for k in self.K:
                            for o in self.O:
                                val = self._get_val(f"y[{i},{j},{k},{t},{o}]")
                                if val > 0.1:
                                    write(f"  {val:.0f} of Type {k} from Hub {i} → Hub {j} at t={t}, scenario={o}")
            write("")






    


    @staticmethod
    def rebalancing_plan(y, scenario, t_start, t_end):
        """
        Print the rebalancing plan from pre-extracted y dict.
        y: dict with keys (i, j, k, t, o) from extract_MRP()
        """
        print(f"\nRebalancing Plan — Scenario {scenario}, periods [{t_start}, {t_end}]")
        print("=" * 60)
        for t in range(t_start, t_end + 1):
            transfers = []
            for (i, j, k, tp, o), val in y.items():
                if o == scenario and tp == t:
                    transfers.append(f"  Hub {i} -> Hub {j}: {val:.0f} of type {k}")
            print(f"\nPeriod {t}:")
            if transfers:
                print("\n".join(sorted(transfers)))
            else:
                print("  No transfers")

    def save_instance(self, filepath):
        """Save all data attributes (everything from generate_data) to a pickle file."""
        data = {k: v for k, v in self.__dict__.items() if k != 'model'}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Instance saved to {filepath}")

    def load_instance(self, filepath):
        """Load data attributes from a pickle file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.__dict__.update(data)
        print(f"Instance loaded from {filepath}")


if __name__ == "__main__":
    from plots import (extract_MRP, extract_static, extract_MNP,
                        extract_MRP_costs, extract_MNP_costs,
                        plot_compare_subcontracting, plot_compare_resource, plot_compare_costs,
                        plot_compare_subcontracting_3way, plot_compare_resource_3way, plot_compare_costs_3way)
    # options = {
    #     'WLSACCESSID': "30bca212-81df-41cc-a94e-a0269b14a3ec",
    #     'WLSSECRET': "215eee4c-3130-4a8b-8156-898521b84f16",
    #     'LICENSEID': 2738996,
    #     'WLSTOKENDURATION': 10 #mins
    # }
    options = None
    N, K, T, O = 3, 3, 52, 50
    M = VehicleAllocationModel(N, K, T, O, seed=42)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join("experimentsIPIC", f"exp{N}_{K}_{T}_{O}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    M.generate_data()
    M.save_instance(os.path.join(exp_dir, "instance.pkl"))

    start = time.time()
    M.solve_static(params={"TimeLimit": 3600, "MIPGap": 0.01}, options=options)
    end1 = time.time()
    static_x, static_s = extract_static(M)
    static_obj = M.model.ObjVal
    print(f'solve_static in {round(end1 - start, 2)} seconds')

    M.solve_MNP(params={"TimeLimit": 3600, "MIPGap": 0.01}, options=options)
    end2 = time.time()
    mnp_x, mnp_s = extract_MNP(M)
    mnp_costs = extract_MNP_costs(M)
    mnp_obj = M.model.ObjVal
    print(f'solve_MNP in {round(end2 - end1, 2)} seconds')

    M.solve_MRP(params={"TimeLimit": 3600, "MIPGap": 0.01}, options=options)
    end3 = time.time()
    M.export_solution_summaryuiui(filename=os.path.join(exp_dir, "ui.txt"))
    mrp_x, mrp_s, mrp_y = extract_MRP(M)
    mrp_costs = extract_MRP_costs(M)
    print(f'solve_MRP in {round(end3 - end2, 2)} seconds')

    # Save MRP rebalancing solution
    with open(os.path.join(exp_dir, "mrp_y.pkl"), 'wb') as f:
        pickle.dump(mrp_y, f)

    # --- Fleet summary: sum_i x[i,k] for Static/MNP, X[k] for MRP ---
    print("\n=== Fleet allocation per vehicle type ===")
    print(f"{'Type':<6} {'Static sum_i x[i,k]':>22} {'MNP sum_i x[i,k]':>20} {'MRP X[k]':>12}")
    print("-" * 64)
    for k in M.K:
        s_static = static_x[k].sum(axis=0)[0]   # x[i,k] constant over t
        s_mnp    = mnp_x[k].sum(axis=0)[0]      # x[i,k] constant over t
        X_mrp    = M._get_val(f"X[{k}]")        # fleet size variable
        print(f"{k:<6} {s_static:>22.0f} {s_mnp:>20.0f} {X_mrp:>12.0f}")
    print()

    # 2-way comparisons (MRP vs Static, as before)
    plot_compare_subcontracting(M, mrp_s, static_s, output_dir=exp_dir)
    plot_compare_resource(M, mrp_x, static_x, output_dir=exp_dir)
    plot_compare_costs(M, mrp_costs, static_obj, output_dir=exp_dir)

    # 3-way comparisons (MRP vs MNP vs Static)
    plot_compare_subcontracting_3way(M, mrp_s, mnp_s, static_s, output_dir=exp_dir)
    plot_compare_resource_3way(M, mrp_x, mnp_x, static_x, output_dir=exp_dir)
    plot_compare_costs_3way(M, mrp_costs, mnp_costs, static_obj, output_dir=exp_dir)

    # Example: rebalancing plan for scenario 4, periods 5 to 15
    M.rebalancing_plan(mrp_y, scenario=4, t_start=5, t_end=15)

