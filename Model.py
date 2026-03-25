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
import pandas as pd
import plotly.graph_objects as go
import time

import matplotlib.pyplot as plt
from matplotlib import cm
import networkx as nx
import numpy as np
import re

import csv
import os
import pickle

from plots import (extract_ST, extract_static, extract_ST_costs,
                    plot_compare_subcontracting, plot_compare_resource, plot_compare_costs)


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
        # part fix with K = 2
        self.Ki = {i: self.K for i in self.N}

        self.q = {0:10, 1:120}
        self.beta = {0:500, 1:5000}
        self.alpha = {(i, j, 0): 1 for i in self.N for j in self.N}
        self.alpha.update({(i, j, 1): 5 for i in self.N for j in self.N})
        self.gamma = {0:20, 1:200}
        self.gamma_corr = {0:30, 1:250}

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
        
        for i in self.N:
            self.d_pred = {(i, t): random.randint(200+i*100, 400+i*100) for i in self.N for t in self.T}

        self.theta = {i: 0.3 for i in self.N}
        self.l = {(i, k): 1 for i in self.N for k in self.K}
        self.g = {0:1, 1:0}

        self.M1 = {0:80, 1:60}
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
                    noise = random.gauss(0, 1)  # Standard normal noise scaled by std_dev and direction
                    print(f"Hub {i}, Time {t}: Predicted={self.d_pred[i, t]}, Noise={noise:.2f}, StdDev={std_dev:.2f}")
                    noise = std_dev * noise
                    perturbed = max(0, self.d_pred[i, t] + noise)
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



    def build_model_ST(self, env=None):
        """
        Le BON
        
        """
        if env is not None:
            self.model = Model(env=env, name="DynamicVehicleAllocation")
        else:
            self.model = Model(name="DynamicVehicleAllocation")

        # VARIABLES
        X = self.model.addVars(self.K, vtype=GRB.INTEGER, name="X")
        x = self.model.addVars(self.N, self.K, self.T, vtype=GRB.INTEGER, name="x")
        s = self.model.addVars(self.N, self.K, self.T, vtype=GRB.INTEGER, name="s")
        s_corr = self.model.addVars(self.N, self.K, self.T, self.O, vtype=GRB.INTEGER, name="s_corr")
        y = self.model.addVars(self.N, self.N, self.K, self.T, self.O, vtype=GRB.INTEGER, name="y")

        # OBJECTIVE
        self.model.setObjective(
            quicksum(self.beta[k] * X[k] for k in self.K) +
            quicksum(self.gamma[k] * s[i, k, t] for i in self.N for k in self.K for t in self.T) +
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

        # Planification support
        for k in self.K:
            self.model.addConstr(X[k] <= self.S[k], name=f"stock_max_{k}")
            for t in self.T:
                self.model.addConstr(quicksum(x[i, k, t] for i in self.N) <= X[k], name=f"stock_sum_{k}_{t}")

        # Predictive demand and green coverage
        for i in self.N:
            for t in self.T:
                self.model.addConstr(
                    quicksum(self.q[k] * (x[i, k, t] + s[i, k, t]) for k in self.Ki[i]) >= self.d_pred[i, t],
                    name=f"pred_demand_{i}_{t}"
                )
                self.model.addConstr(
                    quicksum(self.g[k] * self.q[k] * (x[i, k, t] + s[i, k, t]) for k in self.Ki[i]) >= self.theta[i] * self.d_pred[i, t],
                    name=f"green_pred_{i}_{t}"
                )

        # Real demand satisfaction and green constraint
        for i in self.N:
            for t in self.T:
                for o in self.O:
                    if t == 0:
                        inflow_k = 0
                        outflow_k = 0
                    else:
                        inflow_k = quicksum(self.q[k]* y[j, i, k, t - 1, o] for j in self.N for k in self.Ki[i])
                        outflow_k = quicksum(self.q[k]*y[i, j, k, t - 1, o] for j in self.N for k in self.Ki[i])
                    self.model.addConstr(
                        quicksum(self.q[k] * (
                            x[i, k, t]  + s[i, k, t] + s_corr[i, k, t, o]) for k in self.Ki[i]) + inflow_k - outflow_k
                         >= self.d_real[i, t, o],
                        name=f"real_demand_{i}_{k}_{t}_{o}"
                    )
                    self.model.addConstr(
                        quicksum(self.g[k] * self.q[k] * (
                            x[i, k, t]+ s[i, k, t] + s_corr[i, k, t, o]) for k in self.Ki[i]) + inflow_k - outflow_k 
                         >= self.theta[i] * self.d_real[i, t, o],
                        name=f"green_real_{i}_{k}_{t}_{o}"
                    )

        ##precedence
        for i in self.N:
            for k in self.K:
                for t in self.T:
                    if t>0:
                        for o in self.O:
                            inflow_k = quicksum(y[j, i, k, t - 1, o] for j in self.N)
                            outflow_k = quicksum(y[i, j, k, t - 1, o] for j in self.N)
                            self.model.addConstr(x[i, k, t] == x[i, k, t-1] + inflow_k - outflow_k, name="precedence")

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
        s = self.model.addVars(self.N, self.K, self.T, vtype=GRB.INTEGER, name="s")

        # OBJECTIVE
        self.model.setObjective(
            quicksum(self.beta[k] * X[k] for k in self.K) +
            quicksum(self.gamma[k] * s[i, k, t] for i in self.N for k in self.K for t in self.T) ,
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
                quicksum(self.q[k] * x[i, k] for k in self.Ki[i]) >= d_max[i],
                name=f"peak_demand_{i}"
            )
            self.model.addConstr(
                quicksum(self.g[k] * self.q[k] * x[i, k] for k in self.Ki[i]) >= self.theta[i] * d_max[i],
                name=f"green_peak_{i}"
            )

        # # Per-period demand coverage (x + s must still cover each period)
        # for i in self.N:
        #     for t in self.T:
        #         self.model.addConstr(
        #             quicksum(self.q[k] * (x[i, k] + s[i, k, t]) for k in self.Ki[i]) >= self.d_pred[i, t],
        #             name=f"pred_demand_{i}_{t}"
        #         )
        #         self.model.addConstr(
        #             quicksum(self.g[k] * self.q[k] * (x[i, k] + s[i, k, t]) for k in self.Ki[i]) >= self.theta[i] * self.d_pred[i, t],
        #             name=f"green_pred_{i}_{t}"
        #         )

    
        

    
        
    def solve_ST(self, params=None, options=None):
        """
        Build and solve the model.

        params : dict – Gurobi parameters, e.g.
            {"TimeLimit": 500, "MIPGap": 0.01, "Threads": 4,
             "OutputFlag": 0, "LogFile": "gurobi_log.txt"}
        """
        if options is not None:
            env = Env(params=options)
        else:
            env = Env()

        self.build_model_ST(env=env)

        if params:
            for name, value in params.items():
                self.model.setParam(name, value)

        print('opt start')
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal solution found: {self.model.ObjVal}")
        elif self.model.status == GRB.TIME_LIMIT:
            print(f"Time limit reached. Best objective: {self.model.ObjVal}, Gap: {self.model.MIPGap:.2%}")
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print(f"Optimization ended with status {self.model.status}")

    def solve_static(self, params=None, options=None):
        """
        Build and solve the model.

        params : dict – Gurobi parameters, e.g.
            {"TimeLimit": 500, "MIPGap": 0.01, "Threads": 4,
             "OutputFlag": 0, "LogFile": "gurobi_log.txt"}
        """
        if options is not None:
            env = Env(params=options)
        else:
            env = Env()
        self.build_model_static(env=env)

        if params:
            for name, value in params.items():
                self.model.setParam(name, value)

        print('opt start')
        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            print(f"Optimal solution found: {self.model.ObjVal}")
        elif self.model.status == GRB.TIME_LIMIT:
            print(f"Time limit reached. Best objective: {self.model.ObjVal}, Gap: {self.model.MIPGap:.2%}")
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible.")
        else:
            print(f"Optimization ended with status {self.model.status}")

    

    
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

            write("🔹 Allocation per Hub and Time (x[i,k,t]):")
            for t in self.T:
                for k in self.K:
                    for i in self.N:
                        val = self._get_val(f"x[{i},{k},{t}]")
                        if val > 0.1:
                            write(f"  Hub {i}, Type {k}, Time {t}: {val:.0f}")
                write("\n")
            write("")

            write("🔹 Anticipated Subcontracting (s[i,k,t]):")
            for t in self.T:
                for k in self.K:
                    for i in self.N:
                        val = self._get_val(f"s[{i},{k},{t}]")
                        if val > 0.1:
                            write(f"  Hub {i}, Type {k}, Time {t}: {val:.0f}")
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
        y: dict with keys (i, j, k, t, o) from extract_ST()
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
    options = {
        'WLSACCESSID': "30bca212-81df-41cc-a94e-a0269b14a3ec",
        'WLSSECRET': "215eee4c-3130-4a8b-8156-898521b84f16",
        'LICENSEID': 2738996,
        'WLSTOKENDURATION': 10 #mins
    }
    N, K, T, O = 3, 2, 52, 100
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
    
    M.solve_ST(params={"TimeLimit": 3600, "MIPGap": 0.01}, options=options)
    end2 = time.time()
    print(f'solve_static in {round(end2 - end1, 2)} seconds')
    print(f'solve_ST in {round(end1 - start, 2)} seconds')
    M.export_solution_summaryuiui(filename=os.path.join(exp_dir, "ui.txt"))
    st_x, st_s, st_y = extract_ST(M)
    st_costs = extract_ST_costs(M)

    # Save ST rebalancing solution
    with open(os.path.join(exp_dir, "st_y.pkl"), 'wb') as f:
        pickle.dump(st_y, f)

    plot_compare_subcontracting(M, st_s, static_s, output_dir=exp_dir)
    plot_compare_resource(M, st_x, static_x, output_dir=exp_dir)
    plot_compare_costs(M, st_costs, static_obj, output_dir=exp_dir)

    # Example: rebalancing plan for scenario 4, periods 5 to 15
    M.rebalancing_plan(st_y, scenario=4, t_start=5, t_end=15)






    # by maher
    #M.plot_solution_map(scenario=0)
    #M.plot_rebalancing_solution(scenario=0)
    #plot_solution_graph_from_file(filename="bismillah.txt",N=5,T=5)

