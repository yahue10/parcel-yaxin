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





from gurobipy import Model, GRB, quicksum
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
        self.Ki = self.K

        self.q = {k: random.randint(10, 30) for k in self.K}
        self.beta = {k: random.randint(5, 10) for k in self.K}
        self.alpha = {(i, j, k): random.randint(1, 5) for i in self.N for j in self.N for k in self.K}
        self.gamma = {k: random.randint(15, 20) for k in self.K}
        self.gamma_corr = {k: self.gamma[k] + random.randint(5, 10) for k in self.K}

        self.d_pred = {(i, t): random.randint(40, 80) for i in self.N for t in self.T}
        self.theta = {i: round(random.uniform(0.3, 0.6), 2) for i in self.N}
        self.l = {(i, k): 1 for i in self.N for k in self.K}
        self.g = {k: 1 for k in self.K}

        self.lat = {i: random.uniform(43.0, 50.0) for i in self.N}
        self.lon = {i: random.uniform(0.5, 6.0) for i in self.N}

        self.M1 = {k: 10 for k in self.K}
        self.S = self.M1
        self.M2 = {k: 100 for k in self.K}
        self.M3 = {k: 100 for k in self.K}
        self.M4 = {k: 100 for k in self.K}

        self.generate_scenarios_from_dict()


    def generate_scenarios(self):
        self.d_real = []
        for o in self.O:
            do = []
            for i in self.N:
                doi = []
                di = [self.d_pred[i][t] for t in self.T]
                #print(di)
                for t in self.T:
                    noise = np.std(di)
                    doi.append( max(0, self.d_pred[i][t] + noise))
                do.append(doi)
            self.d_real.append(do)
        print(np.shape(self.d_real))


        
    def generate_scenarios_from_dict(self):
        self.d_real = {}
        for o in self.O:
            for i in self.N:
                di = [self.d_pred[i, t] for t in self.T]
                std_dev = np.std(di)
                for t in self.T:
                    perturbed = max(0, self.d_pred[i, t] + std_dev)
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



    def build_model1(self):
        """
        Le BON
        
        """
        self.model = Model("VehicleAllocation")

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
                    quicksum(self.q[k] * (x[i, k, t] + s[i, k, t]) for k in self.Ki) >= self.d_pred[i, t],
                    name=f"pred_demand_{i}_{t}"
                )
                self.model.addConstr(
                    quicksum(self.g[k] * self.q[k] * (x[i, k, t] + s[i, k, t]) for k in self.Ki) >= self.theta[i] * self.d_pred[i, t],
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
                        inflow_k = quicksum(self.q[k]* y[j, i, k, t - 1, o] for j in self.N for k in self.Ki)
                        outflow_k = quicksum(self.q[k]*y[i, j, k, t - 1, o] for j in self.N for k in self.Ki)
                    self.model.addConstr(
                        quicksum(self.q[k] * (
                            x[i, k, t]  + s[i, k, t] + s_corr[i, k, t, o]) for k in self.Ki) + inflow_k - outflow_k
                         >= self.d_real[i, t, o],
                        name=f"real_demand_{i}_{k}_{t}_{o}"
                    )
                    self.model.addConstr(
                        quicksum(self.g[k] * self.q[k] * (
                            x[i, k, t]+ s[i, k, t] + s_corr[i, k, t, o]) for k in self.Ki) + inflow_k - outflow_k 
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

        

    
        
    def solve(self,log:bool,timelimit):
        """
        marche avec build1
        """
        if log:
            self.build_model1()
            self.model.setParam("TimeLimit", timelimit)
            self.model.setParam("OutputFlag", 0)
            self.model.setParam("LogFile", "gurobi_log.txt")
            print('opt start')
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                print("✅ Optimal solution found:", self.model.ObjVal)
            else:
                print("❌ No optimal solution.")
        else:
            self.build_model1()
            #self.model.setParam("TimeLimit", 200)
            print('opt start')
            self.model.optimize()
            if self.model.status == GRB.OPTIMAL:
                print("✅ Optimal solution found:", self.model.ObjVal)
            else:
                print("❌ No optimal solution.")


    

    
    def _get_val(self, var_name):
        var = self.model.getVarByName(var_name)
        return var.X if var else 0.0



    def export_solution_summary1(self, filename="solution_summary.txt"):
        with open(filename, "w") as f:
            def write(line=""):
                f.write(line + "\n")

            write("Resource Allocation, Rebalancing Plan, Outsourcing, and Demand Analysis")
            write("=" * 80)

            write("\nAllocated Resources:")
            for i in self.N:
                allocations = [f"{self._get_val(f'x[{i},{k}]'):.0f} of type {k}" for k in self.K]
                write(f"Hub {i} has: " + ", ".join(allocations))

            
            write("\n" + "=" * 80 + "\n\n Anticipated Outsourced Resources:")
            for t in self.T:
                write(f"Week {t+1}:")
                found = False
                for i in self.N:
                    for k in self.K:
                        val = self._get_val(f's[{i},{k},{t}]')
                        if val >= 0.5:
                            write(f"  Hub {i} outsourced {int(val)} of type {k} (week {t})")
                            found = True
                if not found:
                    write("  No outsourcing")
            
            
            
            write("\n" + "=" * 80 + "\n\nRebalancing Plan:")
            for t in self.T:
                write(f"Week {t+1}:")
                found = False
                for o in self.O:
                    for i in self.N:
                        for j in self.N:
                            if i != j:
                                for k in self.K:
                                    name = f"y[{i},{j},{k},{t},{o}]"
                                    val = self._get_val(name)
                                    if val >= 0.5:
                                        write(f"  Hub {i} sends {int(val)} of type {k} to Hub {j} (Scenario {o})")
                                        found = True
                if not found:
                    write("  No transfers")   
            
            write("\n" + "=" * 80 + "\n\nOutsourced Resources Corrected:")
            for t in self.T:
                write(f"Week {t+1}:")
                found = False
                for o in self.O:
                    for i in self.N:
                        for k in self.K:
                            val = self._get_val(f"s_corr[{i},{k},{t},{o}]")
                            if val >= 0.5:
                                write(f"  Hub {i} outsourced {int(val)} of type {k} (Scenario {o}, week {t})")
                                found = True
                if not found:
                    write("  No outsourcing")

            write("\n" + "=" * 80 + "\n\nDemand Fulfillment Summary:")
            for t in self.T:
                for o in self.O:
                    for i in self.N:
                        total_available = sum(
                            self.q[k] * (self._get_val(f"v[{i},{k},{t},{o}]")
                                        + self._get_val(f"s[{i},{k},{t}]")
                                        + self._get_val(f"s_corr[{i},{k},{t},{o}]"))
                            for k in self.K)
                        demand = self.d_real[i, t, o]
                        write(f"Hub {i} (t={t}, scenario={o}): demand={demand}, fulfilled={int(total_available)}")

            write("\n" + "=" * 80 + "\n\nGreen Vehicle Contribution:")
            for t in self.T:
                for i in self.N:
                    green_capacity = sum(
                        self.g[k] * self.q[k] * (
                            self._get_val(f"x[{i},{k}]") +
                            self._get_val(f"s[{i},{k},{t}]") +
                            self._get_val(f"s_corr[{i},{k},{t},0]"))
                        for k in self.K)
                    required = self.theta[i] * self.d_pred[i, t]
                    write(f"Hub {i} (t={t}): Green capacity = {green_capacity:.1f}, Required = {required:.1f}")

    def export_solution_summaryuiui(self, filename="solution_summary.txt"): # simple et efficace
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






    


if __name__ == "__main__":
    """M = VehicleAllocationModel(100,3,25,40,seed=111) #(self, N, K, T, O, seed=42):
    #M.DataManoTest(True)
    start = time.time()
    M.generate_data()
    M.solve(False)
    end = time.time()

    runtime = round(end - start, 2)
    print(f'solve + gen data = {runtime}')"""
    #M.export_solution_summaryuiui(filename="ui.txt")
    #M.plot_solution_map(scenario=0)
    #M.plot_rebalancing_solution(scenario=0)
    #plot_solution_graph_from_file(filename="bismillah.txt",N=5,T=5)


