import pickle
from Model import VehicleAllocationModel

with open("experimentsIPIC/exp3_2_30_100_.../st_y.pkl", "rb") as f:
    st_y = pickle.load(f)

VehicleAllocationModel.rebalancing_plan(st_y, scenario=4, t_start=5, t_end=15)
