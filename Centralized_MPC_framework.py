# ---- Initiate all required packages ----
import gurobipy as gp
from gurobipy import GRB
import pandas as pd 
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import scipy.stats as stats
import json

# ---- Create a new model ----
m = gp.Model("MPC")

# ---- Prediction Horizon ---- 
N = 168

# ---- Define the container availability probability distribution ----
kans = [0, 1]
sequence = np.random.choice([0, 1], size=N, p=kans)

# ---- Bounds ----
umax = np.array([200,400,400])  
xmax = np.array([200, 200, 400, 400, 400, 5000, 10000])
x_max_total = np.array([600, 600, 600, 600, 20000, 30000, 50000])


# ---- Insert Data ----
df = pd.read_csv("YOUR_DATA.csv")   # pallet production data
df = df.loc[:, (df != 0).any(axis=0)]
df_sku = pd.read_csv("YOUR_SKU_DATA.csv") # SKU specific data


# ----  Production Data and Container Data Preperation ---- 
df  = df.drop(['Unnamed: 0'], axis=1)
df_sku = df_sku.set_index(df_sku['Material\nMaterial'])
df_sku.index = df_sku.index.astype('object')
df_sku = df_sku.T
df_sku.columns = df_sku.columns.map(lambda x: f'{x}')
common_cols = [col for col in df.columns if col in df_sku.columns]
df1 = df[common_cols]
df2 = df_sku[common_cols]
df2 = df2.reindex(columns=df1.columns)
df2 = df2.drop(df2.index[0])

# ---- DataFrames to lists ----
df_array = np.array(df1.values.reshape(-1, len(df1.columns), 1).tolist())
df_sequence = np.array(df2.values.reshape(-1, len(df2.columns), 1).tolist())

# ---- numbers of SKUs in data ---- 
num_products = df_array.shape[1]

# ---- Setting initial conditions ---- 
array1 = 3*df_array[0]
arr2 = np.zeros_like(array1)
arr3 = np.full_like(array1, 400)
arr4 = np.zeros_like(array1)
x_initial = np.array([array1, array1, array1, array1, array1, arr4, arr2])
x_initial = x_initial.T

# ---- Create variables ----
x = []
u = []
l = []

for i in range(num_products):
    x_i = np.array([])
    u_i = np.array([])
    l_i = np.array([])
 
    for k in range(N+1):
        x_i = np.append(x_i,[m.addMVar(7, lb=0, vtype=GRB.CONTINUOUS)])
        u_i = np.append(u_i,[m.addMVar(3, lb=0, vtype=GRB.CONTINUOUS)])
        l_i = np.append(l_i,[m.addMVar(9, lb=0, vtype=GRB.BINARY)])

    l.append(l_i)
    x.append(x_i)
    u.append(u_i)

# ---- Cost Function ----

Q = sp.diags([10, 10, 10, 10, 100, 100, 1])

# ---- Define The System Matrices For Each Product ---- 
A_lin   =    []
Bu_lin  =    []
Bl_lin  =    []
Bp_lin  =    []

for i in range(num_products):
    A_lin_i = sp.csc_matrix([
        [0., 0., 0., 0, 0, 0, 0],
        [1, 1, 0., 0, 0, 0, 0],
        [0., 0., 0., 0, 0, 0, 0],
        [0., 0., 1, 1, 0, 0, 0],
        [0., 0., 0., 0, 0, 0, 0],
        [0., 0., 0, 0,1, 1, 0],
        [0., 0., 0., 0, 0, 0, 1]])

    Bu_lin_i = sp.csc_matrix([
                        [0,0,0],
                        [-1,0,0],
                        [1, 0, 1],
                        [0,-1,0],
                        [0,1,0],
                        [0,0,-1],
                        [0,0,0]
                        ])

    Bl_lin_i = sp.csc_matrix([
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [-1,-1,-1,-1,-1,-1,-1,-1,-1],
                        [0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0],
                        [1,1,1,1,1,1,1,1,1],])

    Bp_lin_i = sp.csc_matrix([
                        [1],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0],
                        [0]])
    
    A_lin.append(A_lin_i)
    Bu_lin.append(Bu_lin_i)
    Bl_lin.append(Bl_lin_i)
    Bp_lin.append(Bp_lin_i)

print ('\n--------------------------------------------------------------------\n')

print('\n Number of SKUs in production data: \n', num_products)
print('\n Number of produced pallets in production data: \n', df_array.sum().sum())


print ('\n--------------------------------------------------------------------\n')

# ---- MPC Formulation ----

obj = 0

for k in range(N):

    for p in range(num_products):
        obj = obj + x[p][k] @ Q 
        m.addConstr(u[p][k+1] <= umax)
        m.addConstr(sum(x[p][k+1] for p in range(num_products)) <= x_max_total)
        m.addConstr(x[p][k+1] <= xmax)
        m.addConstr(x[p][0]   == x_initial[0][p])
        m.addConstr(x[p][k+1] == A_lin[p] @ x[p][k] + Bu_lin[p] @ u[p][k] + Bp_lin[p] @ df_array[k][p] + Bl_lin[p] @ l[p][k]*sequence[k]*df_sequence[0][p])
        m.addConstr(sum(l[p][k] for p in range(num_products)) <= 1)

m.setParam("MIPGAP", 0.0005)
m.setObjective(sum(obj))
m.modelSense = GRB.MINIMIZE
m.update()
m.optimize()

gap = m.getAttr("MIPGap")
print(f"Optimization Gap: {gap}")

# ---- Print results ----
print ('\n--------------------------------------------------------------------\n')

print('\n Number of SKUs in production data is \n', num_products)
    
if m.status == GRB.Status.OPTIMAL: # If optimal solution is found
    print ('Total Cost : %10.2f' % m.objVal)
    x[0][0].X
    dict_state, dict_action, dict_truck = {}, {}, {}
    for sku in range(num_products):
        state_data = pd.DataFrame()
        action_data = pd.DataFrame()
        truck_data = pd.DataFrame()
        for k in range(N+1):
            state = x[sku][k].X
            action = u[sku][k].X
            truck = l[sku][k].X
            state = pd.DataFrame(np.reshape(state, (1,len(state))))
            action = pd.DataFrame(np.reshape(action, (1,len(action))))
            truck = pd.DataFrame(np.reshape(truck, (1,len(truck))))
            state_data = state_data.append(state)
            action_data = action_data.append(action)
            truck_data = truck_data.append(truck)
        dict_state[sku] = state_data
        dict_action[sku] = action_data
        dict_truck[sku] = truck_data

# ---- Save Model Output in .json Files ----

dict_state = {k: v.to_dict("records") for k, v in dict_state.items()}
dict_action = {k: v.to_dict("records") for k, v in dict_action.items()}
dict_truck = {k: v.to_dict("records") for k, v in dict_truck.items()}

json_string1 = json.dumps(dict_state)
json_string2 = json.dumps(dict_action)
json_string3 = json.dumps(dict_truck)

with open("dict_state_run8_9trucks.json", "w") as f:
    f.write(json_string1)

with open("dict_action_run8_9trucks.json", "w") as f:
    f.write(json_string2)

with open("dict_truck_run8_9trucks.json", "w") as f:
    f.write(json_string3)

