"""
Testing the baseline bike rebalancing problem
"""
import cvxpy as cp
import numpy as np
import pandas as pd

# -----Sets / indices-----
T = 30 # Num time periods
S = 30 # Num stations
V = 1 # Num vehicles

# -----Reading input data-----
# Initial inventory
INVENTORY_FILEPATH = "./Initial_Inven.json"
inven_init = pd.read_json(INVENTORY_FILEPATH)
RENTALS_FILEPATH = "./rentals_classic.csv"
rentals_df = pd.read_csv(RENTALS_FILEPATH, usecols=["rentals", "time_period", "station_id"])
RENTALS_E_FILEPATH = "./rentals_electric.csv"
rentals_e_df = pd.read_csv(RENTALS_E_FILEPATH, usecols=["rentals", "time_period", "station_id"])
RETURNS_FILEPATH = "./returns_classic.csv"
returns_df = pd.read_csv(RETURNS_FILEPATH, usecols=["returns", "time_period", "station_id"])
RETURNS_E_FILEPATH = "./returns_electric.csv"
returns_e_df = pd.read_csv(RETURNS_E_FILEPATH, usecols=["returns", "time_period", "station_id"])


# -----Setting parameters / input data-----
C_s = np.concatenate([[40] * 5, [20] * 25]) # Capacity of each station s
C_s = C_s[:S] # Subset number of stations for a toy model
C_hat_v = np.array([40, 40, 40, 40, 40]) # Capacity of each vehicle v for classic bikes
C_hat_v = C_hat_v[:V] # Subset when we want a toy model with a small number of vehicles
C_tilde_v = np.array([40, 40, 40, 40, 40]) # Capacity of each vehicle for e-bikes
C_tilde_v = C_tilde_v[:V]

d_s_1 = inven_init[0].to_numpy() - 3 # Initial num of classic bikes at station s
d_s_1_mask = d_s_1 < 0
d_s_1[d_s_1_mask] = 0
d_s_1 = d_s_1[:S]
d_bar_s_1 = np.ones(S) * 3 # Initial num of e-bikes at station s
d_hat_v_1 = np.array([0, 0, 3, 3, 1]) # Initial num classic bikes in each vehicle v
d_hat_v_1 = d_hat_v_1[:V] # Subset when we want a toy model with a small number of vehicles
d_tilde_v_1 = np.array([0, 0, 2, 3, 0]) # Initial num e-bikes in each vehicle v
d_tilde_v_1 = d_tilde_v_1[:V] # Subset when we want a toy model with a small number of vehicles

a_classic = 1 # Value of classic ride
a_electric = 2 # Value of electric ride
w_s = np.random.randint(1, size=(S, T)) # Num "dead" e-bikes at each station at time t

time_periods = list(range(96*500))
f_plus = pd.pivot_table(
    rentals_df, values="rentals", index="station_id", columns="time_period"
).reindex(time_periods, axis='columns').fillna(0).to_numpy()[:S, :T] # Subset when we want a toy model with a small number of time periods
f_minus = pd.pivot_table(
    returns_df, values="returns", index="station_id", columns="time_period"
).reindex(time_periods, axis='columns').fillna(0).to_numpy()[:S, :T] # Subset when we want a toy model with a small number of time periods
f_bar_plus = pd.pivot_table(
    rentals_e_df, values="rentals", index="station_id", columns="time_period"
).reindex(time_periods, axis='columns').fillna(0).to_numpy()[:S, :T]
f_bar_minus = pd.pivot_table(
    returns_e_df, values="returns", index="station_id", columns="time_period"
).reindex(time_periods, axis='columns').fillna(0).to_numpy()[:S, :T]

# -----Variables-----
d = cp.Variable((S, T), integer=True) # Num classic bikes at station s at time t
d_bar = cp.Variable((S, T), integer=True) # Num e-bikes at s at t
d_hat = cp.Variable((V, T), integer=True) # Num classic bikes in vehicle v at time t
d_tilde = cp.Variable((V, T), integer=True) # Num e-bikes in v at t

x_plus = cp.Variable((S, T), integer=True) # Num of successful classic bike trips starting at stations s at time t
x_minus = cp.Variable((S, T), integer=True) # Num of successful classic returns starting at stations s at time t
x_bar_plus = cp.Variable((S, T), integer=True) # Num of successful e-bike trips starting at stations s at time t
x_bar_minus = cp.Variable((S, T), integer=True) # Num of successful e-bike returns starting at stations s at time t

# Need to Frankenstein n-dimensional variables for n>2 in cvxpy...
r_plus = {} # Num of bikes vehicle v picks up at stations s at time t
r_minus = {} # Num of bikes vehicle v unloads up at stations s at time t
r_bar_plus = {} # Num e-bikes vehicle v picks up at s at t
r_bar_minus = {} # Num e-bikes vehicle v drops off at s at t
z = {} # 1 if vehicle v is at station s at time t
for t in range(T):
    r_plus[t] = cp.Variable((S, V), integer=True)
    r_minus[t] = cp.Variable((S, V), integer=True)
    r_bar_plus[t] = cp.Variable((S, V), integer=True)
    r_bar_minus[t] = cp.Variable((S, V), integer=True)
    z[t] = cp.Variable((S, V), boolean=True)


# -----Constraints-----
# Initial d_hat
d_hat_constraint = [
    d_hat[:, 0] == d_hat_v_1
]
# Other d_hat
d_hat_constraint.extend([
    d_hat[:, t+1] == (
        d_hat[:, t]
        + sum([r_plus[t][s, :] - r_minus[t][s, :] for s in range(S)])
    )

    for t in range(T-1)
])

# Initial d_tilde
d_tilde_constraint = [
    d_tilde[:, 0] == d_tilde_v_1
]
# Other d_tilde
d_tilde_constraint.extend([
    d_tilde[:, t+1] == (
        d_tilde[:, t]
        + sum([r_bar_plus[t][s, :] - r_bar_minus[t][s, :] for s in range(S)])
    )

    for t in range(T-1)
])

# Initial d
d_constraint = [
    d[:, 0] == d_s_1
]
# Other d
d_constraint.extend([
    d[:, t+1] == (
        d[:, t]
        - sum([r_plus[t][:, v] - r_minus[t][:, v] for v in range(V)])
        - x_plus[:, t] + x_minus[:, t]
    )

    for t in range(T-1)
])

# Initial d_bar
d_bar_constraint = [
    d_bar[:, 0] == d_bar_s_1
]
# Other d_bar
d_bar_constraint.extend([
    d_bar[:, t+1] == (
        d_bar[:, t]
        - sum([r_bar_plus[t][:, v] - r_bar_minus[t][:, v] for v in range(V)])
        - x_bar_plus[:, t] + x_bar_minus[:, t] - w_s[:, t]
    )

    for t in range(T-1)
])

# z - each vehicle can only be in one location at a time
z_constraint = [
    sum([z[t][s, :] for s in range(S)]) == 1
    for t in range(T)
]

# Capacity constraint classic
capacity_constraint = [
    r_plus[t][s, :] + r_minus[t][s, :] <= cp.multiply(C_hat_v,  z[t][s, :])
    for s in range(S) for t in range(T)
]
# Capacity constraint electric
capacity_electric_constraint = [
    r_bar_plus[t][s, :] + r_bar_minus[t][s, :] <= cp.multiply(C_tilde_v,  z[t][s, :])
    for s in range(S) for t in range(T)
]

# Limits on variables
d_hat_limits = [
    d_hat >= 0
]
d_hat_limits.extend([
    d_hat[:, t] <= C_hat_v
    for t in range(T)
])

d_tilde_limits = [
    d_tilde >= 0
]
d_tilde_limits.extend([
    d_tilde[:, t] <= C_tilde_v
    for t in range(T)
])

d_limits = [
    d >= 0 #TODO: Need to index here?
]
d_limits.extend([d_bar >= 0])
d_limits.extend([
    d[:, t] + d_bar[:, t] <= C_s
    for t in range(T)
])

x_plus_limits = [
    x_plus >= 0
]
x_plus_limits.extend([
    x_plus <= f_plus
])

x_minus_limits = [
    x_minus >= 0
]
x_minus_limits.extend([
    x_minus <= f_minus
])

x_bar_plus_limits = [
    x_bar_plus >= 0
]
x_bar_plus_limits.extend([
    x_bar_plus <= f_bar_plus
])

x_bar_minus_limits = [
    x_bar_minus >= 0
]
x_bar_minus_limits.extend([
    x_bar_minus <= f_bar_minus
])

r_plus_limits = [
    0 <= r_plus[t] for t in range(T)
]
r_plus_limits.extend([
    r_plus[t][s, :] <= C_hat_v
    for s in range(S) for t in range(T)
])

r_minus_limits = [
    r_minus[t][s, :] <= C_hat_v
    for s in range(S) for t in range(T)
]
r_minus_limits.extend([
    r_minus[t][s, :] >= 0
    for s in range(S) for t in range(T)
])

r_bar_plus_limits = [
    0 <= r_bar_plus[t] for t in range(T)
]
r_bar_plus_limits.extend([
    r_bar_plus[t][s, :] <= C_tilde_v
    for s in range(S) for t in range(T)
])

r_bar_minus_limits = [
    r_bar_minus[t][s, :] <= C_tilde_v
    for s in range(S) for t in range(T)
]
r_bar_minus_limits.extend([
    r_bar_minus[t][s, :] >= 0
    for s in range(S) for t in range(T)
])

# Sum all constraints
constraints = (
    d_hat_constraint + d_tilde_constraint + d_constraint + d_bar_constraint + z_constraint
    + capacity_constraint + capacity_electric_constraint + d_hat_limits + d_tilde_limits
    + d_limits + x_plus_limits + x_minus_limits + x_bar_plus_limits + x_bar_minus_limits
    + r_plus_limits + r_minus_limits + r_bar_plus_limits + r_bar_minus_limits
)

# -----Objective-----
objective = (
    (sum(sum(f_plus - x_plus)) + sum(sum(f_minus - x_minus))) * a_classic
    + (sum(sum(f_bar_plus - x_bar_plus)) + sum(sum(f_bar_minus - x_bar_minus))) * a_electric
)

objective = cp.Minimize(objective)

# -----Define problem-----
prob = cp.Problem(objective, constraints)

# -----Define problem-----
prob.solve(solver=cp.GUROBI, verbose=True)

# -----Visualize results-----
station_ids = [f"s{i}" for i in range(S)]
vehicle_ids = [f"v{i}" for i in range(V)]
time_ids = [f"t{i}" for i in range(T)]

print("-----Num classic bikes in each station over time-----")
print(pd.DataFrame(d.value, index=station_ids, columns=time_ids))

print("-----Num e-bikes in each station over time-----")
print(pd.DataFrame(d_bar.value, index=station_ids, columns=time_ids))

print("-----Num of bikes in each vehicle-----")
print(pd.DataFrame(d_hat.value, index=vehicle_ids, columns=time_ids))

print("-----Num of e-bikes in each vehicle-----")
print(pd.DataFrame(d_tilde.value, index=vehicle_ids, columns=time_ids))

# print("-----How many e-bikes vehicles pick up-----")
# for t in range(T):
#     print(f"Time = {t}")
#     print(pd.DataFrame(r_bar_plus[t].value, index=station_ids, columns=vehicle_ids))

print("-----Successful classic trips-----")
print(pd.DataFrame(x_plus.value, index=station_ids, columns=time_ids))

print("-----Successful classic returns-----")
print(pd.DataFrame(x_minus.value, index=station_ids, columns=time_ids))

print("-----Successful e-bike trips-----")
print(pd.DataFrame(x_bar_plus.value, index=station_ids, columns=time_ids))

print("-----Successful e-bike returns-----")
print(pd.DataFrame(x_bar_minus.value, index=station_ids, columns=time_ids))

lost_rental_demand =  f_plus - x_plus.value
print("-----Lost classic rental demand-----")
print(pd.DataFrame(lost_rental_demand, index=station_ids, columns=time_ids))

lost_return_demand = f_minus - x_minus.value
print("-----Lost classic return demand-----")
print(pd.DataFrame(lost_return_demand, index=station_ids, columns=time_ids))

lost_ebike_rental_demand =  f_bar_plus - x_bar_plus.value
print("-----Lost e-bike rental demand-----")
print(pd.DataFrame(lost_ebike_rental_demand, index=station_ids, columns=time_ids))

lost_ebike_return_demand = f_bar_minus - x_bar_minus.value
print("-----Lost e-bike return demand-----")
print(pd.DataFrame(lost_ebike_return_demand, index=station_ids, columns=time_ids))
print("-----Ebike + Classic-----")
print()
print("-----Objective value-----")
print(objective.value)
print()

def visualize():
    print("-----Trip info-----")
    print('Successful classic trips:', np.sum(x_plus.value))
    print('Successful classic returns:', np.sum(x_minus.value))
    print('Successful ebike trips:', np.sum(x_bar_plus.value))
    print('Successful ebike returns:', np.sum(x_bar_minus.value))
    print()
    print('Lost classic rental demand:', np.sum(lost_rental_demand))
    print('Lost classic return demand:', np.sum(lost_return_demand))
    print('Lost ebike rental demand:', np.sum(lost_ebike_rental_demand))
    print('Lost ebike return demand:', np.sum(lost_ebike_return_demand))
    print()

    print('Successful total trips:', np.sum(x_plus.value) + np.sum(x_bar_plus.value))
    print('Successful total returns:', np.sum(x_minus.value)+np.sum(x_bar_minus.value))
    print()
    diff = 0
    print("-----Station Volatility -----")
    for s in range(S):
        diff += np.max(d.value[s] + d_bar.value[s]) - np.min(d.value[s] + d_bar.value[s])
    print(f'total volatility of classic + ebikes: {diff}')
    print()
    vehs = [[] for y in range(V)]
    for t in range(T):
        # print(f"Time = {t}")
        # print(pd.DataFrame(z[t].value, index=station_ids, columns=vehicle_ids))
        for v in range(V):
            if np.where(z[t].value!=0)[0][v] not in vehs[v]:
                vehs[v] = np.hstack((vehs[v],np.where(z[t].value!=0)[0][v]))
    vehs = vehs[::-1] # swap axes.

    print("-----Where did the Vehicles Go?-----")
    for v in range(V):
        cnt = len(vehs[v])
        print(f'Total unique stations visited by vehicle {v}: {cnt}')

visualize()

# print("==========Results over time==========")
# for t in range(T):
#     print(f"*****In time period {t}:*****")
#     for v in range(V):
#         print(f"Vehicle {v} is at station: {z[t].value[:, v]}")
#         print(f"Vehicle {v} has {d_hat.value[v, t]} bikes in the vehicle")
#         print(f"Vehicle {v} leaves {r_minus[t].value[:, v]} at the station")
#         print(f"Vehicle {v} picks up {r_plus[t].value[:, v]} at the station")
#     for s in range(S):
#         print(f"At station {s}:")
#         print(f"There are {d.value[s, t]} bikes already")
#         print(f"We expect a rental demand of {f_plus[s, t]}")
#         print(f"We expect a return demand of {f_minus[s, t]}")
#         print(f"We have {x_plus.value[s, t]} successful trips leaving the station")
#         print(f"We have {x_minus.value[s, t]} successful returns to the station")
#         print()
