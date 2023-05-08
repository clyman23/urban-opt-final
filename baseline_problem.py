"""
Testing the baseline bike rebalancing problem
"""
import cvxpy as cp
import numpy as np
import pandas as pd

# -----Sets / indices-----
T = 3 # Num time periods
S = 5 # Num stations
V = 2 # Num vehicles

# -----Parameters / input data-----
D_ij = [] # Distance between stations i and j; may not be used
C_s = np.array([10, 10, 10, 10, 10]) # Capacity of each station s
C_hat_v = np.array([10, 10]) # Capacity of each vehicle v
L_t = [] # Length in minutes of time-period t (can probably just be a constant); may not be used
d_s_1 = np.array([5, 9, 10, 4, 3]) # Initial num bikes at each station s
d_hat_v_1 = np.array([3, 10]) # Initial num bikes in each vehicle v
z_sv_1 = np.array([]) # Initial conditions of z; 1 if vehicle v initially at station s; may not be used
f_plus = np.array([
    [11, 1, 1, 14, 3],
    [6, 14, 24, 1, 5],
    [0, 0, 3, 3, 7]
]).T # Expected rental demand at station s at time t
f_minus = np.array([
    [4, 2, 1, 5, 2],
    [0, 10, 3, 6, 0],
    [0, 0, 0, 1, 1]
]).T # Expected return demand at station s at time t

# -----Variables-----
d = cp.Variable((S, T)) # Num bikes at station s at time t
d_hat = cp.Variable((V, T)) # Num bikes in vehicle v at time t
x_plus = cp.Variable((S, T)) # Num of successful bike trips starting at stations s at time t
x_minus = cp.Variable((S, T)) # Num of successful returns starting at stations s at time t
# Need to Frankenstein n-dimensional variables for n>2 in cvxpy...
r_plus = {} # Num of bikes vehicle v picks up at stations s at time t
r_minus = {} # Num of bikes vehicle v unloads up at stations s at time t
z = {} # 1 if vehicle v is at station s at time t
for t in range(T):
    r_plus[t] = cp.Variable((S, V))
    r_minus[t] = cp.Variable((S, V))
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

# z - each vehicle can only be in one location at a time
z_constraint = [
    sum([z[t][s, :] for s in range(S)]) == 1
    for t in range(T)
]

# Capacity constraint
capacity_constraint = [
    r_plus[t][s, :] + r_minus[t][s, :] <= cp.multiply(C_hat_v,  z[t][s, :])
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

d_limits = [
    d >= 0
]
d_limits.extend([
    d[:, t] <= C_s
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

r_plus_limits = [
    0 <= r_plus[t] for t in range(T)
]

r_minus_limits = [
    r_minus[t][s, :] <= C_hat_v
    for s in range(S) for t in range(T)
]
r_minus_limits.extend([
    r_minus[t][s, :] >= 0
    for s in range(S) for t in range(T)
])

# Sum all constraints
constraints = (
    d_hat_constraint + d_constraint + z_constraint + capacity_constraint + d_hat_limits
    + d_limits + x_plus_limits + x_minus_limits + r_plus_limits + r_minus_limits
)

# -----Objective-----
objective = (
    sum([f_plus - x_plus])
    + sum([f_minus - x_minus])
)

objective = cp.Minimize(sum(sum(objective)))

# -----Define problem-----
prob = cp.Problem(objective, constraints)

# -----Define problem-----
prob.solve(solver=cp.GUROBI, verbose=True)

# -----Visualize results-----
station_ids = [f"s{i}" for i in range(S)]
vehicle_ids = [f"v{i}" for i in range(V)]
time_ids = [f"t{i}" for i in range(T)]

print("-----Num bikes in each station over time-----")
print(pd.DataFrame(d.value, index=station_ids, columns=time_ids))

print("-----Num of bikes in each vehicle-----")
print(pd.DataFrame(d_hat.value, index=vehicle_ids, columns=time_ids))

print("-----Where the vehicles are over time-----")
for t in range(T):
    print(f"Time = {t}")
    print(pd.DataFrame(z[t].value, index=station_ids, columns=vehicle_ids))

print("-----How many bikes vehicles pick up-----")
for t in range(T):
    print(f"Time = {t}")
    print(pd.DataFrame(r_plus[t].value, index=station_ids, columns=vehicle_ids))

print("-----How many bikes vehicles drop off-----")
for t in range(T):
    print(f"Time = {t}")
    print(pd.DataFrame(r_minus[t].value, index=station_ids, columns=vehicle_ids))

print("-----Successful trips-----")
print(pd.DataFrame(x_plus.value, index=station_ids, columns=time_ids))

print("-----Successful returns-----")
print(pd.DataFrame(x_minus.value, index=station_ids, columns=time_ids))

lost_rental_demand =  f_plus - x_plus.value
print("-----Lost rental demand-----")
print(pd.DataFrame(lost_rental_demand, index=station_ids, columns=time_ids))

lost_return_demand = f_minus - x_minus.value
print("-----Lost return demand-----")
print(pd.DataFrame(lost_return_demand, index=station_ids, columns=time_ids))

print("-----Objective value-----")
print(objective.value)


print("==========Results over time==========")
for t in range(T):
    print(f"*****In time period {t}:*****")
    for v in range(V):
        print(f"Vehicle {v} is at station: {z[t].value[:, v]}")
        print(f"Vehicle {v} has {d_hat.value[v, t]} bikes in the vehicle")
        print(f"Vehicle {v} leaves {r_minus[t].value[:, v]} at the station")
        print(f"Vehicle {v} picks up {r_plus[t].value[:, v]} at the station")
    for s in range(S):
        print(f"At station {s}:")
        print(f"There are {d.value[s, t]} bikes already")
        print(f"We expect a rental demand of {f_plus[s, t]}")
        print(f"We expect a return demand of {f_minus[s, t]}")
        print(f"We have {x_plus.value[s, t]} successful trips leaving the station")
        print(f"We have {x_minus.value[s, t]} successful returns to the station")
        print()
