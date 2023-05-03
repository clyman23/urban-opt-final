"""
Testing the baseline bike rebalancing problem
"""
import cvxpy as cp
import numpy as np

# -----Sets / indices-----
T = 3 # Num time periods
S = 5 # Num stations
V = 2 # Num vehicles

# -----Parameters / input data-----
D_ij = [] # Distance between stations i and j; may not be used
C_s = np.array([10, 10, 10, 10, 10]) # Capacity of each station s
C_hat_v = np.array([20, 20]) # Capacity of each vehicle v
L_t = [] # Length in minutes of time-period t (can probably just be a constant); may not be used
d_s_1 = np.array([5, 9, 10, 4, 3]) # Initial num bikes at each station s
d_hat_v_1 = np.array([3, 10]) # Initial num bikes in each vehicle v
z_sv_1 = np.array([]) # Initial conditions of z; 1 if vehicle v initially at station s; may not be used
f_plus = np.array([
    [6, 1, 1, 1, 1],
    [0, 4, 1, 1, 1],
    [0, 0, 0, 3, 7]
]).T # Expected rental demand at station s at time t
f_minus = np.array([
    [0, 2, 1, 0, 2],
    [0, 0, 3, 0, 0],
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
        + sum([r_plus[t][:, v] - r_minus[t][:, v] for v in range(V)])
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
print("Num bikes in each station over time:")
print(d.value)

print("Where the vehicles are over time:")
for t in range(T):
    print(z[t].value)
