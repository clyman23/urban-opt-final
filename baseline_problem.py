"""
Testing the baseline bike rebalancing problem
"""
import cvxpy as cp
import numpy as np

# -----Sets / indices-----
t = 3 # Num time periods
s = 4 # Num stations
v = 1 # Num vehicles

# -----Parameters / input data-----
D_ij = [] # Distance between stations i and j; may not be used
C_s = [] # Capacity of each station s
C_hat_v = [] # Capacity of each vehicle v
L_t = [] # Length in minutes of time-period t (can probably just be a constant); may not be used
d_s_1 = [] # Initial num bikes at each station s
d_hat_v_1 = np.ones(v) # Initial num bikes in each vehicle v
z_sv_1 = [] # Initial conditions of z; 1 if vehicle v initially at station s
f_plus = [] # Expected rental demand at station s at time t
f_minus = [] # Expected return demand at station s at time t

# -----Variables-----
d = cp.Variable((s, t)) # Num bikes at station s at time t
d_hat = cp.Variable((v, t)) # Num bikes in vehicle v at time t
x_plus = cp.Variable((s, t)) # Num of successful bike trips starting at stations s at time t
x_minus = cp.Variable((s, t)) # Num of successful returns starting at stations s at time t
r_plus = cp.Variable((s, v, t)) # Num of bikes vehicle v picks up at stations s at time t
r_minus = cp.Variable((s, v, t)) # Num of bikes vehicle v unloads up at stations s at time t
z = cp.Variable((s, v, t), integer=True) # 1 if vehicle v is at station s at time t

# -----Constraints-----
constraints = []
# Initial d_hat
d_hat_constraint = [
    d_hat[:, 0] == d_hat_v_1
]
# Other d_hat
d_hat_constraint.extend([
    d_hat[:, i+1] == (
        d_hat[:, i]
        + sum([r_plus[j, :, i] - r_minus[j, :, i] for j in range(s)])
    )

    for i in range(t)
])

# prob = cp.Problem()
# prob.solve(solver=cp.GUROBI)