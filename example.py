import cvxpy as cp
import numpy as np

t = 3
p = 2

cost = np.array([
    [1, 4, 1],
    [3, 5, 5.5]
])

revenue = cp.Variable(t, name="revenue")
producing = cp.Variable((p, t), boolean=True, name="is_producing")

production_constraint = [sum(producing[:, i] for i in range(t)) <= 1]
revenue_constraint = [
    revenue == sum(cp.multiply(cost[i, :], producing[i, :]) for i in range(p))
]
constraints = production_constraint + revenue_constraint

total_revenue = sum(revenue)

objective = cp.Minimize(-total_revenue)

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI)

print(revenue.value)
print(producing.value)
print(total_revenue.value)
