import cvxpy as cp
import numpy as np

t = 3

cost = np.array([1, 4, 1])

revenue = cp.Variable(t)
producing = cp.Variable(t, boolean=True)

production_constraint = [sum(producing) <= 1]
revenue_constraint = [revenue == cp.multiply(cost, producing)]
constraints = production_constraint + revenue_constraint

objective = cp.Minimize(-sum(revenue))

prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.GUROBI)

print(revenue.value)
print(producing.value)
