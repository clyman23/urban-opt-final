"""
Example with advanced indexing (for n>2 variables)
"""

import cvxpy as cp
import numpy as np

T = 3
V = 2
S = 5

c = np.array([4, 1])

z = {}
r_plus = {}
r_minus = {}
for t in range(T):
    z[t] = cp.Variable((S, V), boolean=True, name="z-{}".format(t))
    r_plus[t] = cp.Variable((S, V), name="r_plus-{}".format(t))
    r_minus[t] = cp.Variable((S, V), name="r_minus-{}".format(t))

constraints = [
    r_plus[t][s, :] + r_minus[t][s, :] <= cp.multiply(c, z[t][s, :])
    for t in range(T) for s in range(S)
]

str(constraints[0])
