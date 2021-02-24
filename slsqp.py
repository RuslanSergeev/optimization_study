import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import Bounds
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#We'll search a minimal satisfiing control vector.
def opt_fun(u):
    return np.dot(u, u)

# derivatives of the function been optimized.
def opt_der(u):
    der = np.empty_like(u)
    der[0] = 2 * u[0]
    der[1] = 2 * u[1]
    der[2] = 2 * u[2]
    der[3] = 2 * u[3]
    return der

# Define a random orthogonal actuators installation matrix.
A = np.eye(3, 4)
A[:, 3] = np.full(3, 1.0)

# Define random desired torque.
M = np.random.randn(3)*0.03

# Constraints:
eq_cons = {'type': 'eq', 'fun': lambda u: np.dot(A, u) - M, 'jac': lambda x: A}

#We'll find u0, u1, u2 between [-0.03, 0.03]
bounds = Bounds(np.full(4, -0.03), np.full(4, 0.03))

# Initial aproximation to solution
u0 = np.zeros(4)

res = minimize(opt_fun, u0, method='SLSQP', jac=opt_der,
    constraints=[eq_cons], options={'disp': True, 'ftol': 1.0e-8},
    bounds=bounds, tol=1.0e-12)

# Optimization terminated successfully.    (Exit mode 0)
#             Current function value: 0.00022926613141697832
#             Iterations: 3
#             Function evaluations: 4
#             Gradient evaluations: 3

print('final u: ', res.x)
print('result torque: ', np.dot(A, res.x))
print('desired torque: ', M)
