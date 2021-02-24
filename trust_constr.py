import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import SR1
from scipy.stats import ortho_group
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

# Define a random orthogonal actuators installation matrix.
A = np.eye(3, 4)
A[:, 3] = np.full(3, 1.0)

# Define random desired torque.
M = np.random.randn(3)*0.03*np.sqrt(3)

#We'll find u0, u1, u2 between [-0.03, 0.03]
bounds = Bounds(np.full(4, -0.03), np.full(4, 0.03))

#A*u should be exactly M
lin_constr = LinearConstraint(A, M, M)

#We'll search a minimal satisfiing control vector.
def opt_fun(u):
    return la.norm(u)

def opt_der(u):
    if np.isclose(u, np.zeros_like(u)).all():
        return np.zeros_like(u)
    norm = la.norm(u)
    der = np.zeros_like(u)
    der[0] = -1 * u[0] / norm
    der[1] = -1 * u[1] / norm
    der[2] = -1 * u[2] / norm
    der[3] = -1 * u[3] / norm
    return der

#Initial control vector - not known.
u0 = np.full(4, 0.1)

# optimize with predefined jacobian. (derivative)
# res = minimize(opt_fun, u0, method='trust-constr', jac=opt_der, hess=SR1(),
#     constraints=[lin_constr], bounds=bounds, options={'verbose':1})

# we can also estimate the jacobian and hessian with finite differences.
res = minimize(opt_fun, u0, method='trust-constr', jac='2-point', hess=SR1(),
    constraints=[lin_constr], bounds=bounds, options={'verbose':1})

print('final u: ', res.x)
print('result torque: ', np.dot(A, res.x))
print('desired torque: ', M)
