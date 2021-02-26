# In this example we'l solve the same problem as in trust_constr.py:
# Given an actuators installation matrix A, and penalty bilinear form W,
# find a control vector to produce desired torque M (or force F).
# So that:
# A * u - M = 0
# In order to minimize:
# u.T * W * u -> min
# This method works much faster than the general optimization methods,
# so is based on straightforward analitical solution.

#In math terms, we have a system:
# \begin{cases}
# g(u) = A \cdot u - M \\
# f(u) = u^T \cdot W \cdot u
# \end{cases}
# The system Lagrangian is following:
# \mathcal L = f(u) - g(u)^T \cdot \lambda
# In order to solve the constrained system,
# we need to sort the next system of equations, respect to u:
# \begin{cases}
# (W+W^T) \cdot u - A^T \cdot \lambda = 0 \\
# A \cdot u -M = 0
# \end{cases}
# The control vector will be:
# u = (W+W^T)^{-1} \cdot A^T \cdot (A \cdot (W+W^T)^{-1} \cdot A^T)^{-1} \cdot M

import numpy as np
from numpy.linalg import inv

#installation matrix: tetrahedron
A = np.eye(3, 4)
A[:, 3] = np.full(3, 1.0/np.sqrt(3))

#Penalty matrix: use all the actuators equally:
W = np.eye(4)

#We can disable the second actuator for instance, with:
#W = np.diag(1, 1000000, 1, 1)

#set a random desired torque:
M = np.random.randn(3)

#finally calculate the control vector and obtained torque:
u = inv(W+W.T) @ A.T @ inv(A@ inv(W+W.T)@ A.T) @ M
M_ = A @ u

print('optimal control vector: ', u)
print('torque error: ', M - M_)
