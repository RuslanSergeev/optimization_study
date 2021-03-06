import scipy as sp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# In this example will fit a cos(t) into
# set of data measured with noise.
# Data source can be Hall sensor for instance.
# This snippet will use BFGS algorithm with no gradient given.
# BFGS estimates the gradient by finite differences if it is not given.
# Even with estimated gradient, algorithm converges faster than Nelder-Mead.

# Initial parameters and time series.
a = 3.0
b = 1.5
phi = 2*np.pi/3
t = np.linspace(0, 2*np.pi, num=100)

# compute the cos with given parameters.
# a*cos(t + phi) + b
def yfun(p, t):
    return p[0]*np.cos(t + p[1]) + p[2]

# compute the error of fitting the cos with
# parameters into a set of data-points.
def err(p):
    # p[0] = a, p[1] = phi, p[2] = b
    y_cur = yfun(p, t)
    return sum((y_cur - o)**2)

#define ideal and noised measurements
y = yfun(np.array([a, phi, b]), t)
o = y + np.random.randn(*t.shape)*0.1

#initial aproximation to parameters.
p0 = np.array([2.0, np.pi / 6, 0.0])

#optimization via BFGS with no gradient given.
res = minimize(err, p0, method='BFGS',
   options={'disp': True})

# Optimization terminated successfully.
#          Current function value: 0.957516
#          Iterations: 11
#          Function evaluations: 70
#          Gradient evaluations: 14
# [2.99116415 2.0883387  1.48700633]

print(res.x)
oopt = yfun(res.x, t)

#plot measurements, fit curve and the ideal curve
plt.plot(t, o, 'green')
plt.plot(t, oopt, 'red')
plt.plot(t, y, 'blue')
plt.show()
