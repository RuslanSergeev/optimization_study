import scipy as sp
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

# In this example will fit a cos(t) into
# set of data measured with noise.
# Data source can be Hall sensor for instance.
# This snippet will use BFGS algorithm with gradient function defined.
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

def gradfun(p):
    grd = np.zeros_like(p)
    y_cur = yfun(p, t)
    grd[0] = 2 * np.dot(y_cur-o, np.cos(t+p[1]))
    grd[1] = 2 * np.dot(y_cur-o, -1 * p[0]*np.sin(t+p[1]))
    grd[2] = 2 * np.dot(y_cur-o, np.full_like(y_cur, 1.0))
    return grd

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
res = minimize(err, p0, method='BFGS', jac=gradfun,
   options={'disp': True})

# Optimization terminated successfully.
#          Current function value: 1.106916
#          Iterations: 11
#          Function evaluations: 14
#          Gradient evaluations: 14
# [2.9932548  2.09502411 1.49198071]

print(res.x)
oopt = yfun(res.x, t)

#plot measurements, fit curve and the ideal curve
plt.plot(t, o, 'green')
plt.plot(t, oopt, 'red')
plt.plot(t, y, 'blue')
plt.show()
