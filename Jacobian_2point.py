import numpy as np

#in this example wе'll numerically compute the Jacobian
#for a nonlinear vector function.

#straightforward non accurate algorithm
#This algorithm uses finite differences:
# df = J * dx
# J[i][j] = df[i] / dx[j]; dk[k] = 0 (k!=j)
def jacobian_2points(fun, x0, delta=1.0e-5):
    f_x0 = fun(x0)
    m = len(f_x0)
    n = len(x0)
    J = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            dx = np.zeros_like(x0)
            dx[j] = delta
            J[i, j] = (fun(x0+dx) - fun(x0))[i]/delta
    return J

if __name__ == '__main__':
    A1 = np.random.randn(3, 4)
    x0 = np.zeros(4)
    #example one: Linear operator.
    def fun1(x):
        return A1 @ x

    jac1 = jacobian_2points(fun1, x0)
    print('jac1 obtained: ', jac1)
    print('original jac: ', A1)
    # jac1 obtained:  [[ 0.14716741 -0.86943084 -0.04878433 -0.24217711]
    #  [ 1.55359453 -0.90044994  0.71382048 -1.44701175]
    #  [-0.66301039  0.40495456  0.6987578   1.37086411]]
    # original jac:  [[ 0.14716741 -0.86943084 -0.04878433 -0.24217711]
    #  [ 1.55359453 -0.90044994  0.71382048 -1.44701175]
    #  [-0.66301039  0.40495456  0.6987578   1.37086411]]

    A2 = np.random.randn(4, 4)
    A2 = A2 + A2.T
    x0 = np.random.randn(4)
    #example two: bilinear function.
    def fun2(x):
        return np.array([x.T @ A2 @ x])

    jac2 = jacobian_2points(fun2, x0)
    print('jac2 obtained: ', jac2)
    print('original jac:', 2 * A2 @ x0)
    # The output may vary.
    # jac2 obtained:  [[-1.99956821  5.57034706 -3.30215737  1.21133863]]
    # original jac: [-1.99956693  5.57036051 -3.30214894  1.21129958]
