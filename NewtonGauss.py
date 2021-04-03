import numpy as np
from numpy.linalg import inv
from matplotlib import pyplot as plt

# This is the direct implementation of
# https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm
# For the a*cos(b*x+c) fitting problem, the initial parameters guess
# must be close enough to the optimal ones.
# In this sense SGD is more stable on this example problem.

#estimate a function gradient if gradient is not given
def estimate_grad(fun, step_size=0.001):
    def decorated_fun(x, theta):
        d_theta = np.ones((theta.size, theta.size))*theta+np.eye(theta.size)*step_size
        d_fun = [(fun(x, d_theta_i)-fun(x, theta))/step_size \
                 for d_theta_i in d_theta]
        return np.array(d_fun)
    return decorated_fun

#get an iterator of size batch_size, returning x, y.
def get_batch(x, y, batch_size):
     idx = np.arange(x.shape[0])
     np.random.shuffle(idx)
     for i in idx[:batch_size]:
         yield (x[i], y[i])

#test on the entire test dataset.
def test_all(x_test, y_test, theta, fun):
    total_loss = 0
    test_dset = get_batch(x_test, y_test, x_test.shape[0])
    for x_i, y_i in test_dset:
        total_loss += (y_i - fun(x_i, theta))**2
    return total_loss / x_test.shape[0]

#This is a stochastic Newton-Gauss method, accepting lr and momentum parameters.
#It shuffles the dataset on each batch.
def train_batch(x_train, y_train, theta, fun, gradfun, lr, momentum, batch_size, dtheta):
    train_batch = get_batch(x_train, y_train, batch_size)
    jac = np.empty((min(x.shape[0], batch_size), theta.shape[0]))
    r = np.empty(min(x.shape[0], batch_size))
    for idx, (x_i, y_i) in enumerate(train_batch):
        r[idx] = y_i - fun(x_i, theta)
        jac[idx] = -1*gradfun(x_i, theta)
    dtheta *= momentum
    dtheta += lr * inv(jac.T @ jac) @ jac.T @ r
    theta -= dtheta


def minimize(x_train, y_train, x_test, y_test, theta, fun, gradfun=None,
             lr=0.01, momentum=0.9,
             num_epochs=100, batch_size = None, test_any=1,
             weight_decay=0.9, decr_any=5):
    if gradfun is None:
        gradfun = estimate_grad(fun)
    if batch_size is None:
        batch_size = x_train.shape[0]
    prev_loss = 0
    num_bad_loss = 0
    dtheta = np.zeros_like(theta)
    for epoch in range(num_epochs):
        train_batch(x_train, y_train, theta, fun, gradfun, lr, momentum, batch_size, dtheta)
        if not (epoch+1) % test_any:
            train_loss = test_all(x_train, y_train, theta, fun)
            test_loss =  test_all(x_test, y_test, theta, fun)
            print(f'epoch {epoch}: train_loss={train_loss}, test_loss={test_loss}')
            print(lr)
            if train_loss > prev_loss:
                num_bad_loss += 1
            if not (num_bad_loss+1)%decr_any:
                lr = lr * weight_decay
            prev_loss = train_loss


if __name__ == '__main__':

    #simple a*cos(bx+c) function to be optimized.
    #we searching for a, b, and c parameters.
    def y_fun(x_i, theta):
        return theta[0]*np.cos(theta[1]*x_i+theta[2])

    #loss function for the y_fun.
    def grad_fun(x_i, theta):
        return np.array([np.cos(theta[1]*x_i+theta[2]),
                         -1*theta[0]*np.sin(theta[1]*x_i+theta[2])*x_i,
                         theta[0]*np.sin(theta[1]*x_i+theta[2])])

    x = np.linspace(0, 10, 1000)
    phi_opt = 10*np.random.rand(3)
    y = y_fun(x, phi_opt)
    y_data = y + 0.3*(np.random.rand(y.shape[0])-0.5)

    train_part = 0.9
    # attention: The initial parameters estimation need to be close enough
    # to optimal ones.
    phi = phi_opt + 0.9*(np.random.rand(3)-0.5)
    x_train = x[:int(x.shape[0]*train_part)]
    x_test = x[int(x.shape[0]*train_part):]
    y_train = y_data[:int(y.shape[0]*train_part)]
    y_test =  y_data[int(y.shape[0]*train_part):]
    minimize(x_train, y_train, x_test, y_test, phi, y_fun,
                lr=0.05, momentum=0.9, weight_decay=0.9,
                decr_any=100, num_epochs=500, batch_size=100)

    y_est = y_fun(x, phi)
    print('parameters found: ', phi)

    plt.plot(x, y, color='green')
    plt.plot(x, y_data, color='blue')
    plt.plot(x, y_est, color='red')
    plt.show()
