import numpy as np
from matplotlib import pyplot as plt

#estimate a function gradient if gradient is not given
def estimate_grad(loss, step_size=0.001):
    def decorated_fun(x, y, theta):
        d_theta = np.ones((theta.size, theta.size))*theta+np.eye(theta.size)*step_size
        d_fun = [(loss(x, y, d_theta_i)-loss(x, y, theta))/step_size \
                 for d_theta_i in d_theta]
        return np.array(d_fun)
    return decorated_fun

#get an iterator of size batch_size, returning x, y.
def get_batch(x, y, batch_size):
     idx = np.arange(x.shape[0])
     np.random.shuffle(idx)
     for i in idx[:batch_size]:
         yield (x[i], y[i])

#train on batch_size samples of x_train, y_train
#dtheta - is the previous theta increment. used for momentum.
def train_batch(x_train, y_train, theta, gradfun, lr, momentum, batch_size, dtheta):
    train_dset = get_batch(x_train, y_train, batch_size)
    for x_i, y_i in train_dset:
        dtheta*=momentum
        dtheta+=lr*gradfun(x_i, y_i, theta)
        theta -= dtheta

#perform loss calculation all over the test dataset.
def test_all(x_test, y_test, theta, loss):
    total_loss = 0
    test_dset = get_batch(x_train, y_train, x_test.shape[0])
    for x_i, y_i in test_dset:
        total_loss += loss(x_i, y_i, theta)
    return total_loss / x_test.shape[0]


#minimize over the traind dataset.
#any epoch operates on batch_size randomly selected test samples.
def minimize(x_train, y_train, x_test, y_test, theta, loss, gradfun=None,
             lr=0.01, momentum=0.9,
             num_epochs=100, batch_size = 100, test_any=1,
             weight_decay=0.9, decr_any=5):
    if not gradfun:
        gradfun = estimate_grad(loss)
    #perform a simple learning rate scheduling technique
    prev_loss = 0
    num_bad_loss = 0
    #initial direction of theta change.
    dtheta = np.zeros_like(theta)
    for epoch in range(num_epochs):
        train_batch(x_train, y_train, theta, gradfun, lr, momentum, batch_size, dtheta)
        if not (epoch+1) % test_any:
            train_loss = test_all(x_train, y_train, theta, loss)
            test_loss =  test_all(x_test, y_test, theta, loss)
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
    def y_fun(x, theta):
        return theta[0]*np.cos(theta[1]*x+theta[2])

    #loss function for the y_fun.
    def y_loss(x_i, y_i, theta):
        err = y_i - y_fun(x_i, theta)
        return 0.5 * err * err

    #analitical gradient for y_fun.
    def grad_loss(x_i, y_i, theta):
        err = y_i - y_fun(x_i, theta)
        return err * np.array([-np.cos(theta[1]*x_i+theta[2]),
                                theta[0]*np.sin(theta[1]*x_i+theta[2])*x_i,
                                theta[0]*np.sin(theta[1]*x_i+theta[2])])

    x = np.linspace(0, 10, 1000)
    phi_opt = np.array([3, 2, 1])
    y = y_fun(x, phi_opt)
    y_data = y + 0.3*(np.random.rand(y.shape[0])-0.5)

    train_part = 0.9
    phi = np.random.rand(3)
    x_train = x[:int(x.shape[0]*train_part)]
    x_test = x[int(x.shape[0]*train_part):]
    y_train = y_data[:int(y.shape[0]*train_part)]
    y_test =  y_data[int(y.shape[0]*train_part):]
    minimize(x_train, y_train, x_test, y_test, phi, y_loss, lr=1.0e-3,
        num_epochs=200, batch_size=500, decr_any=30)

    y_est = y_fun(x, phi)
    print('parameters found: ', phi)

    plt.plot(x, y, color='green')
    plt.plot(x, y_data, color='blue')
    plt.plot(x, y_est, color='red')
    plt.show()
