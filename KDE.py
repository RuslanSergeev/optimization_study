import numpy as np
import matplotlib.pyplot as plt

# In this example we use the KDE to estimate the probability density
# function of a randomly generated data.
# We use a triangular distribution to generate the data.
# The intuition around the KDE: we can think of each data point as
# a small normal distribution centered at that point.
# The KDE is the sum of all these normal distributions: how likely
# is the given point was produced by all the distributions
# placed at data points.
# The bandwidth is the standard deviation of the normal distribution.
# The kernel we use at each data-point to approximate the distribution
# is a normal distribution by default, but can be any other distribution.
# Small bandwidths (narrow distributions) will better fit the data,
# but will be more sensitive to noise.

def p_std(mu, sigma):
    ''' Given mean and standard deviation, return a function that
        computes the probability density function of a normal
        distribution with that mean and standard deviation.
    '''
    def p(x):
        return 1/np.sqrt(2*np.pi*(sigma**2))*np.exp(-1*((x-mu)**2)/(2*(sigma**2)))
    return p


def kde(m, k=p_std(0, 1), h=1.0):
    ''' Given a list of numbers, return a function that computes
        the kernel density estimate of those numbers using a
        given kernel with bandwidt h.

        By default, the probability density function is the standard
        normal distribution and the bandwidth is 1.0.
    '''
    def K(x):
        return np.sum(k((x[:, np.newaxis]-m)/h), axis=1)/(len(m)*h)
    return K

def test_kde():
    m = np.random.triangular(-3, 0, 8, 200)
    K01 = kde(m, h=0.1)
    K05 = kde(m, h=0.5)
    K10 = kde(m, h=1.0)
    x = np.linspace(-10, 10, 1000)
    y01 = K01(x)
    y05 = K05(x)
    y10 = K10(x)
    plt.plot(x, y01, label='h=0.1')
    plt.plot(x, y05, label='h=0.5')
    plt.plot(x, y10, label='h=1.0')
    plt.legend(title='Bandwidth comparison')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    test_kde()
