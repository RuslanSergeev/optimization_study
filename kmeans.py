import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, dist=None):
        if dist is None:
            self.dist = lambda x, y: (x-y) @ (x-y)
        self.k = k

    def fit_predict(self, x, max_iterations=50):
        #potential noncritical error here.
        y = np.random.randint(self.k, size=x.shape[0])
        for i in range(max_iterations):
            self._calc_centers(x, y)
            y = self._regroup(x)
        return y

    def _calc_centers(self, x, y):
        self.centers = np.array([x[y == i].mean(0) for i in range(self.k)])

    def _regroup(self, x):
        get_cent = lambda vec: min(range(self.k), \
            key=lambda j: self.dist(self.centers[j], vec))
        return np.array(list(map(get_cent, x)))


if __name__ == "__main__":
    num_clusters = 5
    num_samples = 2000
    c = np.random.rand(num_clusters, 2)*20
    #also a potential noncritical error
    y = np.random.randint(num_clusters, size=num_samples)
    x = np.array([c[y[i]]+3*(np.random.rand(2)-0.5) for i in range(num_samples)])

    km = KMeans(num_clusters)
    yp = km.fit_predict(x)

    for i in range(num_clusters):
        plt.scatter(x[yp==i, 0], x[yp==i, 1])
    plt.show()
