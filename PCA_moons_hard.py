import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as la
import scipy as sc
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# See this article:
# https://en.wikipedia.org/wiki/Kernel_principal_component_analysis
# A good explanation on kernelized PCA also available on Sebastian Rashka's book.

#number of samples
N=500

#toy data to clasify. Non linearly separable.
X, Y = make_moons(n_samples=N, noise=0.07)
x_tr, x_tst, y_tr, y_tst = \
    train_test_split(X, Y, test_size=0.3, random_state=42)

#kernel function to be used for kernel trick.
def rbf(x, y, gamma=2):
    d = x-y
    return np.exp(-1*gamma*np.dot(d, d))

#number of elements in train dataset.
n = x_tr.shape[0]

#kernelized covariation matrix, without Phi(x) computation.
#Kernel trick.
K = np.empty((n, n))
for i in range(n):
    for j in range(n):
        K[i, j] = rbf(x_tr[i], x_tr[j])
# Centralized covariation matrix.
# https://en.wikipedia.org/wiki/Centering_matrix
O = np.full((n, n), 1.0/n)
K = K - O@K - K@O + O@K@O

#eigen directions in kernel space.
w, v = la.eigh(K)

#select only dims most important directions.
important_dirs = 10
v = v[:, -1*important_dirs:]

#features projections onto most important directions. (high dimensional space)
def features_projections(x_lowdim, x_train, eig):
    dims = eig.shape[1]
    N = x_train.shape[0]
    M = x_lowdim.shape[0]
    x_highdim = np.zeros((x_lowdim.shape[0], dims))
    for m in range(M):
        for k in range(dims):
            for i in range(N):
                x_highdim[m, k] += v[i, k]*rbf(x_train[i], x_lowdim[m])
    return x_highdim

#create high dimensional train and test features.
x_train_phi = features_projections(x_tr, x_tr, v)
x_test_phi = features_projections(x_tst, x_tr, v)

#make a simple linear model.
clf = SVC(kernel='linear',  C=0.025)
clf.fit(x_train_phi, y_tr)

#evaluate model on train dataset (just for test on correctness)
y_tr_eval = clf.predict(x_train_phi)

#evaluate model on test data.
y_tst_eval = clf.predict(x_test_phi)

plt.scatter(x_tr[y_tr_eval==0, 0], x_tr[y_tr_eval==0, 1], color='red')
plt.scatter(x_tr[y_tr_eval==1, 0], x_tr[y_tr_eval==1, 1], color='blue')
plt.scatter(x_tst[y_tst_eval==0, 0], x_tst[y_tst_eval==0, 1], color='green')
plt.scatter(x_tst[y_tst_eval==1, 0], x_tst[y_tst_eval==1, 1], color='black')
plt.show()
