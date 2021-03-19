import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, Y = make_moons(n_samples=500, noise=0.1)
x_tr, x_tst, y_tr, y_tst = \
    train_test_split(X, Y, test_size=0.3, random_state=42)

clf = SVC(gamma=2, C=1)
clf.fit(x_tr, y_tr)

y_tr_eval = clf.predict(x_tr)
y_tst_eval = clf.predict(x_tst)
plt.scatter(x_tr[y_tr_eval==0, 0], x_tr[y_tr_eval==0, 1], color='red')
plt.scatter(x_tr[y_tr_eval==1, 0], x_tr[y_tr_eval==1, 1], color='blue')
plt.scatter(x_tst[y_tst_eval==0, 0], x_tst[y_tst_eval==0, 1], color='green')
plt.scatter(x_tst[y_tst_eval==1, 0], x_tst[y_tst_eval==1, 1], color='yellow')

plt.show()
