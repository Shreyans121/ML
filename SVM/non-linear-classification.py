# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 


# %%
X, Y = make_circles(n_samples=500, noise=0.05)


# %%
X.shape, Y.shape


# %%
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.show()


# %%
def phi(X):
    X_1 = X[:, 0]
    X_2 = X[:, 1]
    X_3 = X_1**2 + X_2**2

    X_ = np.zeros((X.shape[0], 3))

    X_[:, 0] = X_1  
    X_[:, 1] = X_2
    X_[:, 2] = X_3

    return X_


# %%
X_ = phi(X)


# %%
X_


# %%
def plot3D(X):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    X1 = X[:, 0]
    X2 = X[:, 1]
    X3 = X[:, 2]

    ax.scatter(X1, X2, X3, zdir='z', s=20, c=Y, depthshade = True)
    plt.show()

    return ax


# %%
plot3D(X_)


# %%



