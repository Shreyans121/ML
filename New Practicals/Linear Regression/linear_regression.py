import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

X,Y = make_regression(n_samples=1000,n_features=20,n_informative=20,noise=0.2,random_state=1)
print(X.shape,Y.shape)

u = np.mean(X,axis = 0)
std = np.std(X,axis = 0)
X = (X-u)/std

ones = np.ones((X.shape[0],1))
X = np.hstack((ones,X))
print(X.shape,Y.shape)

def hypothesis(X,theta):
    return np.dot(X,theta)

def error(X,y,theta):
    e = 0.0
    m = X.shape[0]
    y_ = hypothesis(X,theta)
    e = (np.sum((y-y_)**2))
    return e/m

def gradient(X,y,theta):
    y_ = hypothesis(X,theta)
    grad = np.dot(X.T,(y_-y))
    return grad/X.shape[0]

def gradient_descent(X,y,lr=0.2,epochs=100):
    n = X.shape[1]
    theta = np.zeros((n,))
    errList = []
    for i in range(epochs):
        e = error(X,y,theta)
        errList.append(e)

        grad = gradient(X,y,theta)
        theta = theta - lr*grad
    return theta,errList

theta,errList = gradient_descent(X,Y)

plt.style.use('seaborn')
plt.figure()
plt.plot(np.arange(len(errList)),errList)
plt.show()
