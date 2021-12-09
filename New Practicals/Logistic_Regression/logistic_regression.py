import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

x_train = pd.read_csv("Diabetes_XTrain.csv")
y_train = pd.read_csv("Diabetes_YTrain.csv")
x_test = pd.read_csv("Diabetes_Xtest.csv")

print(x_train.shape,y_train.shape,x_test.shape)

x = x_train.values
y = y_train.values
xt = x_test.values

u = x.mean(axis=0)
std = x.std(axis=0)
x = (x-u)/std
xt = (xt-u)/std

def sigmoid(z):
    return 1.0/(1 + np.exp(-z))

def hypothesis(X,theta):
    return sigmoid(np.dot(X,theta))

def error(X,y,theta):
    hi = hypothesis(X,theta)
    e = -1.0 * np.mean((y*np.log(hi)) + ((1-y)*np.log(1-hi)))
    return e

def gradient(X,y,theta):
    hi = hypothesis(X,theta)
    grad = np.dot(X.T,(hi-y))
    return grad/X.shape[0]

def gradient_descent(X,y,lr=0.5,epoch=300):
    n = X.shape[1]
    theta = np.zeros((n,1))
    error_list = []
    for i in range(epoch):
        err = error(X,y,theta)
        error_list.append(err)
        grad = gradient(X,y,theta)
        theta = theta - lr*grad
    return theta, error_list

ones = np.ones((x.shape[0],1))
x_ntrain = np.hstack((ones,x))
Y_train = y.reshape((-1,1))

theta,errorList = gradient_descent(x_ntrain,Y_train)

plt.style.use('seaborn')
plt.plot(errorList)
plt.show()

