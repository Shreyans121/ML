import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

X = pd.read_csv('xdata.csv')
Y = pd.read_csv('ydata.csv')

x = X.values
y = Y.values
x = x[:,1:]
y = y[:,1:].reshape((-1,))

print(x.shape,y.shape)

plt.style.use('seaborn')
plt.scatter(x[:,0],x[:,1],c = y)
plt.show()

query_x = np.array([2,3])
plt.scatter(query_x[0],query_x[1],c='red')
plt.scatter(x[:,0],x[:,1],c = y)
plt.show()

def distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

def knn(x,y,query,k=5):
    dist = []
    m = x.shape[0]
    for i in range(m):
        d = distance(query,x[i])
        dist.append((d,y[i]))
    dist = sorted(dist)
    dist = dist[:k]
    vals = np.array(dist)
    new_vals = np.unique(vals[:,1],return_counts=True)
    idx = np.argmax(new_vals[1])
    return idx

pred = knn(x,y,query_x)
print(int(pred))
