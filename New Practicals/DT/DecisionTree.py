import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")

columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "Embarked"]

data_clean = data.drop(columns_to_drop,axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data_clean["Sex"] = le.fit_transform(data_clean["Sex"])
data_clean = data_clean.fillna(data_clean["Age"].mean())

input_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
output_cols = ["Survived"]

X = data_clean[input_cols]
Y = data_clean[output_cols]

def entropy(col):
    counts = np.unique(col,return_counts=True)
    N = float(col.shape[0])
    ent = 0.0
    for ix in counts[1]:
        p = ix/N
        ent += (-1.0*p*np.log2(p))
    return ent

def divide_data(x_data,fkey,fval):
    x_right = pd.DataFrame([],columns=x_data.columns)
    x_left = pd.DataFrame([],columns=x_data.columns)

    for ix in range(x_data.shape[0]):
        val = x_data[fkey].loc[ix]

        if val > fval:
            x_right = x_right.append(x_data.loc[ix])
        else:
            x_left = x_left.append(x_data.loc[ix])
    return x_left,x_right

def info_gain(x_data,fkey,fval):
    left,right = divide_data(x_data,fkey,fval)
    l = float(left.shape[0]/x_data.shape[0])
    r = float(right.shape[0]/x_data.shape[0])

    if left.shape==0 or right.shape==0:
        return -100000
    i_gain = entropy(x_data.Survived) - (l*entropy(left.Survived) + r*entropy(right.Survived))
    return i_gain
 
for fx in X.columns:
    print(fx)
    print(info_gain(data_clean,fx,data_clean[fx].mean()))