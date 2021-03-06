#!/usr/bin/env python
# coding: utf-8
'''
MOGA data preprocessing
Reduce to 43.37% of 1124 data 
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2019
Last Update: 12/03/2019
'''

# load libs
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle

# load files
path = '/home/ylu/data/dat/'
fs = os.listdir(path)
data = []

# read files
for f in fs:
    tmp_df = pd.read_csv(path+f, header=None)
    data.append(tmp_df)
df = pd.concat(data, ignore_index=True, sort =False)

# get X and Y
data = df.to_numpy()
X = data[:,20:31]
Y_t = data[:,15:17]
Y = np.mean(Y_t, axis=1)

# split data into training set and test set
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X, Y, test_size=0.4337, random_state=2019)
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(x_test_1, y_test_1, test_size=0.20, random_state=2019)

# data normalization to [0, 1]
x_mean = np.mean(x_train_o, axis=0)
x_std = np.std(x_train_o, axis=0)

x_train = (x_train_o - x_mean) / x_std
x_test = (x_test_o - x_mean) / x_std
y_train = y_train_o
y_test = y_test_o 

moga = {
    "x_train": x_train,
    "x_test": x_test,
    "y_train": y_train,
    "y_test": y_test,
    "x_mean": x_mean,
    "x_std": x_std,
}

# save data
pickle.dump(moga, open("moga.1203.test1.pkl", "wb" ))