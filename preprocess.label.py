#!/usr/bin/env python
# coding: utf-8
'''
MOGA data preprocessing, extract label of each input
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2019
Last Update: 11/24/2019
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
data0 = df.to_numpy()
data1 = data0[data0[:,-3] < 0.1]
data = data1[data1[:,-3] > -1.1]

X = data[:,20:31]
Y = abs(data[:,-3])
Y = Y.astype(int)

# split data into training set and test set
x_train_o, x_test_o, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=2019)

# data normalization to [0, 1]
x_mean = np.mean(x_train_o, axis=0)
x_std = np.std(x_train_o, axis=0)

x_train = (x_train_o - x_mean) / x_std
x_test = (x_test_o - x_mean) / x_std

moga = {
    "x_train": x_train,
    "x_test": x_test,
    "y_train": y_train,
    "y_test": y_test,
    "x_mean": x_mean,
    "x_std": x_std,
}

# save data
pickle.dump(moga, open("moga.label.0121.pkl", "wb" ))