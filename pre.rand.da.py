#!/usr/bin/env python
# coding: utf-8
'''
MOGA data preprocessing for dynamic aperture
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2020
Last Update: 03/17/2020
'''

# load libs
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle

# load files
path = '/home/ylu/data/moga_random/'
fs = os.listdir(path)
data = []

# read files
for f in fs:
    tmp_df = pd.read_csv(path+f, header=None, sep='\t')
    data.append(tmp_df)
df = pd.concat(data, ignore_index=True, sort =False)

# get X and Y
data = df.to_numpy()
X = data[:,1:3]
Y_t = data[:,3:5]
Y = np.mean(Y_t, axis=1)

# split data into training set and test set
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(X, Y, test_size=0.20, random_state=2020)

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
pickle.dump(moga, open("rand.da.0317.pkl", "wb" ))