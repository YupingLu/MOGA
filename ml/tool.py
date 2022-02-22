#!/usr/bin/env python
# coding: utf-8
'''
MOGA data preprocessing
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2020
Last Update: 10/12/2020
'''

# load libs
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import pickle

# load files
path = './dat/'
fs = os.listdir(path)
data = []

# read files
for f in fs:
    tmp_df = pd.read_csv(path+f, header=None)
    data.append(tmp_df)
df = pd.concat(data, ignore_index=True, sort =False)

# get X and Y
data = df.to_numpy()
x = data[:, -3]
data = data[x == 0]
X = data[:,20:31]
Y_t = data[:,15:17]
Y = np.mean(Y_t, axis=1)

# split data into training set and test set
x_train_o, x_test_o, y_train_o, y_test_o = train_test_split(X, Y, test_size=0.20, random_state=2020)

# data normalization to [0, 1]
x_mean = np.mean(x_train_o, axis=0)
x_std = np.std(x_train_o, axis=0)
print(x_mean)
print(x_std)

print("\n")

s1 = "x_mean = np.array(["
s2 = "x_std = np.array(["

for i in x_mean:
    s1 += "{:.8}".format(i) + ", "
s1 = s1[:-2]

for i in x_std:
    s2 += "{:.8e}".format(i) + ", "
s2 = s2[:-2]

s1 += "])"
s2 += "])"

print(s1)
print(s2)

print("\n")

pool = ["aa", "bb", "cc", "dd", "ee", "ff", "gg", "hh", "ii", "jj", "kk"]
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for idx in range(len(x_mean)):
    print("    float " + pool[idx] + " = (xvar[" + str(nums[idx]) + "] - " + "{:.8}".format(x_mean[idx]) + ") / " + "{:.8e}".format(x_std[idx]) + ";")
    