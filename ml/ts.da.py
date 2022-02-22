#!/usr/bin/env python
# coding: utf-8
'''
Converting to Torch Script via Tracing
Run trained model for Dynamic Aperture
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2021
Last Update: 01/12/2021
'''

# load libs
import sys 
import torch
import torch.nn as nn
import numpy as np

# definition of a simple fc model [2, 32, 64, 1]
class MOGANet(nn.Module):
    def __init__(self):
        super(MOGANet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x

# Use CUDA
device = torch.device("cpu")
model = MOGANet().to(device)
criterion = nn.MSELoss()
model.load_state_dict(torch.load("da.1208.pth"))
model.eval()

x_mean = np.array([13.954896, 9.5851928, 11.948139, 15.474422, 15.717001, 15.650609, -14.815044, -2.3404391, -7.1923663, -57.846782, -182.95906])
x_std = np.array([3.24855981e-01, 4.78972654e-01, 8.30422947e-01, 2.37172289e-01, 2.00656458e-01, 3.35015730e-01, 3.68295868e-01, 2.26876614e-01, 2.19417668e-01, 2.72298625e+02, 3.30543704e+02])

a = 13
b = 9
c = 12
d = 15
e = 15
f = 15
g = -14
h = -2
i = -7
j = -13
k = -31

sample = np.array([a, b, c, d, e, f, g, h, i, j, k])
sample = (sample - x_mean) / x_std
x = torch.from_numpy(sample).float()

traced_script_module = torch.jit.trace(model, x)
traced_script_module.save("da.1208.pt")