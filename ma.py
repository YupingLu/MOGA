#!/usr/bin/env python
# coding: utf-8
'''
Run trained model for Momentum Aperture
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2020
Last Update: 03/02/2020
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
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x

# Use CUDA
device = torch.device("cpu")
model = MOGANet().to(device)
criterion = nn.MSELoss()
model.load_state_dict(torch.load("ma-0211.pth"))
model.eval()

x_mean = np.array([-36.62848469, -395.5610711])
x_std = np.array([86.29351497, 116.12076713])

a = float(sys.argv[1])
b = float(sys.argv[2])

sample = np.array([a, b])
sample = (sample - x_mean) / x_std
x = torch.from_numpy(sample).float()
y = model(x)
print(y.detach().numpy()[0])
