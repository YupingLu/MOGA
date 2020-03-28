#!/usr/bin/env python
# coding: utf-8
'''
Converting to Torch Script via Tracing
Run trained model for Dynamic Aperture
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2020
Last Update: 03/24/2020
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
model.load_state_dict(torch.load("random_gen/rand.da.0317.pth"))
model.eval()

x_mean = np.array([-0.36242136, 0.10387579])
x_std = np.array([335.14358049, 335.16320865])

a = -500.262
b = -425.707

sample = np.array([a, b])
sample = (sample - x_mean) / x_std
x = torch.from_numpy(sample).float()

traced_script_module = torch.jit.trace(model, x)
traced_script_module.save("rand.da.0317.pt")