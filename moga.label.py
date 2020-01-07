#!/usr/bin/env python
# coding: utf-8
'''
MOGA data label classification
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2020
Last Update: 01/07/2020
'''

# load libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

# definition of a simple fc model [11, 32, 64, 2]
class MOGANet(nn.Module):
    def __init__(self):
        super(MOGANet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(11, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x

class MOGADataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
    
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
# define train and test functions
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    correct = 0
    acc = 0
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # compute output
        outputs = model(x)
        loss = criterion(outputs, y)
        train_loss += loss.item()
        pred = outputs.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(y).sum().item()
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader)
    acc = 100. * correct / len(train_loader.dataset)
    print('Train set: Average loss:\t'
          '{:.3f}\t'
          'Accuracy: {}/{}\t'
          '{:.3f}'.format(train_loss, correct, len(train_loader.dataset), acc))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            # compute output
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            pred = outputs.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(y).sum().item()
            
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss:\t'
          '{:.3f}\t'
          'Accuracy: {}/{}\t'
          '{:.3f}'.format(test_loss, correct, len(test_loader.dataset), acc))
    
    return test_loss
    
# Use CUDA
#use_cuda = torch.cuda.is_available()
device = torch.device("cpu")

random.seed(2020) 
torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)

# load files
moga = pickle.load(open("moga.label.pkl", "rb"))
x_train, x_test = moga["x_train"], moga["x_test"]
y_train, y_test = moga["y_train"], moga["y_test"]

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test)

# Parameters
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 4}

# dataloader
train_data = MOGADataset(x_train, y_train)
train_loader = DataLoader(dataset=train_data, **params)

test_data = MOGADataset(x_test, y_test)
test_loader = DataLoader(dataset=test_data, **params)

model = MOGANet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5)

# Training here
for t in range(100):
    train(model, device, train_loader, optimizer, criterion)
    test_loss = test(model, device, test_loader, criterion)
    scheduler.step(test_loss)
        
# save trained model
torch.save(model.state_dict(), 'moga-label-01072020.pth')

model.eval()
with torch.no_grad():
    Yp = model(x_test)
    pred = Yp.max(1)[1] # get the index of the max log-probability
    print(pred.eq(y_test).sum().item() / len(Yp))