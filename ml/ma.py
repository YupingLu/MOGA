#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load libs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import matplotlib.pyplot as plt
import pickle


# In[ ]:


# definition of a simple fc model [11, 32, 64, 1]
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

class MOGADataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
    
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])


# In[ ]:


# define train and test functions
def train(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        # compute output
        outputs = model(x)
        loss = criterion(outputs, y)
        train_loss += loss.item()
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader)
    return train_loss

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            
            # compute output
            outputs = model(x)
            loss = criterion(outputs, y)
            test_loss += loss.item()
            
    test_loss /= len(test_loader)
    return test_loss


# In[ ]:


# Use CUDA
device = torch.device("cpu")

random.seed(2020) 
torch.manual_seed(2020)
torch.cuda.manual_seed_all(2020)

# load files
moga = pickle.load(open("ma.1208.pkl", "rb"))
x_train, x_test = moga["x_train"], moga["x_test"]
y_train, y_test = moga["y_train"], moga["y_test"]
x_mean, x_std = moga["x_mean"], moga["x_std"]

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train.reshape(-1,1)).float()
x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test.reshape(-1,1)).float()


# In[ ]:


# Parameters
params = {'batch_size': 128,
          'shuffle': True,
          'num_workers': 4}

# dataloader
train_data = MOGADataset(x_train, y_train)
train_loader = DataLoader(dataset=train_data, **params)

test_data = MOGADataset(x_test, y_test)
test_loader = DataLoader(dataset=test_data, **params)


# In[ ]:


model = MOGANet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8)


# In[ ]:


# Training here
for t in range(500):
    train_loss = train(model, device, train_loader, optimizer, criterion) 
    test_loss = test(model, device, test_loader, criterion)
    if t%10 == 0:
        print(t, train_loss, test_loss)
    
    scheduler.step(test_loss)


# In[ ]:


# save trained model
torch.save(model.state_dict(), 'ma.1208.pth')
#model.load_state_dict(torch.load("ma.1208.pth"))


# In[ ]:


model.eval()
Yp = model(x_test)
print("test MSE from criterion")
MSE = criterion(Yp, y_test).item()
print(MSE)

Yp = Yp[:,0].detach().numpy()
y_test = y_test[:,0].detach().numpy()

delta_y = Yp - y_test
RMSE = np.sqrt(MSE)
print("test RMSE of original data")
print(RMSE)


# In[ ]:


# plot RMSE
plt.figure(figsize=(6,4))
num_bins = 200
plt.hist(delta_y, num_bins, facecolor='blue')
plt.xlabel('Prediction Error')
plt.xlim(-0.0015,0.0015)
plt.grid(True)
#plt.title('RMSE: %.4f, N=%d' % (RMSE,len(x_test)))
plt.title('RMSE: '+'{:.2e}'.format(RMSE)+', N=%d' % (len(x_test)))
plt.savefig('ma1.png')


# In[ ]:


# check mean ans std
print(np.std(y_test))
print(np.mean(y_test))
print(max(y_test), min(y_test))


# In[ ]:


# Viz Yp and |Error|
l = np.array([x for x in range(len(y_test))])
plt.figure(figsize=(13,4))
plt.title("Blue:Predicted, Red:|Error|")
plt.plot(l, Yp, 'b-', linewidth=1)
plt.plot(l, abs(delta_y), 'r-', linewidth=1)
plt.grid(True)
plt.savefig('ma2.png')


# In[ ]:


# Viz y_test hist
plt.figure(figsize=(6,4))
num_bins = 200
plt.hist(y_test, num_bins, facecolor='blue')
plt.xlabel('Y_test histogram')
plt.grid(True)
plt.savefig('ma3.png')


# In[ ]:




