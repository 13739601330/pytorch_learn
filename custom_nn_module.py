# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:23:05 2020

@author: Administrator
"""

import torch
# import torch.nn as nn


class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
    
    def forward(self, x):
        y_pred = self.linear2(self.linear1(x).clamp(min=0))
        return y_pred
    

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).to("cuda")
y = torch.randn(N, D_out).to("cuda")    

model = TwoLayerNet(D_in, H, D_out).to("cuda") 
learning_rate = 1e-4
loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)    
    
for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)  
    print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    
    
    
    
    
    
    
    
    