# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:45:56 2020

@author: Administrator
"""

import torch
# import torch.nn as nn

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in).to("cuda")
y = torch.randn(N, D_out).to("cuda")

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H), # w1*x + b
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)
'''
torch.nn.init.normal_(model[0].weight)
torch.nn.init.normal_(model[2].weight)
'''
model = model.to("cuda")

learning_rate = 8e-4

loss_fn = torch.nn.MSELoss(reduction='sum')

for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    print(t, loss.item())
    
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad
    model.zero_grad()

'''
print(model[0])
print(model[0].weight)
print(model[0].bias)
'''




