# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:14:19 2020

@author: Administrator
"""
import torch
N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in);
y = torch.randn(N, D_out);

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-6

for it in range(500):
    '''
    h = x.mm(w1)
    h_relu = h.clamp(min = 0)
    y_pred = h_relu.mm(w2)
    '''
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()
    print(it, loss.item())
    
    
    '''    
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)
    '''
    
    # 将最后更新w1, w2时的计算过程屏蔽，不算梯度
    with torch.no_grad():
        loss.backward()
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # 清0梯度
        w1.grad.zero_()
        w2.grad.zero_()