# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 12:48:07 2020
用numpy手工bp实现只有1个隐藏层的NN
@author: Administrator
"""

import numpy as np
# 64个 1000 维的数据, 隐藏层有100个神经元, 10个输出
N, D_in, H, D_out = 64, 1000, 100, 10

# 随机生成 输入输出数据
x = np.random.randn(N, D_in);
y = np.random.randn(N, D_out);

# 随机初始化weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

# 设定参数更新速率系数，或叫学习率
learning_rate = 1e-6


for it in range(500):
    
    # forward pass(省略bias)
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    
    # compute loss
    loss = np.square(y_pred - y).sum()
    print(it, loss)
    
    # backforward pass
    # compute the gradients
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)
    
    # update w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2




























