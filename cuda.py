# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 10:01:38 2020
cuda
@author: Administrator
"""
import torch
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5, device=device)
    z = x + y
    print(z)
    print(z.to('cpu'), torch.double)
    
    #先放到CPU再转成numpy, (numpy不支持CPU)
    print(y.to('cpu').data.numpy())
    print(y.cpu().data.numpy())
    
    
    # 模型直接放到GPU
#   model = model.cuda