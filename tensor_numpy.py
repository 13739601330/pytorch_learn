# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 09:48:54 2020
tensor 和 numpy的转换

@author: Administrator
"""

import torch
import numpy as np

# 共享内存空间
a = np.ones(5)
b = torch.from_numpy(a)
torch.add(b, 1, out=b)
#print(a)

# 重新创建变量，分配内存空间
c = np.ones(5)
d = torch.from_numpy(c)
d = d + 1 
print(c)
print(d)
