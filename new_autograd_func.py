# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 09:19:33 2020

@author: Administrator
"""

import torch

class MyReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min = 0)
    
    @staticmethod
    def backward(ctx, grad_outputs):
        input = ctx.saved.tensors
        grad_input = grad_outputs.clone()
        grad_input[input < 0] = 0
        return grad_input