import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from src.IP import *

def get_activation(activation):
    activation_fn = {
        'StepSigmoid': StepSigmoid,
        'ReLU': nn.ReLU,
        'Sigmoid': nn.Sigmoid,
        'CReLU': CReLU,
    }
    if activation in activation_fn:
        return activation_fn[activation]()
    else:
        raise NotImplementedError(f"{activation} not supported")

class StepSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        sigmoid_backward = input.sigmoid() * (1 - input.sigmoid())
        return grad_input * sigmoid_backward

class StepSigmoid(nn.Module):
    def forward(self, input):
        return StepSigmoidFunction.apply(input)

class FloorFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input

class Floor(nn.Module):
    def forward(self, input):
        return FloorFunction.apply(input)


class CReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0, max=6)/6

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        grad_input[input > 6] = 0
        return grad_input

class CReLU(nn.Module):
    def forward(self, input):
        return CReLUFunction.apply(input)
