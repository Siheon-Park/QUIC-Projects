import logging
import numpy as np
import torch
from torch.autograd import Function
from torch import nn
from typing import Callable
from .qasvm import QASVM

logger = logging.getLogger(__name__)


# noinspection PyMethodOverriding
class GeneralFunction(Function):

    @staticmethod
    def forward(ctx, params, cost_fn: Callable, grad_fn: Callable):
        ctx.cost_fn = cost_fn
        ctx.grad_fn = grad_fn
        evaluation = torch.tensor([ctx.cost_fn(params.tolist())])
        ctx.save_for_backward(params)
        return evaluation

    @staticmethod
    def backward(ctx, grad_output):
        params, = ctx.saved_tensors

        if ctx.grad_fn is not None:
            gradients = ctx.grad_fn(params.tolist())
        else:
            input_list = np.array(params.tolist())
            input_list_rights = input_list + 1e-3 * np.eye(len(input_list))
            input_list_lefts = input_list + 1e-3 * np.eye(len(input_list))

            gradients = (np.array(list(map(ctx.cost_fn, input_list_rights))) - np.array(
                list(map(ctx.cost_fn, input_list_lefts)))) / (2 * 1e-3)

        return grad_output * gradients, None, None


class QASVM_model(nn.Module):
    def __init__(self, qasvm: QASVM) -> None:
        super().__init__()
        self.qasvm = qasvm
        if self.qasvm.initial_point is not None:
            self.params = nn.Parameter(list(self.qasvm.initial_point.values()))
        else:
            self.params = nn.Parameter(torch.randn(self.qasvm.num_parameters))

    def forward(self):
        return GeneralFunction.apply(self.params, self.qasvm.cost_fn, self.qasvm.grad_fn)
